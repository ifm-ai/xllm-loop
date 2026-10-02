from typing import Optional, Tuple, List, Any, Sequence
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from xllm.configuration import ConfStore
from xllm.data.dataset_streamer.tokenizer import Tokenizer
from xllm.modules import (
    MultiheadAttention,
    NormalizedFeedForwardNetwork,
    MOVAttention,
    NormalizedMoE,
    RotaryEmbedding,
    GroupLayerNorm,
    GroupRMSNorm,
)
from xllm.modules.residual import num_params_in_residual
from xllm.modules.fused_ops import memory_efficient_dropout
from xllm.modules.model_parallel import (
    ParallelEmbedding,
    ColumnParallelLinear,
    vocab_parallel_cross_entropy,
    gather_copy_model_parallel_region,
)
from xllm.modules.context_parallel import gather_from_context_parallel_region
from xllm.config import ModelConf
from xllm.models.xllm import XLLModel
from xllm.distributed import (
    get_context_parallel_world_size,
    get_context_parallel_rank,
)
from xllm.models.fused_blocks import (
    TransformerBlockFunction,
    TransformerMoEBlockFunction,
    TransformerMoVABlockFunction,
    TransformerOutputLayerFunction,
)
from xllm.utils import get_init_fn


class TransformerBlock(nn.Module):

    def __init__(self, cfg: ModelConf, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.residual_func = cfg.residual_func
        self.residual_heads = cfg.residual_heads
        self.fused_block = cfg.fused_block
        self.recompute_q = cfg.recompute_q
        self.recompute_kv = cfg.recompute_v
        self.recompute_attention = cfg.recompute_attention
        self.recompute_fc1_out = cfg.recompute_fc1_out
        self.recompute_fc3_out = cfg.recompute_fc3_out
        self.dropout = cfg.dropout
        self.apply_rmsnorm = cfg.apply_rmsnorm
        self.layernorm_eps = cfg.layernorm_eps
        self.rmsnorm_eps = cfg.rmsnorm_eps

        if self.fused_block:
            assert cfg.causal_attn_backend in ['flash', 'xattn'], f"flash or xattn attention backend is required for fused block."

        self.attention = MultiheadAttention(
            layer_id=layer_id,
            mdim=cfg.model_dim,
            n_heads=cfg.num_heads,
            n_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            rope_head_dim=cfg.rope_head_dim,
            attn_act_func=cfg.attn_act_func,
            qknorm=cfg.qknorm,
            norm_num_groups=cfg.layernorm_num_groups,
            norm_affine=cfg.norm_affine,
            layernorm_eps=cfg.layernorm_eps,
            rmsnorm_eps=cfg.rmsnorm_eps,
            memory_efficient_norm=cfg.memory_efficient_norm,
            apply_rmsnorm=cfg.apply_rmsnorm,
            apply_attn_gate=cfg.apply_attn_gate,
            attn_gate_func=cfg.attn_gate_func,
            causal_attn_backend=cfg.causal_attn_backend,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            hidden_dropout=cfg.hidden_dropout,
            residual_func=cfg.residual_func,
            residual_heads=cfg.residual_heads,
            init_mode=cfg.init_mode,
            init_std=cfg.init_std,
        )

        self.nffn = NormalizedFeedForwardNetwork(
            layer_id=layer_id,
            model_dim=cfg.model_dim,
            ffn_hidden_dim=cfg.ffn_hidden_dim,
            swiglu=cfg.swiglu,
            dropout=cfg.dropout,
            hidden_dropout=cfg.hidden_dropout,
            norm_num_groups=cfg.layernorm_num_groups,
            norm_affine=cfg.norm_affine,
            layernorm_eps=cfg.layernorm_eps,
            rmsnorm_eps=cfg.rmsnorm_eps,
            memory_efficient_norm=cfg.memory_efficient_norm,
            apply_rmsnorm=cfg.apply_rmsnorm,
            residual_func=cfg.residual_func,
            residual_heads=cfg.residual_heads,
            init_mode=cfg.init_mode,
            init_std=cfg.init_std,
        )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Optional[Tensor],
        segments: Optional[Any] = None,
        moe_router_load_balancing_type: Optional[str] = None,
        fp32_attn_output: bool = False,
        deterministic: bool = True,
        cache: Optional[Tuple[Tensor, Tensor, int]] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Any]]:
        if self.fused_block and self.training:
            fn = TransformerBlockFunction
            return fn.apply(
                x,
                freqs_cis,
                segments,
                self.attention.norm.weight,
                self.attention.norm.bias,
                self.attention.wq.weight,
                self.attention.wk.weight,
                self.attention.wv.weight,
                self.attention.wr.weight if self.attention.apply_attn_gate else None,
                self.attention.wg.weight if self.attention.attn_act_func == 'softdelta' else None,
                self.attention.wo.weight,
                self.attention.query_norm.weight if self.attention.qknorm else None,
                self.attention.key_norm.weight if self.attention.qknorm else None,
                self.attention.local_heads,
                self.attention.local_kv_heads,
                self.attention.head_dim,
                self.attention.rope_head_dim,
                self.attention.attn_gate_fn,
                self.attention.causal_attn_backend,
                None,
                self.nffn.norm.weight,
                self.nffn.norm.bias,
                self.nffn.norm.groups_per_partition,
                self.nffn.fc1.weight,
                self.nffn.fc2.weight,
                self.nffn.fc3.weight if self.nffn.swiglu else None,
                None,
                self.dropout,
                self.attention.attention_dropout,
                self.attention.hidden_dropout,
                self.nffn.swiglu,
                self.layernorm_eps,
                self.rmsnorm_eps,
                self.apply_rmsnorm,
                self.attention.norm.gather_input,
                self.residual_func,
                self.residual_heads,
                fp32_attn_output,
                deterministic,
                self.recompute_q,
                self.recompute_kv,
                self.recompute_attention,
                self.recompute_fc1_out,
                self.recompute_fc3_out
            )

        y, cache = self.attention(
            x, freqs_cis, segments, fp32_attn_output, deterministic, cache
        )
        # FFN
        out = self.nffn(y)

        return out, None, cache


class TransformerMoEBlock(nn.Module):

    def __init__(self, cfg: ModelConf, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.residual_func = cfg.residual_func
        self.residual_heads = cfg.residual_heads
        self.fused_block = cfg.fused_block
        self.recompute_q = cfg.recompute_q
        self.recompute_kv = cfg.recompute_v
        self.recompute_attention = cfg.recompute_attention
        self.recompute_fc1_out = cfg.recompute_fc1_out
        self.recompute_fc3_out = cfg.recompute_fc3_out
        self.recompute_router = cfg.recompute_router
        self.dropout = cfg.dropout
        self.apply_rmsnorm = cfg.apply_rmsnorm
        self.layernorm_eps = cfg.layernorm_eps
        self.rmsnorm_eps = cfg.rmsnorm_eps

        if self.fused_block:
            assert cfg.causal_attn_backend in ['flash', 'xattn'], f"flash or xattn attention backend is required for fused block."

        self.attention = MultiheadAttention(
            layer_id=layer_id,
            mdim=cfg.model_dim,
            n_heads=cfg.num_heads,
            n_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            rope_head_dim=cfg.rope_head_dim,
            attn_act_func=cfg.attn_act_func,
            qknorm=cfg.qknorm,
            norm_num_groups=cfg.layernorm_num_groups,
            norm_affine=cfg.norm_affine,
            layernorm_eps=cfg.layernorm_eps,
            rmsnorm_eps=cfg.rmsnorm_eps,
            memory_efficient_norm=cfg.memory_efficient_norm,
            apply_rmsnorm=cfg.apply_rmsnorm,
            apply_attn_gate=cfg.apply_attn_gate,
            attn_gate_func=cfg.attn_gate_func,
            causal_attn_backend=cfg.causal_attn_backend,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            hidden_dropout=cfg.hidden_dropout,
            residual_func=cfg.residual_func,
            residual_heads=cfg.residual_heads,
            init_mode=cfg.init_mode,
            init_std=cfg.init_std,
        )

        self.moe = NormalizedMoE(
            layer_id=layer_id,
            model_dim=cfg.model_dim,
            expert_inter_dim=cfg.expert_inter_dim,
            num_experts=cfg.num_experts,
            num_activated_experts=cfg.num_activated_experts,
            num_shared_experts=cfg.num_shared_experts,
            expert_backend=cfg.moe_expert_backend,
            permutation_backend=cfg.moe_permutation_backend,
            router_score_func=cfg.moe_router_score_func,
            router_bias=cfg.moe_router_bias,
            router_bias_update_rate=cfg.moe_router_bias_update_rate,
            router_scaling_factor=cfg.moe_router_scaling_factor,
            dropout=cfg.dropout,
            hidden_dropout=cfg.hidden_dropout,
            norm_num_groups=cfg.layernorm_num_groups,
            norm_affine=cfg.norm_affine,
            layernorm_eps=cfg.layernorm_eps,
            rmsnorm_eps=cfg.rmsnorm_eps,
            memory_efficient_norm=cfg.memory_efficient_norm,
            apply_rmsnorm=cfg.apply_rmsnorm,
            residual_func=cfg.residual_func,
            residual_heads=cfg.residual_heads,
            init_mode=cfg.init_mode,
            init_std=cfg.init_std
        )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Optional[Tensor],
        segments: Optional[Any] = None,
        moe_router_load_balancing_type: Optional[str] = None,
        fp32_attn_output: bool = False,
        deterministic: bool = True,
        cache: Optional[Tuple[Tensor, Tensor, int]] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Any]]:
        if self.fused_block and self.training:
            fn = TransformerMoEBlockFunction
            return fn.apply(
                x,
                freqs_cis,
                segments,
                self.attention.norm.weight,
                self.attention.norm.bias,
                self.attention.wq.weight,
                self.attention.wk.weight,
                self.attention.wv.weight,
                self.attention.wr.weight if self.attention.apply_attn_gate else None,
                self.attention.wg.weight if self.attention.attn_act_func == 'softdelta' else None,
                self.attention.wo.weight,
                self.attention.query_norm.weight if self.attention.qknorm else None,
                self.attention.key_norm.weight if self.attention.qknorm else None,
                self.attention.local_heads,
                self.attention.local_kv_heads,
                self.attention.head_dim,
                self.attention.rope_head_dim,
                self.attention.attn_gate_fn,
                self.attention.causal_attn_backend,
                None,
                self.moe.norm.weight,
                self.moe.norm.bias,
                self.moe.norm.groups_per_partition,
                self.moe.fc1.weight if self.moe.fc1 is not None else None,
                self.moe.fc2.weight if self.moe.fc2 is not None else None,
                self.moe.fc3.weight if self.moe.fc3 is not None else None,
                self.moe.router.weight,
                self.moe.n_experts,
                self.moe.n_local_experts,
                self.moe.expert_start_idx,
                self.moe.expert_end_idx,
                self.moe.topk,
                self.moe.permutation_backend,
                self.moe.router.score_func,
                self.moe.router.bias,
                self.moe.router.bias_update_rate,
                self.moe.router.scaling_factor,
                moe_router_load_balancing_type,
                self.moe.experts.weight1,
                self.moe.experts.weight2,
                self.moe.experts.weight3,
                self.moe.experts.backend,
                None,
                self.dropout,
                self.attention.attention_dropout,
                self.attention.hidden_dropout,
                self.layernorm_eps,
                self.rmsnorm_eps,
                self.apply_rmsnorm,
                self.attention.norm.gather_input,
                self.residual_func,
                self.residual_heads,
                fp32_attn_output,
                deterministic,
                self.recompute_q,
                self.recompute_kv,
                self.recompute_attention,
                self.recompute_fc1_out,
                self.recompute_fc3_out,
                self.recompute_router,
            )

        y, cache = self.attention(
            x, freqs_cis, segments, fp32_attn_output, deterministic, cache
        )
        # MoE
        out, aux_loss = self.moe(y, moe_router_load_balancing_type)

        return out, aux_loss, cache


class TransformerMoVABlock(nn.Module):

    def __init__(self, cfg: ModelConf, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.residual_func = cfg.residual_func
        self.residual_heads = cfg.residual_heads
        self.fused_block = cfg.fused_block
        self.recompute_q = cfg.recompute_q
        self.recompute_kv = cfg.recompute_v
        self.recompute_attention = cfg.recompute_attention
        self.recompute_fc1_out = cfg.recompute_fc1_out
        self.recompute_fc3_out = cfg.recompute_fc3_out
        self.recompute_router = cfg.recompute_router
        self.dropout = cfg.dropout
        self.apply_rmsnorm = cfg.apply_rmsnorm
        self.layernorm_eps = cfg.layernorm_eps
        self.rmsnorm_eps = cfg.rmsnorm_eps

        if self.fused_block:
            assert cfg.causal_attn_backend in ['flash', 'xattn'], f"flash or xattn attention backend is required for fused block."

        self.mova = MOVAttention(
            layer_id=layer_id,
            mdim=cfg.model_dim,
            n_heads=cfg.num_heads,
            n_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            rope_head_dim=cfg.rope_head_dim,
            attn_act_func=cfg.attn_act_func,
            num_values=cfg.num_values,
            num_activated_values=cfg.num_activated_values,
            qknorm=cfg.qknorm,
            norm_num_groups=cfg.layernorm_num_groups,
            norm_affine=cfg.norm_affine,
            layernorm_eps=cfg.layernorm_eps,
            rmsnorm_eps=cfg.rmsnorm_eps,
            memory_efficient_norm=cfg.memory_efficient_norm,
            apply_rmsnorm=cfg.apply_rmsnorm,
            apply_attn_gate=cfg.apply_attn_gate,
            attn_gate_func=cfg.attn_gate_func,
            causal_attn_backend=cfg.causal_attn_backend,
            value_backend=cfg.mova_backend,
            permutation_backend=cfg.moe_permutation_backend,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            hidden_dropout=cfg.hidden_dropout,
            router_score_func=cfg.moe_router_score_func,
            router_bias=cfg.moe_router_bias,
            router_bias_update_rate=cfg.moe_router_bias_update_rate,
            router_scaling_factor=cfg.moe_router_scaling_factor,
            residual_func=cfg.residual_func,
            residual_heads=cfg.residual_heads,
            init_mode=cfg.init_mode,
            init_std=cfg.init_std,
        )

        self.moe = NormalizedMoE(
            layer_id=layer_id,
            model_dim=cfg.model_dim,
            expert_inter_dim=cfg.expert_inter_dim,
            num_experts=cfg.num_experts,
            num_activated_experts=cfg.num_activated_experts,
            num_shared_experts=cfg.num_shared_experts,
            expert_backend=cfg.moe_expert_backend,
            permutation_backend=cfg.moe_permutation_backend,
            router_score_func=cfg.moe_router_score_func,
            router_bias=cfg.moe_router_bias,
            router_bias_update_rate=cfg.moe_router_bias_update_rate,
            router_scaling_factor=cfg.moe_router_scaling_factor,
            dropout=cfg.dropout,
            hidden_dropout=cfg.hidden_dropout,
            norm_num_groups=cfg.layernorm_num_groups,
            norm_affine=cfg.norm_affine,
            layernorm_eps=cfg.layernorm_eps,
            rmsnorm_eps=cfg.rmsnorm_eps,
            memory_efficient_norm=cfg.memory_efficient_norm,
            apply_rmsnorm=cfg.apply_rmsnorm,
            residual_func=cfg.residual_func,
            residual_heads=cfg.residual_heads,
            init_mode=cfg.init_mode,
            init_std=cfg.init_std
        )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Optional[Tensor],
        segments: Optional[Any] = None,
        moe_router_load_balancing_type: Optional[str] = None,
        fp32_attn_output: bool = False,
        deterministic: bool = True,
        cache: Optional[Tuple[Tensor, Tensor, int]] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Any]]:
        if self.fused_block and self.training:
            fn = TransformerMoVABlockFunction
            return fn.apply(
                x,
                freqs_cis,
                segments,
                self.mova.norm.weight,
                self.mova.norm.bias,
                self.mova.wq.weight,
                self.mova.wk.weight,
                self.mova.wv.weight,
                self.mova.wr.weight if self.mova.apply_attn_gate else None,
                self.mova.wg.weight if self.mova.attn_act_func == 'softdelta' else None,
                self.mova.wo.weight,
                self.mova.query_norm.weight if self.mova.qknorm else None,
                self.mova.key_norm.weight if self.mova.qknorm else None,
                self.mova.local_heads,
                self.mova.local_kv_heads,
                self.mova.head_dim,
                self.mova.rope_head_dim,
                self.mova.router.weight,
                self.mova.n_values,
                self.mova.topk,
                self.mova.router.bias,
                self.mova.value_backend,
                self.mova.attn_gate_fn,
                self.mova.causal_attention,
                None,
                self.moe.norm.weight,
                self.moe.norm.bias,
                self.moe.norm.groups_per_partition,
                self.moe.fc1.weight if self.moe.fc1 is not None else None,
                self.moe.fc2.weight if self.moe.fc2 is not None else None,
                self.moe.fc3.weight if self.moe.fc3 is not None else None,
                self.moe.router.weight,
                self.moe.n_experts,
                self.moe.n_local_experts,
                self.moe.expert_start_idx,
                self.moe.expert_end_idx,
                self.moe.topk,
                self.moe.permutation_backend,
                self.moe.router.score_func,
                self.moe.router.bias,
                self.moe.router.bias_update_rate,
                self.moe.router.scaling_factor,
                moe_router_load_balancing_type,
                self.moe.experts.weight1,
                self.moe.experts.weight2,
                self.moe.experts.weight3,
                self.moe.experts.backend,
                None,
                self.dropout,
                self.mova.attention_dropout,
                self.mova.hidden_dropout,
                self.layernorm_eps,
                self.rmsnorm_eps,
                self.apply_rmsnorm,
                self.mova.norm.gather_input,
                self.residual_func,
                self.residual_heads,
                fp32_attn_output,
                deterministic,
                self.recompute_q,
                self.recompute_kv,
                self.recompute_attention,
                self.recompute_fc1_out,
                self.recompute_fc3_out,
                self.recompute_router,
            )

        y, aux_loss_mova, cache = self.mova(
            x, freqs_cis, segments, fp32_attn_output, deterministic, cache, moe_router_load_balancing_type
        )
        # MoE
        out, aux_loss_moe = self.moe(y, moe_router_load_balancing_type)
        aux_loss = None if aux_loss_mova is None else (aux_loss_mova + aux_loss_moe) * 0.5

        return out, aux_loss, cache


class TransformerOutputLayer(nn.Module):
    def __init__(self, cfg: ModelConf):
        super().__init__()

        self.model_dim = cfg.model_dim
        self.output_size = cfg.vocab_size if cfg.output_size == -1 else cfg.output_size
        self.apply_rmsnorm = cfg.apply_rmsnorm
        self.layernorm_eps = cfg.layernorm_eps
        self.rmsnorm_eps = cfg.rmsnorm_eps
        self.recompute_logits = cfg.recompute_logits
        self.fused_output_layer = cfg.fused_output_layer

        norm_cls = GroupRMSNorm if cfg.apply_rmsnorm else GroupLayerNorm
        norm_eps = cfg.rmsnorm_eps if self.apply_rmsnorm else cfg.layernorm_eps
        self.final_norm = norm_cls(
            cfg.model_dim,
            num_groups=cfg.layernorm_num_groups,
            elementwise_affine=cfg.norm_affine,
            eps=norm_eps,
            memory_efficient=cfg.memory_efficient_norm
        )

        init_fn = get_init_fn('gaussian', dim=self.model_dim, std=cfg.init_logits_std)
        self.output = ColumnParallelLinear(
            self.model_dim,
            self.output_size,
            bias=False,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=init_fn
        )

    def forward(
        self,
        x: Tensor,
        y: Optional[Tensor],
        mask: Optional[Tensor] = None,
    ):
        if self.fused_output_layer and self.training:
            assert y is not None
            fn = TransformerOutputLayerFunction
            return fn.apply(
                x,
                y,
                mask,
                self.final_norm.weight,
                self.final_norm.bias,
                self.final_norm.groups_per_partition,
                self.output.weight,
                self.layernorm_eps,
                self.rmsnorm_eps,
                self.apply_rmsnorm,
                self.final_norm.gather_input,
                self.recompute_logits,
            )

        x = self.final_norm(x)
        if not self.final_norm.gather_input:
            x = gather_copy_model_parallel_region(x)
        logits = self.output(x).float()
        if y is None:
            return logits
        else:
            loss = vocab_parallel_cross_entropy(logits, y)
            if mask is not None:
                loss = loss * mask.to(loss)
            return loss


class Transformer(XLLModel):
    def __init__(self, cfg: ModelConf, tokenizer: Tokenizer):
        super().__init__(cfg, tokenizer)
        self.chunk_size = cfg.chunk_size
        self.context_parallel_size = get_context_parallel_world_size()
        self.context_parallel_rank = get_context_parallel_rank()

        if cfg.ddp_backend == 'fsdp1':
            from torch.distributed.fsdp.wrap import wrap
        else:
            assert cfg.ddp_backend == 'fsdp2'
            from xllm.distributed.wrap import wrap

        init_fn = get_init_fn('gaussian', dim=self.model_dim, std=cfg.init_embed_std)
        self.embed = wrap(ParallelEmbedding(
            self.vocab_size, self.model_dim, scale_emb=cfg.scale_emb, gather_output=False, init_method=init_fn
        ))

        assert self.v_head_dim == self.head_dim
        if self.rope_head_dim > 0:
            self.rope = RotaryEmbedding(self.rope_head_dim, cfg.chunk_size * 16, base=cfg.rope_base)
        else:
            self.rope = None
        self.qknorm = cfg.qknorm
        self.apply_attn_gate = cfg.apply_attn_gate

        self.layers = nn.ModuleList()
        for layer_id in range(self.num_layers):
            block_cls = TransformerBlock if layer_id < self.num_dense_layers else \
                (TransformerMoVABlock if self.num_values > 0 else TransformerMoEBlock)
            layer = block_cls(cfg, layer_id)
            self.layers.append(wrap(layer))

        self.output = wrap(TransformerOutputLayer(cfg))

    def forward(
        self,
        tokens: Tensor,
        multi_segments: bool,
        targets: Optional[Tensor] = None,
        token_mask: Optional[Tensor] = None,
        moe_router_load_balancing_type: Optional[str] = None,
        fp32_attn_output: bool = False,
        deterministic: bool = True,
        cache: Optional[List[Tuple[Tensor, Tensor, int]]] = None,
    ):
        bsz, seq_len = tokens.shape

        if targets is None:
            assert token_mask is None

        if multi_segments:
            assert cache is None
            bos_mask = torch.eq(tokens, self.bos_id)
            segments = self.get_segment_arg(bos_mask)
        else:
            bos_mask = None
            segments = None

        if self.training:
            assert cache is None, "training model does not support kv cache."
            assert seq_len % self.context_parallel_size == 0
            seq_len = seq_len // self.context_parallel_size
            assert seq_len % self.chunk_size == 0
            cache_len = seq_len * self.context_parallel_rank
            start = self.context_parallel_rank * seq_len
            end = (self.context_parallel_rank + 1) * seq_len
            tokens = tokens[:, start:end]
            if bos_mask is not None:
                bos_mask = bos_mask[:, :end]
            if targets is not None:
                targets = targets[:, start:end]
                token_mask = token_mask[:, start:end] if token_mask is not None else None
        else:
            assert self.context_parallel_size == 1, "inference mode does not support context parallel."
            cache_len = 0 if cache is None else cache[0][-1]

        # embeddings
        emb = self.embed(tokens)
        x = memory_efficient_dropout(emb, self.dropout, self.training)
        # rope frequencies
        freq_cis = None if self.rope is None else self.rope.get_freqs_cis(cache_len, cache_len + seq_len, x.device)

        aux_loss_sum = None
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            if self.layerwise_ckpt:
                x, aux_loss, layer_cache = checkpoint(
                    layer, x, freq_cis, segments, moe_router_load_balancing_type,
                    fp32_attn_output, deterministic, layer_cache,
                    use_reentrant=False, preserve_rng_state=True
                )
            else:
                x, aux_loss, layer_cache = layer(
                    x, freq_cis, segments, moe_router_load_balancing_type,
                    fp32_attn_output, deterministic, layer_cache
                )

            if aux_loss is not None:
                aux_loss_sum = aux_loss if aux_loss_sum is None else aux_loss_sum + aux_loss

            if cache is not None:
                cache[i] = layer_cache

        logits_or_loss = self.output(x, targets, token_mask)
        if targets is not None and self.context_parallel_size > 1:
            logits_or_loss = gather_from_context_parallel_region(logits_or_loss)

        return logits_or_loss, aux_loss_sum, cache

    def support_multi_segments_with_cache(self) -> bool:
        # TODO: causal attn w. xattn backend supports it.
        return False

    def get_segment_arg(self, bos_mask):
        bsz, seq_len = bos_mask.shape
        seq_len = seq_len // self.context_parallel_size
        start = self.context_parallel_rank * seq_len
        end = (self.context_parallel_rank + 1) * seq_len
        if self.causal_attn_backend == 'flash':
            if self.context_parallel_size > 1:
                assert bsz == 1, f"Flash attention does not support context parallel w. bsz {bsz} > 1"
            # query seqlens
            bos_idx_q = torch.nonzero(F.pad(bos_mask[:, (start + 1):end], (1, 0), value=1))
            cu_seqlens_q = bos_idx_q[:, 0] * seq_len + bos_idx_q[:, 1]
            cu_seqlens_q = F.pad(cu_seqlens_q, (0, 1), value=bsz * seq_len)
            max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
            # key seqlens
            bos_idx_k = torch.nonzero(F.pad(bos_mask[:, 1:end], (1, 0), value=1))
            cu_seqlens_k = bos_idx_k[:, 0] * end + bos_idx_k[:, 1]
            cu_seqlens_k = F.pad(cu_seqlens_k, (0, 1), value=bsz * end)
            num_q_seqs = cu_seqlens_q.shape[0]
            cu_seqlens_k = cu_seqlens_k[-num_q_seqs:]
            offset = cu_seqlens_k[0]
            cu_seqlens_k = cu_seqlens_k - offset
            max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
            total_seqlen_k = cu_seqlens_k[-1].item()
            return cu_seqlens_q.int(), cu_seqlens_k.int(), max_seqlen_q, max_seqlen_k, total_seqlen_k
        elif self.causal_attn_backend == 'swift':
            segment_idx = torch.cumsum(bos_mask, dim=-1)
            # B x L1
            q_segment_idx = segment_idx[:, start:end]
            # B x L2
            k_segment_idx = segment_idx[:, :end]
            return q_segment_idx, k_segment_idx
        elif self.causal_attn_backend == 'xattn':
            segment_idx = torch.cumsum(bos_mask, dim=-1)
            return segment_idx[:, :end]
        else:
            assert self.causal_attn_backend is None
            segment_idx = torch.cumsum(bos_mask, dim=-1)
            # B x L1
            q_segment_idx = segment_idx[:, start:end]
            # B x L2
            k_segment_idx = segment_idx[:, :end]
            # B x L1 x L2
            seg_mask = torch.ne(q_segment_idx.unsqueeze(2), k_segment_idx.unsqueeze(1))
            return seg_mask

    def num_parameters(self):
        embed_params = self.model_dim * self.vocab_size * 2
        attn_params_per_block = self.model_dim * (self.num_heads + self.num_kv_heads) * self.head_dim * 2
        if self.attn_act_func == 'softdelta':
            attn_params_per_block += self.model_dim * self.num_heads * (self.head_dim + 1)
        if self.apply_attn_gate:
            attn_params_per_block += self.model_dim * self.num_heads * self.head_dim
        attn_params_per_block += num_params_in_residual(self.residual_func, self.model_dim, self.residual_heads, self.num_heads * self.head_dim)
        # Norm
        norm_params_per_block = self.model_dim * (2 if self.apply_rmsnorm else 4)
        if self.qknorm:
            norm_params_per_block += self.head_dim * (self.num_heads + self.num_kv_heads)
        # FFN
        ffn_params_per_block = self.model_dim * self.ffn_hidden_dim * (3 if self.swiglu else 2)
        ffn_params_per_block += num_params_in_residual(self.residual_func, self.model_dim, self.residual_heads, self.ffn_hidden_dim)

        activated_params = norm_params_per_block * self.num_layers
        activated_params += (attn_params_per_block + ffn_params_per_block) * self.num_dense_layers + embed_params
        total_params = activated_params
        if self.num_values > 0:
            mova_qko_params = self.model_dim * (self.num_heads * 2 + self.num_kv_heads) * self.head_dim
            if self.attn_act_func == 'softdelta':
                mova_qko_params += self.model_dim * self.num_heads * (self.head_dim + 1)
            if self.apply_attn_gate:
                mova_qko_params += self.model_dim * self.num_heads * self.head_dim
            mova_qko_params += num_params_in_residual(self.residual_func, self.model_dim, self.residual_heads, self.num_heads * self.head_dim)
            # Value
            routed_value_params = self.model_dim * self.num_kv_heads * self.head_dim * self.num_values
            activated_value_params = self.model_dim * self.num_kv_heads * self.head_dim * self.num_activated_values
            router_params = self.model_dim * self.num_values
            activated_params += (mova_qko_params + activated_value_params + router_params) * (self.num_layers - self.num_dense_layers)
            total_params += (mova_qko_params + routed_value_params + router_params) * (self.num_layers - self.num_dense_layers)
        else:
            activated_params += attn_params_per_block * (self.num_layers - self.num_dense_layers)
            total_params += attn_params_per_block * (self.num_layers - self.num_dense_layers)
        if self.num_experts > 0:
            shared_expert_params = self.model_dim * self.expert_inter_dim * self.num_shared_experts * 3
            routed_expert_params = self.model_dim * self.expert_inter_dim * self.num_experts * 3
            activated_expert_params = self.model_dim * self.expert_inter_dim * self.num_activated_experts * 3
            router_params = self.model_dim * self.num_experts
            residual_params = num_params_in_residual(self.residual_func, self.model_dim, self.residual_heads, self.model_dim)
            activated_params += (shared_expert_params + activated_expert_params + router_params + residual_params) * (self.num_layers - self.num_dense_layers)
            total_params += (shared_expert_params + routed_expert_params + router_params) * (self.num_layers - self.num_dense_layers)
        # final norm
        activated_params += self.model_dim * (1 if self.apply_rmsnorm else 2)
        total_params += self.model_dim * (1 if self.apply_rmsnorm else 2)
        return total_params, activated_params, embed_params

    def tflops_per_token(self, seq_len: int, layer_schedule: Optional[Sequence[int]] = None):
        # `layer_schedule` lists executed layer ids; looped models run layers repeatedly.
        if layer_schedule is None:
            layer_schedule = list(range(self.num_layers))
        num_layer_calls = len(layer_schedule)
        num_dense_calls = sum(layer_id < self.num_dense_layers for layer_id in layer_schedule)
        expansion_factor = 6
        embed_flops = self.model_dim + self.vocab_size
        logits_flops = self.model_dim * self.vocab_size
        norm_flops = self.model_dim * (4 if self.apply_rmsnorm else 8)
        if self.qknorm:
            norm_flops += self.head_dim * (self.num_heads + self.num_kv_heads) * 2
        attn_flops = self.model_dim * (self.num_heads + self.num_kv_heads) * self.head_dim * 2

        # FFN
        ffn_flops = self.model_dim * self.ffn_hidden_dim * (3 if self.swiglu else 2)

        # residual
        # TODO
        res_flops = 0.0

        # MoVA
        mova_flops = self.model_dim * (self.num_heads * 2 + self.num_kv_heads) * self.head_dim
        mova_flops += self.model_dim * self.num_kv_heads * self.head_dim * self.num_activated_values
        mova_flops += self.model_dim * self.num_values

        if self.apply_attn_gate:
            attn_flops += self.model_dim * self.num_heads * self.head_dim
            mova_flops += self.model_dim * self.num_heads * self.head_dim

        if self.attn_act_func == 'softmax':
            attn_flops += self.num_heads * self.head_dim * seq_len
            mova_flops += self.num_heads * self.head_dim * seq_len
        elif self.attn_act_func == 'softdelta':
            attn_flops += self.model_dim * self.num_heads * (self.head_dim + 1)
            attn_flops += self.num_heads * self.head_dim * seq_len * 2  # two qkv
            mova_flops += self.model_dim * self.num_heads * (self.head_dim + 1)
            mova_flops += self.num_heads * self.head_dim * seq_len * 2  # two qkv
        else:
            raise ValueError(f"Unknown attention activation function: {self.attn_act_func}")

        # MoE
        moe_flops = self.model_dim * self.expert_inter_dim * (self.num_activated_experts + self.num_shared_experts) * 3
        moe_flops += self.model_dim * self.num_experts

        total_tflops = embed_flops + logits_flops + norm_flops
        total_tflops += num_layer_calls * (norm_flops + res_flops)
        total_tflops += num_dense_calls * (attn_flops + ffn_flops)
        total_tflops += (num_layer_calls - num_dense_calls) * (moe_flops + (mova_flops if self.num_values > 0 else attn_flops))
        total_tflops = total_tflops * expansion_factor / 10**12
        return total_tflops


# register some models configurations
# fmt: off

# llama models
ConfStore["llama3-7B"] = ModelConf(arch='transformer', num_layers=32, model_dim=4096, num_heads=32, num_kv_heads=8,
                                   rope_base=500000, ffn_hidden_dim=14336, swiglu=True, apply_rmsnorm=True, layernorm_num_groups=1)

ConfStore["llama3-70B"] = ModelConf(arch='transformer', num_layers=80, model_dim=8192, num_heads=64, num_kv_heads=8,
                                    rope_base=500000, ffn_hidden_dim=28672, swiglu=True, apply_rmsnorm=True, layernorm_num_groups=1)

ConfStore["llama3-400B"] = ModelConf(arch='transformer', num_layers=126, model_dim=16384, num_heads=128, num_kv_heads=8,
                                     rope_base=500000, ffn_hidden_dim=53248, swiglu=True, apply_rmsnorm=True, layernorm_num_groups=8)

# K2 models
ConfStore["k2v2-70B"] = ConfStore["llama3-70B"]

ConfStore["k2-horizon-0.9B"] = ModelConf(arch='transformer', num_layers=28, model_dim=1536,
                                         num_heads=32, num_kv_heads=8, head_dim=64,
                                         rope_base=500000, ffn_hidden_dim=5120, swiglu=True,
                                         apply_rmsnorm=True, layernorm_num_groups=1, rmsnorm_eps=1e-6)

ConfStore["k2-horizon-3.7B"] = ModelConf(arch='transformer', num_layers=36, model_dim=2560,
                                         num_heads=32, num_kv_heads=8, head_dim=128,
                                         rope_base=500000, ffn_hidden_dim=10240, swiglu=True,
                                         apply_rmsnorm=True, layernorm_num_groups=2, rmsnorm_eps=1e-6)

ConfStore["k2-horizon-7B"] = ModelConf(arch='transformer', num_layers=36, model_dim=4096,
                                       num_heads=32, num_kv_heads=8, head_dim=128,
                                       rope_base=500000, ffn_hidden_dim=12288, swiglu=True,
                                       apply_rmsnorm=True, layernorm_num_groups=4, rmsnorm_eps=1e-6)

ConfStore["k2-horizon-32B"] = ModelConf(arch='transformer', num_layers=64, model_dim=5120,
                                        num_heads=64, num_kv_heads=8, head_dim=128,
                                        rope_base=500000, ffn_hidden_dim=26624, swiglu=True,
                                        apply_rmsnorm=True, layernorm_num_groups=4, rmsnorm_eps=1e-6)

# MoE
ConfStore["mixtral8x7B"] = ModelConf(arch='transformer', num_layers=32, model_dim=4096, num_heads=32, num_kv_heads=8,
                                     ffn_hidden_dim=14336, swiglu=True, rope_base=500000, apply_rmsnorm=True, layernorm_num_groups=1,
                                     num_experts=8, num_activated_experts=2, expert_inter_dim=14336, moe_router_score_func='softmax')

ConfStore["k2-horizon-moe-36B-a4B"] = ModelConf(arch='transformer', num_layers=48, num_dense_layers=3, model_dim=2560,
                                                num_heads=32, num_kv_heads=8, head_dim=128, rope_head_dim=128, rope_base=500000,
                                                apply_attn_gate=True, attn_gate_func='softplus', ffn_hidden_dim=6144, swiglu=True,
                                                num_experts=128, num_activated_experts=8, num_shared_experts=1, expert_inter_dim=768,
                                                moe_router_score_func='sigmoid', moe_router_bias=True, moe_router_bias_update_rate=1e-3,
                                                moe_router_scaling_factor=2.5, apply_rmsnorm=True, layernorm_num_groups=2, rmsnorm_eps=1e-6)

ConfStore["k2-horizon-mova-36B-a4B"] = ModelConf(arch='transformer', num_layers=48, num_dense_layers=3, model_dim=2560,
                                                 num_heads=32, num_kv_heads=8, head_dim=128, rope_head_dim=128, rope_base=500000,
                                                 apply_attn_gate=True, attn_gate_func='softplus', ffn_hidden_dim=6144, swiglu=True,
                                                 num_values=64, num_activated_values=4,
                                                 num_experts=100, num_activated_experts=8, num_shared_experts=1, expert_inter_dim=768,
                                                 moe_router_score_func='sigmoid', moe_router_bias=True, moe_router_bias_update_rate=1e-3,
                                                 moe_router_scaling_factor=2.5, apply_rmsnorm=True, layernorm_num_groups=2, rmsnorm_eps=1e-6)


ConfStore["k2-horizon-375B-a23B"] = ModelConf(arch='transformer', num_layers=61, num_dense_layers=3, model_dim=6144,
                                              num_heads=48, num_kv_heads=8, head_dim=128, rope_head_dim=64, rope_base=500000,
                                              apply_attn_gate=False, ffn_hidden_dim=16384, swiglu=True,
                                              num_experts=192, num_activated_experts=8, num_shared_experts=1, expert_inter_dim=1792,
                                              moe_router_score_func='sigmoid', moe_router_bias=True, moe_router_bias_update_rate=1e-3,
                                              moe_router_scaling_factor=2.5, apply_rmsnorm=True, layernorm_num_groups=1, rmsnorm_eps=1e-6)
