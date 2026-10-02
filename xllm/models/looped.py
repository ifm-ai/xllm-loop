"""Looped language models built from the Transformer's layers.

- `LoopedTransformer` (arch='transformer'): layers [loop_start_layer, loop_end_layers) run
  loop_times times, optionally re-injecting the embeddings at each loop entry. It also
  implements the unrolled controls that inject an earlier layer's output into later layers.
- `Huginn` (arch='huginn'): a prelude, a recurrent block iterated loop_times times from an
  initial state with the prelude output injected at every recurrence, and a coda; the
  recurrent state can be split into hierarchical H/L states.
- `DepthControlledHuginn` (arch='huginn', huginn_depth_control=True): Huginn whose training
  depth is fixed, sampled or learned, with truncated backpropagation and further input
  injections. It is the model of Part II; the others are the models of Part I.

Parameter names match the released checkpoints of both papers.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from xllm.config import ModelConf
from xllm.distributed import get_model_parallel_rank, get_model_parallel_world_size
from xllm.modules import GroupRMSNorm
from xllm.modules.context_parallel import gather_from_context_parallel_region
from xllm.modules.fused_ops import memory_efficient_dropout
from xllm.modules.model_parallel import (
    ColumnParallelLinear,
    gather_copy_model_parallel_region,
)
from xllm.models.transformer import Transformer
from xllm.models.looped_depth import cumulative_distribution, pln_pmf, sample_depth
from xllm.models.looped_depth_prior import DepthPriorConfig, LearnedDepthPrior
from xllm.models.looped_flops import HuginnShape
from xllm.utils import get_init_fn


def _diagonal_input_injection_parameter_count(model_dim: int) -> int:
    return model_dim * model_dim + 2 * model_dim


def _input_injection_tflops_per_token(model_dim: int, num_injections: int) -> float:
    # Matches Transformer.tflops_per_token: 6 FLOPs per multiply-add (fwd + bwd).
    return num_injections * model_dim * model_dim * 6 / 10**12


def build_loop_execution_schedule(
    *,
    num_layers: int,
    loop_times: int,
    loop_start_layer: int,
    loop_end_layers: int,
) -> Tuple[int, ...]:
    """Return the exact physical-layer order for a fixed loop schedule."""
    if loop_times < 1:
        raise ValueError("loop_times must be positive")
    if not 0 <= loop_start_layer < loop_end_layers <= num_layers:
        raise ValueError("invalid loop layer span")

    prefix = tuple(range(loop_start_layer))
    loop = tuple(range(loop_start_layer, loop_end_layers))
    suffix = tuple(range(loop_end_layers, num_layers))
    return prefix + loop * loop_times + suffix


class InputInjection(nn.Module):
    """Diagonal input injection: h <- exp(-dt * a) * h + dt * (W e).

    dt = softplus(dt_bias) and a = exp(A_log) are per channel, and W starts as the
    identity. Like every other parameter, A_log, dt_bias and W take AdamW weight decay.
    """

    def __init__(
        self,
        cfg: ModelConf,
        *,
        diagonal_dt_init: float,
        diagonal_a_init: float,
    ) -> None:
        super().__init__()
        self.input_adapter = ColumnParallelLinear(
            cfg.model_dim,
            cfg.model_dim,
            bias=False,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=self._identity_init,
        )
        local_dim = self.input_adapter.output_size_per_partition
        self.A_log = nn.Parameter(torch.empty(local_dim))
        self.dt_bias = nn.Parameter(torch.empty(local_dim))
        with torch.no_grad():
            self.A_log.fill_(math.log(diagonal_a_init))
            self.dt_bias.fill_(self._inverse_softplus(diagonal_dt_init))

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        return value + math.log(-math.expm1(-value))

    @staticmethod
    def _identity_init(weight: Tensor) -> Tensor:
        with torch.no_grad():
            weight.zero_()
            diagonal = min(weight.shape)
            weight[:diagonal, :diagonal] = torch.eye(
                diagonal,
                dtype=weight.dtype,
                device=weight.device,
            )
        return weight

    def _decayed_state_and_update(self, state: Tensor, input_embeds: Tensor) -> Tuple[Tensor, Tensor]:
        """Return exp(-dt * a) * h and dt * (W e)."""
        full_input = gather_copy_model_parallel_region(input_embeds)
        if full_input.dtype != self.input_adapter.weight.dtype:
            raise TypeError(
                "input injection dtype mismatch: "
                f"input={full_input.dtype}, weight={self.input_adapter.weight.dtype}"
            )
        input_update = self.input_adapter(full_input)
        if input_update.dtype != state.dtype:
            raise TypeError(
                "input injection output dtype mismatch: "
                f"output={input_update.dtype}, state={state.dtype}"
            )

        dt = F.softplus(self.dt_bias).to(dtype=state.dtype)
        decay_rate = torch.exp(self.A_log).to(dtype=state.dtype)
        decay = torch.exp(-dt * decay_rate)
        return state * decay, dt * input_update

    def forward(self, state: Tensor, input_embeds: Tensor) -> Tensor:
        retained, injected = self._decayed_state_and_update(state, input_embeds)
        return retained + injected


class LinearInputInjection(nn.Module):
    """Huginn-Linear injection W[h; e] + b, 2d -> d; parameter names match the paper checkpoints."""

    def __init__(self, cfg: ModelConf) -> None:
        super().__init__()
        self.adapter = ColumnParallelLinear(
            cfg.model_dim * 2,
            cfg.model_dim,
            bias=cfg.apply_bias_term,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=get_init_fn(cfg.init_mode, dim=cfg.model_dim, std=cfg.init_std),
        )

    def forward(self, state: Tensor, input_embeds: Tensor) -> Tensor:
        full_state = gather_copy_model_parallel_region(state)
        full_input = gather_copy_model_parallel_region(input_embeds)
        return self.adapter(torch.cat([full_state, full_input], dim=-1))


class OrthogonalInputInjection(InputInjection):
    """OrthoInj: diagonal injection whose decayed state loses its component along the injected update."""

    def __init__(self, cfg: ModelConf) -> None:
        super().__init__(cfg, diagonal_dt_init=cfg.huginn_diagonal_dt_init,
                         diagonal_a_init=cfg.huginn_diagonal_a_init)
        if get_model_parallel_world_size() != 1:
            raise ValueError("the orthogonal projection needs the full hidden state (model parallel size 1)")
        self.eps = cfg.huginn_ortho_projection_eps

    def forward(self, state: Tensor, input_embeds: Tensor) -> Tensor:
        retained, injected = self._decayed_state_and_update(state, input_embeds)
        # Per token, in FP32 under mixed precision; gradients flow through both terms.
        compute_dtype = torch.float64 if retained.dtype == torch.float64 else torch.float32
        value, anchor = retained.to(compute_dtype), injected.to(compute_dtype)
        coefficient = (value * anchor).sum(dim=-1, keepdim=True) / (anchor.square().sum(dim=-1, keepdim=True) + self.eps)
        return (value - coefficient * anchor).to(retained.dtype) + injected


@dataclass
class _ForwardContext:
    """Per-forward inputs shared by every layer call, and the MoE aux losses in execution order."""

    freq_cis: Optional[Tensor]
    segments: Optional[Any]
    layer_kwargs: Dict[str, Any]
    aux_losses: List[Tensor] = field(default_factory=list)

    def aux_loss_sum(self) -> Optional[Tensor]:
        total = None
        for aux_loss in self.aux_losses:
            total = aux_loss if total is None else total + aux_loss
        return total


class _LoopedModel(Transformer):
    """Forward steps the looped models share: input slicing, embedding, layer calls and the output head."""

    def _begin_forward(
        self,
        tokens: Tensor,
        multi_segments: bool,
        targets: Optional[Tensor],
        token_mask: Optional[Tensor],
        moe_router_load_balancing_type: Optional[str],
        fp32_attn_output: bool,
        deterministic: bool,
        cache: Optional[List[Tuple[Tensor, Tensor, int]]],
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], _ForwardContext]:
        """Check the inputs, keep this context-parallel rank's slice and embed it.

        Segments come from the full sequence, before the slice. Returns the embeddings,
        the sliced targets and token mask, and the context for `_run_layers`.
        """
        if targets is None:
            assert token_mask is None
        if cache is not None:
            raise NotImplementedError(f"{type(self).__name__}.forward does not take a KV cache")

        _, seq_len = tokens.shape
        segments = self.get_segment_arg(torch.eq(tokens, self.bos_id)) if multi_segments else None
        if self.training:
            assert seq_len % self.context_parallel_size == 0
            seq_len = seq_len // self.context_parallel_size
            assert seq_len % self.chunk_size == 0
            cache_len = seq_len * self.context_parallel_rank
            start = self.context_parallel_rank * seq_len
            end = (self.context_parallel_rank + 1) * seq_len
            tokens = tokens[:, start:end]
            if targets is not None:
                targets = targets[:, start:end]
                token_mask = token_mask[:, start:end] if token_mask is not None else None
        else:
            assert self.context_parallel_size == 1
            cache_len = 0

        hidden = memory_efficient_dropout(self.embed(tokens), self.dropout, self.training)
        freq_cis = None if self.rope is None else self.rope.get_freqs_cis(cache_len, cache_len + seq_len, hidden.device)
        layer_kwargs = dict(moe_router_load_balancing_type=moe_router_load_balancing_type,
                            fp32_attn_output=fp32_attn_output, deterministic=deterministic)
        return hidden, targets, token_mask, _ForwardContext(freq_cis, segments, layer_kwargs)

    def _run_layers(self, layer_ids: Sequence[int], hidden: Tensor, ctx: _ForwardContext) -> Tensor:
        """Run the given physical layers in order, with per-layer activation checkpointing if enabled."""
        for layer_id in layer_ids:
            layer = self.layers[layer_id]
            if self.layerwise_ckpt:
                hidden, aux_loss, _ = checkpoint(layer, hidden, ctx.freq_cis, ctx.segments, **ctx.layer_kwargs,
                                                 cache=None, use_reentrant=False, preserve_rng_state=True)
            else:
                hidden, aux_loss, _ = layer(hidden, ctx.freq_cis, ctx.segments, **ctx.layer_kwargs, cache=None)
            if aux_loss is not None:
                ctx.aux_losses.append(aux_loss)
        return hidden

    def _finish_forward(self, hidden: Tensor, targets: Optional[Tensor], token_mask: Optional[Tensor]) -> Tensor:
        """Output head, checkpointed like the layers, and the context-parallel gather of the loss."""
        if self.layerwise_ckpt:
            logits_or_loss = checkpoint(self.output, hidden, targets, token_mask,
                                        use_reentrant=False, preserve_rng_state=True)
        else:
            logits_or_loss = self.output(hidden, targets, token_mask)
        if targets is not None and self.context_parallel_size > 1:
            logits_or_loss = gather_from_context_parallel_region(logits_or_loss)
        return logits_or_loss


class LoopedTransformer(_LoopedModel):
    """Transformer whose layers [loop_start_layer, loop_end_layers) run loop_times times.

    With loop input injection, the embeddings are injected at every loop entry. With the
    unrolled-control injection (loop_times 1), the output of `dense_prelude_source_layer`
    is injected before each of `dense_prelude_input_layers`.
    """

    def __init__(self, cfg: ModelConf, tokenizer) -> None:
        if cfg.arch != "transformer":
            raise NotImplementedError("LoopedTransformer does not implement Huginn; use Huginn")

        super().__init__(cfg, tokenizer)
        self.loop_times = cfg.loop_times
        self.loop_start_layer = cfg.loop_start_layer
        self.loop_end_layers = (
            self.num_layers if cfg.loop_end_layers is None else cfg.loop_end_layers
        )
        self.execution_schedule = build_loop_execution_schedule(
            num_layers=self.num_layers,
            loop_times=self.loop_times,
            loop_start_layer=self.loop_start_layer,
            loop_end_layers=self.loop_end_layers,
        )
        self.loop_input_injection_module = (
            InputInjection(
                cfg,
                diagonal_dt_init=cfg.loop_diagonal_dt_init,
                diagonal_a_init=cfg.loop_diagonal_a_init,
            )
            if cfg.loop_input_injection != "none"
            else None
        )
        self.dense_prelude_source_layer = cfg.dense_prelude_source_layer
        self.dense_prelude_input_layers = {
            int(layer)
            for layer in cfg.dense_prelude_input_layers.split(",")
            if layer.strip()
        }
        self.dense_prelude_input_injection_module = (
            InputInjection(
                cfg,
                diagonal_dt_init=cfg.dense_prelude_diagonal_dt_init,
                diagonal_a_init=cfg.dense_prelude_diagonal_a_init,
            )
            if cfg.dense_prelude_input_injection != "none"
            else None
        )
        self.num_aux_loss_layers = sum(
            layer_id >= self.num_dense_layers
            for layer_id in self.execution_schedule
        )

    def num_parameters(self):
        total_params, activated_params, embed_params = super().num_parameters()
        num_input_injections = sum(
            module is not None
            for module in (
                self.loop_input_injection_module,
                self.dense_prelude_input_injection_module,
            )
        )
        injection_params = num_input_injections * (
            _diagonal_input_injection_parameter_count(self.model_dim)
        )
        return (
            total_params + injection_params,
            activated_params + injection_params,
            embed_params,
        )

    def tflops_per_token(self, seq_len: int) -> float:
        num_injections = 0
        if self.loop_input_injection_module is not None:
            num_injections += self.loop_times
        if self.dense_prelude_input_injection_module is not None:
            num_injections += sum(
                layer_id in self.dense_prelude_input_layers
                for layer_id in self.execution_schedule
            )
        return super().tflops_per_token(
            seq_len, self.execution_schedule
        ) + _input_injection_tflops_per_token(self.model_dim, num_injections)

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
        hidden, targets, token_mask, ctx = self._begin_forward(
            tokens, multi_segments, targets, token_mask, moe_router_load_balancing_type,
            fp32_attn_output, deterministic, cache,
        )
        input_embeddings = hidden
        dense_prelude_embeddings = None
        for layer_id in self.execution_schedule:
            if (
                self.loop_input_injection_module is not None
                and layer_id == self.loop_start_layer
            ):
                hidden = self.loop_input_injection_module(hidden, input_embeddings)
            if (
                self.dense_prelude_input_injection_module is not None
                and layer_id in self.dense_prelude_input_layers
            ):
                hidden = self.dense_prelude_input_injection_module(hidden, dense_prelude_embeddings)
            hidden = self._run_layers((layer_id,), hidden, ctx)
            if (
                self.dense_prelude_input_injection_module is not None
                and layer_id == self.dense_prelude_source_layer
            ):
                dense_prelude_embeddings = hidden
        return self._finish_forward(hidden, targets, token_mask), ctx.aux_loss_sum(), None


class Huginn(_LoopedModel):
    """Prelude, a recurrent block iterated at fixed depth (optionally with H/L states), and coda.

    Huginn's recurrent span comes from the huginn_* layer counts; config.loop_start_layer
    and config.loop_end_layers do not apply.
    """

    def __init__(self, cfg: ModelConf, tokenizer) -> None:
        if cfg.arch != "huginn":
            raise ValueError(f"Huginn requires arch='huginn', got {cfg.arch!r}")

        super().__init__(cfg, tokenizer)
        self.loop_times = cfg.loop_times
        self.huginn_prelude_layers = cfg.huginn_prelude_layers
        self.huginn_recurrent_layers = (
            self.num_layers
            - cfg.huginn_prelude_layers
            - cfg.huginn_coda_layers
            if cfg.huginn_recurrent_layers is None
            else cfg.huginn_recurrent_layers
        )
        self.huginn_coda_layers = cfg.huginn_coda_layers
        self.huginn_state_init = cfg.huginn_state_init
        self.huginn_hierarchical_state = cfg.huginn_hierarchical_state
        self.huginn_hierarchical_h_cycles = cfg.huginn_hierarchical_h_cycles
        self.huginn_hierarchical_l_cycles = cfg.huginn_hierarchical_l_cycles
        self.huginn_split_module_repeats = cfg.huginn_split_module_repeats
        self.loop_start_layer = self.huginn_prelude_layers
        self.loop_end_layers = self.loop_start_layer + self.huginn_recurrent_layers
        if self.huginn_hierarchical_state == "split_hl":
            split_layer = self.loop_start_layer + self.huginn_recurrent_layers // 2
            h_layers = tuple(range(self.loop_start_layer, split_layer)) * self.huginn_split_module_repeats
            l_layers = tuple(range(split_layer, self.loop_end_layers)) * self.huginn_split_module_repeats
            hierarchical_layers = ()
            for _ in range(self.huginn_hierarchical_h_cycles):
                hierarchical_layers += (
                    l_layers * self.huginn_hierarchical_l_cycles + h_layers
                )
            self.execution_schedule = (
                tuple(range(self.loop_start_layer))
                + hierarchical_layers
                + tuple(range(self.loop_end_layers, self.num_layers))
            )
        else:
            self.execution_schedule = build_loop_execution_schedule(
                num_layers=self.num_layers,
                loop_times=self.loop_times,
                loop_start_layer=self.loop_start_layer,
                loop_end_layers=self.loop_end_layers,
            )
        self.input_injection = self.build_input_injection(cfg)
        self.num_aux_loss_layers = sum(
            layer_id >= self.num_dense_layers
            for layer_id in self.execution_schedule
        )

    def build_input_injection(self, cfg: ModelConf) -> Optional[nn.Module]:
        """Input injection of the recurrent block; subclasses override this to add other methods."""
        if cfg.huginn_input_injection == "none":
            return None
        return InputInjection(
            cfg,
            diagonal_dt_init=cfg.huginn_diagonal_dt_init,
            diagonal_a_init=cfg.huginn_diagonal_a_init,
        )

    def _added_parameter_count(self) -> int:
        """Parameters that the Transformer's count leaves out."""
        if self.input_injection is None:
            return 0
        return _diagonal_input_injection_parameter_count(self.model_dim)

    def num_parameters(self):
        total_params, activated_params, embed_params = super().num_parameters()
        added = self._added_parameter_count()
        return total_params + added, activated_params + added, embed_params

    def tflops_per_token(self, seq_len: int) -> float:
        # One input injection per recurrent state update.
        if self.input_injection is None:
            num_injections = 0
        elif self.huginn_hierarchical_state == "none":
            num_injections = self.loop_times
        else:
            num_injections = self.huginn_hierarchical_h_cycles * (
                self.huginn_hierarchical_l_cycles + 1
            )
        return super().tflops_per_token(
            seq_len, self.execution_schedule
        ) + _input_injection_tflops_per_token(self.model_dim, num_injections)

    def initialize_recurrent_state(
        self,
        input_embeds: Tensor,
        cache_len: int = 0,
    ) -> Tensor:
        if self.huginn_state_init == "zero":
            return torch.zeros_like(input_embeds)
        if self.huginn_state_init == "input":
            return input_embeds
        if self.huginn_state_init == "normal":
            if not self.training:
                batch = torch.arange(input_embeds.shape[0], device=input_embeds.device, dtype=torch.float32)
                return self._hashed_normal_state(
                    input_embeds, cache_len, batch.view(-1, 1, 1) * 0.037719, math.sqrt(2.0 / self.model_dim))
            return torch.randn_like(input_embeds).mul(self.model_dim**-0.5)
        raise ValueError(
            f"unknown Huginn state init: {self.huginn_state_init!r}"
        )

    def _hashed_normal_state(self, input_embeds: Tensor, cache_len: int, batch_term: Tensor, scale: float) -> Tensor:
        """Deterministic stand-in for the training state N(0, 1/d), used at evaluation.

        Each entry is a hash of its hidden index, its position and a per-row term:
        fract(43758.5453123 * sin(x)) with x = 0.013579 * dim + 0.021713 * pos + batch_term
        + 0.123457 is roughly uniform on (0, 1), and sqrt(2) * erfinv(2u - 1) maps it to a
        standard normal before `scale`. Hidden indices are global, so every model-parallel
        rank computes its slice of the same state; positions start at `cache_len`.
        """
        batch_size, seq_len, hidden_dim = input_embeds.shape
        device = input_embeds.device
        dim_offset = get_model_parallel_rank() * hidden_dim
        dimension = torch.arange(dim_offset, dim_offset + hidden_dim, device=device, dtype=torch.float32).view(1, 1, -1)
        position = torch.arange(cache_len, cache_len + seq_len, device=device, dtype=torch.float32).view(1, -1, 1)

        state = dimension.expand(batch_size, seq_len, hidden_dim).clone()
        state.mul_(0.013579)
        state.add_(position * 0.021713)
        state.add_(batch_term)
        state.add_(0.123457)
        state.sin_()
        state.mul_(43758.5453123)
        state.sub_(state.floor())
        state.clamp_(1e-6, 1.0 - 1e-6)
        state.mul_(2.0).sub_(1.0)
        state.erfinv_()
        state.mul_(scale)
        return state.to(dtype=input_embeds.dtype)

    def _recurrence_input(self, prelude_output: Tensor) -> Tensor:
        """The prelude output that every recurrence injects."""
        return prelude_output

    def _recur(
        self,
        state: Tensor,
        input_embeds: Tensor,
        start_layer: int,
        end_layer: int,
        ctx: _ForwardContext,
        module_repeats: int = 1,
    ) -> Tensor:
        """One state update: the input injection, then layers [start_layer, end_layer) module_repeats times."""
        if self.input_injection is not None:
            state = self.input_injection(state, input_embeds)
        for _ in range(module_repeats):
            state = self._run_layers(range(start_layer, end_layer), state, ctx)
        return state

    def _run_recurrences(self, state: Tensor, input_embeds: Tensor, ctx: _ForwardContext) -> Tensor:
        """loop_times state updates of one state, or the H/L cycles of two."""
        if self.huginn_hierarchical_state == "none":
            for _ in range(self.loop_times):
                state = self._recur(state, input_embeds, self.loop_start_layer, self.loop_end_layers, ctx)
            return state

        # The caller's state is not used: z_H starts from the prelude output and z_L from a
        # second draw. Both draws are kept, so seeded runs reproduce the released H/L models.
        z_h = input_embeds
        z_l = self.initialize_recurrent_state(input_embeds)
        if self.huginn_hierarchical_state == "split_hl":
            split_layer = self.loop_start_layer + self.huginn_recurrent_layers // 2
            h_start_layer, h_end_layer = self.loop_start_layer, split_layer
            l_start_layer, l_end_layer = split_layer, self.loop_end_layers
        else:
            h_start_layer = l_start_layer = self.loop_start_layer
            h_end_layer = l_end_layer = self.loop_end_layers
        for _ in range(self.huginn_hierarchical_h_cycles):
            for _ in range(self.huginn_hierarchical_l_cycles):
                z_l = self._recur(z_l + z_h, input_embeds, l_start_layer, l_end_layer, ctx,
                                  self.huginn_split_module_repeats)
            z_h = self._recur(z_h + z_l, input_embeds, h_start_layer, h_end_layer, ctx,
                              self.huginn_split_module_repeats)
        return z_h

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
        hidden, targets, token_mask, ctx = self._begin_forward(
            tokens, multi_segments, targets, token_mask, moe_router_load_balancing_type,
            fp32_attn_output, deterministic, cache,
        )
        hidden = self._run_layers(range(self.loop_start_layer), hidden, ctx)
        input_embeds = self._recurrence_input(hidden)
        hidden = self.initialize_recurrent_state(input_embeds)
        hidden = self._run_recurrences(hidden, input_embeds, ctx)
        hidden = self._run_layers(range(self.loop_end_layers, self.num_layers), hidden, ctx)
        return self._finish_forward(hidden, targets, token_mask), ctx.aux_loss_sum(), None


class DepthControlledHuginn(Huginn):
    """Huginn whose training depth is fixed, sampled or learned, with truncated backpropagation.

    Each training forward draws its depth R (loop_times, or capped PLN with mean
    loop_times) from its forward index `iteration_step`, or from the learned
    depth prior, then runs R - b recurrences without gradients and the last b
    with them. Evaluation runs loop_times recurrences. The prior learns through
    the `train.py` step hooks. The Table 3 variants change the input injection
    (linear, or orthogonal), normalize the injected prelude output, or normalize
    the state after every recurrence. The model is dense with one recurrent state.
    """

    def __init__(self, cfg: ModelConf, tokenizer) -> None:
        super().__init__(cfg, tokenizer)
        self.flops_shape = HuginnShape(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            ffn_hidden_dim=self.ffn_hidden_dim,
            vocab_size=self.vocab_size,
            output_size=self.output_size,
            prelude_layers=self.huginn_prelude_layers,
            recurrent_layers=self.huginn_recurrent_layers,
            coda_layers=self.huginn_coda_layers,
            input_injection=cfg.huginn_input_injection,
            apply_rmsnorm=self.apply_rmsnorm,
            qknorm=cfg.qknorm,
            apply_attn_gate=self.apply_attn_gate,
            swiglu=self.swiglu,
            layerwise_ckpt=self.layerwise_ckpt,
            prelude_norm=cfg.huginn_prelude_norm != "none",
            orthogonal_injection=cfg.huginn_prelude_orthogonal,
        )
        self.sampling_scheme = cfg.huginn_sampling_scheme
        self.antithetic_sampling = cfg.huginn_antithetic_sampling
        self.backprop_depth = cfg.huginn_backprop_depth
        self.depth_cdf = None
        self.depth_prior = None
        if self.sampling_scheme == "fixed":
            self.depth_distribution = ((self.loop_times, 1.0),)
        else:
            pmf = pln_pmf(cfg.huginn_poisson_lognormal_target_mean, cfg.huginn_poisson_lognormal_sigma,
                          cfg.huginn_poisson_lognormal_max)
            self.depth_distribution = tuple(enumerate(pmf, start=1))
            self.depth_cdf = cumulative_distribution(pmf)
            if cfg.huginn_depth_prior == "learned":
                self.depth_prior = LearnedDepthPrior(
                    DepthPriorConfig(entropy_coefficient=cfg.huginn_depth_prior_entropy,
                                     target_mean=cfg.huginn_poisson_lognormal_target_mean), initial_pmf=pmf)
        self.iteration_step = 0  # Training forwards run so far; seeds the depth draws.
        norm = dict(num_groups=cfg.layernorm_num_groups, eps=cfg.rmsnorm_eps, memory_efficient=cfg.memory_efficient_norm)
        self.huginn_prelude_norm_module = (
            GroupRMSNorm(self.model_dim, elementwise_affine=True, **norm) if cfg.huginn_prelude_norm == "rms" else None)
        self.huginn_recurrent_exit_norm_module = (
            GroupRMSNorm(self.model_dim, elementwise_affine=False, **norm)
            if cfg.huginn_recurrent_exit_norm == "parameterless_rms" else None)

    def build_input_injection(self, cfg: ModelConf) -> Optional[nn.Module]:
        if cfg.huginn_input_injection == "linear":
            return LinearInputInjection(cfg)
        if cfg.huginn_prelude_orthogonal:
            return OrthogonalInputInjection(cfg)
        return super().build_input_injection(cfg)

    def _added_parameter_count(self) -> int:
        """The injection's own size and the prelude norm, as the paper counted them."""
        if isinstance(self.input_injection, LinearInputInjection):
            bias = self.input_injection.adapter.bias
            added = 2 * self.model_dim * self.model_dim + (0 if bias is None else self.model_dim)
        else:
            added = super()._added_parameter_count()
        if self.huginn_prelude_norm_module is not None:
            added += self.model_dim
        return added

    def tflops_per_token(self, seq_len: int) -> float:
        """Training TFLOPs per token as the paper counted them: expected over the depth distribution."""
        return self.flops_shape.expected_tflops_per_token(self.depth_distribution, self.backprop_depth, seq_len)

    def step_train_tflops_per_token(self, seq_len: int) -> float:
        """Training TFLOPs per token of this step: the drawn depths' mean with a learned prior, else expected."""
        if self.depth_prior is None:
            return self.tflops_per_token(seq_len)
        depths = self.depth_prior.pending_depths
        return sum(self.flops_shape.depth_tflops_per_token(depth, self.backprop_depth, seq_len)
                   for depth in depths) / len(depths)

    def after_forward(self, tok_loss: Tensor, token_mask: Optional[Tensor]) -> None:
        """`train.py` hook after every training forward: the prior records the forward's mean token loss."""
        if self.depth_prior is not None:
            self.depth_prior.observe(tok_loss, token_mask)

    def after_optimizer_step(self) -> dict:
        """`train.py` hook after every optimizer step: the prior's REINFORCE update and its metrics."""
        if self.depth_prior is None:
            return {}
        return {f"depth_prior/{key}": value for key, value in self.depth_prior.finish_step().items()}

    def training_state_dict(self) -> dict:
        """Training state the weights do not hold; `train.py` saves it in each checkpoint's training state."""
        state = {"iteration_step": self.iteration_step}
        if self.depth_prior is not None:
            state["depth_prior"] = self.depth_prior.state_dict()
        return state

    def load_training_state_dict(self, state: dict) -> None:
        self.iteration_step = int(state["iteration_step"])
        if self.depth_prior is not None:
            self.depth_prior.load_state_dict(state["depth_prior"])

    def _draw_recurrence_steps(self) -> Tuple[int, int]:
        """Return the recurrences to run without and with gradients; a training forward advances the draws."""
        depth = self.loop_times
        if self.training:
            if self.depth_prior is not None:
                depth = self.depth_prior.sample()
            elif self.sampling_scheme != "fixed":
                depth = sample_depth(self.depth_cdf, self.iteration_step, antithetic=self.antithetic_sampling)
            self.iteration_step += 1
        with_grad = depth if self.backprop_depth is None else min(depth, self.backprop_depth)
        return depth - with_grad, with_grad

    def _recurrence_input(self, prelude_output: Tensor) -> Tensor:
        if self.huginn_prelude_norm_module is None:
            return prelude_output
        return self.huginn_prelude_norm_module(prelude_output)

    def _recur(
        self,
        state: Tensor,
        input_embeds: Tensor,
        start_layer: int,
        end_layer: int,
        ctx: _ForwardContext,
        module_repeats: int = 1,
    ) -> Tensor:
        state = super()._recur(state, input_embeds, start_layer, end_layer, ctx, module_repeats)
        if self.huginn_recurrent_exit_norm_module is not None:
            state = self.huginn_recurrent_exit_norm_module(state)
        return state

    def _run_recurrences(self, state: Tensor, input_embeds: Tensor, ctx: _ForwardContext) -> Tensor:
        """The drawn depth: the first recurrences without gradients, the last `backprop_depth` with them."""
        steps_without_grad, steps_with_grad = self._draw_recurrence_steps()
        with torch.no_grad():
            for _ in range(steps_without_grad):
                state = self._recur(state, input_embeds, self.loop_start_layer, self.loop_end_layers, ctx)
        for _ in range(steps_with_grad):
            state = self._recur(state, input_embeds, self.loop_start_layer, self.loop_end_layers, ctx)
        return state

    def identity_recurrent_state(self, input_embeds: Tensor, cache_len: int, example_ids: Tensor, seed: int) -> Tensor:
        """Evaluation state of the paper: normal-like, fixed by each example's global identity, seed and position.

        Unlike `Huginn.initialize_recurrent_state`, whose evaluation state depends on the row's
        position in the batch, the per-row term is a phase in [0, 2 pi) derived from the example
        id and the seed, so an example gets the same state in any batch. 16,777,213 is the
        largest prime below 2**24, which keeps the phase exact in FP32; 104,729 and 130,363
        are mixing primes.
        """
        modulus = 16_777_213
        identity = torch.remainder(example_ids.to(device=input_embeds.device, dtype=torch.long), modulus)
        mixed = torch.remainder(identity * 104_729 + (int(seed) % modulus) * 130_363, modulus)
        batch_phase = mixed.to(dtype=torch.float32).mul_(2.0 * math.pi / modulus).view(-1, 1, 1)
        return self._hashed_normal_state(
            input_embeds, cache_len, batch_phase, math.sqrt(2.0) * (1.0 / (self.model_dim ** 0.5)))

    def terminal_kv_forward(
        self,
        tokens: Tensor,
        example_ids: Tensor,
        seed: int,
        cache: Optional[List[Tuple[Optional[Tensor], Optional[Tensor], int]]] = None,
        full_logits: bool = False,
        depth: Optional[int] = None,
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor, int]]]:
        """Evaluation forward with terminal KV: earlier tokens keep only their last-recurrence KV.

        The cache holds one bank per physical layer (prelude, recurrent blocks, coda), each (k, v, length).
        Prelude and coda layers append the chunk's KV. In the recurrent blocks, every recurrence but the
        last attends to the cached terminal KV plus the chunk's own KV of that recurrence (only within the
        chunk when the cache is empty), and the last recurrence appends. The state starts from
        `identity_recurrent_state` and runs `depth` recurrences (default `loop_times`). Returns the logits
        (last position unless `full_logits`) and the cache.
        """
        if self.training or self.context_parallel_size != 1:
            raise RuntimeError("terminal-KV evaluation needs eval mode and no context parallel")
        _, seq_len = tokens.shape
        cache = [(None, None, 0)] * self.num_layers if cache is None else list(cache)
        cache_len = cache[0][2]
        hidden = self.embed(tokens)
        freq_cis = None if self.rope is None else self.rope.get_freqs_cis(cache_len, cache_len + seq_len, hidden.device)

        def run_layer(layer_id: int, layer_cache, keep: bool) -> None:
            nonlocal hidden
            hidden, _, updated = self.layers[layer_id](
                hidden, freq_cis, None, moe_router_load_balancing_type=None, fp32_attn_output=False,
                deterministic=True, cache=layer_cache,
            )
            if keep:
                cache[layer_id] = updated

        for layer_id in range(self.loop_start_layer):
            run_layer(layer_id, cache[layer_id], keep=True)
        input_embeds = self._recurrence_input(hidden)
        hidden = self.identity_recurrent_state(input_embeds, cache_len, example_ids, seed)
        depth = self.loop_times if depth is None else depth
        for step in range(depth):
            last = step + 1 == depth
            if self.input_injection is not None:
                hidden = self.input_injection(hidden, input_embeds)
            for layer_id in range(self.loop_start_layer, self.loop_end_layers):
                terminal = cache[layer_id]
                run_layer(layer_id, terminal if last or cache_len > 0 else None, keep=last)
            if self.huginn_recurrent_exit_norm_module is not None:
                hidden = self.huginn_recurrent_exit_norm_module(hidden)
        for layer_id in range(self.loop_end_layers, self.num_layers):
            run_layer(layer_id, cache[layer_id], keep=True)
        return self.output(hidden if full_logits else hidden[:, -1:], None, None), cache
