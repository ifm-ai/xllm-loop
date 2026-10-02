from dataclasses import dataclass
from logging import getLogger
from typing import Optional
from pathlib import Path
from typing import List, Dict
import os
import json

from xllm.configuration import Config
from xllm.distributed import (
    model_parallel_is_initialized,
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_context_parallel_rank,
    get_model_parallel_rank
)
from xllm.utils import get_default_training_half


logger = getLogger()


@dataclass
class TokenizerConf(Config):
    # Config for processing different types of data
    type: str = 'llama2'  # Supports llama2/sentencepiece, llama3, or huggingface
    path: Optional[str] = None  # Path to the tokenizer folder
    num_reserved_special_tokens: int = 0  # Reserved tokens used by SentencePiece tokenizer, multiples of 256
    template_dict_json: Optional[str] = None  # Allows different chat templates to be used in runtime for HF tokenizer. The json str should have a template name to template file relative path mapping.

    @property
    def tokenizer_path(self) -> str:
        assert self.path is not None
        return self.path

    def __post_init__(self):
        if self.template_dict_json is not None:
            json.loads(self.template_dict_json)


@dataclass
class DataLoaderConfig(Config):
    buffer_size: int = 512  # The num of tokens in buffer is buffer_size * seq_len
    num_workers: int = 1  # Tokenizer worker threads during buffer refill
    packing_type: str = "simple"  # Sequence packing strategy: simple = concatenation, bestfit = find best sequence to pack into to reduce truncations
    skip_long_docs: bool = False  # True skips overlong non-text docs; otherwise they are split into seq_len chunks.
    max_consecutive_skips: int = 10000  # Fail a source that repeatedly produces no tokenized sample


@dataclass
class SlurmConf(Config):
    is_slurm_job: bool = False
    global_rank: int = -1  # The rank of current process in entire distributed training job. 0 is the main process
    world_size: int = -1  # The total ranks of the entire distributed training job

    # Slurm parameters for async eval jobs
    partition: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    mem_gb: Optional[int] = None
    time: int = 3 * 24 * 60  # 3 days for default
    timeout_min: int = 3 * 24 * 60  # 3 days for default
    gpus_per_node: int = 8
    cpus_per_task: int = 10
    additional_params: Optional[str] = None

    def set_values(self, is_slurm_job, global_rank, world_size):
        self.is_slurm_job = is_slurm_job
        self.global_rank = global_rank
        self.world_size = world_size

    @property
    def is_master(self) -> bool:
        return self.global_rank == 0

    def __post_init__(self):
        if self.additional_params is not None:
            json.loads(self.additional_params)


@dataclass
class OptimConf(Config):
    lr: float = 1e-3  # AdamW learning rate after warmup
    warmup: int = 2000  # Num of steps for LR warmup. 0 means no warmup
    weight_decay: float = 0.1  # Decoupled weight decay
    epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.98
    adamw_fused: Optional[bool] = None  # None uses PyTorch's default AdamW kernel; True selects the fused kernel
    clip: float = 1.0  # Global gradient L2 norm clipping threshold, before applying loss scale.
    scheduler: str = "cosine"  # LR scheduler: linear, cosine, constant, or wsd (warmup-stable-decay)
    lr_init_ratio: float = 1e-4  # The initial LR = lr_init_ratio * lr
    lr_end_ratio: float = 0.01  # The ending LR = lr_end_ratio * lr. Also the end value if using a cosine scheduler
    cycles: float = 1.0  # Cycles of half period of cosine scheduler. 1 means decaying from peak to lr_end_ratio
    wsd_decay_duration_steps: Optional[int] = None  # Number of steps in the WSD decay phase. Decay starts at total_steps - wsd_decay_duration_steps.
    wsd_decay_style: str = 'cosine'  # The curve type of decay phase in WSD scheduler: cosine or linear

    def __post_init__(self):
        assert self.scheduler in ['linear', 'cosine', 'constant', 'wsd']
        assert self.warmup >= 0
        if self.scheduler == 'wsd':
            assert self.wsd_decay_duration_steps is not None, 'decay duration steps is required for WSD.'


@dataclass
class ModelConf(Config):
    arch: str = "transformer"  # Architecture: transformer or gekko
    ddp_backend: str = 'fsdp1'  # Distributed parameter sharding implementation: fsdp1 or fsdp2
    num_layers: int = 8
    model_dim: int = 1024  # Dimensions of token hidden state and residual stream
    # Attention
    chunk_size: int = 2048  # Used for local attention, memory update in Gekko and prefilling.
    num_heads: int = 1  # Number of Query attention heads
    num_kv_heads: Optional[int] = None  # Number of Key/Value heads. None means = num_heads. Uses GQA when this is small.
    head_dim: Optional[int] = None  # Dimension of each Q/K head. None means = model_dim // num_heads
    v_head_dim: Optional[int] = None  # Dimension of each V head in Gekko. None means = head_dim
    qknorm: bool = False  # Whether to apply RMSNorm to Q/K heads in Transformer
    attn_act_func: str = 'softmax'  # softmax or softdelta
    apply_attn_gate: bool = False  # Whether to apply gating to attention output in Transformer
    attn_gate_func: str = 'softplus'  # silu or softplus
    causal_attn_backend: Optional[str] = None  # Attention operator backend: None = regular implementation, or choose between swift/flash/xattn
    sca_backend: str = 'swift'  # Sliding Chunk Attention (SCA) operator backend (swift/flash/xattn).
    apply_bias_term: bool = False  # Whether the Gekko attention output gating projection W_r includes a bias term.
    ffn_hidden_dim: int = 2048  # Hidden dimension in dense FFN. MoE expert dimensions is set by expert_inter_dim.
    # Adaptive Working Memory (AWM)
    awm_orthogonal_update: bool = False  # Whether to apply orthogonal update when updating the memory in AWM
    # MoE & MoVA
    num_values: int = 0  # Num of value projections available for routing to in MoVA. 0 means MoVA off. Requires using MoE as well when using MoVA.
    num_activated_values: int = 0  # Num of activated value projections per token (top-k) in MoVA
    mova_backend: str = 'sequential'  # sequential, nested, torch, or te
    num_experts: int = 0  # Num of experts to be routed in each MoE layer (not counting shared experts). 0 means using dense FFN
    num_activated_experts: int = 0  # Num of activated experts per token (top-k)
    num_shared_experts: int = 0  # Num of shared experts for each token, in addition to routed experts
    num_dense_layers: Optional[int] = None  # Number of initial dense layers. If None, defaults to 0 when MoE is enabled, otherwise to all layers.
    expert_inter_dim: int = 0  # Hidden dimension of the FFN in a single expert.
    moe_router_score_func: str = 'sigmoid'  # sigmoid or softmax
    moe_router_bias: bool = False  # Whether to include the router bias, only affects top-k selection.
    moe_router_bias_update_rate: Optional[float] = None  # Update rate for the router bias during backprop
    moe_router_scaling_factor: Optional[float] = None  # The scaling factor for outputs from selected experts. None means no scaling
    moe_expert_backend: str = 'sequential'  # sequential, nested, torch, or te
    moe_permutation_backend: str = 'torch'  # Token permutation and unpermutation backend: torch or triton
    # Causal conv1d
    causal_conv_width: int = 4  # Width of Q/K/V causal convolution kernel in Gekko. Currently supports 1-4 tokens
    causal_conv_backend: str = 'triton'  # triton or fla
    causal_conv_weight_normalization: bool = True  # Whether to apply softmax normalization for causal convolution weights
    # Initialization
    init_mode: str = 'he'  # he, gaussian, xavier, or none (skipping init)
    init_std: Optional[float] = None  # Init std for gaussian; If None, deriving from input dimensions. he/xavier requires None
    init_embed_std: Optional[float] = None  # Std used to truncate normal init of embedding weights. If None, = 1 / sqrt(model_dim)
    init_logits_std: Optional[float] = None  # Std used to truncate normal init of output projection weights. If None, = 1 / sqrt(model_dim)
    # Input & output
    vocab_size: int = -1  # Defined later by tokenizer
    output_size: int = -1  # Output size. -1 means using vocab_size
    # Normalization
    timenorm_num_groups: int = 32  # Num of groups of features used in timestep decay normalization in Gekko.
    timenorm_beta1: float = 0.999  # Used in timestep decay normalization.
    timenorm_beta2: float = 0.9999  # Used in timestep decay normalization.
    timenorm_backend: str = 'cub'  # cub or chunkwise
    layernorm_num_groups: int = 1  # Num of groups of features in LayerNorm/RMSNorm.
    memory_efficient_norm: bool = False  # Whether to use norm output for backward to save activation memory. Unsupported by 'cub' timestep norm backend.
    norm_affine: bool = True  # Enable learnable scaling in attention/FFN input norms (plus bias for LayerNorm).
    timenorm_eps: float = 1e-5  # Used in timestep decay normalization.
    layernorm_eps: float = 1e-5
    rmsnorm_eps: float = 1e-6
    apply_rmsnorm: bool = False  # True = RMSNorm, False = LayerNorm
    residual_func: str = "base"  # Residual connection type: base/add means standard addition, ortho for orthogonal residuals.
    residual_heads: Optional[int] = None  # Num of feature groups for orthogonal residuals. Required when using residual_func='ortho'.
    # Rope base
    rope_base: float = 100000  # RoPE frequency base
    rope_head_dim: Optional[int] = None  # RoPE dimensions per Q/K head. None means full head_dim, 0 disables RoPE.
    # Dropout rates
    dropout: float = 0.0  # Dropout rate for embeddings and for sublayer outputs before residual connections.
    hidden_dropout: float = 0.0  # Dropout rate for attention activations before output projection and for FFN intermediate activations.
    attention_dropout: float = 0.0  # Dropout rate for attention weights after softmax.

    swiglu: bool = False  # Whether to use SwiGLU in dense FFN. Otherwise uses SiLU.
    scale_emb: bool = False  # Whether to scale embedding output by sqrt(model_dim).
    layerwise_ckpt: bool = False  # Whether to enable activation checkpointing for each layer to save memory by recomputing forward during backprop.

    # Looped models (xllm/models/looped.py). For arch='transformer', layers [loop_start_layer, loop_end_layers)
    # run loop_times times; for arch='huginn', loop_times is the number of recurrences and the recurrent span
    # comes from the huginn_* layer counts, so loop_start_layer and loop_end_layers do not apply.
    loop_times: int = 1
    loop_start_layer: int = 0
    loop_end_layers: Optional[int] = None  # None means num_layers
    loop_input_injection: str = "none"  # none or diagonal; re-injects the embeddings at each loop entry
    loop_diagonal_dt_init: float = 1.0
    loop_diagonal_a_init: float = 1.0

    # Input injection for the unrolled (loop_times 1) controls, dense or MoE: the output of
    # dense_prelude_source_layer is injected before each of dense_prelude_input_layers.
    dense_prelude_input_injection: str = "none"  # none or diagonal
    dense_prelude_input_layers: str = ""  # Comma separated layer ids receiving the injection
    dense_prelude_source_layer: int = -1  # Layer whose output is injected
    dense_prelude_diagonal_dt_init: float = 1.0
    dense_prelude_diagonal_a_init: float = 1.0

    # Huginn (arch='huginn'): prelude, recurrent block and coda, with an optional H/L state split.
    huginn_prelude_layers: int = 1
    huginn_recurrent_layers: Optional[int] = None  # None means num_layers - prelude - coda
    huginn_coda_layers: int = 1
    huginn_state_init: str = "zero"  # zero, input or normal
    huginn_input_injection: str = "diagonal"  # none or diagonal
    huginn_diagonal_dt_init: float = 1.0
    huginn_diagonal_a_init: float = 1.0
    huginn_hierarchical_state: str = "none"  # none, shared_hl or split_hl
    huginn_hierarchical_h_cycles: int = 2
    huginn_hierarchical_l_cycles: int = 3
    huginn_split_module_repeats: int = 1  # Repeats of each split H/L module per state update
    huginn_depth_control: bool = False  # Use DepthControlledHuginn: the training depth and injection options below (dense, one state)
    # Training depth (huginn_depth_control only): fixed loop_times, or capped Poisson-lognormal draws with mean loop_times.
    huginn_sampling_scheme: str = "fixed"  # fixed or poisson-lognormal-capped
    huginn_antithetic_sampling: bool = False  # Pair consecutive draws as u and 1 - u
    huginn_poisson_lognormal_target_mean: float = 5.0
    huginn_poisson_lognormal_sigma: float = 0.5
    huginn_poisson_lognormal_max: int = 64
    huginn_backprop_depth: Optional[int] = None  # Backpropagate through the last N recurrences; None means all
    huginn_depth_prior: str = "none"  # none or learned: training depths come from a categorical prior trained by REINFORCE
    huginn_depth_prior_entropy: float = 0.0  # Entropy coefficient lambda_H of the learned depth prior
    # Input-injection variants (huginn_depth_control only).
    huginn_prelude_norm: str = "none"  # none or rms: RMSNorm with weight on the injected prelude output (Parcae-Decay)
    huginn_recurrent_exit_norm: str = "none"  # none or parameterless_rms: RMSNorm after every recurrence (Huginn-Linear)
    huginn_prelude_orthogonal: bool = False  # OrthoInj: drop the decayed state's component along the injected update
    huginn_ortho_projection_eps: float = 1e-6  # Added to the injected update's squared norm in that projection

    fused_block: bool = False  # Whether to use custom fused forward/backward implementations for training blocks; incompatible with layerwise_ckpt.
    # Recomputation switches
    recompute_q: bool = False  # Whether to recompute Q activations in fused block during backprop to save activation memory.
    recompute_v: bool = False  # Whether to recompute K/V activations in fused block.
    recompute_attention: bool = False  # Whether to recompute attention outputs in fused block. Applies to sliding chunk attention in Gekko.
    recompute_awk: bool = False  # Whether to recompute AWM outputs in fused block in Gekko.
    recompute_fc1_out: bool = False  # Whether to recompute fc1 outputs in dense FFNs or shared experts in fused block.
    recompute_fc3_out: bool = False  # Whether to recompute fc3 outputs in dense SwiGLU FFNs or shared experts in fused block.
    recompute_router: bool = False  # Whether to recompute router scores for MoE/MoVA in fused block.

    recompute_logits: bool = False  # Whether to recompute logits in output layer. Independent of fused_block. True = recompute logits and compute grad_X/grad_W during backward, False = precompute grad_X/grad_W during forward.
    fused_output_layer: bool = True  # Fused norm/logits/cross-entropy in training. False computes them with separate modules, which is numerically different.

    def __post_init__(self):
        assert self.arch in ['transformer', 'huginn', 'gekko']
        assert self.ddp_backend in ['fsdp1', 'fsdp2']

        assert self.causal_attn_backend in [None, "swift", "flash", 'xattn']
        assert self.sca_backend in ["swift", "flash", 'xattn']
        assert self.mova_backend in ['sequential', 'nested', 'torch', 'te']
        assert self.moe_expert_backend in ['sequential', 'nested', 'torch', 'te']
        assert self.moe_permutation_backend in ['torch', 'triton']
        assert self.timenorm_backend in ['cub', 'chunkwise']
        assert self.causal_conv_backend in ['fla', 'triton']

        if self.fused_block:
            assert not self.layerwise_ckpt, 'disable layer-wise checkpointing when using fused block'

        if self.num_kv_heads is not None:
            assert self.num_heads % self.num_kv_heads == 0

        assert self.attn_act_func in ['softmax', 'softdelta']
        assert self.attn_gate_func in ['silu', 'softplus']
        assert self.residual_func in ['base', 'add', 'ortho']

        if self.num_values > 0:
            assert 0 < self.num_activated_values <= self.num_values
            assert self.num_experts > 0, "MoVA requires MoE."

        if self.num_experts > 0:
            assert 0 < self.num_activated_experts <= self.num_experts
            assert self.num_dense_layers is None or self.num_dense_layers <= self.num_layers
            assert self.expert_inter_dim > 0
            assert self.moe_router_score_func in ['softmax', 'sigmoid']
            if self.moe_router_bias:
                assert self.moe_router_bias_update_rate > 0.0, f"Invalid MoE router bias update rate: {self.moe_router_bias_update_rate}"
        else:
            assert self.num_dense_layers is None

        assert self.timenorm_num_groups <= 256

        assert 0 <= self.dropout < 1
        assert 0 <= self.attention_dropout < 1
        assert 0 <= self.hidden_dropout < 1

        if self.arch in ['transformer', 'huginn']:
            if self.fused_block:
                assert self.causal_attn_backend is not None, 'requiring efficient attention when using fused block'
                if self.qknorm:
                    assert self.recompute_v and self.recompute_q, "QK-norm w. fused block requires qkv re-computation."

        assert self.loop_times >= 1
        assert 0 <= self.loop_start_layer < self.num_layers
        loop_end_layers = self.num_layers if self.loop_end_layers is None else self.loop_end_layers
        assert self.loop_start_layer < loop_end_layers <= self.num_layers
        assert self.loop_input_injection in ['none', 'diagonal']
        assert self.loop_diagonal_dt_init > 0
        assert self.loop_diagonal_a_init > 0

        assert self.dense_prelude_input_injection in ['none', 'diagonal']
        assert self.dense_prelude_diagonal_dt_init > 0
        assert self.dense_prelude_diagonal_a_init > 0
        dense_prelude_input_layers = [
            int(layer) for layer in self.dense_prelude_input_layers.split(',') if layer.strip()
        ]
        if self.dense_prelude_input_injection == 'none':
            assert not dense_prelude_input_layers
            assert self.dense_prelude_source_layer < 0
        else:
            assert self.arch == 'transformer'
            assert self.loop_input_injection == 'none'
            assert self.loop_times == 1
            assert dense_prelude_input_layers
            assert 0 <= self.dense_prelude_source_layer < min(dense_prelude_input_layers)
            assert all(layer < self.num_layers for layer in dense_prelude_input_layers)

        if self.huginn_depth_control:
            assert self.arch == 'huginn', "huginn_depth_control requires arch='huginn'."
            assert self.num_experts == 0 and self.huginn_hierarchical_state == 'none', \
                "DepthControlledHuginn is dense and has one recurrent state."
        else:
            depth_control_defaults = dict(
                huginn_sampling_scheme='fixed', huginn_antithetic_sampling=False, huginn_backprop_depth=None,
                huginn_depth_prior='none', huginn_depth_prior_entropy=0.0, huginn_prelude_norm='none',
                huginn_recurrent_exit_norm='none', huginn_prelude_orthogonal=False,
            )
            changed = [name for name, default in depth_control_defaults.items() if getattr(self, name) != default]
            assert not changed, f"{changed} require huginn_depth_control=True."
        assert self.huginn_sampling_scheme in ['fixed', 'poisson-lognormal-capped']
        assert self.huginn_backprop_depth is None or self.huginn_backprop_depth >= 1
        if self.huginn_sampling_scheme == 'poisson-lognormal-capped':
            # Evaluation runs loop_times recurrences, the mean of the training depth.
            assert self.huginn_poisson_lognormal_target_mean == self.loop_times
            assert 1 < self.loop_times < self.huginn_poisson_lognormal_max
            assert self.huginn_poisson_lognormal_sigma > 0
        else:
            assert not self.huginn_antithetic_sampling, "antithetic draws require a sampled depth."
        assert self.huginn_depth_prior in ['none', 'learned']
        if self.huginn_depth_prior == 'learned':
            # The prior starts from the capped PLN distribution over depths 1..64 and draws every depth itself.
            assert self.huginn_sampling_scheme == 'poisson-lognormal-capped' and self.huginn_poisson_lognormal_max == 64
            assert not self.huginn_antithetic_sampling
            assert self.huginn_depth_prior_entropy >= 0
        assert self.huginn_prelude_norm in ['none', 'rms']
        assert self.huginn_recurrent_exit_norm in ['none', 'parameterless_rms']
        if self.huginn_prelude_orthogonal:
            assert self.huginn_input_injection == 'diagonal' and self.huginn_ortho_projection_eps > 0
        assert self.huginn_split_module_repeats >= 1
        if self.huginn_split_module_repeats != 1:
            assert self.arch == 'huginn' and self.huginn_hierarchical_state == 'split_hl', (
                "huginn_split_module_repeats > 1 is only valid for split H/L."
            )

        if self.arch == 'huginn':
            assert self.loop_input_injection == 'none'
            recurrent_layers = (
                self.num_layers - self.huginn_prelude_layers - self.huginn_coda_layers
                if self.huginn_recurrent_layers is None
                else self.huginn_recurrent_layers
            )
            assert recurrent_layers > 0
            assert self.huginn_prelude_layers + recurrent_layers + self.huginn_coda_layers == self.num_layers
            assert self.huginn_state_init in ['zero', 'input', 'normal']
            assert self.huginn_input_injection in ['none', 'diagonal'] + (['linear'] if self.huginn_depth_control else [])
            assert self.huginn_diagonal_dt_init > 0
            assert self.huginn_diagonal_a_init > 0
            assert self.huginn_hierarchical_state in ['none', 'shared_hl', 'split_hl']
            assert self.huginn_hierarchical_h_cycles >= 1
            assert self.huginn_hierarchical_l_cycles >= 1
            if self.huginn_hierarchical_state != 'none':
                if self.huginn_hierarchical_state == 'split_hl':
                    assert recurrent_layers % 2 == 0, "Split H/L state requires an even number of recurrent layers."
                assert self.loop_times == self.huginn_hierarchical_h_cycles * (
                    self.huginn_hierarchical_l_cycles + 1
                )


@dataclass
class ValidConf(Config):
    # Tasks to evaluate
    ppl_file_list: str = ""  # Comma separated list of files to eval PPL
    task_root: str = ""  # Root directory of tasks
    task_list: str = ""  # Comma separated list of tasks

    batch_size: int = 32  # Eval batch size per DP process.
    seq_len: int = 8192  # Input seq_len in tokens for PPL eval.
    n_batches: int = -1  # Max batches per PPL file per DP process used for PPL eval (<= 0 for full evaluation).
    add_template: bool = False  # Applies chat template when tokenizing eval examples.

    # Decoding parameters
    use_sampling: bool = False  # Whether to sample during generations. False uses argmax greedy decoding.
    temperature: float = 1.0  # Divide logits by this value before sampling. Must be positive when sampling.
    top_k: int = 0  # Restrict sampling to the top-k tokens. 0 disables it.
    top_p: float = 0.0  # Cumulative prob threshold nucleus sampling. 0 disables it. >0 takes precedence over top_k.

    save_eval: bool = False  # Whether to save per-sample results to the eval output dir.

    @property
    def ppl_files(self) -> List[str]:
        paths = [path for path in self.ppl_file_list.split(",") if len(path) > 0]
        return paths

    @property
    def tasks(self) -> List[str]:
        tasks = [task for task in self.task_list.split(",") if len(task) > 0]
        assert len(tasks) == len(set(tasks))
        if "mmlu" in set(tasks):
            from xllm.eval.task.mmlu import MMLUTask
            tasks = [t for t in tasks if t != "mmlu"] + list(MMLUTask.mmlu_tasks.keys())
        
        if "arabic_mmlu" in set(tasks):
            from xllm.eval.task.arabic_mmlu import ArabicMMLUTask
            tasks = [t for t in tasks if t != "arabic_mmlu"] + list(ArabicMMLUTask.arabic_mmlu_tasks.keys())

        if "ruler" in set(tasks):
            from xllm.eval.task.ruler import RulerTask
            tasks = [t for t in tasks if t != "ruler"] + list(
                RulerTask.get_sub_tasks.keys()
            )
        return tasks

    def should_do_eval(self):
        return len(self.ppl_files) > 0 or len(self.tasks) > 0

    def __post_init__(self):
        # decoding params
        assert self.temperature >= 0
        assert self.top_k >= 0
        assert 0 <= self.top_p < 1

        # ppl files
        if len(self.ppl_files) > 0:
            assert len(self.ppl_files) == len(set(self.ppl_files))
            assert all(path.endswith(".jsonl") for path in self.ppl_files)
            assert all(Path(path).is_file() for path in self.ppl_files), self.ppl_files

        # tasks root dir
        if len(self.task_root) > 0:
            assert os.path.isdir(self.task_root), self.task_root

        # task dirs
        for task in self.tasks:
            task_dir = os.path.join(self.task_root, task)
            assert os.path.isdir(task_dir), task_dir


@dataclass
class TrainerConf(Config):
    slurm: Optional[SlurmConf]
    model: ModelConf
    optim: OptimConf
    valid: ValidConf
    tokenizer: TokenizerConf
    dataloader: DataLoaderConfig  # Configs for multi-source data loading.

    # Training data / iterator
    data: str = ""  # Comma seperated data mix. Each mix is formatted as path:weight:json_key:source_format. Weights need not sum to 1.

    # Output dir
    dump_dir: str = ""  # Root dir for training logs and checkpoints. Also searched for checkpoints to resume training.

    # Base model dir
    base_model_dir: Optional[str] = None  # Initializes training when dump_dir has no checkpoints to resume. Useful for a new training stage.
    reload_base_optim_state: bool = False  # Whether to also load optimizer state from the base model checkpoint.

    # Parallelism
    model_parallel_size: int = 1  # Num of model/tensor parallel.
    context_parallel_size: int = 1  # Num of context parallel. Splits sequences along the token dimension.
    fully_sharded_size: Optional[int] = None  # Num of FSDP shard group. If None, = DP.

    # Logging
    disable_workers_print: bool = False  # Whether to disable logging from non-main training processes (workers).
    disable_logging: bool = False  # Whether to disable metric logging via MetricLogger, including JSONL, TensorBoard, and W&B.

    # Training hyperparameters
    batch_size: Optional[int] = None  # Num of sequences per DP rank per step. Set either this or global_batch_size.
    global_batch_size: Optional[int] = None  # Total batch size across DP ranks. Must be divisible by DP * gradient_accumulation_steps; batch_size is derived from this.
    gradient_accumulation_steps: int = 1  # Microbatches per optimizer step; global_batch_size = DP * batch_size * gradient_accumulation_steps.
    seq_len: int = 8192  # Num of tokens in each sequence. Must be divisible by context_parallel_size * model.chunk_size.
    multi_segments: bool = True  # Whether to segment docs using BOS token. This will prevent attention across document boundaries.
    deterministic: bool = False  # Whether to request deterministic computations when supported.
    fp32_attn_output: bool = False # Whether to use high precision attention output (only supported in xattn backend).

    steps: int = 100000  # Total number of training steps.
    stop_step: Optional[int] = None  # Stops training early at this step, with a checkpoint (and eval); the LR schedule still spans `steps`. None means `steps`.
    dtype: str = get_default_training_half()  # Training compute dtype: bf16 (default), fp16, or fp32.
    loss_rescaling: bool = False  # Whether to enable dynamic loss scaling and overflow handling. Required for fp16.

    # MoE hyps
    moe_router_load_balancing_type: Optional[str] = None  # Router load balancing aux loss: dot or entropy. None disables it.
    moe_aux_loss_coeff: float = 0.0  # Coeff of router load balancing aux loss in the total training loss.
    moe_z_loss_coeff: float = 0.0  # Coeff for router z-loss.

    # FSDP parameters
    fp32_reduce_scatter: bool = True  # Whether to use fp32 for FSDP gradient reduction. Otherwise uses the training compute dtype.
    reshard_after_forward: bool = True  # Whether to release unsharded parameters after forward to save memory. Needs to gather them again for backward.
    forward_prefetch: bool = False  # Whether to prefetch parameters for better overlapping. Not supported by the current FSDP2 implementation.

    log_freq: int = 10  # Logs training metric summaries every N steps.
    dump_freq: int = 1000  # Saves checkpoint every N steps. Also save at the final step.
    eval_freq: int = 1000  # Evaluates every N steps. <= 0 disables it.
    gc_collect_freq: int = 1000  # Calls gc.collect() every N steps.
    sync_check_freq: int = 10000  # Interval for checking batch consistency within MP/CP groups before sequence splitting; <= 0 disables it.

    keep_n_last_checkpoints: int = 2  # Keeps the latest N checkpoints plus any retained eval checkpoints; -1 keeps all.
    keep_eval_checkpoints: bool = True  # Whether to additionally retain checkpoints used for evals.
    dcp_for_optimizer: bool = False  # Whether to use DCP optimizer state handling and loading. Also selects DCP for synchronous saves.
    async_checkpointing: bool = False  # Whether to save model and optimizer states async to reduce training stalls.
    async_eval_ngpus: int = -1  # Total GPUs for separate async eval jobs; <= 0 uses synchronous eval.

    # Wandb
    log_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

    # Random seed
    seed: int = 1
    restore_rng_state: bool = False  # Whether resume restores the per-rank Python/NumPy/torch/CUDA RNG states saved in training states.
    nccl_timeout: int = 3600  # NCCL timeout in seconds.

    cluster_check_level: int = 1  # Cluster check level at start. 0 will only log env. >=1 will run communication benchmarks in addition.

    @property
    def data_parallel_size(self) -> int:
        assert model_parallel_is_initialized()
        total_model_parallel_size = self.context_parallel_size * self.model_parallel_size
        assert self.slurm.world_size % total_model_parallel_size == 0
        size = self.slurm.world_size // total_model_parallel_size
        assert size == get_data_parallel_world_size()
        return size

    @property
    def data_parallel_rank(self) -> int:
        assert model_parallel_is_initialized()
        rank = self.slurm.global_rank // self.model_parallel_size // self.context_parallel_size
        assert rank == get_data_parallel_rank()
        return rank

    @property
    def context_parallel_rank(self) -> int:
        assert model_parallel_is_initialized()
        rid = (self.slurm.global_rank // self.model_parallel_size) % self.context_parallel_size
        assert rid == get_context_parallel_rank()
        return rid

    @property
    def model_parallel_rank(self) -> int:
        assert model_parallel_is_initialized()
        rid = self.slurm.global_rank % self.model_parallel_size
        assert rid == get_model_parallel_rank()
        return rid

    def __post_init__(self):
        # sanity check
        assert self.dtype in ["bf16", "fp16", "fp32"]
        if self.dtype == "fp16":
            assert self.loss_rescaling, "loss rescaling needed for fp16!"
        assert not (self.loss_rescaling and self.model.huginn_depth_prior == 'learned'), \
            "the learned depth prior must see every optimizer step, and loss rescaling skips steps on overflow."

        # MoE
        if self.model.num_experts > 0:
            assert self.model.num_experts % self.model_parallel_size == 0, f"{self.model.num_experts}/{self.model_parallel_size}"
        if self.moe_router_load_balancing_type is not None:
            assert self.moe_router_load_balancing_type in ['dot', 'entropy']
            assert self.moe_aux_loss_coeff >= 0.0
        assert self.gradient_accumulation_steps >= 1
        assert self.stop_step is None or 0 < self.stop_step <= self.steps, f"{self.stop_step}/{self.steps}"
        # freq checks
        assert self.dump_freq % self.log_freq == 0
        assert self.eval_freq < 0 or (self.eval_freq % self.log_freq == 0)

        # async eval
        if self.async_eval_ngpus > 0:
            assert self.keep_eval_checkpoints
            assert self.async_eval_ngpus % self.model_parallel_size == 0, f"{self.async_eval_ngpus}/{self.model_parallel_size}"
            assert self.async_eval_ngpus % self.slurm.gpus_per_node == 0, f"{self.async_eval_ngpus}/{self.slurm.gpus_per_node}"
        if self.async_eval_ngpus > 0 or self.keep_eval_checkpoints:
            # be sure that we dump for each eval
            assert self.eval_freq < 0 or (self.eval_freq % self.dump_freq == 0)

        assert self.seq_len % self.model.chunk_size == 0, f"{self.seq_len}/{self.model.chunk_size}"
        assert self.valid.seq_len % self.model.chunk_size == 0, f"{self.valid.seq_len}/{self.model.chunk_size}"

        assert self.model.num_heads % self.model_parallel_size == 0, f"{self.model.num_heads}/{self.model_parallel_size}"
        if self.model.arch in ['transformer', 'huginn']:
            assert self.model.layernorm_num_groups == 1 or self.model.layernorm_num_groups % self.model_parallel_size == 0, \
                f"{self.model.layernorm_num_groups}/{self.model_parallel_size}"
        elif self.model.arch in ['gekko']:
            assert self.model.layernorm_num_groups % self.model_parallel_size == 0, \
                f"{self.model.layernorm_num_groups}/{self.model_parallel_size}"
