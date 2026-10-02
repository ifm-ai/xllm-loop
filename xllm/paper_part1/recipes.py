"""Recipe registry for the xLLM Part I release.

Values are copied from the Part 1 paper code; model fields use the current
`ModelConf` field names.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

from xllm.config import ModelConf, OptimConf


OptimizerKernel = Literal["pytorch_default", "pytorch_fused"]

SCHEDULER_STEPS = 119_210
# Execution stops on the shared 500B schedule: tokens per activated parameter, or an exact step.
TPP_TARGETS: Mapping[str, float] = MappingProxyType({"20tpp": 20, "1tpp": 1})
ENDPOINT_STEPS: Mapping[str, int] = MappingProxyType({"336b": 80_000, "500b": SCHEDULER_STEPS})

_COMMON_TRAINER_FIELDS: Mapping[str, object] = MappingProxyType({
    "seed": 1,
    "dtype": "bf16",
    "deterministic": True,
    "multi_segments": True,
    "loss_rescaling": False,
    "fp32_reduce_scatter": True,
    "fully_sharded_size": None,
    "reshard_after_forward": True,
    "forward_prefetch": False,
    "async_checkpointing": False,
    # RNG states are not restored on resume.
    "restore_rng_state": False,
})
_DENSE_FAMILY_FIELDS: Mapping[str, object] = MappingProxyType({
    "moe_router_load_balancing_type": None,
    "moe_aux_loss_coeff": 0.0,
    "moe_z_loss_coeff": 0.0,
})
_MOE_FAMILY_FIELDS: Mapping[str, object] = MappingProxyType({
    "moe_router_load_balancing_type": "dot",
    "moe_aux_loss_coeff": 0.001,
    "moe_z_loss_coeff": 0.0,
})


@dataclass(frozen=True)
class Recipe:
    """One Part 1 recipe: model fields and optimizer kernel; the schedule is shared."""

    name: str
    optimizer_kernel: OptimizerKernel
    model_fields: Mapping[str, object]  # ModelConf fields
    scheduler_steps: int = SCHEDULER_STEPS

    @property
    def trainer_fields(self) -> Mapping[str, object]:
        family = _MOE_FAMILY_FIELDS if self.model_fields["num_experts"] else _DENSE_FAMILY_FIELDS
        return MappingProxyType({**_COMMON_TRAINER_FIELDS, **family})

    def model_config(self) -> ModelConf:
        return ModelConf.from_dict(dict(self.model_fields))

    def optimizer_config(self) -> OptimConf:
        """AdamW, 200 warmup steps, cosine decay to 0.01x over the scheduler steps (no WSD)."""
        return OptimConf(
            lr=6e-4,
            warmup=200,
            weight_decay=0.1,
            epsilon=1e-8,
            beta1=0.9,
            beta2=0.98,
            adamw_fused=True if self.optimizer_kernel == "pytorch_fused" else None,
            clip=1.0,
            scheduler="cosine",
            lr_init_ratio=1e-4,
            lr_end_ratio=0.01,
        )


def _model_fields(
    *,
    arch: str = "transformer",
    num_layers: int = 28,
    loop_times: int = 1,
    moe: bool = False,
    recompute: bool = True,
    **overrides: object,
) -> Mapping[str, object]:
    fields: dict[str, object] = {
        "arch": arch, "num_layers": num_layers, "model_dim": 1536, "num_heads": 24, "num_kv_heads": 6,
        "head_dim": 64, "attn_act_func": "softmax", "qknorm": False, "apply_attn_gate": False,
        "attn_gate_func": "silu", "ffn_hidden_dim": 4352, "vocab_size": 250624, "output_size": -1,
        "apply_bias_term": True, "apply_rmsnorm": True, "layernorm_num_groups": 1,
        "memory_efficient_norm": False, "norm_affine": True, "rmsnorm_eps": 1e-6,
        "dropout": 0.0, "hidden_dropout": 0.0, "attention_dropout": 0.0,
        "rope_base": 1_000_000, "rope_head_dim": 64, "swiglu": True, "scale_emb": False,
        "layerwise_ckpt": False, "causal_attn_backend": "flash", "chunk_size": 1024,
        "init_mode": "gaussian", "init_std": 0.02, "init_embed_std": None, "init_logits_std": None,
        "fused_block": True, "fused_output_layer": True,
        "loop_times": loop_times, "loop_start_layer": 0, "loop_end_layers": None, "loop_input_injection": "none",
        "loop_diagonal_dt_init": 1.0, "loop_diagonal_a_init": 1.0,
        "dense_prelude_input_injection": "none", "dense_prelude_input_layers": "",
        "dense_prelude_source_layer": -1, "dense_prelude_diagonal_dt_init": 1.0,
        "dense_prelude_diagonal_a_init": 1.0,
        "num_experts": 25 if moe else 0, "num_activated_experts": 2 if moe else 0, "num_shared_experts": 0,
        "num_values": 0, "num_activated_values": 0, "num_dense_layers": None,
        "expert_inter_dim": 2432 if moe else 0, "moe_router_score_func": "sigmoid", "moe_router_bias": False,
        "moe_router_bias_update_rate": None, "moe_router_scaling_factor": None,
        "moe_expert_backend": "torch" if moe else "sequential", "moe_permutation_backend": "torch",
        "recompute_q": recompute, "recompute_v": recompute, "recompute_attention": recompute,
        "recompute_fc1_out": recompute, "recompute_fc3_out": recompute,
        "recompute_router": False, "recompute_logits": False,
    }
    if arch == "huginn":
        fields.update({
            "huginn_prelude_layers": 8, "huginn_recurrent_layers": 12, "huginn_coda_layers": 8,
            "huginn_state_init": "normal", "huginn_input_injection": "diagonal",
            "huginn_diagonal_dt_init": 1.0, "huginn_diagonal_a_init": 1.0,
            "huginn_hierarchical_state": "none", "huginn_hierarchical_h_cycles": 2,
            "huginn_hierarchical_l_cycles": 3, "huginn_split_module_repeats": 1,
        })
    fields.update(overrides)
    return MappingProxyType(fields)


# name: (optimizer kernel, model fields)
_RECIPES: Mapping[str, Recipe] = MappingProxyType({
    name: Recipe(name, kernel, fields)
    for name, (kernel, fields) in {
        "dense_d28": ("pytorch_default", _model_fields()),
        "dense_d112": ("pytorch_default", _model_fields(num_layers=112)),
        "dense_ouro": ("pytorch_default", _model_fields(loop_times=4, recompute=False)),
        "dense_middle": ("pytorch_default", _model_fields(
            arch="huginn", loop_times=8, recompute=False, huginn_state_init="input",
            huginn_input_injection="none")),
        "dense_ouro_raw_injection": ("pytorch_default", _model_fields(
            loop_times=4, loop_input_injection="diagonal")),
        "dense_middle_prelude_injection": ("pytorch_default", _model_fields(
            arch="huginn", loop_times=8, huginn_state_init="input")),
        "dense_huginn": ("pytorch_default", _model_fields(arch="huginn", loop_times=8)),
        "dense_shared_hl": ("pytorch_default", _model_fields(
            arch="huginn", loop_times=8, huginn_hierarchical_state="shared_hl")),
        "dense_split_hl": ("pytorch_default", _model_fields(
            arch="huginn", loop_times=8, huginn_hierarchical_state="split_hl")),
        "moe_ff112": ("pytorch_fused", _model_fields(num_layers=112, moe=True, recompute=False)),
        "moe_ff112_prelude_injection": ("pytorch_fused", _model_fields(
            num_layers=112, moe=True, recompute=False, dense_prelude_input_injection="diagonal",
            dense_prelude_input_layers="8,20,32,44,56,68,80,92", dense_prelude_source_layer=7)),
        "moe_ouro": ("pytorch_default", _model_fields(loop_times=4, moe=True)),
        "moe_middle": ("pytorch_default", _model_fields(
            arch="huginn", loop_times=8, moe=True, huginn_state_init="input", huginn_input_injection="none")),
        "moe_middle_input_injection": ("pytorch_default", _model_fields(
            arch="huginn", loop_times=8, moe=True, huginn_state_init="input")),
        "moe_huginn": ("pytorch_default", _model_fields(arch="huginn", loop_times=8, moe=True)),
        "moe_shared_hl": ("pytorch_default", _model_fields(
            arch="huginn", loop_times=8, moe=True, huginn_hierarchical_state="shared_hl")),
        # loop_end_layers has no effect for arch=huginn; it is kept to match the paper config.
        "dense_split_hl_x2": ("pytorch_default", _model_fields(
            arch="huginn", loop_times=8, loop_end_layers=28, huginn_hierarchical_state="split_hl",
            huginn_split_module_repeats=2)),
    }.items()
})


def recipe_names() -> tuple[str, ...]:
    """Return the recipe names in registry order."""
    return tuple(_RECIPES)


def target_names() -> tuple[str, ...]:
    """Return the training targets."""
    return ("20tpp", *ENDPOINT_STEPS, "1tpp")


def get_recipe(name: str) -> Recipe:
    return _RECIPES[name]
