"""Recipe registry for the xLLM Part II release.

Each recipe reproduces one main-text row of *Towards Looped Models Done Right,
Part II: Rethinking at Fixed Points*: its model, optimizer and schedule. Every
recipe trains on a global batch of 512 sequences of 8,192 tokens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from xllm.config import ModelConf, OptimConf
from xllm.paper_part2.eval_protocol import SEQUENCE_LENGTH


GLOBAL_BATCH_SEQUENCES = 512


@dataclass(frozen=True)
class Scale:
    """Width and training length of one model scale."""

    model_dim: int
    num_heads: int
    num_kv_heads: int
    ffn_hidden_dim: int
    steps: int


SCALES: Mapping[str, Scale] = MappingProxyType({
    "S": Scale(1536, 24, 6, 4352, 5_120),
    "M": Scale(3072, 48, 12, 8704, 20_480),
    "L": Scale(6144, 96, 24, 17408, 81_920),
})


# Fixed PLN-5: depth ~ capped Poisson-lognormal with mean 5 (sigma 0.5, cap 64), backpropagating the last 10.
PLN5_FIELDS: Mapping[str, object] = MappingProxyType({
    "huginn_sampling_scheme": "poisson-lognormal-capped",
    "huginn_poisson_lognormal_target_mean": 5.0,
    "huginn_poisson_lognormal_sigma": 0.5,
    "huginn_poisson_lognormal_max": 64,
    "huginn_backprop_depth": 10,
})


# Table 3 input-injection variants, each trained with PLN-5 depths.
PARCAE_DT_INIT = 0.5 * math.log(5)
INJECTION_VARIANTS: Mapping[str, Mapping[str, object]] = MappingProxyType({
    # W[h; e] + b, with a parameter-free RMSNorm after every recurrence.
    "huginn_linear": MappingProxyType({"huginn_input_injection": "linear",
                                       "huginn_recurrent_exit_norm": "parameterless_rms"}),
    # Diagonal injection of the RMS-normalized prelude output, dt initialized at 1/2 log 5.
    "parcae": MappingProxyType({"huginn_diagonal_dt_init": PARCAE_DT_INIT, "huginn_prelude_norm": "rms"}),
    # Diagonal injection whose decayed state drops its component along the injected update.
    "orthoinj": MappingProxyType({"huginn_diagonal_dt_init": PARCAE_DT_INIT, "huginn_prelude_orthogonal": True}),
})


def learned_prior_fields(entropy_coefficient: float) -> Mapping[str, object]:
    """Learned depth prior initialized at PLN-5, with entropy coefficient lambda_H."""
    return MappingProxyType({**PLN5_FIELDS, "huginn_depth_prior": "learned",
                             "huginn_depth_prior_entropy": entropy_coefficient})


def huginn_model_fields(scale: Scale, **overrides: object) -> Mapping[str, object]:
    """Prelude/core/coda of 1/2/1 Llama blocks trained at R=5 with full backpropagation, as in the paper."""
    fields: dict[str, object] = {
        "arch": "huginn",
        "huginn_depth_control": True,
        "num_layers": 4,
        "model_dim": scale.model_dim,
        "num_heads": scale.num_heads,
        "num_kv_heads": scale.num_kv_heads,
        "head_dim": 64,
        "ffn_hidden_dim": scale.ffn_hidden_dim,
        "vocab_size": 64256,
        "swiglu": True,
        "apply_bias_term": True,
        "apply_rmsnorm": True,
        "rmsnorm_eps": 1e-6,
        "attn_gate_func": "silu",
        "rope_base": 1_000_000.0,
        "rope_head_dim": 64,
        "chunk_size": 1024,
        "causal_attn_backend": "flash",
        "fused_block": False,
        "init_mode": "gaussian",
        "init_std": 0.02,
        "loop_times": 5,
        "loop_start_layer": 1,
        "loop_end_layers": 3,
        "huginn_prelude_layers": 1,
        "huginn_recurrent_layers": 2,
        "huginn_coda_layers": 1,
        "huginn_state_init": "normal",
        "huginn_input_injection": "diagonal",
        "huginn_sampling_scheme": "fixed",
        "huginn_backprop_depth": 5,
    }
    fields.update(overrides)
    return MappingProxyType(fields)


def dense_model_fields(scale: Scale) -> Mapping[str, object]:
    """The dense baseline D4: the four Huginn blocks' width and numerics, run once in sequence, untied head."""
    fields = {key: value for key, value in huginn_model_fields(scale).items()
              if not key.startswith(("huginn_", "loop_"))}
    return MappingProxyType({**fields, "arch": "transformer"})


@dataclass(frozen=True)
class Recipe:
    """One Part 2 training row."""

    name: str
    table_row: str
    scale: str
    lr: float
    model_fields: Mapping[str, object]

    @property
    def steps(self) -> int:
        return SCALES[self.scale].steps

    def model_config(self) -> ModelConf:
        return ModelConf.from_dict(dict(self.model_fields))

    def differing_fields(self, model: Mapping[str, object]) -> list[str]:
        """Recipe model fields that `model`, such as a checkpoint's saved model config, sets differently."""
        return sorted(key for key, value in self.model_fields.items() if model.get(key) != value)

    def optimizer_config(self) -> OptimConf:
        """AdamW with warmup-stable-decay: 5% linear warmup, cosine decay to 0.1x over the last 10%."""
        return OptimConf(
            lr=self.lr,
            warmup=self.steps // 20,
            weight_decay=0.1,
            epsilon=1e-8,
            beta1=0.9,
            beta2=0.98,
            clip=1.0,
            scheduler="wsd",
            lr_init_ratio=1e-4,
            lr_end_ratio=0.1,
            cycles=1.0,
            wsd_decay_duration_steps=self.steps // 10,
            wsd_decay_style="cosine",
        )


_RECIPES: Mapping[str, Recipe] = MappingProxyType({
    recipe.name: recipe
    for recipe in (
        Recipe("s_fixed_r5", "Table 2, S, Fixed R=5", "S", 1.5e-3, huginn_model_fields(SCALES["S"])),
        Recipe("m_fixed_r5", "Table 2, M, Fixed R=5", "M", 1.5e-3, huginn_model_fields(SCALES["M"])),
        Recipe("l_fixed_r5", "Table 2, L, Fixed R=5", "L", 1e-3, huginn_model_fields(SCALES["L"])),
        Recipe("s_fixed_pln5", "Table 2, S, Fixed PLN-5", "S", 1.5e-3, huginn_model_fields(SCALES["S"], **PLN5_FIELDS)),
        Recipe("m_fixed_pln5", "Table 2, M, Fixed PLN-5", "M", 1.5e-3, huginn_model_fields(SCALES["M"], **PLN5_FIELDS)),
        Recipe("l_fixed_pln5", "Table 2, L, Fixed PLN-5", "L", 5e-4, huginn_model_fields(SCALES["L"], **PLN5_FIELDS)),
        Recipe("s_learned_entropy0", "Table 2, S, Learned lambda_H=0", "S", 2e-3,
               huginn_model_fields(SCALES["S"], **learned_prior_fields(0.0))),
        Recipe("s_learned_entropy0p01", "Table 2, S, Learned lambda_H=0.01", "S", 2e-3,
               huginn_model_fields(SCALES["S"], **learned_prior_fields(0.01))),
        Recipe("m_learned_entropy0", "Table 2, M, Learned lambda_H=0", "M", 1.5e-3,
               huginn_model_fields(SCALES["M"], **learned_prior_fields(0.0))),
        Recipe("m_learned_entropy0p01", "Table 2, M, Learned lambda_H=0.01", "M", 1.5e-3,
               huginn_model_fields(SCALES["M"], **learned_prior_fields(0.01))),
        Recipe("l_learned_entropy0", "Table 2, L, Learned lambda_H=0", "L", 1e-3,
               huginn_model_fields(SCALES["L"], **learned_prior_fields(0.0))),
        Recipe("l_learned_entropy0p01", "Table 2, L, Learned lambda_H=0.01", "L", 1e-3,
               huginn_model_fields(SCALES["L"], **learned_prior_fields(0.01))),
        Recipe("s_huginn_linear", "Table 3, S, Huginn-Linear", "S", 1e-3,
               huginn_model_fields(SCALES["S"], **PLN5_FIELDS, **INJECTION_VARIANTS["huginn_linear"])),
        Recipe("s_parcae", "Table 3, S, Parcae-Decay", "S", 1.5e-3,
               huginn_model_fields(SCALES["S"], **PLN5_FIELDS, **INJECTION_VARIANTS["parcae"])),
        Recipe("s_orthoinj", "Table 3, S, OrthoInj", "S", 2e-3,
               huginn_model_fields(SCALES["S"], **PLN5_FIELDS, **INJECTION_VARIANTS["orthoinj"])),
        Recipe("m_huginn_linear", "Table 3, M, Huginn-Linear", "M", 7.5e-4,
               huginn_model_fields(SCALES["M"], **PLN5_FIELDS, **INJECTION_VARIANTS["huginn_linear"])),
        Recipe("m_parcae", "Table 3, M, Parcae-Decay", "M", 1e-3,
               huginn_model_fields(SCALES["M"], **PLN5_FIELDS, **INJECTION_VARIANTS["parcae"])),
        Recipe("m_orthoinj", "Table 3, M, OrthoInj", "M", 1.5e-3,
               huginn_model_fields(SCALES["M"], **PLN5_FIELDS, **INJECTION_VARIANTS["orthoinj"])),
        Recipe("s_d4", "Table 4, S, D4", "S", 2e-3, dense_model_fields(SCALES["S"])),
        Recipe("m_d4", "Table 4, M, D4", "M", 1e-3, dense_model_fields(SCALES["M"])),
    )
})


def recipe_names() -> tuple[str, ...]:
    """Return recipe names in table order."""
    return tuple(_RECIPES)


def get_recipe(name: str) -> Recipe:
    return _RECIPES[name]
