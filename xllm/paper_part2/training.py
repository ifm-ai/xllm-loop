"""Complete training configs for Part 2 recipes.

The recipe owns the model, optimizer and schedule, steps, the global batch of
512 sequences, the seed, and the switches the paper runs fixed: deterministic
kernels, attention within documents, no loss rescaling, RNG states restored on
resume, and no model or context parallelism. The caller's base config owns data
and tokenizer, logging, checkpoint retention, evaluation and
`gradient_accumulation_steps` (default 1); every other TrainerConf setting, such
as the dtype, keeps its default unless the base config sets it. At launch the
data-parallel ranks times the accumulation must divide 512, which gives the
local batch; with sampled depths, every accumulation step draws its own depth.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from xllm.config import TrainerConf
from xllm.configuration.configuration import flatten_dict
from xllm.paper_part2.recipes import GLOBAL_BATCH_SEQUENCES, SEQUENCE_LENGTH, Recipe


RECIPE_OWNED_FIELDS = frozenset({
    "model", "optim", "steps", "seq_len", "batch_size", "global_batch_size", "seed", "deterministic",
    "multi_segments", "loss_rescaling", "restore_rng_state", "dump_dir", "model_parallel_size", "context_parallel_size",
})


def training_config(recipe: Recipe, base: Mapping[str, Any], dump_dir: str) -> TrainerConf:
    """Return the full `TrainerConf` of a recipe on top of a caller's base config."""
    owned = sorted(RECIPE_OWNED_FIELDS & set(base))
    if owned:
        raise ValueError(f"the recipe sets {owned}; remove them from the base config")
    cfg = TrainerConf.from_dict({
        **base,
        "model": dict(recipe.model_fields),
        "optim": dataclasses.asdict(recipe.optimizer_config()),
        "steps": recipe.steps,
        "seq_len": SEQUENCE_LENGTH,
        "global_batch_size": GLOBAL_BATCH_SEQUENCES,
        "model_parallel_size": 1,
        "context_parallel_size": 1,
        "seed": 1,
        "deterministic": True,
        "multi_segments": True,
        "loss_rescaling": False,
        # The recurrent state is drawn from the CUDA RNG on every forward.
        "restore_rng_state": True,
        "dump_dir": dump_dir,
    })
    if GLOBAL_BATCH_SEQUENCES % cfg.gradient_accumulation_steps:
        raise ValueError(f"gradient_accumulation_steps must divide the global batch of {GLOBAL_BATCH_SEQUENCES}, "
                         f"got {cfg.gradient_accumulation_steps}")
    # TrainerConf.from_dict only warns about unknown fields.
    unknown = sorted(set(flatten_dict(dict(base))) - set(flatten_dict(cfg.to_dict())))
    if unknown:
        raise ValueError(f"unknown base config fields: {unknown}")
    return cfg
