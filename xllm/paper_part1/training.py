"""Complete training configs for Part 1 recipes.

The recipe owns what defines the paper run: model, optimizer and schedule, the
target's stop step, global batch, seed and numerics. The caller's base config
owns the rest: data and tokenizer, logging, checkpoint retention, evaluation and
`gradient_accumulation_steps` (default 1).

Every recipe uses data parallelism only and a global batch of 512 sequences of 8192
tokens: the ranks times the accumulation must divide 512, which gives each rank's
batch.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from xllm.config import ModelConf, TrainerConf
from xllm.configuration.configuration import flatten_dict
from xllm.models.build import get_model_cls
from xllm.paper_part1.recipes import ENDPOINT_STEPS, TPP_TARGETS, Recipe, get_recipe, recipe_names


GLOBAL_BATCH_SIZE = 512
SEQUENCE_LENGTH = 8192
GLOBAL_TOKENS_PER_STEP = GLOBAL_BATCH_SIZE * SEQUENCE_LENGTH

RECIPE_OWNED_FIELDS = frozenset({
    "model", "optim", "steps", "stop_step", "seq_len", "batch_size", "global_batch_size",
    "model_parallel_size", "context_parallel_size", "dump_dir",
}).union(*(get_recipe(name).trainer_fields for name in recipe_names()))


def activated_parameters(model_cfg: ModelConf) -> int:
    """Activated parameters as train.py counts them, from a model built on the meta device."""
    with torch.device("meta"):
        model = get_model_cls(model_cfg.arch, model_cfg)(model_cfg, None)
    return model.num_parameters()[1]


def stop_step(recipe: Recipe, target: str) -> int:
    """Last optimizer step of a target; a TPP target stops at the first step reaching its tokens."""
    if target in ENDPOINT_STEPS:
        step = ENDPOINT_STEPS[target]
    else:
        tokens = TPP_TARGETS[target] * activated_parameters(recipe.model_config())
        step = math.ceil(tokens / GLOBAL_TOKENS_PER_STEP)
    assert 0 < step <= recipe.scheduler_steps, f"{target}: step {step}/{recipe.scheduler_steps}"
    return step


def training_config(recipe: Recipe, target: str, base: Mapping[str, Any], dump_dir: str) -> TrainerConf:
    """Return the full `TrainerConf` of a recipe and target on top of a caller's base config."""
    owned = sorted(RECIPE_OWNED_FIELDS & set(base))
    if owned:
        raise ValueError(f"the recipe sets {owned}; remove them from the base config")
    cfg = TrainerConf.from_dict({
        **base,
        "model": recipe.model_config().to_dict(),
        "optim": recipe.optimizer_config().to_dict(),
        "steps": recipe.scheduler_steps,
        "stop_step": stop_step(recipe, target),
        "seq_len": SEQUENCE_LENGTH,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "model_parallel_size": 1,
        "context_parallel_size": 1,
        "dump_dir": dump_dir,
        **recipe.trainer_fields,
    })
    if GLOBAL_BATCH_SIZE % cfg.gradient_accumulation_steps:
        raise ValueError(f"gradient_accumulation_steps must divide the global batch of {GLOBAL_BATCH_SIZE}, "
                         f"got {cfg.gradient_accumulation_steps}")
    # TrainerConf.from_dict only warns about unknown fields; reject them here.
    unknown = sorted(set(flatten_dict(dict(base))) - set(flatten_dict(cfg.to_dict())))
    if unknown:
        raise ValueError(f"unknown base config fields: {unknown}")
    return cfg
