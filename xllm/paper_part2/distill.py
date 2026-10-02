"""Distilled prefill of Table 4: a two-block student learns the teacher's pre-coda state.

The teacher is a Table 2 learned-prior model (lambda_H = 0.01), released or
exported from a run of its recipe, with one prelude, two recurrent and one
coda layer. For each 8,192-token row, the teacher's prelude output `e` is the
student input, and the state entering the coda after five recurrences from the
identity-seeded state (seed 42) is the target `h`. The student is a `d x d`
input map followed by two Transformer blocks of the teacher's width, and
minimizes

    MSE(student(e), h) / E  +  KL(coda(h) || coda(student(e)))

where `E` is the mean squared target over 64 calibration rows and `coda(.)`
applies the teacher's coda layer and head. The teacher initialization sets the
input map to `diag(softplus(dt_bias)) W_in`, the injected input at a zero state,
and copies the two recurrent blocks.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import math
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from xllm.config import ModelConf
from xllm.models.transformer import TransformerBlock
from xllm.paper_part2.artifacts import open_artifact
from xllm.paper_part2.eval_protocol import EVAL_DEPTH, SEQUENCE_LENGTH, STATE_SEED


ROWS_PER_UPDATE = 256  # token-cache rows per student update
STUDENT_SEED = 23
CALIBRATION_ROWS = 64
VALIDATION_ROWS = 128
TRAIN_ROW_OFFSET = 128  # training reads rows 128 on; of the first 128, rows 0-63 calibrate E
KL_WEIGHT = 1.0


@dataclasses.dataclass(frozen=True)
class DistillRecipe:
    teacher: str  # released teacher artifact
    teacher_recipe: str  # the teacher's training recipe
    teacher_init: bool
    steps: int
    lr: float = 1e-3
    warmup_steps: int = 8
    schedule_steps: int = 2048  # cosine to min_lr_ratio * lr at this step, constant after
    min_lr_ratio: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0


DISTILL_RECIPES = {
    "s_distill_random_init": DistillRecipe("huginn-s-learned-entropy0p01", "s_learned_entropy0p01", False, 2560),
    "s_distill_teacher_init": DistillRecipe("huginn-s-learned-entropy0p01", "s_learned_entropy0p01", True, 2560),
    "m_distill_random_init": DistillRecipe("huginn-m-learned-entropy0p01", "m_learned_entropy0p01", False, 10240),
    "m_distill_teacher_init": DistillRecipe("huginn-m-learned-entropy0p01", "m_learned_entropy0p01", True, 10240),
}


def lr_scale(step: int, recipe: DistillRecipe) -> float:
    """Multiplier of the peak LR at update `step` (1-based): linear warmup, then cosine to the floor."""
    if step <= recipe.warmup_steps:
        return step / recipe.warmup_steps
    fraction = min(1., (step - recipe.warmup_steps) / (recipe.schedule_steps - recipe.warmup_steps))
    return recipe.min_lr_ratio + (1 - recipe.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * fraction))


def training_rows(step: int, micro: int, local: int, rank: int, world: int,
                  global_batch: int = ROWS_PER_UPDATE) -> list[int]:
    """Logical rows of one rank's micro-batch: each update reads the next rows, independent of topology."""
    if global_batch % (world * local):
        raise ValueError(f"{global_batch} rows do not split into {world} ranks x {local}")
    first = TRAIN_ROW_OFFSET + (step - 1) * global_batch + (micro * world + rank) * local
    return list(range(first, first + local))


class TokenCache:
    """The distillation token cache: int32 shards of 8,208-token rows, row `i` in shard `i % n`.

    Rows are 8,208 tokens wide, as in the paper's cache; the trainer reads the
    first 8,192. Example ids, which seed the teacher's recurrent state, are the
    row numbers, plus 10^9 for validation rows so they never repeat a training id.
    """

    ROW_TOKENS = 8208
    LAYOUT = "round_robin_shards_v1"

    def __init__(self, manifest: str | Path, split: str) -> None:
        manifest = Path(manifest)
        self.meta = json.loads(manifest.read_text())
        if (self.meta["split"], self.meta["dtype"], self.meta["layout"], self.meta["sequence_tokens"]) != (
                split, "int32", self.LAYOUT, self.ROW_TOKENS):
            raise ValueError(f"{manifest} is not a {split} token cache")
        self.split = split
        self.shards = [np.memmap(manifest.parent / s["path"], dtype="<i4", mode="r",
                                 shape=(s["stored_rows"], self.ROW_TOKENS)) for s in self.meta["shards"]]

    def batch(self, rows: Sequence[int], length: int = SEQUENCE_LENGTH) -> tuple[Tensor, Tensor]:
        n = self.meta["shard_count"]
        if not all(0 <= i < self.meta["rows"] for i in rows):
            raise IndexError(f"rows outside the {self.meta['rows']}-row cache")
        tokens = np.stack([self.shards[i % n][self.meta["shards"][i % n]["row_offset"] + i // n, :length]
                           for i in rows])
        ids = np.asarray(rows, dtype=np.int64) + (10**9 if self.split == "valid" else 0)
        return (torch.as_tensor(tokens.astype(np.int64), device="cuda"), torch.as_tensor(ids, device="cuda"))


def student_config(teacher_config: ModelConf) -> ModelConf:
    """Block configuration of the student: the teacher's width, heads and numerics, two layers."""
    fields = {field.name: getattr(teacher_config, field.name) for field in dataclasses.fields(ModelConf)
              if not field.name.startswith(("huginn_", "loop_"))}
    return ModelConf.from_dict({**fields, "arch": "transformer", "num_layers": 2})


class Student(nn.Module):
    """The `d x d` input map and two Transformer blocks; random init draws from the CPU RNG under `seed`."""

    def __init__(self, cfg: ModelConf, seed: int = STUDENT_SEED) -> None:
        super().__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.in_proj = nn.Linear(cfg.model_dim, cfg.model_dim, bias=False)
        self.layers = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.num_layers)])

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        x = self.in_proj(x)
        for layer in self.layers:
            x, _, _ = layer(x, freqs_cis, None, None, False, True, None)
        return x


def wrap_student(student: Student) -> nn.Module:
    """FSDP NO_SHARD with FP32 weights and reductions and BF16 compute, as the paper trained the student."""
    from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, ShardingStrategy

    return FullyShardedDataParallel(
        student, device_id=torch.cuda.current_device(), use_orig_params=True,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                       buffer_dtype=torch.float32))


@torch.no_grad()
def init_from_teacher(student: Student, teacher_state: Mapping[str, Tensor], prelude_layers: int = 1) -> None:
    """Teacher initialization from the teacher's state dict.

    The paper passed the teacher run's FP32 weights; an artifact holds them rounded to BF16.
    """
    dt = F.softplus(teacher_state["input_injection.dt_bias"].float().cuda())
    student.in_proj.weight.copy_(dt[:, None] * teacher_state["input_injection.input_adapter.weight"].float().cuda())
    for j, layer in enumerate(student.layers):
        prefix = f"layers.{prelude_layers + j}."
        source = {key[len(prefix):]: value for key, value in teacher_state.items() if key.startswith(prefix)}
        target = dict(layer.named_parameters())
        if set(source) != set(target):
            raise KeyError(f"teacher layer {prelude_layers + j} and student layer {j} differ: "
                           f"{sorted(set(source) ^ set(target))}")
        for name, parameter in target.items():
            parameter.copy_(source[name])


def _run_layer(teacher: nn.Module, layer_id: int, hidden: Tensor, freqs_cis: Tensor) -> Tensor:
    hidden, _, _ = teacher.layers[layer_id](hidden, freqs_cis, None, None, False, True, None)
    return hidden


@torch.no_grad()
def capture(teacher: nn.Module, tokens: Tensor, ids: Tensor, seed: int = STATE_SEED,
            depth: int = EVAL_DEPTH) -> tuple[Tensor, Tensor, Tensor]:
    """The teacher's prelude output (student input), pre-coda state after `depth` recurrences (target) and RoPE."""
    hidden = teacher.embed(tokens)
    freqs_cis = teacher.rope.get_freqs_cis(0, tokens.shape[1], hidden.device)
    for layer_id in range(teacher.loop_start_layer):
        hidden = _run_layer(teacher, layer_id, hidden, freqs_cis)
    inputs = hidden
    norm = teacher.huginn_prelude_norm_module
    input_embeds = inputs if norm is None else norm(inputs)
    state = teacher.identity_recurrent_state(input_embeds, 0, ids, seed)
    for _ in range(depth):
        state = teacher.input_injection(state, input_embeds)
        for layer_id in range(teacher.loop_start_layer, teacher.loop_end_layers):
            state = _run_layer(teacher, layer_id, state, freqs_cis)
        if teacher.huginn_recurrent_exit_norm_module is not None:
            state = teacher.huginn_recurrent_exit_norm_module(state)
    return inputs, state, freqs_cis


def coda_logits(teacher: nn.Module, state: Tensor, freqs_cis: Tensor) -> Tensor:
    """The teacher's coda layer and head on a pre-coda state, at every position."""
    for layer_id in range(teacher.loop_end_layers, teacher.num_layers):
        state = _run_layer(teacher, layer_id, state, freqs_cis)
    return teacher.output(state, None, None)


def teacher_kl(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    """KL(teacher || student) of the next-token distributions, averaged over positions but the last."""
    s = F.log_softmax(student_logits[:, :-1].float(), -1)
    t = F.log_softmax(teacher_logits[:, :-1].float(), -1)
    return (t.exp() * (t - s)).sum(-1).mean()


def distillation_loss(teacher: nn.Module, student: nn.Module, tokens: Tensor, ids: Tensor, energy: float) -> Tensor:
    """Energy-normalized hidden MSE plus the KL term on one micro-batch."""
    inputs, target, freqs_cis = capture(teacher, tokens, ids)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = student(inputs, freqs_cis)
        loss = F.mse_loss(pred.float(), target.float()) / energy
        logits = coda_logits(teacher, pred, freqs_cis)
        with torch.no_grad():
            reference = coda_logits(teacher, target, freqs_cis)
        loss = loss + KL_WEIGHT * teacher_kl(logits, reference)
    return loss


def _pooled_rows(count: int, rank: int, world: int) -> Iterator[tuple[list[int], int]]:
    """Two-row batches dealt round-robin to ranks; each yields (rows, how many of them count). Rows past
    `count` wrap to the start and do not count."""
    for base in range(0, count, world * 2):
        first = base + rank * 2
        yield [(first + i) % count for i in range(2)], max(0, min(2, count - first))


@torch.no_grad()
def hidden_energy(teacher: nn.Module, cache: TokenCache, rank: int = 0, world: int = 1,
                  count: int = CALIBRATION_ROWS) -> Tensor:
    """Squared-target sum and element count over the calibration rows 0 to `count` - 1; the caller sums them over
    ranks."""
    totals = torch.zeros(2, device="cuda", dtype=torch.float64)
    for rows, used in _pooled_rows(count, rank, world):
        tokens, ids = cache.batch(rows)
        _, target, _ = capture(teacher, tokens, ids)
        target = target[:used].float()
        totals[0] += target.square().sum().double()
        totals[1] += target.numel()
    return totals


@torch.no_grad()
def validation_totals(teacher: nn.Module, student: nn.Module, cache: TokenCache, rank: int = 0,
                      world: int = 1) -> Tensor:
    """Squared error, target energy and KL sums over the 128 validation rows; the caller sums over ranks."""
    totals = torch.zeros(4, device="cuda", dtype=torch.float64)
    for rows, used in _pooled_rows(VALIDATION_ROWS, rank, world):
        tokens, ids = cache.batch(rows)
        inputs, target, freqs_cis = capture(teacher, tokens, ids)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = student(inputs, freqs_cis)
            logits = coda_logits(teacher, pred, freqs_cis)
        kl = teacher_kl(logits[:used], coda_logits(teacher, target, freqs_cis)[:used])
        pred, target = pred[:used].float(), target[:used].float()
        totals += torch.stack([(pred - target).square().sum().double(), target.square().sum().double(),
                               kl.double() * used, torch.tensor(float(used), device="cuda", dtype=torch.float64)])
    return totals


def train_step(teacher: nn.Module, student: nn.Module, optimizer: torch.optim.Optimizer, cache: TokenCache,
               recipe: DistillRecipe, step: int, local: int, energy: float,
               global_batch: int = ROWS_PER_UPDATE) -> dict:
    """One update of the FSDP-wrapped student over `global_batch` rows, with gradient accumulation."""
    rank, world = dist.get_rank(), dist.get_world_size()
    accumulation = global_batch // (world * local)
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = recipe.lr * lr_scale(step, recipe)
    total = 0.0
    for micro in range(accumulation):
        tokens, ids = cache.batch(training_rows(step, micro, local, rank, world, global_batch))
        with contextlib.nullcontext() if micro == accumulation - 1 else student.no_sync():
            loss = distillation_loss(teacher, student, tokens, ids, energy)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            (loss / accumulation).backward()
        total += float(loss.detach()) / accumulation
    grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), recipe.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    mean = torch.tensor(total, device="cuda", dtype=torch.float64)
    dist.all_reduce(mean)
    return dict(loss=float(mean / world), grad_norm=float(grad_norm), learning_rate=optimizer.param_groups[0]["lr"])


@torch.no_grad()
def prefill_banks(teacher: nn.Module, student: nn.Module, tokens: Tensor) -> list[tuple]:
    """Distilled prefill: the student's state, one teacher recurrence and the coda write the terminal banks.

    The teacher's prelude layer, its two recurrent layers (after injecting the prelude output into the
    student's predicted pre-coda state) and its coda layer each append their KV to an empty cache,
    giving the four banks the teacher decodes from at R=5.
    """
    cache = [(None, None, 0)] * teacher.num_layers
    hidden = teacher.embed(tokens)
    freqs_cis = teacher.rope.get_freqs_cis(0, tokens.shape[1], hidden.device)

    def run(layer_id: int, state: Tensor) -> Tensor:
        state, _, cache[layer_id] = teacher.layers[layer_id](state, freqs_cis, None, None, False, True,
                                                               cache[layer_id])
        return state

    for layer_id in range(teacher.loop_start_layer):
        hidden = run(layer_id, hidden)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = student(hidden, freqs_cis)
    norm = teacher.huginn_prelude_norm_module
    state = teacher.input_injection(state, hidden if norm is None else norm(hidden))
    for layer_id in range(teacher.loop_start_layer, teacher.loop_end_layers):
        state = run(layer_id, state)
    if teacher.huginn_recurrent_exit_norm_module is not None:
        state = teacher.huginn_recurrent_exit_norm_module(state)
    for layer_id in range(teacher.loop_end_layers, teacher.num_layers):
        state = run(layer_id, state)
    return cache


def load_student(path: str | Path, teacher_manifest: Mapping[str, object],
                 teacher_fields: Mapping[str, object]) -> Student:
    """A trained student in BF16 for evaluation with the teacher artifact of the given manifest and model fields.

    `path` is a student artifact, which must pin that teacher's manifest SHA-256, or the `student_final.pt`
    of distill_paper_part2.py.
    """
    if Path(path).is_dir():
        _, config, state = open_artifact(path)
        if config.get("student", {}).get("teacher_manifest_sha256") != teacher_manifest["manifest_sha256"]:
            raise ValueError(f"{path} was not distilled from this teacher artifact")
        cfg = ModelConf.from_dict(config["model"])
    else:
        cfg = student_config(ModelConf.from_dict(dict(teacher_fields)))
        state = torch.load(path, map_location="cpu", weights_only=True)
    student = Student(cfg)
    student.load_state_dict(state, strict=True)
    return student.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
