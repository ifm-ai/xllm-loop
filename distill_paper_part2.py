"""Train a Table 4 distilled-prefill student against a released learned-prior teacher.

Every update reads the next 256 cache rows. Ranks split them into micro-batches
of `--local-batch` rows with gradient accumulation, so ranks x local batch x
accumulation is 256. The student keeps FP32 weights and computes in BF16, as in
the paper. Writes OUT/metrics.jsonl, OUT/checkpoint.pt every 128 updates
(resumed automatically) and OUT/student_final.pt.

    ENABLE_FLASH_ATTENTION_3=true torchrun --nnodes N --nproc-per-node G ... distill_paper_part2.py \\
        --recipe s_distill_teacher_init --teacher ARTIFACTS/huginn-s-learned-entropy0p01 \\
        --train TRAIN_MANIFEST --valid VALID_MANIFEST --out OUT
"""

import argparse
import json
import signal
import time
from pathlib import Path

import torch
import torch.distributed as dist

from xllm.config import ModelConf, TokenizerConf, TrainerConf
from xllm.data.dataset_streamer.tokenizer import build_tokenizer
from xllm.distributed import init_torch_distributed, initialize_model_parallel
from xllm.paper_part2.artifacts import build_model, open_artifact
from xllm.paper_part2.recipes import get_recipe
from xllm.paper_part2.distill import (
    DISTILL_RECIPES,
    ROWS_PER_UPDATE,
    SEQUENCE_LENGTH,
    Student,
    TokenCache,
    hidden_energy,
    init_from_teacher,
    student_config,
    train_step,
    validation_totals,
    wrap_student,
)

CHECKPOINT_EVERY = VALIDATE_EVERY = 128
LOG_EVERY = 10


def log(out: Path, row: dict) -> None:
    if dist.get_rank() == 0:
        with (out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


def save(path: Path, value) -> None:
    if dist.get_rank() == 0:
        torch.save(value, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
    dist.barrier()


def validate(teacher, student, valid: TokenCache) -> dict:
    student.eval()
    totals = validation_totals(teacher, student, valid, dist.get_rank(), dist.get_world_size())
    student.train()
    dist.all_reduce(totals)
    return dict(relative_mse=float(totals[0] / totals[1]), kl_to_teacher=float(totals[2] / totals[3]))


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True, choices=sorted(DISTILL_RECIPES))
    parser.add_argument("--teacher", required=True, type=Path,
                        help="the recipe's teacher artifact: released, or exported from a run of its training recipe")
    parser.add_argument("--tokenizer", help="tokenizer directory (default: the teacher artifact's)")
    parser.add_argument("--train", required=True, type=Path, help="training token-cache manifest")
    parser.add_argument("--valid", required=True, type=Path, help="validation token-cache manifest")
    parser.add_argument("--local-batch", type=int, help="rows per rank and micro-batch (default: no accumulation)")
    parser.add_argument("--stop-step", type=int, help="stop after this update (default: the last); rerun to continue")
    parser.add_argument("--out", required=True, type=Path)
    return parser, parser.parse_args()


def build(args, recipe, config: dict, teacher_state: dict):
    """The BF16 teacher, the FSDP-wrapped student (from the teacher's weights for a teacher-init recipe) and AdamW."""
    tokenizer = build_tokenizer(TokenizerConf(type="huggingface", path=args.tokenizer or str(
        args.teacher / config["tokenizer"]["path"])))
    teacher = build_model(config["model"], teacher_state, tokenizer)
    student = Student(student_config(ModelConf.from_dict(config["model"]))).cuda()
    if recipe.teacher_init:
        init_from_teacher(student, teacher_state, teacher.loop_start_layer)
    student = wrap_student(student)
    optimizer = torch.optim.AdamW(student.parameters(), lr=recipe.lr, betas=recipe.betas, eps=recipe.eps,
                                  weight_decay=recipe.weight_decay)
    return teacher, student, optimizer


def resume(args, teacher, student, optimizer, train: TokenCache) -> tuple[int, float]:
    """The last saved update and the hidden energy E: from OUT/checkpoint.pt, or E over the calibration rows."""
    checkpoint = args.out / "checkpoint.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["recipe"] != args.recipe:
            raise ValueError(f"{checkpoint} belongs to {saved['recipe']}")
        student.load_state_dict(saved["student"])
        optimizer.load_state_dict(saved["optimizer"])
        return saved["step"], saved["hidden_energy"]
    totals = hidden_energy(teacher, train, dist.get_rank(), dist.get_world_size())
    dist.all_reduce(totals)
    return 0, float(totals[0] / totals[1])


def train_loop(args, recipe, teacher, student, optimizer, train: TokenCache, valid: TokenCache, local: int,
               done: int, energy: float) -> None:
    """Updates `done + 1` to the last, with logging, validation and checkpoints; a signal checkpoints and stops."""
    signals = []
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, lambda number, _frame: signals.append(number))
    checkpoint = args.out / "checkpoint.pt"
    last = recipe.steps if args.stop_step is None else min(args.stop_step, recipe.steps)
    for step in range(done + 1, last + 1):
        started = time.perf_counter()
        values = train_step(teacher, student, optimizer, train, recipe, step, local, energy)
        if step == done + 1 or step % LOG_EVERY == 0:
            log(args.out, dict(kind="train", step=step, seconds=time.perf_counter() - started, **values))
        if step % VALIDATE_EVERY == 0 or step == last:
            log(args.out, dict(kind="valid", step=step, **validate(teacher, student, valid)))
        stop = torch.tensor(len(signals), device="cuda")
        dist.all_reduce(stop)
        if step % CHECKPOINT_EVERY == 0 or step == last or stop:
            save(checkpoint, dict(recipe=args.recipe, step=step, hidden_energy=energy,
                                  student=student.state_dict(), optimizer=optimizer.state_dict()))
        if stop:
            raise SystemExit(f"stopped by a signal after step {step}; rerun to resume")
    if last == recipe.steps:
        save(args.out / "student_final.pt", {k: v.detach().cpu().clone() for k, v in student.state_dict().items()})


def main() -> None:
    parser, args = parse_args()
    recipe = DISTILL_RECIPES[args.recipe]
    _, config, teacher_state = open_artifact(args.teacher)
    differing = get_recipe(recipe.teacher_recipe).differing_fields(config["model"])
    if differing:
        parser.error(f"--teacher is not a {recipe.teacher_recipe} model: {', '.join(differing)} differ")
    init_torch_distributed(TrainerConf.nccl_timeout)
    initialize_model_parallel(1, 1, TrainerConf.nccl_timeout)
    world = dist.get_world_size()
    local = args.local_batch or ROWS_PER_UPDATE // world
    if ROWS_PER_UPDATE % (world * local):
        raise ValueError(f"{ROWS_PER_UPDATE} rows do not split into {world} ranks x {local}")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "student_final.pt").exists():
        raise FileExistsError(f"{args.out} already holds a finished student")
    teacher, student, optimizer = build(args, recipe, config, teacher_state)
    del teacher_state
    train = TokenCache(args.train, "train")
    valid = TokenCache(args.valid, "valid")
    done, energy = resume(args, teacher, student, optimizer, train)
    log(args.out, dict(kind="setup", recipe=args.recipe, world_size=world, local_batch=local,
                       gradient_accumulation=ROWS_PER_UPDATE // (world * local),
                       global_batch=ROWS_PER_UPDATE, global_tokens=ROWS_PER_UPDATE * SEQUENCE_LENGTH,
                       hidden_energy=energy, resumed_from=done,
                       student_parameters=sum(p.numel() for p in student.parameters())))
    train_loop(args, recipe, teacher, student, optimizer, train, valid, local, done, energy)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
