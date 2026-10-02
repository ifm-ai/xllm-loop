"""Evaluate a Part 2 artifact with terminal KV sharing, as in the paper's Tables 2-4.

Stages (one GPU each; shard the task and generation stages across jobs):

    ppl                       Wiki prompt-conditioned PPL -> OUT/ppl.json
    tasks SHARD SHARDS        the seven choice tasks -> OUT/tasks/<task>-shard<i>of<n>.jsonl
    gen TASK SHARD SHARDS     GSM8K, DROP or MBPP+ -> OUT/gen/<task>/shard<i>of<n>/
    latency                   prefill latency of the first Wiki batch -> OUT/latency.json
    summary                   the table row from the outputs above -> OUT/summary.json (no GPU)

`--data` is a JSON file with the evaluation inputs: {"wiki": file, "tasks": {task: file},
"gsm8k": file, "drop": file, "mbpp": file, "mbpp_plus": file};
every input is checked against its protocol SHA-256. `xllm.paper_part2.eval_summary`
turns the outputs into a table row.

Table 4 rows: `--prefill r1` prefills with one teacher recurrence, `--student
STUDENT` prefills with a distilled student; the teacher decodes both at
R=5. A dense artifact (D4) decodes itself with its full KV cache. The latency
stage times the chosen path together with the teacher's own prefill.

    ENABLE_FLASH_ATTENTION_3=true python eval_paper_part2.py --artifact ARTIFACT --data data.json \\
        --out OUT tasks 0 4
"""

import argparse
import json
import shlex
from pathlib import Path

import torch

from xllm.config import TokenizerConf, TrainerConf
from xllm.data.dataset_streamer.tokenizer import build_tokenizer
from xllm.distributed import init_torch_distributed, initialize_model_parallel
from xllm.paper_part2.artifacts import build_model, open_artifact
from xllm.paper_part2.choice_tasks import load_choice_examples, prepare_requests
from xllm.paper_part2.distill import load_student
from xllm.paper_part2.eval_data import wiki_batches
from xllm.paper_part2.eval_protocol import CHOICE_TASKS, GENERATION
from xllm.paper_part2.eval_summary import table_row
from xllm.paper_part2.latency import prefill_latency
from xllm.paper_part2.evaluator import (
    DenseKV,
    DistilledPrefill,
    TeacherR1Prefill,
    TerminalKV,
    evaluate_mbpp,
    generate,
    generation_items,
    score_generations,
    score_requests,
    segment_ppl,
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


STAGE_ARGUMENTS = {"ppl": "", "tasks": "SHARD SHARDS", "gen": "TASK SHARD SHARDS", "latency": "", "summary": ""}


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """The options, with the stage's arguments checked and parsed into `task`, `shard` and `shards`."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--tokenizer", help="tokenizer directory (default: the artifact's)")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mbpp-sandbox", default="", help="command prefix that sandboxes EvalPlus, e.g. bwrap ...")
    parser.add_argument("--prefill", choices=["teacher", "r1"], default="teacher",
                        help="Table 4: prefill with the teacher at R=5 or with one recurrence")
    parser.add_argument("--student", type=Path,
                        help="Table 4: prefill with this distilled student (artifact or student_final.pt)")
    parser.add_argument("stage", choices=list(STAGE_ARGUMENTS))
    parser.add_argument("args", nargs="*", help="the stage's arguments (see above)")
    args = parser.parse_args()
    values = list(args.args)
    args.task = values.pop(0) if args.stage == "gen" and values else None
    counts = {"tasks": (2,), "gen": (2,)}.get(args.stage, (0,))
    if len(values) not in counts or not all(value.isdigit() for value in values):
        parser.error(f"{args.stage} takes {STAGE_ARGUMENTS[args.stage] or 'no arguments'}")
    args.shard, args.shards = map(int, values) if values else (0, 1)
    if args.shard >= args.shards:
        parser.error("SHARD must be below SHARDS")
    if args.stage == "gen" and args.task not in GENERATION:
        parser.error(f"TASK is one of {', '.join(GENERATION)}")
    if args.student and args.prefill == "r1":
        parser.error("--student prefills with the distilled student; drop --prefill r1")
    return parser, args


def make_arm(model, args, manifest: dict, config: dict) -> TerminalKV:
    """The evaluated path: D4 with its own KV, a Table 4 prefill, or the teacher's terminal KV at R=5."""
    if model.arch == "transformer":
        return DenseKV(model)
    if args.student:
        return DistilledPrefill(model, load_student(args.student, manifest, config["model"]))
    if args.prefill == "r1":
        return TeacherR1Prefill(model)
    return TerminalKV(model)


def run_ppl(arm, tokenizer, data: dict, args) -> None:
    wiki = segment_ppl(arm, wiki_batches(tokenizer, data["wiki"]), tokenizer.bos_id)
    write_json(args.out / "ppl.json", {"wiki": wiki})


def run_latency(arm, model, tokenizer, data: dict, args) -> None:
    """Time the arm's prefill, and the teacher's own for a Table 4 prefill."""
    arms = {"teacher": TerminalKV(model)} if type(arm) in (TeacherR1Prefill, DistilledPrefill) else {}
    x, _, _, ids, _ = next(wiki_batches(tokenizer, data["wiki"]))
    tokens = torch.as_tensor(x, dtype=torch.long, device="cuda")
    write_json(args.out / "latency.json", prefill_latency(
        {**arms, type(arm).__name__: arm}, tokens, torch.tensor(ids, device="cuda")))


def run_tasks(arm, tokenizer, data: dict, args) -> None:
    shard, shards = args.shard, args.shards
    for task in CHOICE_TASKS:
        requests = prepare_requests(load_choice_examples(task, data["tasks"][task]), tokenizer)
        records = score_requests(arm, requests, shard, shards)
        path = args.out / "tasks" / f"{task}-shard{shard}of{shards}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
    write_json(args.out / "tasks" / f"done-shard{shard}of{shards}.json", dict(ok=True))


def run_generation(arm, tokenizer, data: dict, args) -> None:
    task = args.task
    items, results, stats = generate(arm, generation_items(task, tokenizer, data[task]), task, tokenizer,
                                     args.shard, args.shards)
    out = args.out / "gen" / task / f"shard{args.shard}of{args.shards}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    if task == "mbpp":
        metrics = evaluate_mbpp(results, items, out, data["mbpp_plus"], shlex.split(args.mbpp_sandbox))
    else:
        scored, metrics = score_generations(task, items, results)
        (out / "scored.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scored))
    write_json(out / "result.json", dict(ok=True, task=task, examples=len(items), metrics=metrics, stats=stats))


def main() -> None:
    parser, args = parse_args()
    data = json.loads(args.data.read_text())
    if args.stage == "summary":
        write_json(args.out / "summary.json", table_row(args.out, data))
        return
    manifest, config, state = open_artifact(args.artifact)
    if config["model"]["arch"] == "transformer" and (args.student or args.prefill == "r1"):
        parser.error("a dense artifact (D4) decodes itself; --student and --prefill r1 apply to Huginn artifacts")
    init_torch_distributed(TrainerConf.nccl_timeout)
    initialize_model_parallel(1, 1, TrainerConf.nccl_timeout)
    tokenizer = build_tokenizer(TokenizerConf(type="huggingface", path=args.tokenizer or str(
        args.artifact / config["tokenizer"]["path"])))
    model = build_model(config["model"], state, tokenizer)
    del state
    arm = make_arm(model, args, manifest, config)
    if args.stage == "ppl":
        run_ppl(arm, tokenizer, data, args)
    elif args.stage == "latency":
        run_latency(arm, model, tokenizer, data, args)
    elif args.stage == "tasks":
        run_tasks(arm, tokenizer, data, args)
    else:
        run_generation(arm, tokenizer, data, args)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
