"""Evaluate a native XLLM Part 1 artifact with strict item-level resume."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from xllm.paper_part1.tasks.task_paths import task_data_dir_name
from xllm.paper_part1.artifacts import ARTIFACT_MANIFEST
from xllm.paper_part1.eval_data_contract import (
    EvalDataContract,
    verified_eval_data_snapshot,
)
from xllm.paper_part1.eval_protocol import (
    PAPER_PROTOCOL,
    paper_reported_metrics,
)
from xllm.paper_part1.eval_tasks import (
    expand_task_names,
)
from xllm.paper_part1.evaluator import (
    NativeGenerationConfig,
    NativeTaskEvaluation,
    evaluate_native_task,
    native_evaluator_hash,
)
from xllm.paper_part1.files import sha256_file, sha256_json, write_json
from xllm.paper_part1.native_inference import load_native_model


_SUITE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NativeSuiteEvaluation:
    model_hash: str
    evaluator_hash: str
    suite_hash: str
    task_names: tuple[str, ...]
    tasks: dict[str, NativeTaskEvaluation]
    output_path: str
    protocol_name: str | None = None
    reported_metrics: dict[str, float] = field(default_factory=dict)
    eval_data_tree_hash: str | None = None


def _expand_task_names(
    tasks_root_dir: Path,
    task_names: Sequence[str],
) -> tuple[str, ...]:
    expanded = expand_task_names(task_names)
    for task_name in expanded:
        task_dir = tasks_root_dir / task_data_dir_name(task_name)
        if not task_dir.is_dir():
            raise ValueError(f"evaluation task directory does not exist: {task_dir}")
    return expanded


def _suite_hash(
    task_names: tuple[str, ...],
    generation: NativeGenerationConfig,
    eval_data: EvalDataContract | None,
) -> str:
    return sha256_json(
        {
            "schema_version": _SUITE_SCHEMA_VERSION,
            "task_names": task_names,
            "generation": asdict(generation),
            "protocol": PAPER_PROTOCOL.to_dict(),
            "eval_data": (
                None
                if eval_data is None
                else {
                    "tree_sha256": eval_data.tree_sha256,
                }
            ),
        }
    )


def _evaluate_native_suite_from_root(
    *,
    artifact: Path,
    tasks_root: Path,
    output: Path,
    generation: NativeGenerationConfig,
    requested_tasks: tuple[str, ...],
    eval_data: EvalDataContract | None,
) -> NativeSuiteEvaluation:
    """Evaluate ``requested_tasks`` under the paper task specs.

    ``eval_data`` is the read-only snapshot of the full protocol, or ``None``
    for a subset read directly from ``tasks_root``.
    """
    expanded_tasks = _expand_task_names(tasks_root, requested_tasks)
    manifest_path = artifact / ARTIFACT_MANIFEST
    model_hash = sha256_file(manifest_path)
    model, tokenizer, _ = load_native_model(artifact)
    if sha256_file(manifest_path) != model_hash:
        raise RuntimeError("artifact manifest changed during model load")

    evaluator_hash = native_evaluator_hash()
    suite_hash = _suite_hash(expanded_tasks, generation, eval_data)
    task_results: dict[str, NativeTaskEvaluation] = {}
    for task_name in expanded_tasks:
        task_results[task_name] = evaluate_native_task(
            model=model,
            tokenizer=tokenizer,
            task_name=task_name,
            tasks_root_dir=tasks_root,
            output_dir=output,
            model_hash=model_hash,
            evaluator_hash=evaluator_hash,
            suite_hash=suite_hash,
            generation=generation,
            task_spec=PAPER_PROTOCOL.spec_for_task(task_name),
            expected_dataset_hash=(
                None
                if eval_data is None
                else eval_data.expected_dataset_hash(task_name)
            ),
        )

    result_path = output / "results.native.json"
    evaluation = NativeSuiteEvaluation(
        model_hash=model_hash,
        evaluator_hash=evaluator_hash,
        suite_hash=suite_hash,
        task_names=expanded_tasks,
        tasks=task_results,
        output_path=str(result_path),
        protocol_name=None if eval_data is None else PAPER_PROTOCOL.name,
        reported_metrics=paper_reported_metrics(
            task_results,
            require_all_tasks=eval_data is not None,
        ),
        eval_data_tree_hash=None if eval_data is None else eval_data.tree_sha256,
    )
    write_json(result_path, asdict(evaluation))
    return evaluation


def evaluate_native_suite(
    *,
    artifact_dir: str | Path,
    tasks_root_dir: str | Path,
    output_dir: str | Path,
    generation: NativeGenerationConfig,
    task_names: Sequence[str] | None = None,
    protocol_name: str | None = None,
) -> NativeSuiteEvaluation:
    """Load one artifact once and evaluate the paper protocol or a subset of its tasks.

    Every task uses the paper task spec (prompt and generation limits, scoring
    rule). ``protocol_name`` runs every task on a read-only snapshot of the task
    files; ``task_names`` runs the named tasks on the files as they are.
    """
    artifact = Path(artifact_dir)
    source_tasks_root = Path(tasks_root_dir)
    output = Path(output_dir)
    if protocol_name is None:
        return _evaluate_native_suite_from_root(
            artifact=artifact,
            tasks_root=source_tasks_root,
            output=output,
            generation=generation,
            requested_tasks=tuple(task_names or ()),
            eval_data=None,
        )
    if protocol_name != PAPER_PROTOCOL.name:
        raise ValueError(f"unknown evaluation protocol: {protocol_name}")
    if task_names:
        raise ValueError("task_names and protocol_name are mutually exclusive")
    with verified_eval_data_snapshot(source_tasks_root) as (
        snapshot_root,
        eval_data,
    ):
        return _evaluate_native_suite_from_root(
            artifact=artifact,
            tasks_root=snapshot_root,
            output=output,
            generation=generation,
            requested_tasks=tuple(PAPER_PROTOCOL),
            eval_data=eval_data,
        )


_SAMPLING_OPTIONS = ("use_sampling", "temperature", "top_k", "top_p")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        required=True,
        type=Path,
        help="artifact directory (a released or exported checkpoint)",
    )
    parser.add_argument(
        "--tasks-root",
        required=True,
        type=Path,
        help="directory with one prepared data directory per task",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="output directory; rerunning into it resumes finished items",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--tasks",
        nargs="+",
        help=(
            "evaluate these protocol tasks with the paper task specs"
            " ('mmlu' expands to its 57 subjects)"
        ),
    )
    selection.add_argument(
        "--protocol",
        choices=(PAPER_PROTOCOL.name,),
        help="evaluate every protocol task on a read-only snapshot of the task files",
    )
    parser.add_argument(
        "--use-sampling",
        action="store_true",
        default=None,
        help="sample instead of greedy decoding (not with --protocol)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="sampling temperature (default 1.0; not with --protocol)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="sample from the k most likely tokens (default 0: off; not with --protocol)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="nucleus sampling threshold (default 0.0: off; not with --protocol)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="base seed of the per-item random state (default 42)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    sampling = {
        name: getattr(args, name)
        for name in _SAMPLING_OPTIONS
        if getattr(args, name) is not None
    }
    if args.protocol is not None and sampling:
        parser.error("--protocol decodes greedily and takes no sampling options")
    evaluation = evaluate_native_suite(
        artifact_dir=args.artifact,
        tasks_root_dir=args.tasks_root,
        output_dir=args.output,
        generation=NativeGenerationConfig(**sampling, seed=args.seed),
        task_names=None if args.tasks is None else tuple(args.tasks),
        protocol_name=args.protocol,
    )
    print(
        json.dumps(
            {
                "model_hash": evaluation.model_hash,
                "protocol_name": evaluation.protocol_name,
                "eval_data_tree_hash": evaluation.eval_data_tree_hash,
                "suite_hash": evaluation.suite_hash,
                "output_path": evaluation.output_path,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
