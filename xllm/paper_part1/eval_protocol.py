"""Frozen evaluator contract for the XLLM Part 1 paper tables."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from xllm.paper_part1.tasks.evalplus_rescore import (
    EVALPLUS_DATASET_HASHES,
    HUMANEVAL_DATASET,
    MBPP_DATASET,
)
from xllm.paper_part1.tasks.base import BaseTask, GenerationTask
from xllm.paper_part1.eval_tasks import MMLU_TASK_NAMES


@dataclass(frozen=True)
class TaskSpec:
    """One paper-facing task, metric, token cap, and dataset cardinality."""

    task_name: str
    metric: str
    max_text_len: int
    max_gen_len: int | None
    num_items: int
    scorer_dataset_hash: str | None = None
    scoring_rule: str | None = None

    def __post_init__(self) -> None:
        if not self.task_name or not self.metric:
            raise ValueError("task_name and metric must be non-empty")
        if self.max_text_len <= 0:
            raise ValueError("max_text_len must be positive")
        if self.max_gen_len is not None and self.max_gen_len <= 0:
            raise ValueError("max_gen_len must be positive when present")
        if self.num_items <= 0:
            raise ValueError("num_items must be positive")
        if self.scoring_rule is not None and (
            not self.scoring_rule or self.scoring_rule != self.scoring_rule.strip()
        ):
            raise ValueError("scoring_rule must be non-empty when present")
        if self.scorer_dataset_hash is not None and (
            len(self.scorer_dataset_hash) != 32
            or any(
                character not in "0123456789abcdef"
                for character in self.scorer_dataset_hash
            )
        ):
            raise ValueError(
                "scorer_dataset_hash must be a lowercase 32-character hex digest"
            )


@dataclass(frozen=True)
class EvaluationProtocol(Mapping[str, TaskSpec]):
    """Ordered, serializable protocol used as part of the suite identity."""

    name: str
    schema_version: int
    tasks: tuple[TaskSpec, ...]

    def __post_init__(self) -> None:
        names = tuple(spec.task_name for spec in self.tasks)
        if not self.name or self.schema_version <= 0 or not names:
            raise ValueError("evaluation protocol identity must be complete")
        if len(names) != len(set(names)):
            raise ValueError("evaluation protocol task names must be unique")

    def __getitem__(self, task_name: str) -> TaskSpec:
        for spec in self.tasks:
            if spec.task_name == task_name:
                return spec
        raise KeyError(task_name)

    def __iter__(self) -> Iterator[str]:
        return (spec.task_name for spec in self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def spec_for_task(self, task_name: str) -> TaskSpec:
        if task_name.startswith("mmlu/"):
            return self["mmlu"]
        return self[task_name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "tasks": [asdict(spec) for spec in self.tasks],
        }


PAPER_PROTOCOL = EvaluationProtocol(
    name="paper-part1",
    schema_version=2,
    # TaskSpec(task_name, metric, max_text_len, max_gen_len (None for choice tasks),
    #          num_items, scorer_dataset_hash (EvalPlus only), scoring_rule)
    tasks=(
        TaskSpec("arc_challenge", "acc_token", 512, None, 1_165),
        TaskSpec("hellaswag", "acc", 256, None, 10_042),
        TaskSpec("mmlu", "acc", 2_048, None, 14_042),
        TaskSpec("mmlu_pro", "acc", 4_096, None, 12_032),
        TaskSpec(
            "bbh_cot",
            "em",
            4_096,
            1_024,
            6_511,
            scoring_rule="bbh_cot_first_lowercase_nongreedy_line_answer_v1",
        ),
        TaskSpec("tqa", "f1", 512, 24, 11_313),
        TaskSpec("drop", "f1", 2_500, 64, 9_535),
        TaskSpec("gsm8k", "acc", 2_048, 512, 1_319),
        TaskSpec("math", "em", 4_096, 2_048, 500),
        TaskSpec(
            "human_eval_plus",
            "evalplus/plus/pass_at_1",
            1_024,
            512,
            164,
            EVALPLUS_DATASET_HASHES[HUMANEVAL_DATASET],
        ),
        TaskSpec(
            "mbpp_plus",
            "evalplus/plus/pass_at_1",
            3_096,
            1_024,
            378,
            EVALPLUS_DATASET_HASHES[MBPP_DATASET],
        ),
    ),
)


def apply_task_spec(task: BaseTask, spec: TaskSpec) -> None:
    """Apply paper caps to one task instance without changing class defaults."""
    task.max_text_len = spec.max_text_len
    if isinstance(task, GenerationTask):
        if spec.max_gen_len is None:
            raise TypeError(f"generation task {spec.task_name} lacks a generation cap")
        task.max_gen_len = spec.max_gen_len
    elif spec.max_gen_len is not None:
        raise TypeError(f"choice task {spec.task_name} has a generation cap")
    if spec.scoring_rule is not None:
        set_scoring_rule = getattr(task, "set_scoring_rule", None)
        if not callable(set_scoring_rule):
            raise TypeError(f"task {spec.task_name} does not support a scoring rule")
        set_scoring_rule(spec.scoring_rule)


def _result_metric(
    task_name: str,
    result: object,
    metric: str,
    expected_items: int,
) -> float:
    observed_items = getattr(result, "num_items", None)
    if observed_items != expected_items:
        raise ValueError(
            f"{task_name} expected {expected_items} items, got {observed_items}"
        )
    metrics = getattr(result, "metrics", None)
    if not isinstance(metrics, Mapping) or metric not in metrics:
        raise ValueError(f"{task_name} is missing reported metric {metric}")
    value = metrics[metric]
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{task_name}/{metric} must be finite")
    return float(value)


def paper_reported_metrics(
    task_results: Mapping[str, object],
    *,
    require_all_tasks: bool = True,
) -> dict[str, float]:
    """Validate item counts and select the paper's reported metric of each task.

    MMLU reports the item-weighted micro-average over its subjects. With
    ``require_all_tasks=False`` (a subset of the protocol), tasks without a
    result are skipped, and so is MMLU unless every subject has a result.
    """
    reported: dict[str, float] = {}
    for task_name, spec in PAPER_PROTOCOL.items():
        if task_name != "mmlu":
            if task_name not in task_results:
                if not require_all_tasks:
                    continue
                raise ValueError(f"missing paper task result: {task_name}")
            reported[f"{task_name}/{spec.metric}"] = _result_metric(
                task_name,
                task_results[task_name],
                spec.metric,
                spec.num_items,
            )
            continue

        subjects = [
            (name, result)
            for name, result in task_results.items()
            if name.startswith("mmlu/")
        ]
        if len(subjects) != len(MMLU_TASK_NAMES):
            if not require_all_tasks:
                continue
            raise ValueError(
                "mmlu expected "
                f"{len(MMLU_TASK_NAMES)} subjects, got {len(subjects)}"
            )
        weighted_score = 0.0
        observed_items = 0
        for subject_name, result in subjects:
            num_items = getattr(result, "num_items", None)
            if type(num_items) is not int or num_items <= 0:
                raise ValueError(f"{subject_name} has invalid item count {num_items}")
            score = _result_metric(subject_name, result, spec.metric, num_items)
            observed_items += num_items
            weighted_score += num_items * score
        if observed_items != spec.num_items:
            raise ValueError(
                f"mmlu expected {spec.num_items} items, got {observed_items}"
            )
        reported["mmlu/micro_avg/acc"] = weighted_score / observed_items
    return reported
