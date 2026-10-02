"""Explicit task dispatch for the XLLM Part 1 evaluator."""

from __future__ import annotations

import os
from collections.abc import Sequence

from xllm.data.dataset_streamer.tokenizer import Tokenizer
from xllm.paper_part1.tasks.arc import ARCTask
from xllm.paper_part1.tasks.base import BaseTask
from xllm.paper_part1.tasks.bbh_cot import BBHCoTTask
from xllm.paper_part1.tasks.drop import DROPTask
from xllm.paper_part1.tasks.evalplus import HumanEvalPlusTask, MBPPPlusTask
from xllm.paper_part1.tasks.gsm8k import GSM8KTask
from xllm.paper_part1.tasks.hellaswag import HellaSwagTask
from xllm.paper_part1.tasks.math import MATHTask
from xllm.paper_part1.tasks.mmlu import MMLU_TASKS
from xllm.paper_part1.tasks.mmlu_pro import MMLUProTask
from xllm.paper_part1.tasks.tqa import TQATask
from xllm.paper_part1.tasks.task_paths import task_data_dir_name


_TASK_CLASSES: dict[str, type[BaseTask]] = {
    "arc_challenge": ARCTask,
    "hellaswag": HellaSwagTask,
    "mmlu_pro": MMLUProTask,
    "bbh_cot": BBHCoTTask,
    "tqa": TQATask,
    "drop": DROPTask,
    "gsm8k": GSM8KTask,
    "math": MATHTask,
    "human_eval_plus": HumanEvalPlusTask,
    "mbpp_plus": MBPPPlusTask,
}
MMLU_TASK_NAMES = tuple(f"mmlu/{subject}" for subject in MMLU_TASKS)


def get_task_class(task_name: str) -> type[BaseTask]:
    """Return the exact released class for one expanded Part 1 task name."""
    if not isinstance(task_name, str) or not task_name:
        raise ValueError("Part 1 task name must be a non-empty string")
    if task_name.startswith("mmlu/"):
        subject = task_name.removeprefix("mmlu/")
        if subject and "/" not in subject:
            task_cls = MMLU_TASKS.get(subject)
            if task_cls is not None:
                return task_cls
    else:
        task_cls = _TASK_CLASSES.get(task_name)
        if task_cls is not None:
            return task_cls
    raise ValueError(f"Unknown Part 1 task: {task_name}")


def expand_task_names(task_names: Sequence[str]) -> tuple[str, ...]:
    """Expand the frozen MMLU aggregate without consulting task registries."""
    requested = tuple(task_names)
    if not requested:
        raise ValueError("task_names must not be empty")
    if any(
        not isinstance(task_name, str)
        or not task_name
        or "," in task_name
        for task_name in requested
    ):
        raise ValueError("task names must be non-empty and must not contain commas")
    if len(requested) != len(set(requested)):
        raise ValueError("task_names must not contain duplicates")

    expanded = tuple(
        expanded_name
        for task_name in requested
        for expanded_name in (
            MMLU_TASK_NAMES if task_name == "mmlu" else (task_name,)
        )
    )
    if len(expanded) != len(set(expanded)):
        raise ValueError("expanded task_names must not contain duplicates")
    for task_name in expanded:
        get_task_class(task_name)
    return expanded


def get_eval_task(
    tasks_root_dir: str, task_name: str, tokenizer: Tokenizer
) -> BaseTask:
    """Construct a Part 1 task without consulting the global task registry."""
    task_cls = get_task_class(task_name)
    return task_cls(tokenizer, os.path.join(tasks_root_dir, task_data_dir_name(task_name)))
