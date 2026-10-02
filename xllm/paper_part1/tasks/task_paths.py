"""Stable task-name to evaluation-data-directory mapping."""

from __future__ import annotations


TASK_DATA_DIR_ALIASES = {
    "human_eval_plus": "human_eval",
}


def task_data_dir_name(task_name: str) -> str:
    """Return the data directory used by one native evaluator task name."""
    if not isinstance(task_name, str) or not task_name:
        raise ValueError("task_name must be a non-empty string")
    return TASK_DATA_DIR_ALIASES.get(task_name, task_name)
