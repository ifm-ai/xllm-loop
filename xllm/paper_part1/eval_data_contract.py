"""Snapshot evaluation inputs and derive cache identities from their bytes."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Iterator, Mapping

from xllm.paper_part1.tasks.task_paths import task_data_dir_name
from xllm.paper_part1.eval_protocol import PAPER_PROTOCOL
from xllm.paper_part1.eval_tasks import expand_task_names, get_task_class
from xllm.paper_part1.files import read_regular_bytes, sha256_json


@dataclass(frozen=True)
class EvalDataContract:
    tree_sha256: str
    task_dataset_hashes: Mapping[str, str]

    def expected_dataset_hash(self, task_name: str) -> str:
        return self.task_dataset_hashes[task_name]


def task_files(task_name: str) -> tuple[str, ...]:
    """Read filenames from the task class used by the evaluator."""
    task = get_task_class(task_name)
    names = [task.eval_file]
    if task.n_fewshot > 0 or task.fewshot_index:
        names.append(task.fewshot_file)
    return tuple(sorted(set(names)))


def dataset_hash(
    file_hashes: Mapping[str, str],
    scorer_dataset_hash: str | None,
) -> str:
    """One task's dataset identity: its files' SHA-256 by name, plus the scorer's dataset if any."""
    local_dataset_hash = sha256_json(dict(file_hashes))
    if scorer_dataset_hash is None:
        return local_dataset_hash
    return sha256_json(
        {
            "local_dataset": local_dataset_hash,
            "scorer_dataset": scorer_dataset_hash,
        }
    )


def required_eval_files() -> tuple[Path, ...]:
    return tuple(
        Path(task_data_dir_name(name)) / filename
        for name in expand_task_names(tuple(PAPER_PROTOCOL))
        for filename in task_files(name)
    )


def _load_contract(tasks_root: Path, snapshot_root: Path) -> EvalDataContract:
    datasets = {}
    tree = {}
    for name in expand_task_names(tuple(PAPER_PROTOCOL)):
        local_files = {}
        for filename in task_files(name):
            relative = Path(task_data_dir_name(name)) / filename
            payload = read_regular_bytes(tasks_root, relative)
            digest = hashlib.sha256(payload).hexdigest()
            local_files[filename] = digest
            tree[relative.as_posix()] = digest
            destination = snapshot_root / relative
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(payload)
            destination.chmod(0o400)
        datasets[name] = dataset_hash(
            local_files,
            PAPER_PROTOCOL.spec_for_task(name).scorer_dataset_hash,
        )
    return EvalDataContract(
        tree_sha256=sha256_json(tree),
        task_dataset_hashes=MappingProxyType(datasets),
    )


def _tasks_root(path: str | Path) -> Path:
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("tasks root must be a non-symlink directory")
    return root.resolve(strict=True)


@contextmanager
def verified_eval_data_snapshot(
    tasks_root_dir: str | Path,
) -> Iterator[tuple[Path, EvalDataContract]]:
    """Evaluate the same bytes that determine the per-item resume keys."""
    root = _tasks_root(tasks_root_dir)
    with TemporaryDirectory(prefix="xllm-part1-eval-") as temporary:
        snapshot = Path(temporary)
        contract = _load_contract(root, snapshot)
        yield snapshot, contract
