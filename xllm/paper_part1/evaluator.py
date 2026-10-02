"""Native single-GPU Part 1 evaluator with strict item-level resume."""

from __future__ import annotations

import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

import torch
import xllm_extension
from torch import nn

from xllm.data.dataset_streamer.tokenizer import Tokenizer
from xllm.paper_part1.tasks.base import ChoiceTask, GenerationTask
from xllm.paper_part1.tasks.human_eval import HumanEvalTask
from xllm.paper_part1.tasks.mbpp import MBPPTask
from xllm.paper_part1.tasks.task_iterator import TaskIterator
from xllm.modules.fused_ops.attention import flash as flash_backend
from xllm.paper_part1.eval_data_contract import dataset_hash, task_files
from xllm.paper_part1.eval_protocol import (
    TaskSpec,
    apply_task_spec,
)
from xllm.paper_part1.eval_resume import (
    EvalResumeHashes,
    EvalResumeStore,
)
from xllm.paper_part1.eval_tasks import get_eval_task
from xllm.paper_part1.files import (
    atomic_write,
    require_sha256,
    sha256_file,
    sha256_json,
    write_json,
)
from xllm.paper_part1.native_inference import (
    configure_native_runtime,
    generate_native,
    model_device,
)


_EVALUATOR_HASH_SCHEMA_VERSION = 2
# Reported for a declared task metric that no item produced, e.g. in a task without items.
_MISSING_METRIC = -1.0


@dataclass(frozen=True)
class NativeGenerationConfig:
    """Generation identity used by both execution and resume hashing."""

    use_sampling: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.use_sampling and self.temperature <= 0:
            raise ValueError("temperature must be positive when sampling")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if self.top_k < 0 or not 0 <= self.top_p < 1:
            raise ValueError("invalid top-k/top-p sampling parameters")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


@dataclass(frozen=True)
class NativeTaskEvaluation:
    task_name: str
    metrics: dict[str, float]
    cache_hits: int
    cache_misses: int
    model_hash: str
    evaluator_hash: str
    suite_hash: str
    dataset_hash: str
    output_path: str
    num_items: int = 0


def _module_binary_identity(module: object, name: str) -> dict[str, str]:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{name} does not expose a module file")
    path = Path(raw_path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{name} module file must be regular: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def _selected_flash_binary_identity() -> dict[str, str]:
    if flash_backend.FLASH_ATTN_4:
        raise RuntimeError("Part 1 evaluation supports FlashAttention 2 or 3, not 4")
    if flash_backend.FLASH_ATTN_3:
        name = "flash_attn_3._C"
        try:
            module = sys.modules[name]
        except KeyError as exc:
            raise RuntimeError(
                f"selected FlashAttention module is missing: {name}"
            ) from exc
        backend = "flash_attn_3"
        distribution = "flash_attn_3"
    else:
        name = "flash_attn_2_cuda"
        module = flash_backend.flash_attn_gpu
        backend = "flash_attn_2"
        distribution = "flash-attn"
    return {
        "backend": backend,
        "distribution": distribution,
        "distribution_version": _package_version(distribution),
        **_module_binary_identity(module, name),
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _cuda_hardware_identity() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("native Part 1 evaluation requires CUDA")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "visible_device_count": torch.cuda.device_count(),
        "name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": int(properties.total_memory),
        "multi_processor_count": int(properties.multi_processor_count),
    }


def native_evaluator_hash() -> str:
    """Hash everything that can change an evaluation result on this machine.

    Covers the SHA-256 of every ``.py`` file under the ``xllm`` package (all of
    it, not only the Part 1 evaluator); the absolute path and SHA-256 of the
    selected FlashAttention and ``xllm_extension`` binaries; the current CUDA
    device's name, compute capability, memory and SM count, and the number of
    visible devices; the TF32, reduced-precision, cuDNN and determinism
    settings and the ``CUBLAS_WORKSPACE_CONFIG`` and ``NVIDIA_TF32_OVERRIDE``
    variables; and the Python, PyTorch, CUDA, EvalPlus, NumPy, Safetensors and
    SentencePiece versions.
    """
    configure_native_runtime()
    package_root = Path(__file__).resolve().parents[1]
    files = []
    for path in sorted(package_root.rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"evaluator source must be a regular file: {path}")
        files.append(
            {
                "path": path.relative_to(package_root).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    numerical_runtime = {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cuda_matmul_allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "cuda_matmul_allow_fp16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "NVIDIA_TF32_OVERRIDE",
            )
        },
    }
    return sha256_json(
        {
            "schema_version": _EVALUATOR_HASH_SCHEMA_VERSION,
            "source_files": files,
            "runtime": {
                "compiled_binaries": {
                    "flash_attention": _selected_flash_binary_identity(),
                    "xllm_extension": _module_binary_identity(
                        xllm_extension,
                        "xllm_extension",
                    ),
                },
                "hardware": _cuda_hardware_identity(),
                "numerical": numerical_runtime,
                "software": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "packages": {
                        name: _package_version(name)
                        for name in (
                            "evalplus",
                            "numpy",
                            "safetensors",
                            "sentencepiece",
                        )
                    },
                },
            },
        }
    )


def _task_file_hashes(task_name: str, task_dir: Path) -> dict[str, str]:
    return {name: sha256_file(task_dir / name) for name in task_files(task_name)}


def _native_token_loss(
    model: nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    device = model_device(model)
    loss, _, cache = model(
        tokens.to(device=device),
        multi_segments=False,
        targets=targets.to(device=device),
        cache=None,
    )
    if cache is not None:
        raise RuntimeError("native no-cache likelihood returned an unexpected cache")
    return loss


def _choice_result(
    model: nn.Module,
    batch: Mapping[str, Any],
    task: ChoiceTask,
) -> dict[str, Any]:
    text_x = batch["text_x"]
    text_y = batch["text_y"]
    text_nll = _native_token_loss(model, text_x, text_y)
    text_nll = text_nll.view(text_x.size()).sum(dim=-1)
    text_ntokens = (text_y != -100).sum(dim=-1).to(text_nll.device)

    completion_x = batch["completion_x"]
    completion_y = batch["completion_y"]
    completion_nll = _native_token_loss(model, completion_x, completion_y)
    completion_nll = completion_nll.view(completion_x.size()).sum(dim=-1)

    start = int(batch["completion_index"][0])
    end = int(batch["completion_index"][1])
    metrics = task.evaluate(
        text_nll[start:end],
        batch["examples"][0]["raw"],
        text_ntokens[start:end],
        batch["completion_text"][0],
        completion_nll[start:end],
    )
    return {"metrics": {key: float(value) for key, value in metrics.items()}}


def _generation_result(
    model: nn.Module,
    tokenizer: Tokenizer,
    batch: Mapping[str, Any],
    task: GenerationTask,
    generation: NativeGenerationConfig,
) -> dict[str, Any]:
    metrics: dict[str, float] = {}
    if "nll" in task.metrics:
        loss = _native_token_loss(model, batch["text_x"], batch["text_y"])
        metrics["nll"] = float(loss.sum().item())

    generated = generate_native(
        model,
        tokenizer,
        batch["prompt"],
        max_prompt_len=task.max_text_len,
        max_gen_len=task.max_gen_len,
        use_sampling=generation.use_sampling,
        temperature=generation.temperature,
        top_k=generation.top_k,
        top_p=generation.top_p,
    )
    if len(generated) != 1:
        raise RuntimeError("native item evaluation expected exactly one generation")
    prediction = task.postprocess(generated[0])
    if not isinstance(prediction, str):
        raise TypeError("generation task postprocess must return text")
    evaluated = task.evaluate(prediction, batch["examples"][0]["raw"])
    metrics.update({key: float(value) for key, value in evaluated.items()})
    return {"generation": prediction, "metrics": metrics}


def _validate_case_result(
    result: object,
    *,
    expects_generation: bool,
) -> dict[str, Any]:
    expected = {"metrics", "generation"} if expects_generation else {"metrics"}
    if not isinstance(result, dict) or set(result) != expected:
        raise ValueError("cached evaluation result has an invalid schema")
    metrics = result["metrics"]
    if not isinstance(metrics, dict) or not all(
        isinstance(key, str)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        for key, value in metrics.items()
    ):
        raise ValueError("cached evaluation metrics must be finite numbers")
    if expects_generation and not isinstance(result["generation"], str):
        raise ValueError("cached generation must be text")
    return result


def _prompt_identity(
    example: Mapping[str, Any],
    task: ChoiceTask | GenerationTask,
) -> object:
    if isinstance(task, GenerationTask):
        return {"prompt": example["prompt"]}
    return {
        "full_text": example["full_text"],
        "completion_text": example["completion_text"],
    }


def _generation_contract(
    task_name: str,
    task: ChoiceTask | GenerationTask,
    generation: NativeGenerationConfig,
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "algorithm": "xllm-native-full-prefix-v1",
        "task": task_name,
        "task_type": "generation" if isinstance(task, GenerationTask) else "choice",
        "max_text_len": task.max_text_len,
        "generation": asdict(generation),
    }
    if isinstance(task, GenerationTask):
        contract["max_gen_len"] = task.max_gen_len
    return contract


def _case_seed(
    generation: NativeGenerationConfig,
    hashes: EvalResumeHashes,
) -> int:
    digest = sha256_json(
        {
            "base_seed": generation.seed,
            "item": hashes.item,
            "prompt": hashes.prompt,
            "generation": hashes.generation,
        }
    )
    # Keep the seed a non-negative signed 64-bit integer.
    return int(digest[:16], 16) % (2**63 - 1)


def _evaluate_case(
    model: nn.Module,
    tokenizer: Tokenizer,
    batch: Mapping[str, Any],
    task: ChoiceTask | GenerationTask,
    generation: NativeGenerationConfig,
    seed: int,
) -> dict[str, Any]:
    device = model_device(model)
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [
            torch.cuda.current_device() if device.index is None else device.index
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        if isinstance(task, ChoiceTask):
            return _choice_result(model, batch, task)
        return _generation_result(model, tokenizer, batch, task, generation)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = b"".join(
        (
            json.dumps(
                row,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )
    atomic_write(path, payload)


def _prepare_task(
    task_name: str,
    tasks_root: Path,
    tokenizer: Tokenizer,
    task_spec: TaskSpec | None,
) -> ChoiceTask | GenerationTask:
    task = get_eval_task(str(tasks_root), task_name, tokenizer)
    if not isinstance(task, (ChoiceTask, GenerationTask)):
        raise TypeError(f"unsupported evaluation task: {type(task)!r}")
    if task_spec is not None:
        if not (
            task_spec.task_name == task_name
            or (
                task_spec.task_name == "mmlu"
                and task_name.startswith("mmlu/")
            )
        ):
            raise ValueError(
                f"task spec {task_spec.task_name} does not match {task_name}"
            )
        apply_task_spec(task, task_spec)
    return task


@dataclass
class _ItemResults:
    rows: list[dict[str, Any]]
    metric_values: dict[str, list[float]]
    cache_hits: int = 0
    cache_misses: int = 0


def _evaluate_items(
    *,
    model: nn.Module,
    tokenizer: Tokenizer,
    task: ChoiceTask | GenerationTask,
    generation: NativeGenerationConfig,
    store: EvalResumeStore,
    identity: Mapping[str, str],
) -> _ItemResults:
    """Evaluate or load every item in file order; ``identity`` holds the item-independent hashes."""
    iterator = TaskIterator(
        path=str(Path(task.task_dir) / task.eval_file),
        batch_size=1,
        task=task,
        world_rank=0,
        world_size=1,
        seed=generation.seed,
    )
    expects_generation = isinstance(task, GenerationTask)
    results = _ItemResults(
        rows=[],
        metric_values={metric: [] for metric in task.metrics},
    )
    for case_index, batch in enumerate(iterator.batch_iterator()):
        examples = batch.get("examples", [])
        if len(examples) != 1:
            raise RuntimeError("native evaluator requires one processed item per batch")
        example = examples[0]
        raw = example["raw"]
        hashes = EvalResumeHashes(
            **identity,
            prompt=sha256_json(_prompt_identity(example, task)),
            item=sha256_json({"case_index": case_index, "raw": raw}),
        )
        cached = store.load(hashes)
        if cached is None:
            results.cache_misses += 1
            result = _evaluate_case(
                model,
                tokenizer,
                batch,
                task,
                generation,
                _case_seed(generation, hashes),
            )
            result = _validate_case_result(
                result,
                expects_generation=expects_generation,
            )
            store.save(hashes, result)
        else:
            results.cache_hits += 1
            result = _validate_case_result(
                cached,
                expects_generation=expects_generation,
            )

        for key, value in result["metrics"].items():
            results.metric_values.setdefault(key, []).append(float(value))
        row = dict(raw)
        row.update(
            {
                "raw": raw,
                "metrics": result["metrics"],
                "resume_hashes": hashes.to_dict(),
            }
        )
        if expects_generation:
            row["prompt"] = example["prompt"]
            row["generation"] = result["generation"]
        else:
            row["full_text"] = example["full_text"]
            row["completion_text"] = example["completion_text"]
        results.rows.append(row)
    return results


@torch.inference_mode()
def evaluate_native_task(
    *,
    model: nn.Module,
    tokenizer: Tokenizer,
    task_name: str,
    tasks_root_dir: str | Path,
    output_dir: str | Path,
    model_hash: str,
    suite_hash: str,
    generation: NativeGenerationConfig,
    evaluator_hash: str | None = None,
    task_spec: TaskSpec | None = None,
    expected_dataset_hash: str | None = None,
) -> NativeTaskEvaluation:
    """Evaluate one Part 1 task with strict per-item resume.

    Writes ``eval_results/<task>-<eval file>`` (one row per item),
    ``<task>.native.json`` and the item records under ``resume/``.
    """
    # Setup: identities, task and dataset hash.
    model.eval()
    model_hash = require_sha256(model_hash, "model_hash")
    suite_hash = require_sha256(suite_hash, "suite_hash")
    evaluator_hash = require_sha256(
        native_evaluator_hash() if evaluator_hash is None else evaluator_hash,
        "evaluator_hash",
    )
    output = Path(output_dir)
    task = _prepare_task(task_name, Path(tasks_root_dir), tokenizer, task_spec)
    task_dir = Path(task.task_dir)
    file_hashes = _task_file_hashes(task_name, task_dir)
    task_dataset_hash = dataset_hash(
        file_hashes,
        None if task_spec is None else task_spec.scorer_dataset_hash,
    )
    if expected_dataset_hash is not None and task_dataset_hash != require_sha256(
        expected_dataset_hash,
        "expected_dataset_hash",
    ):
        raise ValueError(f"Part 1 dataset hash mismatch: {task_name}")

    # Items, in file order.
    results = _evaluate_items(
        model=model,
        tokenizer=tokenizer,
        task=task,
        generation=generation,
        store=EvalResumeStore(output / "resume"),
        identity={
            "model": model_hash,
            "evaluator": evaluator_hash,
            "suite": suite_hash,
            "dataset": task_dataset_hash,
            "generation": sha256_json(
                _generation_contract(task_name, task, generation)
            ),
        },
    )
    if _task_file_hashes(task_name, task_dir) != file_hashes:
        raise RuntimeError(f"evaluation dataset changed while in use: {task_name}")

    # Outputs: item rows, task-level metrics and the task summary.
    metrics = {
        key: fmean(values) if values else _MISSING_METRIC
        for key, values in results.metric_values.items()
    }
    result_path = output / "eval_results" / f"{task_name}-{task.eval_file}"
    _write_rows(result_path, results.rows)
    if isinstance(task, (HumanEvalTask, MBPPTask)):
        metrics.update(task.pass_k_acc(batch_paths=[result_path]))

    evaluation = NativeTaskEvaluation(
        task_name=task_name,
        metrics={key: float(value) for key, value in metrics.items()},
        cache_hits=results.cache_hits,
        cache_misses=results.cache_misses,
        model_hash=model_hash,
        evaluator_hash=evaluator_hash,
        suite_hash=suite_hash,
        dataset_hash=task_dataset_hash,
        output_path=str(result_path),
        num_items=len(results.rows),
    )
    write_json(output / f"{task_name}.native.json", asdict(evaluation))
    return evaluation
