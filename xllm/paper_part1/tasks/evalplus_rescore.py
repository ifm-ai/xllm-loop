"""Utilities for rescoring saved XLLM code generations with EvalPlus."""

from __future__ import annotations

import json
import shutil
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, List


HUMANEVAL_DATASET = "humaneval"
MBPP_DATASET = "mbpp"
SUPPORTED_DATASETS = {HUMANEVAL_DATASET, MBPP_DATASET}
REQUIRED_EVALPLUS_VERSION = "0.3.1"
EVALPLUS_DATASET_HASHES = {
    HUMANEVAL_DATASET: "fe585eb4df8c88d844eeb463ea4d0302",
    MBPP_DATASET: "ee43ecabebf20deef4bb776a405ac5b1",
}


def _require_evalplus_version() -> None:
    try:
        installed_version = metadata.version("evalplus")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "EvalPlus is not installed. Install the pinned evaluator dependency."
        ) from exc
    if installed_version != REQUIRED_EVALPLUS_VERSION:
        raise RuntimeError(
            "EvalPlus version mismatch: expected "
            f"{REQUIRED_EVALPLUS_VERSION}, got {installed_version}"
        )


def normalize_humaneval_completion(completion: str) -> str:
    """Indent a completion by four spaces, as HumanEvalTask.evaluate does."""
    if completion.startswith(" " * 4):
        return completion
    return " " * 4 + completion.lstrip(" ")


def normalize_mbpp_task_id(task_id: object) -> str:
    """Convert local MBPP ids to EvalPlus' ``Mbpp/<id>`` keys."""
    if isinstance(task_id, int):
        return f"Mbpp/{task_id}"
    task_id_str = str(task_id)
    if "/" in task_id_str:
        return f"Mbpp/{task_id_str.split('/')[-1]}"
    return f"Mbpp/{task_id_str}"


def xllm_eval_record_to_evalplus_sample(record: Dict, dataset: str) -> Dict[str, str]:
    """Convert one XLLM ``eval_results`` JSONL row into EvalPlus sample schema."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported EvalPlus dataset: {dataset}")

    raw = record.get("raw")
    if not isinstance(raw, dict):
        raise ValueError("XLLM eval record is missing a raw example dictionary.")
    generation = record.get("generation")
    if not isinstance(generation, str):
        raise ValueError("XLLM eval record is missing a string generation.")

    if dataset == HUMANEVAL_DATASET:
        task_id = raw.get("task_id")
        prompt = raw.get("prompt")
        if not isinstance(task_id, str) or not isinstance(prompt, str):
            raise ValueError("HumanEval record must include raw.task_id and raw.prompt.")
        solution = prompt + normalize_humaneval_completion(generation)
        return {"task_id": task_id, "solution": solution}

    task_id = raw.get("task_id")
    if task_id is None:
        raise ValueError("MBPP record must include raw.task_id.")
    return {"task_id": normalize_mbpp_task_id(task_id), "solution": generation}


def load_xllm_eval_results(path: Path) -> List[Dict]:
    """Load the item rows the Part 1 evaluator writes to ``eval_results/``."""
    rows = []
    with path.open() as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_evalplus_samples(
    input_paths: Iterable[Path],
    output_path: Path,
    dataset: str,
) -> int:
    """Write EvalPlus samples converted from one or more XLLM eval-results files."""
    paths = list(input_paths)
    if not paths:
        raise ValueError("At least one XLLM eval-results path is required.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w") as fout:
        for input_path in paths:
            for record in load_xllm_eval_results(input_path):
                sample = xllm_eval_record_to_evalplus_sample(record, dataset)
                fout.write(json.dumps(sample, sort_keys=True) + "\n")
                count += 1
    return count


def run_evalplus_evaluate(
    dataset: str,
    samples_path: Path,
    output_path: Path,
) -> Dict:
    """Score a samples file with EvalPlus 0.3.1 and copy its results to ``output_path``.

    EvalPlus 0.3.1 writes ``<samples>_eval_results.json`` next to the samples
    and prompts before replacing an existing one, so an earlier result file is
    removed first.
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported EvalPlus dataset: {dataset}")
    _require_evalplus_version()
    try:
        from evalplus.evaluate import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "EvalPlus is not installed. Install `evalplus` before running plus "
            "rescoring, or use the converter without executing the scorer."
        ) from exc

    result_path = Path(str(samples_path).replace(".jsonl", "_eval_results.json"))
    result_path.unlink(missing_ok=True)
    evaluate(dataset=dataset, samples=str(samples_path), i_just_wanna_run=True)
    shutil.copyfile(result_path, output_path)
    with output_path.open() as fin:
        return json.load(fin)


def summarize_evalplus_results(results: Dict) -> Dict[str, float]:
    """Return percentage pass@1 values from an EvalPlus 0.3.1 result dictionary.

    ``plus`` counts a sample as passing only if it passes the base tests too.
    """
    summary: Dict[str, float] = {}
    eval_results = results.get("eval", {})
    if isinstance(eval_results, dict):
        totals = {"base": 0, "plus": 0}
        passed = {"base": 0, "plus": 0}
        for task_results in eval_results.values():
            if not isinstance(task_results, list):
                continue
            for sample_result in task_results:
                if not isinstance(sample_result, dict):
                    continue
                for split_name in totals:
                    status = sample_result.get(f"{split_name}_status")
                    if status is None:
                        continue
                    totals[split_name] += 1
                    if split_name == "plus":
                        base_status = sample_result.get("base_status")
                        if base_status == "pass" and status == "pass":
                            passed[split_name] += 1
                    elif status == "pass":
                        passed[split_name] += 1
        for split_name, total in totals.items():
            if total:
                summary[f"{split_name}/pass@1"] = 100.0 * passed[split_name] / total
    return summary
