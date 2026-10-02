"""EvalPlus-backed variants of existing code generation tasks."""

from pathlib import Path
from typing import Dict, List

from xllm.paper_part1.tasks.evalplus_rescore import (
    EVALPLUS_DATASET_HASHES,
    HUMANEVAL_DATASET,
    MBPP_DATASET,
    run_evalplus_evaluate,
    summarize_evalplus_results,
    write_evalplus_samples,
)
from xllm.paper_part1.tasks.human_eval import HumanEvalTask
from xllm.paper_part1.tasks.mbpp import MBPPTask


def _evalplus_pass_k_acc(
    batch_paths: List[str],
    *,
    dataset: str,
    pass_at_k: int,
) -> Dict[str, float]:
    first_path = Path(batch_paths[0])
    samples_path = first_path.with_name(f"{first_path.stem}.evalplus_samples.jsonl")
    results_path = first_path.with_name(f"{first_path.stem}.evalplus_results.json")

    sample_count = write_evalplus_samples([Path(path) for path in batch_paths], samples_path, dataset)
    results = run_evalplus_evaluate(dataset, samples_path, results_path)
    expected_hash = EVALPLUS_DATASET_HASHES[dataset]
    if results.get("hash") != expected_hash:
        raise RuntimeError(
            f"EvalPlus {dataset} dataset hash mismatch: "
            f"expected {expected_hash}, got {results.get('hash')}"
        )
    evaluations = results.get("eval")
    result_count = (
        sum(len(samples) for samples in evaluations.values())
        if isinstance(evaluations, dict)
        and all(isinstance(samples, list) for samples in evaluations.values())
        else None
    )
    if result_count != sample_count:
        raise RuntimeError(
            f"EvalPlus {dataset} result count mismatch: "
            f"expected {sample_count}, got {result_count}"
        )
    summary = summarize_evalplus_results(results)

    metrics = {
        f"evalplus/{key.replace('pass@', 'pass_at_')}": value
        for key, value in summary.items()
    }
    plus_key = f"plus/pass@{pass_at_k}"
    if plus_key in summary:
        metrics[f"pass_at_{pass_at_k}"] = summary[plus_key]
    return metrics


class HumanEvalPlusTask(HumanEvalTask):
    """HumanEval generations rescored with EvalPlus base+extra tests."""

    dataset_dirs = ["human_eval_plus"]
    metrics = ["em", "f1", "nll"]

    def pass_k_acc(self, batch_paths: List[str]) -> Dict[str, float]:
        return _evalplus_pass_k_acc(
            batch_paths,
            dataset=HUMANEVAL_DATASET,
            pass_at_k=self.pass_at_k,
        )


class MBPPPlusTask(MBPPTask):
    """MBPP generations rescored with EvalPlus base+extra tests."""

    dataset_dirs = ["mbpp_plus"]
    metrics = ["em", "f1", "nll"]

    def pass_k_acc(self, batch_paths: List[str]) -> Dict[str, float]:
        return _evalplus_pass_k_acc(
            batch_paths,
            dataset=MBPP_DATASET,
            pass_at_k=self.pass_at_k,
        )
