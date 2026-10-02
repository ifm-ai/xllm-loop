"""Prefill latency of Table 4, with the paper's timing protocol.

Each sample is one full-prompt prefill of eight rows: the arm's path over all
prompt tokens but the last, then the decoder on the last one (`arm.prefill`).
Every arm runs once to warm up; then six repetitions time the arms in rotating
order, each call after garbage collection, an emptied CUDA cache and a device
synchronization. Table 4 reports the mean and standard deviation.
"""
from __future__ import annotations

import gc
import statistics
import time
from typing import TYPE_CHECKING, Mapping

import torch

if TYPE_CHECKING:
    from xllm.paper_part2.evaluator import TerminalKV


REPEATS = 6


@torch.inference_mode()
def prefill_latency(arms: Mapping[str, "TerminalKV"], tokens: torch.Tensor, ids: torch.Tensor,
                    repeats: int = REPEATS) -> dict[str, dict]:
    """Milliseconds of every arm's prefill of `tokens`: the samples, their mean and standard deviation."""
    names = list(arms)
    for name in names:
        logits, cache = arms[name].prefill(tokens, ids)
        del logits, cache
    samples = {name: [] for name in names}
    for repeat in range(repeats):
        shift = repeat % len(names)
        for name in names[shift:] + names[:shift]:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            start = time.perf_counter()
            logits, cache = arms[name].prefill(tokens, ids)
            torch.cuda.synchronize()
            samples[name].append(1e3 * (time.perf_counter() - start))
            del logits, cache
    return {name: dict(ms=values, mean_ms=statistics.mean(values), sd_ms=statistics.stdev(values),
                       batch=tokens.shape[0], prompt_tokens=tokens.shape[1])
            for name, values in samples.items()}
