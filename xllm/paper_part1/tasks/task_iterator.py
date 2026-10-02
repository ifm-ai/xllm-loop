from typing import Any, List, Dict, Iterator
from collections import defaultdict
from itertools import chain
import torch
import numpy as np
from functools import partial

from .data import ValidJSONLIterator, pad
from .base import BaseTask
from .math import MATHTask
from .gsm8k import GSM8KTask
from .human_eval import HumanEvalTask
from .mbpp import MBPPTask
from .utils import get_task_rng


def _batch_iterator(
    iterator: Iterator[Dict[str, Any]],
    batch_size: int,
    max_length: int,
    drop_last: bool = False,
) -> Iterator[Dict]:
    batch: Dict[str, List] = defaultdict(list)
    curr_bs = 0
    batch_counter = 0
    for example in iterator:
        batch["examples"].append(example)
        for k, v in example.items():
            batch[k].append(v)
        curr_bs += 1
        if curr_bs == batch_size:
            batch_counter += 1
            yield tensorize_batch(batch, max_length=max_length)
            batch = defaultdict(list)
            curr_bs = 0
    if curr_bs > 0 and not drop_last:
        yield tensorize_batch(batch, max_length=max_length)


class TaskIterator:
    def __init__(
        self,
        path: str,
        batch_size: int,
        task: BaseTask,
        world_rank: int,
        world_size: int,
        seed: int,
    ):
        self.world_rank = world_rank
        self.world_size = world_size
        self.path = path
        self.batch_size = batch_size
        self.task = task
        self.rng = get_task_rng(
            base_seed=seed,
            data_parallel_rank=world_rank,
            different_seed_for_tasks_in_job_array=True,
        )

    def batch_iterator(self):
        examples = list(
            ValidJSONLIterator(
                fpath=self.path,
                world_rank=self.world_rank,
                world_size=self.world_size,
                infinite=False,
            )
        )
        process_rng_fn = partial(self.task.process, rng=self.rng)
        processed_example_iterator = map(process_rng_fn, examples)
        # I redo task.process for the case  when prompt is random
        if isinstance(self.task, MATHTask) and self.task.majority_voting_k > 1:
            for i in range(self.task.majority_voting_k - 1):
                processed_example_iterator = chain(
                    processed_example_iterator, map(process_rng_fn, examples)
                )
        elif isinstance(self.task, (HumanEvalTask, MBPPTask)) and self.task.pass_at_k > 0:
            for i in range(self.task.pass_at_k - 1):
                processed_example_iterator = chain(
                    processed_example_iterator, map(process_rng_fn, examples)
                )
        return _batch_iterator(
            processed_example_iterator, self.batch_size, self.task.max_text_len
        )


def tensorize_batch(batch: Dict, max_length: int) -> Dict:
    if not ("text_x" in batch and "text_y" in batch):
        return batch

    key_padding = {"text_x": 0, "text_y": -100, "completion_x": 0, "completion_y": -100}
    for key, padding in key_padding.items():
        if key in batch:
            batch[key] = tensorize(batch[key], max_length, padding)

    if "n_completion" in batch:
        batch["completion_index"] = np.cumsum([0] + batch["n_completion"])

    return batch


def tensorize(
    batch_tokens: List[List[List[int]]], max_length: int, pad_value: int
) -> torch.Tensor:
    # flatten tokens into List[List[int]]
    tokens = [t for ex_tokens in batch_tokens for t in ex_tokens]
    batch_max_length = max([len(t) for t in tokens])

    max_length = min(batch_max_length, max_length)
    # created padded tensors
    padded_tokens = [
        pad(x, max_length=max_length, value=pad_value, truncating="pre") for x in tokens
    ]
    return torch.tensor(padded_tokens, dtype=torch.long)
