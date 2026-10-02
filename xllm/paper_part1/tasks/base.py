import json
import os
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import torch

from xllm.data.dataset_streamer.tokenizer import Tokenizer


Example = Dict[str, Any]


class BaseTask:
    tasks: Dict[str, Type["BaseTask"]] = {}
    dataset_dirs: List[str] = []
    eval_mode: str
    metrics: List
    sep: str
    eval_file: str
    max_text_len: int
    random_fewshots: bool = False
    description: str = ""
    # options for fewshot mode: first, random and index
    # first: select the first n_fewshot examples of the fewshot file as fewshot examples for all test examples
    # random: select n_fewshot random examples from the fewshot file as fewshot examples, examples similar to the current test example will be discarded
    # index: select fewshot examples in the fewshot file at the index specific by fewshot_index list
    # for instance fewshot_index = [1, 4] will use the second and fifth examples as fewshot examples for all examples.
    fewshot_file: str
    fewshot_mode: str
    n_fewshot: int = 0
    fewshot_index: List[int] = []

    def __init__(self, tokenizer: Tokenizer, task_dir: str):
        self.tokenizer = tokenizer
        self.task_dir = task_dir
        self._all_fewshot_examples: List[Example] = []

        assert self.eval_file.endswith(".jsonl")
        if self.n_fewshot > 0 or len(self.fewshot_index) > 0:
            assert self.fewshot_mode in [
                "index",
                "first",
                "random",
            ], f"Fewshot strategy {self.fewshot_mode} not supported"
            assert os.path.isfile(
                os.path.join(self.task_dir, self.fewshot_file)
            ), f"Fewshot file does not exist"

    @classmethod
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        for dirname in cls.dataset_dirs:
            cls.tasks[dirname] = cls

    def get_input_output_tokens(
        self,
        text: str,
        completion: str,
    ) -> Tuple[List[int], List[int]]:
        x = self.tokenizer.encode(text, bos=True, eos=False)
        len_completion = len(self.tokenizer.encode(completion, bos=False, eos=False))
        y = [-100 if k < len(x) - len_completion else t for k, t in enumerate(x)]
        return x[:-1], y[1:]

    def _load_fewshot_file(self) -> List[Example]:
        all_fewshot_examples: List[Example] = []
        if self.n_fewshot > 0 or len(self.fewshot_index) > 0:
            with open(os.path.join(self.task_dir, self.fewshot_file)) as fin:
                for line in fin:
                    if not line:
                        continue
                    all_fewshot_examples.append(json.loads(line))
        return all_fewshot_examples

    def get_fewshot(
        self, example: Example, rng: np.random.RandomState
    ) -> List[Example]:
        if self.n_fewshot == 0 and len(self.fewshot_index) == 0:
            return []
        if len(self._all_fewshot_examples) == 0 and (
            self.n_fewshot > 0 or len(self.fewshot_index) > 0
        ):
            self._all_fewshot_examples = self._load_fewshot_file()
        if self.fewshot_mode == "index":
            fewshot_examples = [
                self._all_fewshot_examples[k] for k in self.fewshot_index
            ]
        elif self.fewshot_mode == "first":
            fewshot_examples = self._all_fewshot_examples[: self.n_fewshot]
        elif self.fewshot_mode == "random":
            while True:
                indices = rng.choice(
                    range(len(self._all_fewshot_examples)),
                    self.n_fewshot,
                    replace=False,
                )
                fewshot_examples = [self._all_fewshot_examples[idx] for idx in indices]
                if all(example != shot for shot in fewshot_examples):
                    break
        else:
            raise ValueError(f"Fewshot strategy {self.fewshot_mode} not supported")
        return fewshot_examples


class GenerationTask(BaseTask):
    max_gen_len: int

    def process(self, example: Example, rng: np.random.RandomState):
        fewshot = [self.textify_example(ex) for ex in self.get_fewshot(example, rng)]

        prompt = self.get_prompt(example)
        prompt = self.description + self.sep.join(fewshot + [prompt])
        target = self.get_target(example)

        text = prompt + " " + target  # TODO: target prompt sep

        input_tokens, target_tokens = self.get_input_output_tokens(text, target)
        return {
            "raw": example,
            "text_x": [input_tokens],
            "text_y": [target_tokens],
            "prompt": prompt,
        }

    @abstractmethod
    def postprocess(self, tokens: List[int]) -> str:
        pass

    @abstractmethod
    def get_prompt(self, example: Example) -> str:
        pass

    @abstractmethod
    def get_target(self, example: Example) -> str:
        pass

    @abstractmethod
    def textify_example(self, example: Example) -> str:
        pass

    @abstractmethod
    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        pass


class ChoiceTask(BaseTask):
    metrics = [
        "acc",
        "acc_norm",
        "acc_token",
        "acc_char",
        "acc_compl",
        "nll",
        "nll_token",
        "nll_char",
        "nll_compl",
    ]

    def process(self, example: Example, rng: np.random.RandomState):
        fewshot = [self.textify_example(ex) for ex in self.get_fewshot(example, rng)]

        text_completion = self.get_text_completion(example)
        text_completion = [
            (self.description + self.sep.join(fewshot + [t]), c) for t, c in text_completion
        ]
        input_tokens, output_tokens = [], []
        input_completion, output_completion = [], []
        for text, completion in text_completion:
            it, ot = self.get_input_output_tokens(text, completion)
            ic, oc = self.get_input_output_tokens("Answer: " + completion, completion)
            input_tokens.append(it)
            output_tokens.append(ot)
            input_completion.append(ic)
            output_completion.append(oc)

        return {
            "raw": example,
            "text_x": input_tokens,
            "text_y": output_tokens,
            "n_completion": len(input_tokens),
            "completion_x": input_completion,
            "completion_y": output_completion,
            "full_text": [t for t, _ in text_completion],
            "completion_text": [c for _, c in text_completion],
        }

    def evaluate(
        self,
        nll: torch.Tensor,
        example: Example,
        n_token: torch.Tensor,
        choices: List[str],
        nll_completion: torch.Tensor,
    ) -> Dict[str, float]:
        label = self.label(example)

        n_char = nll.new_tensor([len(c) for c in choices], dtype=torch.float32)
        if not torch.isfinite(nll).all() or not torch.isfinite(nll_completion).all():
            raise FloatingPointError("Choice evaluation received non-finite NLL values.")
        if not torch.isfinite(n_token).all() or (n_token <= 0).any():
            raise ValueError("Choice evaluation requires positive finite token counts.")
        if (n_char <= 0).any():
            raise ValueError("Choice evaluation requires non-empty choices.")
        nll_val, nll_idx = nll.min(dim=-1)
        nll_char_val, nll_char_idx = (nll / n_char).min(dim=-1)
        nll_token_val, nll_token_idx = (nll / n_token).min(dim=-1)
        nll_compl_val, nll_compl_idx = (nll - nll_completion).min(dim=-1)

        def compute_acc(pred, label):
            return 100.0 * (pred.item() == label)

        sample_metrics = {
            "acc": compute_acc(nll_idx, label),
            "acc_norm": compute_acc(nll_char_idx, label),
            "acc_char": compute_acc(nll_char_idx, label),
            "acc_token": compute_acc(nll_token_idx, label),
            "acc_compl": compute_acc(nll_compl_idx, label),
            "nll": nll_val.item(),
            "nll_char": nll_char_val.item(),
            "nll_token": nll_token_val.item(),
            "nll_compl": nll_compl_val.item(),
        }
        return sample_metrics

    @abstractmethod
    def textify_example(self, example: Example) -> str:
        pass

    @abstractmethod
    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        pass

    @abstractmethod
    def label(self, example: Example) -> int:
        pass
