from typing import List, Dict
import json

from xllm.paper_part1.tasks.base import Example, GenerationTask
from xllm.paper_part1.tasks.utils import exact_match_score, f1_score, normalize_answer, is_valid_code
from collections import defaultdict
from pathlib import Path

# HumanEval example
# {
#   "task_id": "HumanEval/3",
#   "prompt": "from typing import List\n\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n",
#   "entry_point": "below_zero",
#   "canonical_solution": "    balance = 0\n\n    for op in operations:\n        balance += op\n        if balance < 0:\n            return True\n\n    return False\n",
#   "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([]) == False\n    assert candidate([1, 2, -3, 1, 2, -3]) == False\n    assert candidate([1, 2, -4, 5, 6]) == True\n    assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False\n    assert candidate([1, -1, 2, -2, 5, -5, 4, -5]) == True\n    assert candidate([1, -2, 2, -2, 5, -5, 4, -4]) == True\n"
# }


class HumanEvalTask(GenerationTask):
    """
    Code generation task released by OpenAI:
    https://github.com/openai/human-eval
    """

    dataset_dirs = ["human_eval"]
    metrics = ["em", "f1", "nll", "pass_at_1"]

    n_fewshot = 0
    sep = "\n"
    eval_file = "HumanEval.jsonl"
    max_text_len = 1024
    max_gen_len = 512
    pass_at_k: int = 1

    def eval_normalizer(self, answer: str) -> str:
        split = answer.split("\n")
        s = split[0]
        for l in split[1:]:
            if not l or l[0:2] == "  ":
                s += "\n" + l
            else:
                break
        return s

    def textify_example(self, example: Example) -> str:
        context = self.get_prompt(example)
        answer = self.get_target(example)
        return f"{context} {answer}"

    def get_prompt(self, example: Example) -> str:
        return example["prompt"]

    def get_target(self, example: Example) -> str:
        target = example["canonical_solution"]
        return target

    def get_tests(self, example: Example) -> List[str]:
        tests_str = example["test"]
        entry_point = example["entry_point"]
        return [tests_str + f"\ncheck({entry_point})"]

    def postprocess(self, tokens: List[int]) -> str:
        for k, t in enumerate(tokens):
            if t == self.tokenizer.eos_id:
                tokens = tokens[: k + 1]
                break
        generation = self.tokenizer.decode(tokens)

        return self.eval_normalizer(generation)

    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        ground_truths = [example["canonical_solution"]]
        prompt = self.get_prompt(example)
        tests = self.get_tests(example)
        assert (
            len(tests) == 1
        ), f"There should be only 1 test string in human eval. Found {len(tests)}:\n {tests}"

        # the indentation of the prediction is often wrong. It should always be with 4 spaces for HumanEval
        if not prediction[:4] == " " * 4:
            prediction = " " * 4 + prediction.lstrip(" ")

        sample_metrics = {}
        sample_metrics["em"] = 100 * exact_match_score(
            prediction, ground_truths, normalize_answer
        )
        sample_metrics["f1"] = 100 * f1_score(
            prediction, ground_truths, normalize_answer
        )
        for metric in [m for m in self.metrics if m.startswith("pass_at_")]:
            code = prompt + prediction
            sample_metrics[metric] = 100 * is_valid_code(
                code=code, test=tests[0], timeout=5
            )

        return sample_metrics

    def pass_k_acc(self, batch_paths: List[str]) -> Dict[str, float]:
        all_samples = []
        for p in batch_paths:
            with Path(p).open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    all_samples.append(json.loads(line))

        by_task = defaultdict(list)
        for s in all_samples:
            raw = s.get("raw", {})
            task_id = raw.get("task_id") or s.get("task_id")
            if task_id is None:
                task_id = s["prompt"]
            by_task[task_id].append(s)

        per_task_ok = []
        for task_id, samples in by_task.items():
            raw0 = samples[0].get("raw", {})
            prompt = raw0.get("prompt") or samples[0]["prompt"]
            tests_str = raw0.get("test") or samples[0].get("test", "")
            entry_point = raw0.get("entry_point") or samples[0].get("entry_point", "")
            test_code = tests_str + f"\ncheck({entry_point})" if tests_str else ""

            this_ok = False
            for s in samples:
                gen = s["generation"]
                if not gen[:4] == " " * 4:
                    gen = " " * 4 + gen.lstrip()
                code = prompt + gen
                if is_valid_code(code=code, test=test_code, timeout=5):
                    this_ok = True
                    break  
            per_task_ok.append(1.0 if this_ok else 0.0)

        avg = sum(per_task_ok) / len(per_task_ok) if per_task_ok else 0.0
        return {f"pass_at_{self.pass_at_k}": avg * 100.0}
