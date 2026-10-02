from typing import List, Dict
import json

from xllm.paper_part1.tasks.base import Example, GenerationTask
from xllm.paper_part1.tasks.utils import exact_match_score, f1_score, normalize_answer, is_valid_code
from collections import defaultdict
from pathlib import Path

# MBPP example
# {
#   "text": "Write a python function to remove first and last occurrence of a given character from the string.",
#   "code": "def remove_Occ(s,ch): \r\n    for i in range(len(s)): \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    for i in range(len(s) - 1,-1,-1):  \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    return s ",
#   "task_id": 11,
#   "test_setup_code": "",
#   "test_list": [
#     "assert remove_Occ(\"hello\",\"l\") == \"heo\"",
#     "assert remove_Occ(\"abcda\",\"a\") == \"bcd\"",
#     "assert remove_Occ(\"PHP\",\"P\") == \"H\""
#   ],
#   "challenge_test_list": [
#     "assert remove_Occ(\"hellolloll\",\"l\") == \"helollol\"",
#     "assert remove_Occ(\"\",\"l\") == \"\""
#   ]
# }


class MBPPTask(GenerationTask):
    """
    Mostly Basic Programming Problems
    Text to code generation task released by Google Brain:
    https://github.com/google-research/google-research/tree/master/mbpp
    """

    dataset_dirs = ["mbpp"]
    metrics = ["em", "f1", "nll", "pass_at_1"]

    # fewshot idx corresponding to task_idx 2, 3, 4 like in their paper
    fewshot_mode = "index"
    fewshot_index: List[int] = [1, 2, 3]
    sep = "\n"

    fewshot_file = "mbpp_prompting.jsonl"
    eval_file = "mbpp_test.jsonl"
    max_text_len = 3096
    max_gen_len = 256
    prompt_format = "You are an expert Python programmer, and here is your task: {context} Your code should pass these tests:\n\n{tests}\n[BEGIN]\n"
    pass_at_k: int = 1

    def textify_example(self, example: Example) -> str:
        context = example["text"]
        code = self.get_target(example)
        tests_str = "\n".join(self.get_prompt_tests(example))
        return self.clean_mbpp(
            (
                self.prompt_format.format(context=context, tests=tests_str)
                + f"{code}\n[DONE]"
            )
        )

    def get_prompt(self, example: Example) -> str:
        context = example["text"]
        tests_str = "\n".join(self.get_prompt_tests(example))
        return self.clean_mbpp(
            self.prompt_format.format(context=context, tests=tests_str)
        )

    def get_target(self, example: Example) -> str:
        target = self.clean_mbpp(example["code"])
        return target

    def get_tests(self, example: Example) -> List[str]:
        target = example["test_list"] + example["challenge_test_list"]
        return [self.clean_mbpp(t) for t in target]

    def get_prompt_tests(self, example: Example) -> List[str]:
        target = example["test_list"]
        return [self.clean_mbpp(t) for t in target]

    def clean_mbpp(self, t):
        return t.replace("\r", "")

    def postprocess(self, tokens: List[int]) -> str:
        generation = self.tokenizer.decode(tokens, cut_at_eos=True)
        return generation.split("[DONE]")[0]

    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        ground_truths = [self.clean_mbpp(example["code"])]
        tests = "\n".join(self.get_tests(example))
        sample_metrics = {}
        sample_metrics["em"] = 100 * exact_match_score(
            prediction, ground_truths, normalize_answer
        )
        sample_metrics["f1"] = 100 * f1_score(
            prediction, ground_truths, normalize_answer
        )
        for metric in [m for m in self.metrics if m.startswith("pass_at_")]:
            sample_metrics[metric] = 100 * is_valid_code(
                code=prediction, test=tests, timeout=5
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
                task_id = raw.get("text") or s["prompt"]
            by_task[task_id].append(s)

        per_task_ok = []
        for task_id, samples in by_task.items():
            raw0 = samples[0].get("raw", {})
            test_list = raw0.get("test_list") or samples[0].get("test_list", [])
            chall_list = raw0.get("challenge_test_list") or samples[0].get("challenge_test_list", [])
            tests_str = "\n".join(self.clean_mbpp(t) for t in (test_list + chall_list))

            this_ok = False
            for s in samples:
                gen = s["generation"]
                gen = self.clean_mbpp(gen.split("[DONE]")[0])
                if is_valid_code(code=gen, test=tests_str, timeout=5):
                    this_ok = True
                    break

            per_task_ok.append(1.0 if this_ok else 0.0)

        avg = sum(per_task_ok) / len(per_task_ok) if per_task_ok else 0.0
        return {f"pass_at_{self.pass_at_k}": avg * 100.0}
