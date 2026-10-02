from typing import List, Tuple, Dict
import string
import os
import numpy as np

from xllm.data.dataset_streamer.tokenizer import Tokenizer
from xllm.paper_part1.tasks.base import ChoiceTask, Example

# MMLU_Pro example
# {
#     'question_id': 70,
#     'question': 'Typical advertising regulatory bodies suggest, for example that adverts must not: encourage _________, cause unnecessary ________ or _____, and must not cause _______ offence.',
#     'options': 
#         [
#             'Safe practices, Fear, Jealousy, Trivial',
#             'Unsafe practices, Distress, Joy, Trivial',
#             'Safe practices, Wants, Jealousy, Trivial',
#             'Safe practices, Distress, Fear, Trivial',
#             'Unsafe practices, Wants, Jealousy, Serious',
#             'Safe practices, Distress, Jealousy, Serious',
#             'Safe practices, Wants, Fear, Serious',
#             'Unsafe practices, Wants, Fear, Trivial',
#             'Unsafe practices, Distress, Fear, Serious'
#         ],
#     'answer': 'I',
#     'answer_index': 8,
#     'cot_content': '',
#     'category': 'business',
#     'src': 'ori_mmlu-business_ethics'
# }

class MMLUProTask(ChoiceTask):
    dataset_dirs = ["mmlu_pro"]
    fewshot_mode = "first"
    n_fewshot = 5
    fewshot_file = "prompt.jsonl"   
    eval_file = "test.jsonl"
    sep = "\n\n"
    max_text_len = 4096

    def _letters(self, n: int) -> List[str]:
        return list(string.ascii_uppercase[:n])

    def context(self, example: Example) -> str:

        q = example["question"]
        opts: List[str] = example["options"]
        letters = self._letters(len(opts))
        choice_lines = [f"{l}. {opt}" for l, opt in zip(letters, opts)]
        choice_str = "\n".join(choice_lines)
        return f"{q}\n{choice_str}\nAnswer:"

    def textify_example(self, example: Example) -> str:
        ctx = self.context(example)
        ans = example["answer"]  
        return f"{ctx} {ans}"

    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        ctx = self.context(example)
        opts: List[str] = example["options"]
        letters = self._letters(len(opts))
        return [(f"{ctx} {l}", l) for l in letters]

    def label(self, example: Example) -> int:
        if "answer_index" in example:
            return int(example["answer_index"])
        letters = self._letters(len(example["options"]))
        mapping = {l: i for i, l in enumerate(letters)}
        return example["answer"]

    def get_fewshot(self, example: Example, rng):
        if self.n_fewshot == 0:
            return []
        if not self._all_fewshot_examples:
            self._all_fewshot_examples = self._load_fewshot_file()
        cat = example.get("category")
        same_cat = [ex for ex in self._all_fewshot_examples if ex.get("category") == cat]
        return same_cat[: self.n_fewshot]
    
    def process(self, example: Example, rng: np.random.RandomState):
        fewshot = [self.textify_example(ex) for ex in self.get_fewshot(example, rng)]

        text_completion = self.get_text_completion(example)
        text_completion = [
            (self.sep.join([self.description.format(topic=example["category"])] + fewshot + [t]), c) for t, c in text_completion
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