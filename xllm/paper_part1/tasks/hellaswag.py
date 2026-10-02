import re
from typing import List, Tuple

from xllm.paper_part1.tasks.base import ChoiceTask, Example


# HellaSwag example
# {
#     "ind": 4,
#     "activity_label": "Removing ice from car",
#     "ctx_a": "Then, the man writes over the snow covering the window of a car, "
#              "and a woman wearing winter clothes smiles.",
#     "ctx_b": "then",
#     "ctx": "Then, the man writes over the snow covering the window of a car, "
#            "and a woman wearing winter clothes smiles. then",
#     "split": "train",
#     "split_type": "indomain",
#     "label": 3,
#     "endings": [
#         ", the man adds wax to the windshield and cuts it.",
#         ", a person board a ski lift, while [...] snow as the we girls sled.",
#         ", the man puts on a christmas coat, knitted with netting.",
#         ", the man continues removing the snow on his car.",
#     ],
#     "source_id": "activitynet~v_-1IBHYS3L-Y",
# }


class HellaSwagTask(ChoiceTask):
    dataset_dirs = ["hellaswag"]
    fewshot_mode = "first"
    n_fewshot = 0
    sep = " "
    fewshot_file = "train.jsonl"
    eval_file = "val.jsonl"
    max_text_len = 256

    def _context(self, example: Example) -> str:
        """Build the query with the same context normalization as lm-eval."""
        ctx = example["ctx_a"] + " " + example["ctx_b"].capitalize()
        return self.preprocess(f'{example["activity_label"]}: {ctx}')

    def get_input_output_tokens(
        self,
        text: str,
        completion: str,
    ) -> Tuple[List[int], List[int]]:
        """Encode a causal context/continuation pair like lm-eval."""
        if getattr(self, "add_template", False):
            return super().get_input_output_tokens(text, completion)
        if not completion or not text.endswith(completion):
            raise ValueError("Completion must be a non-empty suffix of the scored text")

        context = text[: -len(completion)]
        trailing_whitespace = len(context) - len(context.rstrip())
        if trailing_whitespace:
            completion = context[-trailing_whitespace:] + completion
            context = context[:-trailing_whitespace]

        x = self.tokenizer.encode(context + completion, bos=True, eos=False)
        context_tokens = self.tokenizer.encode(context, bos=True, eos=False)
        y = [
            -100 if k < len(context_tokens) else token
            for k, token in enumerate(x)
        ]
        return x[:-1], y[1:]

    def textify_example(self, example: Example) -> str:
        context = self._context(example)
        answer = self.choices(example)[self.label(example)]
        return f"{context} {answer}"

    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        context = self._context(example)
        return [(f"{context} {e}", e) for e in self.choices(example)]

    def label(self, example: Example) -> int:
        return int(example["label"])

    def choices(self, example: Example) -> List[str]:
        return [self.preprocess(e) for e in example["endings"]]

    # From EleutherAI lm-evaluation-harness (MIT); see THIRD_PARTY_NOTICES.md.
    def preprocess(self, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text
