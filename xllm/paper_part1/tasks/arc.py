from typing import List, Tuple

from xllm.paper_part1.tasks.base import ChoiceTask, Example

# ARC example
#
# {
#    "question": "An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?",
#    "choices": {
#        "A": "Planetary density will decrease.",
#        "B": "Planetary years will become longer.",
#        "C": "Planetary days will become shorter.",
#        "D": "Planetary gravity will become stronger.",
#    },
#    "label": "C",
# }


class ARCTask(ChoiceTask):
    dataset_dirs = ["arc_challenge", "arc_easy"]
    fewshot_mode = "first"
    n_fewshot = 0
    sep = " "
    fewshot_file = "train.jsonl"
    eval_file = "test.jsonl"
    max_text_len = 512

    def textify_example(self, example: Example) -> str:
        context = self.get_context(example)
        answer = self.choices(example)[self.label(example)]
        return f"{context} {answer}"

    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        context = self.get_context(example)
        return [(f"{context} {c}", c) for c in self.choices(example)]

    def label(self, example: Example) -> int:
        mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
        return mapping[example["label"]]

    def choices(self, example: Example) -> List[str]:
        return [example["choices"][c] for c in ["A", "B", "C", "D"]]

    def get_context(self, example: Example) -> str:
        question = example["question"].strip()
        if question.endswith("?"):
            return f"Question: {question}\nAnswer:"
        else:
            return f"{question}"
