from typing import List, Dict

from xllm.paper_part1.tasks.base import GenerationTask, Example
from xllm.paper_part1.tasks.utils import exact_match_score, f1_score, normalize_answer

# TriviaQA example
#
# {
#     "question": "Who was President when the first Peanuts cartoon was published?",
#     "answers": [
#         "Presidency of Harry S. Truman",
#         "Hary truman",
#         "Harry Shipp Truman",
#         "Harry Truman's",
#         "Harry S. Truman",
#         "Harry S.Truman",
#         "Harry S Truman",
#         "H. S. Truman",
#         "President Harry Truman",
#         "Truman administration",
#         "Presidency of Harry Truman",
#         "Mr. Citizen",
#         "HST (president)",
#         "H.S. Truman",
#         "Mary Jane Truman",
#         "Harry Shippe Truman",
#         "S truman",
#         "Harry Truman",
#         "President Truman",
#         "33rd President of the United States",
#         "Truman Administration",
#         "Harry Solomon Truman",
#         "Harold Truman",
#         "Harry truman",
#         "H. Truman",
#     ],
#     "target": "Harry Truman",
# }


class TQATask(GenerationTask):
    dataset_dirs = ["tqa"]
    metrics = ["em", "f1", "nll"]
    fewshot_mode = "first"
    n_fewshot = 5
    sep = "\n"
    fewshot_file = "train.jsonl"
    eval_file = "test.jsonl"
    question_prefix = "Question:"
    target_prefix = "Answer:"
    prompt_format = "{question_prefix} {question}\n{target_prefix}"
    max_text_len = 512
    max_gen_len = 24

    def textify_example(self, example: Example) -> str:
        return f"{self.get_prompt(example)} {self.get_target(example)}"

    def get_prompt(self, example: Example) -> str:
        return self.prompt_format.format(
            question_prefix=self.question_prefix,
            question=example["question"],
            target_prefix=self.target_prefix,
        )

    def get_target(self, example: Example) -> str:
        return example["target"]

    def postprocess(self, tokens: List[int]) -> str:
        generation = self.tokenizer.decode(tokens, cut_at_eos=True)
        generation = generation.split(self.question_prefix)[0].split(
            self.target_prefix
        )[0]
        return generation

    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        ground_truths = example["answers"]
        sample_metrics = {
            "em": 100 * exact_match_score(prediction, ground_truths, normalize_answer),
            "f1": 100 * f1_score(prediction, ground_truths, normalize_answer),
        }
        return sample_metrics
