from typing import List, Dict, Tuple
import re
import numpy as np
import string

from xllm.data.dataset_streamer.tokenizer import Tokenizer
from xllm.paper_part1.tasks.base import GenerationTask, Example
from xllm.paper_part1.tasks.utils import exact_match_score


LEGACY_BBH_COT_SCORING_RULE = "xllm_last_normalized_token_v1"
PAPER_BBH_COT_SCORING_RULE = (
    "bbh_cot_first_lowercase_nongreedy_line_answer_v1"
)
_PAPER_ANSWER_PATTERN = re.compile(
    r"(?<=the answer is )(.*?)(?=\.(?:\s|$)|\s*$)",
    re.MULTILINE,
)

# BBH example
# {
#     "input": "not ( True ) and ( True ) is",
#     "target": "False",
#     "category": "boolean_expressions",
#     "explaination": "",
#     "question_id": 0,
#     "description":
#     "Evaluate the result of a random Boolean expression.\n\n"
# }

def eval_normalizer(s: str) -> str:
    if not s:
        return ""
    text = s.strip()

    pattern = re.compile(
        r"(?:[Tt]he answer is:?\s*)([A-Za-z0-9\-\_]+)[\.\!\s]*"
    )
    m = pattern.search(text)
    if m:
        return m.group(1).strip(" .")

    words = re.findall(r"[A-Za-z0-9\-\_]+", text)
    if not words:
        return ""
    return words[-1].strip(".")


class BBHCoTTask(GenerationTask):
    dataset_dirs = ["bbh_cot"]
    fewshot_mode = "first"
    n_fewshot = 3
    metrics = ["em", "nll"]
    sep = "\n\n"
    eval_file = "test.jsonl"
    fewshot_file = "prompt.jsonl"

    question_prefix = "Q:"
    target_prefix = "A:"

    max_text_len = 4096
    max_gen_len = 256
    scoring_rule = LEGACY_BBH_COT_SCORING_RULE

    def set_scoring_rule(self, scoring_rule: str) -> None:
        if scoring_rule not in {
            LEGACY_BBH_COT_SCORING_RULE,
            PAPER_BBH_COT_SCORING_RULE,
        }:
            raise ValueError(f"Unsupported BBH CoT scoring rule: {scoring_rule}")
        self.scoring_rule = scoring_rule

    def get_fewshot(self, example: Example, rng: np.random.RandomState) -> List[Example]:
        if not self._all_fewshot_examples:
            self._all_fewshot_examples = self._load_fewshot_file()

        if self.n_fewshot == 0:
            return []

        cat = example.get("category")
        if cat is not None:
            same_cat = [
                ex for ex in self._all_fewshot_examples
                if ex.get("category") == cat
            ]
        else:
            same_cat = self._all_fewshot_examples

        n = self.n_fewshot

        if self.fewshot_mode == "first":
            fewshot_examples = same_cat[:n]
        elif self.fewshot_mode == "index":
            fewshot_examples = [same_cat[k] for k in self.fewshot_index]
        elif self.fewshot_mode == "random":
            idxs = rng.choice(range(len(same_cat)), n, replace=False)
            fewshot_examples = [same_cat[i] for i in idxs]
        else:
            raise ValueError(f"Fewshot strategy {self.fewshot_mode} not supported")

        fewshot_examples = [fs for fs in fewshot_examples if fs != example]
        return fewshot_examples

    def textify_example(self, example: Example) -> str:
        return f"{self.get_prompt(example)} {self.get_target(example)}"

    def get_prompt(self, example: Example) -> str:
        desc = example.get("description") or ""
        inp = example["input"]
        parts = []
        if desc:
            parts.append(desc.strip())
        parts.append(f"{self.question_prefix}\n{inp}")
        parts.append(self.target_prefix)
        return "\n".join(parts)

    def process(self, example: Example, rng: np.random.RandomState):
        fewshot = [self.textify_example(ex) for ex in self.get_fewshot(example, rng)]

        prompt = self.get_prompt(example)
        prompt = self.sep.join(fewshot + [prompt])

        target = self.get_target(example)
        text = prompt + " " + target

        input_tokens, target_tokens = self.get_input_output_tokens(text, target)
        return {
            "raw": example,
            "text_x": [input_tokens],
            "text_y": [target_tokens],
            "prompt": prompt,
        }

    def get_target(self, example: Example) -> str:
        explanation = example.get("explaination") or example.get("explanation") or ""
        final_answer = example["target"]
        if explanation:
            return f"{explanation.strip()}\nThe answer is {final_answer}."
        else:
            return f"The answer is {final_answer}."

    def postprocess(self, tokens: List[int]) -> str:
        generation = self.tokenizer.decode(tokens, cut_at_eos=True)
        generation = generation.split("Q:")[0]
        return generation

    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        gold = example["target"]
        if self.scoring_rule == LEGACY_BBH_COT_SCORING_RULE:
            em = exact_match_score(prediction, [gold], eval_normalizer)
        elif self.scoring_rule == PAPER_BBH_COT_SCORING_RULE:
            matches = _PAPER_ANSWER_PATTERN.findall(prediction)
            answer = matches[0].strip() if matches else "[invalid]"
            em = float(answer == gold)
        else:
            raise ValueError(f"Unsupported BBH CoT scoring rule: {self.scoring_rule}")
        return {
            "em": 100 * em,
        }
