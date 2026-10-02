from typing import List, Dict
import json
import random
import re
from logging import getLogger

from xllm.paper_part1.tasks.base import GenerationTask, Example
from xllm.paper_part1.tasks.utils import exact_match_score, f1_score, normalize_answer
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path
from xllm.distributed.slurm import get_global_rank


# MATH example
# {
#     "problem": "The perimeter of a rectangle is 24 inches. What is the number of square inches in the maximum possible area for this rectangle?",
#     "level": "Level 3",
#     "type": "Algebra",
#     "solution": "Let one pair of parallel sides have length $x$ and the other pair of parallel sides have length $12-x$. This means that the perimeter of the rectangle is $x+x+12-x+12-x=24$ as the problem states. The area of this rectangle is $12x-x^2$. Completing the square results in $-(x-6)^2+36\\le 36$ since $(x-6)^2\\ge 0$, so the maximum area of $\\boxed{36}$ is obtained when the rectangle is a square of side length 6 inches."
# }
random.seed(get_global_rank())
logger = getLogger()

# from minerva (Lewkowycz et al., 2022, appendix D; CC BY 4.0);
# see THIRD_PARTY_NOTICES.md.
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def _extract_result_from_boxed(answer: str) -> str:
    box_start = "\\boxed"
    # format is `\\boxed <value>$` or `\\boxed{<value>}`, with potential white spaces framing `<value>`
    start = answer.rfind(box_start)
    if start < 0:
        return ""
    answer = answer[start + len(box_start) :].strip()
    ends_with_curly = answer.startswith("{")
    i = 0
    open_braces = 0
    while i < len(answer):
        if answer[i] == "{":
            open_braces += 1
        elif answer[i] == "}":
            open_braces -= 1
        if open_braces == 0:
            if ends_with_curly:
                answer = answer[: i + 1].strip()
                break
            elif answer[i] == "$":
                answer = answer[:i].strip()
                break
        i += 1
    else:
        return ""
    # remove extra curly braces
    while True:
        if answer.startswith("{") and answer.endswith("}"):
            answer = answer[1:-1].strip()
        else:
            break
    return answer


# _fix_fracs through _normalise_result: adapted from hendrycks/math (MIT);
# see THIRD_PARTY_NOTICES.md.
def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) == 0:
                return string
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        ia = int(a)
        ib = int(b)
        assert string == "{}/{}".format(ia, ib)
        new_string = "\\frac{" + str(ia) + "}{" + str(ib) + "}"
        return new_string
    except (ValueError, AssertionError):
        return string


def _remove_right_units(string: str):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    try:
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string
    except AssertionError:
        return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) == 0:
            return string
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _normalise_result(string: str) -> str:
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")  # the same pattern again, as in the original normalizer

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    string = string.split("=")[-1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# from the Minerva paper; _normalise_result is adapted from hendrycks/math
def normalize_final_answer(final_answer: str) -> str:
    """Extract and normalize a final answer to a quantitative reasoning question."""
    match = re.findall(
        r".*The final answer is (?P<X>.*). I hope it is correct.*", final_answer
    )
    extraction: str
    if len(match) > 0:
        extraction = match[0]
    else:
        extraction = _extract_result_from_boxed(final_answer)

    if len(extraction) == 0:
        return final_answer
    else:
        final_answer = extraction
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return _normalise_result(final_answer)


class MATHTask(GenerationTask):
    dataset_dirs = ["math"]
    metrics = ["em", "f1", "nll"]
    fewshot_mode = "index"
    # Minerva set-up, fewshot_mode = "random" for random n_fewshot fewshots
    fewshot_index: List[int] = [6374, 3321, 6476, 6013]
    n_fewshot = 4
    sep = "\n\n"
    fewshot_file = "train.jsonl"
    eval_file = "test.jsonl"
    question_prefix = "Problem:\n"
    target_prefix = "Solution:"
    prompt_format = "{question_prefix}{question}\n\n{target_prefix}"
    max_text_len: int = 4096
    max_gen_len: int = 512
    answer_format: str = "{solution}\nFinal Answer: The final answer is ${result}$. I hope it is correct."
    majority_voting_k: int = 1

    def process(self, example: Example, rng: np.random.RandomState):
        fewshot = [self.textify_example(ex) for ex in self.get_fewshot(example, rng)]
        target = self.get_target(example)
        prompt = self.get_prompt(example)
        prompt = self.sep.join(fewshot + [prompt])

        text = prompt + "\n" + target

        input_tokens, target_tokens = self.get_input_output_tokens(text, target)
        return {
            "raw": example,
            "prompt": prompt,
            "text_x": [input_tokens],
            "text_y": [target_tokens],
        }

    def textify_example(self, example: Example) -> str:
        return f"{self.get_prompt(example)}\n{self.get_target(example)}"

    def get_prompt(self, example: Example) -> str:
        return self.prompt_format.format(
            question_prefix=self.question_prefix,
            question=example["problem"],
            target_prefix=self.target_prefix,
        )

    def get_target(self, example: Example) -> str:
        solution = example["solution"]
        result = _extract_result_from_boxed(solution)
        return self.answer_format.format(solution=solution, result=result)

    def postprocess(self, tokens: List[int]) -> str:
        generation = self.tokenizer.decode(tokens, cut_at_eos=True)
        generation = generation.split(self.question_prefix)[0].split(
            self.target_prefix
        )[0]
        return generation

    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        prediction = normalize_final_answer(prediction)
        ground_truths = [normalize_final_answer(example["solution"])]
        sample_metrics = {
            # "pred_normalization": prediction,
            # "gold_normalization": ground_truths,
            "em": 100 * exact_match_score(prediction, ground_truths, lambda x: x),
            "f1": 100 * f1_score(prediction, ground_truths, lambda x: x),
        }
        return sample_metrics

    def majority_voting(self, batch_paths):
        if self.majority_voting_k == 1:
            return {}
        # sample = {"input": prompt, "generation": generation, "metrics": metrics}
        # sample["metrics"].keys() = ['pred_normalization', 'gold_normalization', 'em', 'f1', 'nll']
        all_samples = []
        for batch_path in batch_paths:
            print(f"Loading {batch_path}")
            path = Path(batch_path)
            with path.open("r") as fp:
                all_samples += [json.loads(line.strip()) for line in fp.readlines()]
        all_preds: Dict[str, List[str]] = defaultdict(list)
        all_answers: Dict[str, str] = {}

        # regroup preds and answer by question
        print("Regroup preds and answerd by question")
        n_samples = len(all_samples)
        for i, sample in enumerate(all_samples):
            if i % 1000 == 0:
                print(f"Sample # {i} / {n_samples}")
            question = (
                sample["prompt"]
                .split(self.question_prefix)[-1]
                .split(self.target_prefix)[0]
            )
            if question not in all_answers:
                all_answers[question] = [
                    normalize_final_answer(answer) for answer in sample["answers"]
                ]
            all_preds[question].append(normalize_final_answer(sample["generation"]))
        nb_passes = len(all_preds[question])
        if not all(len(preds) == nb_passes for preds in all_preds.values()):
            print("Warning !")
            print(nb_passes)
            print([len(preds) for preds in all_preds.values()])
        print(f"Nb passes = {nb_passes}, Nb samples = {len(all_preds)}.")

        # compute majority voting
        print("Compute majority voting")

        # compute majority voting
        accs = []
        f1s = []
        for question, preds in all_preds.items():
            fst_2_votes = Counter(preds).most_common(2)
            consensus = fst_2_votes[0][0]
            if consensus == "" and len(fst_2_votes) > 1:
                consensus = fst_2_votes[1][0]
            acc = 100 * exact_match_score(consensus, all_answers[question], lambda x: x)
            f1 = 100 * f1_score(consensus, all_answers[question], lambda x: x)

            accs.append(acc)
            f1s.append(f1)

        scores = {
            f"maj1@{nb_passes}_acc": float(np.mean(accs)),
            f"maj1@{nb_passes}_f1": float(np.mean(f1s)),
        }
        print(f"Majority voting score: {scores}")
        return scores
