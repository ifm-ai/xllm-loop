"""Generation tasks of the Part 2 evaluation: prompts, stopping, and scoring.

GSM8K uses a fixed 8-shot chain-of-thought prompt and a numeric-answer scorer;
DROP uses pre-rendered 3-shot prompts and the official AllenAI EM/F1; MBPP+
uses pre-rendered 3-shot prompts, eight samples at temperature 0.8, and
EvalPlus 0.3.1 base and plus tests.
"""
from __future__ import annotations

import hashlib

import numpy as np

from xllm.paper_part2 import drop_eval
from xllm.paper_part2._answer_helpers import exact_match_score, extract_answer, f1_score, math_accuracy_score


STOP_STRINGS = {
    "gsm8k": ["Q: ", "A:"],
    "drop": ["\n", "Question:", "Answer:"],
    "mbpp": ["\n\"\"\"", "\nassert", "\nif __name__", "\ndef main(", "\nprint(", "<|endoftext|>", "<|endofmask|>", "</s>"],
}
SAMPLING_SEED_OFFSET = 20260911  # fixed by the protocol, like the DROP and MBPP+ item selection


def clean(text: str, task: str) -> tuple[str, bool]:
    """Cut a decoded generation at the first stop string; report whether one occurred."""
    end = min((text.find(s) for s in STOP_STRINGS[task] if s in text), default=len(text))
    return text[:end], end < len(text)


def item_identity(item_id: str) -> int:
    """Identity that seeds a generation item's recurrent state and sampling: 7 bytes of the id's SHA-256,
    so it fits in an int64 tensor."""
    return int.from_bytes(hashlib.sha256(item_id.encode()).digest()[:7], "big")


def sampling_uniforms(identity: int, sample: int, max_new_tokens: int) -> np.ndarray:
    """Pre-drawn uniforms for inverse-CDF sampling of one sample's tokens, from a generator seeded per item and
    sample (`identity * 31 + sample`, offset by the protocol's seed)."""
    return np.random.default_rng(identity * 31 + sample + SAMPLING_SEED_OFFSET).random(max_new_tokens)


# Chain-of-thought exemplars of Wei et al. (2022, CC BY 4.0), modified;
# see THIRD_PARTY_NOTICES.md.
GSM8K_FEWSHOT = (
    {
        "question": 'There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?',
        "answer": "Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.\n#### 6",
    },
    {
        "question": 'If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?',
        "answer": "Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5",
    },
    {
        "question": 'Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?',
        "answer": "Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39",
    },
    {
        "question": 'Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?',
        "answer": "Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.\n#### 8",
    },
    {
        "question": 'Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?',
        "answer": "Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.\n#### 9",
    },
    {
        "question": 'There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?',
        "answer": "Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.\n#### 29",
    },
    {
        "question": 'Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?',
        "answer": "Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.\n#### 33",
    },
    {
        "question": 'Olivia has $23. She bought five bagels for $3 each. How much money does she have left?',
        "answer": "Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.\n#### 8",
    },
)

GSM8K_SEP = "\n\n"


def gsm8k_prompt(example: dict) -> str:
    return f"Q: {example['question']}\nA:"


def gsm8k_target(example: dict) -> str:
    split = example["answer"].rsplit("\n#### ", maxsplit=1)
    if len(split) != 2:
        return ""
    solution, result = split
    return f"{solution} The answer is {result}."


def gsm8k_full_prompt(example: dict) -> str:
    """The 8-shot prompt for one test question."""
    shots = [f"{gsm8k_prompt(x)} {gsm8k_target(x)}" for x in GSM8K_FEWSHOT]
    return GSM8K_SEP.join(shots) + GSM8K_SEP + gsm8k_prompt(example)


def score_gsm8k(generation: str, example: dict) -> dict[str, float]:
    ground_truths = [gsm8k_target(example)]
    return {
        "acc": 100 * math_accuracy_score(generation, ground_truths, extract_answer),
        "em": 100 * exact_match_score(generation, ground_truths, extract_answer),
        "f1": 100 * f1_score(generation, ground_truths, extract_answer),
    }


def score_drop(generation: str, item: dict) -> dict[str, float]:
    """Best EM/F1 over the answer and validated answers; ';' separates predicted spans."""
    prediction = generation.strip()
    if ";" in prediction:
        prediction = [span.strip() for span in prediction.split(";") if span.strip()]
    em = f1 = 0.0
    for answer in [item["qa"]["answer"]] + item["qa"].get("validated_answers", []):
        gold, _ = drop_eval.answer_json_to_strings(answer)
        if gold and gold[0].strip():
            exact, overlap = drop_eval.get_metrics(prediction, gold)
            em, f1 = max(em, exact), max(f1, overlap)
    return {"em": 100 * em, "f1": 100 * f1}


def mbpp_pass_rates(eval_results: dict, samples: int = 8) -> dict[str, float]:
    """pass@1 and pass@8 from EvalPlus results: a sample passes both base and plus tests."""
    counts = [sum(x["base_status"] == "pass" and x["plus_status"] == "pass" for x in results)
              for results in eval_results.values()]
    return {"pass1": 100 * sum(counts) / (len(counts) * samples),
            "pass8": 100 * sum(count > 0 for count in counts) / len(counts)}
