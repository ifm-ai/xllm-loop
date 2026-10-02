"""Choice tasks of the Part 2 evaluation: prompts, encoding, and reduction.

Prompts and encoding follow lm-evaluation-harness v0.4.2 (commit 4600d6bf, MIT;
see THIRD_PARTY_NOTICES.md) with no BOS, as the paper's Table 2-4 evaluation
did. Each option's target tokens are scored by the model; `reduce_task` turns
the per-option records into accuracy.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterator, Sequence

from xllm.paper_part2.eval_protocol import (
    CHOICE_TASKS,
    FULL_COUNTS,
    MIN_BATCH,
    PRIMARY_METRIC,
    SEQUENCE_LENGTH,
    check_protocol_file,
)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def stable_id(task: str, index: int) -> int:
    """An example's identity, which seeds its recurrent state: a 31-bit hash of the task in the high 32 bits and
    the row index in the low 32, so every example of every task has its own positive int64."""
    namespace = int.from_bytes(hashlib.sha256(task.encode()).digest()[:4], "big") & 0x7fffffff
    return (namespace << 32) | index


def _hellaswag(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def build_example(task: str, row: dict, index: int) -> dict:
    """Prepare canonical HF rows, without tokenization or model execution."""
    _require(task in CHOICE_TASKS, f"Unknown task {task}")
    if task == "lambada_openai":
        words = row["text"].split(" ")
        _require(len(words) > 1 and words[-1], "LAMBADA needs a nonempty final word")
        contexts, choices, gold = [" ".join(words[:-1])], [words[-1]], 0
    elif task == "hellaswag":
        contexts = [_hellaswag(row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize())]
        choices, gold = [_hellaswag(x) for x in row["endings"]], int(row["label"])
    elif task == "sciq":
        contexts = [row["support"].lstrip() + "\nQuestion: " + row["question"] + "\nAnswer:"]
        choices = [row["distractor1"], row["distractor2"], row["distractor3"], row["correct_answer"]]
        gold = 3
    elif task == "piqa":
        contexts = ["Question: " + row["goal"] + "\nAnswer:"]
        choices, gold = [row["sol1"], row["sol2"]], int(row["label"])
    else:  # arc_easy, arc_challenge, openbookqa
        contexts = [row["question_stem"] if task == "openbookqa" else "Question: " + row["question"] + "\nAnswer:"]
        choices = list(row["choices"]["text"])
        labels = list(row["choices"]["label"])
        _require(len(labels) == len(choices) and len(set(labels)) == len(labels), "Invalid choice labels")
        gold = labels.index(row["answerKey"].lstrip() if task == "openbookqa" else row["answerKey"])
    _require(choices and all(isinstance(c, str) and len(c) > 0 for c in choices), "Empty choice")
    _require(0 <= gold < len(choices), "Gold choice is out of range")
    if len(contexts) == 1:
        contexts *= len(choices)
    return dict(task=task, index=index, example_id=stable_id(task, index),
                native_id=row.get("id", row.get("qID", row.get("ind", index))), raw=row, gold=gold,
                options=[dict(context=c, continuation=" " + a, norm_chars=len(a))
                         for c, a in zip(contexts, choices)])


def encode_pair(tokenizer, context: str, continuation: str, max_length: int = SEQUENCE_LENGTH) -> dict:
    """Reproduce lm-evaluation-harness's _encode_pair and no-BOS HFLM input assembly.

    The denominator is handled separately: trailing prompt whitespace moves into
    the encoded continuation, but does not change len(doc_to_choice).
    """
    spaces = len(context) - len(context.rstrip())
    if spaces:
        continuation = context[-spaces:] + continuation
        context = context[:-spaces]
    whole = list(tokenizer.encode(context + continuation, bos=False, eos=False))
    prefix = list(tokenizer.encode(context, bos=False, eos=False))
    target = whole[len(prefix):]
    _require(prefix and target, "Empty encoded context or continuation")
    _require(all(isinstance(t, int) and t >= 0 for t in prefix + target), "Invalid token IDs")
    # lm-evaluation-harness concatenates the separately encoded context and sliced target.
    tokens = prefix + target
    _require(len(tokens) - 1 <= max_length, f"Sequence exceeds {max_length}; truncation is forbidden")
    return dict(tokens=tokens[:-1], targets=target, target_start=len(prefix) - 1,
                context_tokens=len(prefix), target_tokens=len(target),
                encoded_context=context, encoded_continuation=continuation,
                encoding_sha256=hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest())


def load_choice_examples(task: str, path: str | Path) -> list[dict]:
    """Build a choice task's examples from its hash-pinned JSONL rows."""
    rows = [json.loads(line) for line in check_protocol_file(path, task).read_text().splitlines()]
    if len(rows) != FULL_COUNTS[task]:
        raise ValueError(f"{task} must have {FULL_COUNTS[task]} rows, got {len(rows)}")
    return [build_example(task, row, index) for index, row in enumerate(rows)]


def prepare_requests(examples: list[dict], tokenizer, max_length: int = SEQUENCE_LENGTH) -> list[dict]:
    requests = []
    for example in examples:
        for option_index, option in enumerate(example["options"]):
            requests.append(dict(index=example["index"], example_id=example["example_id"],
                                 option_index=option_index, **option,
                                 **encode_pair(tokenizer, option["context"], option["continuation"], max_length)))
    return requests


def reduce_token_scores(target_ids: Sequence[int], greedy_ids: Sequence[int],
                        logprobs: Sequence[float]) -> tuple[float, bool]:
    _require(len(target_ids) == len(greedy_ids) == len(logprobs) > 0, "Mismatched target subtoken scores")
    _require(all(math.isfinite(x) for x in logprobs), "Nonfinite target log probability")
    return math.fsum(logprobs), list(target_ids) == list(greedy_ids)


def request_groups(requests: list[dict]) -> Iterator[tuple[int, list[dict], list[dict]]]:
    """Yield (context length, real requests, physical batch of eight) as the paper's evaluation batched them."""
    by_length = {}
    for request in requests:
        by_length.setdefault(request["context_tokens"], []).append(request)
    for length, items in sorted(by_length.items()):
        items.sort(key=lambda r: (r["target_tokens"], r["index"], r["option_index"]))
        for offset in range(0, len(items), MIN_BATCH):
            real = items[offset:offset + MIN_BATCH]
            yield length, real, real + [real[-1]] * (MIN_BATCH - len(real))


def reduce_task(task: str, examples: list[dict], option_records: list[dict]) -> tuple[dict, list[dict]]:
    """Reduce per-option records; LAMBADA PPL is per word, not per subtoken."""
    lookup = {(r["index"], r["option_index"]): r for r in option_records}
    _require(len(lookup) == len(option_records), "Duplicate option result")
    _require(len(lookup) == sum(len(e["options"]) for e in examples), "Incomplete option results")
    predictions = []
    for example in examples:
        options = [lookup[(example["index"], j)] for j in range(len(example["options"]))]
        _require(all(r["example_id"] == example["example_id"] for r in options), "State identity drift")
        _require(all(math.isfinite(r["loglikelihood"]) for r in options), "Nonfinite likelihood")
        raw = max(range(len(options)), key=lambda j: options[j]["loglikelihood"])
        norm = max(range(len(options)), key=lambda j: options[j]["loglikelihood_per_char"])
        correct = options[0]["is_greedy"] if task == "lambada_openai" else raw == example["gold"]
        predictions.append(dict(**{k: example[k] for k in ("task", "index", "example_id", "native_id", "raw", "gold")},
            options=options, predicted_raw=raw, predicted_norm=norm,
            correct_raw=bool(correct), correct_norm=bool(correct if task == "lambada_openai" else norm == example["gold"])))
    _require(predictions, "Empty evaluation")
    n = len(predictions)
    result = dict(ok=True, task=task, examples=n, acc=sum(r["correct_raw"] for r in predictions) / n,
                  acc_norm=sum(r["correct_norm"] for r in predictions) / n,
                  primary_metric=PRIMARY_METRIC[task], accuracy_unit="fraction", ties="first option, matching argmax")
    result["primary_score"] = result[result["primary_metric"]]
    if task == "lambada_openai":
        nll = -math.fsum(r["options"][0]["loglikelihood"] for r in predictions)
        targets = sum(r["options"][0]["target_tokens"] for r in predictions)
        result.update(perplexity=math.exp(nll / n), perplexity_unit="last-word (per-example summed target NLL)",
                      target_nll=nll, target_tokens=targets, target_token_ce=nll / targets)
    return result, predictions

