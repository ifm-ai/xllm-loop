"""Frozen evaluation protocol of the Part 2 paper tables (Tables 2-4).

Every model is evaluated at R=5 with terminal KV sharing. Wiki PPL prefills
the first 4,096 tokens of each 8,192-token sequence and scores the rest; the
seven choice tasks and three generation tasks prefill the prompt and decode one
token at a time. The recurrent state is initialized deterministically from each
example's identity with seed 42.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

EVAL_DEPTH = 5
STATE_SEED = 42
SEQUENCE_LENGTH = 8192  # tokens per sequence, in training and evaluation
PPL_PREFIX_TOKENS = 4096
MIN_BATCH = 8  # every evaluation forward runs on at least eight rows, as in the paper

CHOICE_TASKS = ("lambada_openai", "hellaswag", "piqa", "arc_easy", "arc_challenge", "openbookqa", "sciq")
FULL_COUNTS = dict(lambada_openai=5153, hellaswag=10042, piqa=1838, arc_easy=2376,
                   arc_challenge=1172, openbookqa=500, sciq=1000)
PRIMARY_METRIC = {task: "acc" for task in CHOICE_TASKS} | dict(hellaswag="acc_norm", arc_challenge="acc_norm")


@dataclass(frozen=True)
class GenerationSpec:
    """A generation task: its prompts (GSM8K 8-shot, DROP and MBPP+ 3-shot), decoding and table metric."""

    examples: int
    max_new_tokens: int
    max_prompt_tokens: int
    temperature: float
    samples: int
    metric: str


GENERATION = {
    "gsm8k": GenerationSpec(1319, 512, 2048, 0.0, 1, "acc"),
    "drop": GenerationSpec(500, 64, 8128, 0.0, 1, "f1"),
    "mbpp": GenerationSpec(100, 512, 2048, 0.8, 8, "pass1"),
}

# sha256 of every evaluation input. Choice tasks are the lm-evaluation-harness-formatted HF
# rows; DROP and MBPP+ prompts are pre-rendered.
DATA_SHA256 = {
    "lambada_openai": "05ae9755847286957976eca1a513750eeb542019e97af8837baa488483398d78",
    "hellaswag": "2bc407195a57477b62ad939e34237e467aef1bd0116a5459c58900517845ebd7",
    "piqa": "256cd78377a0090efcbc05662c15821c9b60f98a708bf48115d0d9af80c67313",
    "arc_easy": "00feb782af96f8b74a16f63b6cb72246b3ee5c84b74342342302564afad8b846",
    "arc_challenge": "a00c3127fa2437025957049bb97760ce0e7e974bcf5a918b7621f36ab9c3fed8",
    "openbookqa": "2dcf823dba46be337b5b4b35aa199ebf2cba55332286cceb5c6b90c4e225a1c7",
    "sciq": "cd0a2c8be9b2eafc95bd1bab6d2ae553826c1f3b7ddbe5e415e130c30abddddd",
    "gsm8k": "6e1f996338f70458bbb54a5d7b33f9d5bd9c6711716dedd4f9abd8167ae16976",
    "drop": "c7c68cc11e0bebf045971a1d23882ee08f3034f021814b43baeeffa71fa9b402",
    "mbpp": "c5d054492ff927c31e1bf1de7b7eb709b99e5d8aaa383a719928a26a2b38e689",
    "mbpp_plus": "9bee527323281a2b2dd32fc66ae29a1ec32ec9b405bfba3e0b6e453ed205e79f",
    "wikitext103_test": "35c67ef7a6d51fd5a5fc7715a9264b8d3cdc21fe59c4b11bb0ff52202eb1ed87",
}


def check_protocol_file(path: str | Path, key: str) -> Path:
    """Return `path` if the file is the protocol's `key` input, by its SHA-256; raise otherwise."""
    path = Path(path)
    with open(path, "rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != DATA_SHA256[key]:
            raise ValueError(f"{key} input does not match the protocol: {path}")
    return path
