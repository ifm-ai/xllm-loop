"""Evaluation sequences of the Part 2 paper: WikiText-103 test, in batches of eight.

`wiki_batches` yields `(x, y, mask, ids, real)`: 8,192-token inputs and targets, the target
mask, each row's identity for the recurrent state, and the number of real rows
(a short final batch is padded with copies of its first row).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from xllm.paper_part2.eval_protocol import MIN_BATCH as BATCH, SEQUENCE_LENGTH, check_protocol_file


WIKI_SEQUENCES = 38  # the protocol's WikiText-103 test file packs into 38 sequences


def _wiki_sequences(tokenizer, path: Path) -> Iterator[tuple]:
    """Documents encoded with BOS and EOS, concatenated and cut into 8,192-token rows as in the paper's
    evaluation: a buffer of one batch, then a final partial batch and a padded tail."""
    seq_len, buffer_tokens = SEQUENCE_LENGTH, SEQUENCE_LENGTH * BATCH
    tokens: list[int] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tokens.extend(tokenizer.encode(json.loads(line)["text"], bos=True, eos=True))
            while len(tokens) > buffer_tokens:
                x = np.array(tokens[:buffer_tokens]).reshape(BATCH, -1)
                y = np.array(tokens[1:buffer_tokens + 1]).reshape(BATCH, -1)
                tokens = tokens[buffer_tokens:]
                yield x, y, None
    if not tokens:
        return
    num_seqs = (len(tokens) - 1) // seq_len
    num_batches = num_seqs // BATCH
    batched = np.array(tokens[:num_seqs * seq_len + 1])
    tail = tokens[num_seqs * seq_len:]
    for i in range(num_batches):
        start, end = i * BATCH * seq_len, (i + 1) * BATCH * seq_len
        yield batched[start:end].reshape(BATCH, seq_len), batched[start + 1:end + 1].reshape(BATCH, seq_len), None
    if num_batches * BATCH < num_seqs:
        start = num_batches * BATCH * seq_len
        yield batched[start:-1].reshape(-1, seq_len), batched[start + 1:].reshape(-1, seq_len), None
    if len(tail) > 1:
        pad = seq_len - len(tail) + 1
        x = np.array(tail[:-1] + [tokenizer.pad_id] * pad)
        y = np.array(tail[1:] + [tokenizer.pad_id] * pad)
        mask = y != tokenizer.pad_id
        x[~mask] = tokenizer.eos_id
        y[~mask] = tokenizer.eos_id
        yield x.reshape(1, seq_len), y.reshape(1, seq_len), mask.reshape(1, seq_len)


def wiki_batches(tokenizer, path: str | Path) -> Iterator[tuple]:
    """WikiText-103 test rows in batches of eight, identified by their row number."""
    path = check_protocol_file(path, "wikitext103_test")
    offset = 0
    for x, y, mask in _wiki_sequences(tokenizer, path):
        mask = np.ones_like(y, bool) if mask is None else mask
        real = len(x)
        if real < BATCH:
            x, y, mask = (np.concatenate([v, np.repeat(v[:1], BATCH - real, 0)]) for v in (x, y, mask))
        yield x, y, mask, list(range(offset, offset + real)) + [offset] * (BATCH - real), real
        offset += real
