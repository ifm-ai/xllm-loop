"""Terminal-KV evaluation of Part 2 models: prompt-conditioned PPL, the seven choice tasks and generation.

Huginn models run through `DepthControlledHuginn.terminal_kv_forward` at R=5 with the
paper's physical batches (at least eight rows, grouped by length), prefill, one-token
decode and chunk calls, sampler and scoring. Table 4 swaps the prefill path
(`TeacherR1Prefill`, `DistilledPrefill`) or evaluates the dense D4 (`DenseKV`).
"""
from __future__ import annotations

import collections
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from xllm.paper_part2.choice_tasks import reduce_token_scores, request_groups
from xllm.paper_part2.distill import prefill_banks
from xllm.paper_part2.eval_protocol import (
    GENERATION,
    MIN_BATCH,
    PPL_PREFIX_TOKENS,
    SEQUENCE_LENGTH,
    STATE_SEED,
    check_protocol_file,
)
from xllm.paper_part2.generation_tasks import (
    clean,
    gsm8k_full_prompt,
    item_identity,
    mbpp_pass_rates,
    sampling_uniforms,
    score_drop,
    score_gsm8k,
)


def _check_batch(tokens) -> None:
    if tokens.shape[0] < MIN_BATCH:
        raise ValueError(f"evaluation batches have at least {MIN_BATCH} rows, got {tokens.shape[0]}")


class TerminalKV:
    """The standard path of Tables 2 and 3: terminal KV at R=5 with the identity-seeded state. It prefills a
    prompt, decodes one token, or scores a chunk against a cache."""

    def __init__(self, model: nn.Module, seed: int = STATE_SEED) -> None:
        self.model, self.seed = model, seed

    def _forward(self, tokens, ids, cache, full_logits=False):
        _check_batch(tokens)
        return self.model.terminal_kv_forward(tokens, ids, self.seed, cache, full_logits)

    def _prefix(self, tokens, ids):
        """The cache after all prompt tokens but the last."""
        return self._forward(tokens, ids, None)[1]

    def prefill(self, tokens, ids):
        """Next-token logits after the prompt: `prompt[:-1]` from an empty cache, then the last token."""
        cache = self._prefix(tokens[:, :-1], ids) if tokens.shape[1] > 1 else None
        logits, cache = self._forward(tokens[:, -1:], ids, cache)
        return logits[:, -1], cache

    def decode(self, tokens, ids, cache):
        logits, cache = self._forward(tokens, ids, cache)
        return logits[:, -1], cache

    def chunk(self, tokens, ids, cache):
        """Logits at every position of a chunk scored against the cache."""
        return self._forward(tokens, ids, cache, full_logits=True)


class TeacherR1Prefill(TerminalKV):
    """Table 4 "One recurrence": the teacher prefills with one recurrence and decodes at R=5."""

    def _prefix(self, tokens, ids):
        _check_batch(tokens)
        return self.model.terminal_kv_forward(tokens, ids, self.seed, depth=1)[1]


class DistilledPrefill(TerminalKV):
    """Table 4 "Distilled": the student's state, one teacher recurrence and the coda write the banks;
    the teacher decodes at R=5."""

    def __init__(self, model: nn.Module, student: nn.Module, seed: int = STATE_SEED) -> None:
        super().__init__(model, seed)
        self.student = student

    def _prefix(self, tokens, ids):
        _check_batch(tokens)
        return prefill_banks(self.model, self.student, tokens)


class DenseKV(TerminalKV):
    """Table 4 D4: a dense Transformer that decodes itself with its full KV cache."""

    def _forward(self, tokens, ids, cache, full_logits=False):
        _check_batch(tokens)
        model = self.model
        cache = [(None, None, 0)] * model.num_layers if cache is None else list(cache)
        start = cache[0][2]
        hidden = model.embed(tokens)
        freqs_cis = model.rope.get_freqs_cis(start, start + tokens.shape[1], hidden.device)
        for layer_id, layer in enumerate(model.layers):
            hidden, _, cache[layer_id] = layer(hidden, freqs_cis, None, None, False, True, cache[layer_id])
        return model.output(hidden if full_logits else hidden[:, -1:], None, None), cache


def _ce(logits, targets):
    return F.cross_entropy(logits.float(), targets.long(), reduction="none").double().cpu().numpy()


def _documents_around_prompt(row, bos: int) -> tuple[int, list[int]]:
    """Start of the document holding the last prompt token, and starts of documents after the prompt."""
    starts = np.flatnonzero(row == bos)
    before = starts[starts <= PPL_PREFIX_TOKENS - 1]
    return (int(before.max()) if len(before) else 0), [int(v) for v in starts[starts >= PPL_PREFIX_TOKENS]]


def _score_continued(arm: TerminalKV, x: np.ndarray, y: np.ndarray, identity: int, start: int, end: int,
                     ce: np.ndarray) -> None:
    """Prefill the document holding the last prompt token from its start, then score its tokens up to `end`."""
    split = PPL_PREFIX_TOKENS
    lane_ids = torch.tensor([identity] * MIN_BATCH, device="cuda")
    prompt = torch.as_tensor(np.repeat(x[None, start:split], MIN_BATCH, 0), dtype=torch.long, device="cuda")
    last, cache = arm.prefill(prompt, lane_ids)
    ce[0] = _ce(last[:1], torch.as_tensor(y[split - 1:split], device="cuda"))[0]
    if end > split:
        chunk = torch.as_tensor(np.repeat(x[None, split:end], MIN_BATCH, 0), dtype=torch.long, device="cuda")
        logits, cache = arm.chunk(chunk, lane_ids, cache)
        ce[1:end - split + 1] = _ce(logits[0], torch.as_tensor(y[split:end], device="cuda"))
        del logits
    del cache


def _score_new_documents(arm: TerminalKV, rows: list[dict], docs: list[tuple[int, int, int]]) -> None:
    """Score every document `(row, start, end)` that starts after the prompt without a cache, shortest first,
    eight to a batch."""
    split = PPL_PREFIX_TOKENS
    docs = sorted(docs, key=lambda d: d[2] - d[1])
    for offset in range(0, len(docs), MIN_BATCH):
        group = docs[offset:offset + MIN_BATCH]
        physical = group + [group[-1]] * (MIN_BATCH - len(group))
        width = max(s1 - s0 for _, s0, s1 in group)
        tokens = np.zeros((MIN_BATCH, width), np.int64)
        for lane, (key, s0, s1) in enumerate(physical):
            tokens[lane, :s1 - s0] = rows[key]["x"][s0:s1]
        lane_ids = torch.tensor([rows[key]["id"] for key, _, _ in physical], device="cuda")
        logits, cache = arm.chunk(torch.as_tensor(tokens, device="cuda"), lane_ids, None)
        del cache
        for lane, (key, s0, s1) in enumerate(group):
            rows[key]["ce"][s0 - split + 1:s1 - split + 1] = _ce(
                logits[lane, :s1 - s0], torch.as_tensor(rows[key]["y"][s0:s1], device="cuda"))
        del logits


def _ppl_sums(rows: list[dict], documents: int) -> dict:
    """Masked cross-entropy sums `<key>_sum` and target counts `<key>_n` over the scored predictions: `cond` all
    4,097, `first` the one at the last prompt token, `f128` the first 128, `a` those continuing the prompt's
    document (`continued`) and `b` those in documents after the prompt (`new_documents`)."""
    sums = collections.Counter()
    for row in rows:
        mask = row["mask"]
        ce = np.where(mask, row["ce"], 0.)
        continued = row["end"] - PPL_PREFIX_TOKENS + 1
        sums["cond_sum"] += ce.sum()
        sums["cond_n"] += int(mask.sum())
        sums["first_sum"] += ce[0]
        sums["first_n"] += int(mask[0])
        sums["f128_sum"] += ce[:128].sum()
        sums["f128_n"] += int(mask[:128].sum())
        sums["a_sum"] += ce[:continued].sum()
        sums["a_n"] += int(mask[:continued].sum())
        sums["b_sum"] += ce[continued:].sum()
        sums["b_n"] += int(mask[continued:].sum())
        sums["sequences"] += 1
    sums["documents_b"] = documents
    return {key: float(value) for key, value in sums.items()}


@torch.inference_mode()
def segment_ppl(arm: TerminalKV, batches, bos: int) -> dict:
    """Prompt-conditioned PPL of packed 8,192-token sequences in which every prediction sees only its own document.

    Scored are the 4,097 predictions from the last prompt token (position 4,095) on. The document holding position
    4,095 is prefilled by the arm from its own start through the prompt, and its later tokens are scored against
    that cache (`continued`, the Wiki PPL column). Documents that start after the prompt are scored alone
    without a cache (`new_documents`), in length order, eight to a batch. Returns summed cross-entropies and
    counts; `ppl_metrics` turns them into PPL.
    """
    split, length = PPL_PREFIX_TOKENS, SEQUENCE_LENGTH
    rows, docs = [], []
    for x, y, m, ids, real in batches:
        for r in range(real):
            xr, yr, mr = np.asarray(x[r]), np.asarray(y[r]), np.asarray(m[r], bool)
            start, later = _documents_around_prompt(xr, bos)
            ce = np.full(length - split + 1, np.nan)
            if start == split - 1:  # a document starting at the last prompt token shares no prefix
                end, new = start + 1, [start] + later
            else:
                end, new = (later[0] if later else length), later
                _score_continued(arm, xr, yr, ids[r], start, end, ce)
            bounds = new + [length]
            docs += [(len(rows), s0, s1) for s0, s1 in zip(bounds[:-1], bounds[1:])]
            rows.append(dict(ce=ce, mask=mr[split - 1:], x=xr, y=yr, id=ids[r], end=end))
    _score_new_documents(arm, rows, docs)
    return _ppl_sums(rows, len(docs))


def ppl_metrics(sums: dict) -> dict:
    """PPL of summed `segment_ppl` outputs: `continued` is the Wiki PPL column."""
    ppl = lambda name: math.exp(sums[f"{name}_sum"] / sums[f"{name}_n"]) if sums[f"{name}_n"] else None
    return dict(continued=ppl("a"), new_documents=ppl("b"), conditioned=ppl("cond"), first_token=ppl("first"),
                first128=ppl("f128"), continued_share=sums["a_n"] / sums["cond_n"], sequences=int(sums["sequences"]),
                targets=int(sums["cond_n"]))


@torch.inference_mode()
def score_requests(arm: TerminalKV, requests: list[dict], shard: int = 0, shards: int = 1) -> list[dict]:
    """Every option's target log-probabilities: prefill the context, then feed the targets one token at a time."""
    records = []
    for number, (prefix, real, physical) in enumerate(request_groups(requests)):
        if number % shards != shard:
            continue
        ids = torch.tensor([r["example_id"] for r in physical], device="cuda")
        inputs = torch.tensor([r["tokens"][:prefix] for r in physical], device="cuda")
        count = max(r["target_tokens"] for r in real)
        probs, greedy, cache = [[] for _ in real], [[] for _ in real], None
        for pos in range(count):
            if pos == 0:
                logits, cache = arm.prefill(inputs, ids)
            else:
                logits, cache = arm.decode(inputs, ids, cache)
            targets = torch.tensor([r["targets"][pos] if pos < r["target_tokens"] else 0 for r in physical],
                                   device="cuda")
            logprobs = (-F.cross_entropy(logits.float(), targets, reduction="none")).double().cpu().tolist()
            predicted = logits.argmax(-1).cpu().tolist()
            for lane, r in enumerate(real):
                if pos < r["target_tokens"]:
                    probs[lane].append(logprobs[lane])
                    greedy[lane].append(predicted[lane])
            inputs = targets[:, None]
        for lane, r in enumerate(real):
            total, is_greedy = reduce_token_scores(r["targets"], greedy[lane], probs[lane])
            records.append(dict(
                index=r["index"], example_id=r["example_id"], option_index=r["option_index"],
                norm_chars=r["norm_chars"], context_tokens=prefix, target_tokens=r["target_tokens"],
                target_token_ids=r["targets"], greedy_token_ids=greedy[lane], target_logprobs=probs[lane],
                loglikelihood=total, loglikelihood_per_char=total / r["norm_chars"], is_greedy=is_greedy,
                encoding_sha256=r["encoding_sha256"]))
        del cache
    return records


class LaneCache:
    """Terminal KV rows of every generation lane, so lanes of different prompts decode together."""

    def __init__(self, cache, lanes: int, capacity: int) -> None:
        self.kv = [(k.new_empty((lanes, capacity, *k.shape[2:])), v.new_empty((lanes, capacity, *v.shape[2:])))
                   for k, v, _ in cache]

    def add(self, cache, source: int, lanes: list[int], length: int) -> None:
        for (dk, dv), (k, v, _) in zip(self.kv, cache):
            dk[lanes, :length] = k[source, :length]
            dv[lanes, :length] = v[source, :length]

    def get(self, lanes: list[int], count: int):
        index = torch.tensor(lanes, device="cuda")
        return [(dk.index_select(0, index)[:, :count], dv.index_select(0, index)[:, :count], count)
                for dk, dv in self.kv]

    def update(self, cache, lanes: list[int], count: int) -> None:
        n = len(lanes)
        for (dk, dv), (k, v, length) in zip(self.kv, cache):
            assert length == count + 1
            dk[lanes, count] = k[:n, count]
            dv[lanes, count] = v[:n, count]


def generation_items(task: str, tokenizer, data: str | Path) -> list[dict]:
    """The paper's prompts with their identities and token ids (GSM8K 8-shot; DROP and MBPP+ pre-rendered)."""
    spec = GENERATION[task]
    data = check_protocol_file(data, task)
    if task == "gsm8k":
        rows = [json.loads(line) for line in data.read_text().splitlines()]
        items = [dict(id=f"gsm8k/test/{i}", prompt=gsm8k_full_prompt(row), raw=row) for i, row in enumerate(rows)]
    else:
        items = json.loads(data.read_text())
    if len(items) != spec.examples:
        raise ValueError(f"{task} must have {spec.examples} items, got {len(items)}")
    for item in items:
        item["identity"] = item_identity(item["id"])
        tokens = tokenizer.encode(item["prompt"], bos=True, eos=False)
        if len(tokens) > spec.max_prompt_tokens:
            raise ValueError(f"{item['id']}: prompt exceeds {spec.max_prompt_tokens} tokens")
        item["tokens"] = tokens
    return items


PREFILL_BATCH = 16  # prompts of equal length prefilled together
DECODE_BATCH = 64  # lanes of equal cache length decoded together


class _Samples:
    """Every lane's generated tokens and finish reason; lane `i * samples + s` is sample `s` of item `i`."""

    def __init__(self, items: list[dict], task: str, tokenizer) -> None:
        self.spec, self.task, self.tokenizer = GENERATION[task], task, tokenizer
        count, max_new = self.spec.samples, self.spec.max_new_tokens
        self.tokens = [[] for _ in range(len(items) * count)]
        self.reasons = [None] * len(self.tokens)
        self.uniforms = torch.tensor(np.stack([sampling_uniforms(x["identity"], s, max_new) for x in items
                                               for s in range(count)]), dtype=torch.float32, device="cuda")
        self.identities = [x["identity"] for x in items for _ in range(count)]

    def accept(self, logits: torch.Tensor, lanes: list[int]) -> None:
        """Append each lane's next token, greedy or by inverse-CDF sampling from its fixed uniforms, and record
        why a lane stopped."""
        if self.spec.temperature:
            probs = F.softmax(logits.float() / self.spec.temperature, -1)
            cdf = probs.cumsum(-1)
            cdf[:, -1] = 1
            u = self.uniforms[lanes, [len(self.tokens[i]) for i in lanes]]
            chosen = torch.searchsorted(cdf, u[:, None].contiguous()).squeeze(1)
        else:
            chosen = logits.argmax(-1)
        for lane, token in zip(lanes, chosen.tolist()):
            self.tokens[lane].append(token)
            if token == self.tokenizer.eos_id:
                self.reasons[lane] = "eos"
            elif clean(self.tokenizer.decode(self.tokens[lane], cut_at_eos=True), self.task)[1]:
                self.reasons[lane] = "stop_sequence"
            elif len(self.tokens[lane]) == self.spec.max_new_tokens:
                self.reasons[lane] = "length"


def _prefill_prompts(arm: TerminalKV, items: list[dict], samples: _Samples) -> tuple[LaneCache, dict]:
    """Prefill prompts of equal length together and take every lane's first token; return the lanes' KV store
    and the unfinished lanes by cache length."""
    count = samples.spec.samples
    capacity = max(len(x["tokens"]) for x in items) + samples.spec.max_new_tokens
    queues, store = collections.defaultdict(list), None
    by_length = collections.defaultdict(list)
    for i, x in enumerate(items):
        by_length[len(x["tokens"])].append(i)
    for length, indices in sorted(by_length.items()):
        for offset in range(0, len(indices), PREFILL_BATCH):
            real = indices[offset:offset + PREFILL_BATCH]
            physical = real + [real[0]] * max(0, MIN_BATCH - len(real))
            tokens = torch.tensor([items[i]["tokens"] for i in physical], device="cuda")
            ids = torch.tensor([items[i]["identity"] for i in physical], device="cuda")
            logits, cache = arm.prefill(tokens, ids)
            if store is None:
                store = LaneCache(cache, len(samples.tokens), capacity)
            for j, i in enumerate(real):
                lanes = list(range(i * count, (i + 1) * count))
                store.add(cache, j, lanes, length)
                samples.accept(logits[j].expand(count, -1), lanes)
                queues[length].extend(k for k in lanes if samples.reasons[k] is None)
            del logits, cache
    return store, queues


def _decode_lanes(arm: TerminalKV, store: LaneCache, queues: dict, samples: _Samples) -> None:
    """Decode the unfinished lanes one token at a time, shortest cache first, until every lane has finished."""
    while queues:
        count = min(queues)
        pending = queues.pop(count)
        for offset in range(0, len(pending), DECODE_BATCH):
            lanes = pending[offset:offset + DECODE_BATCH]
            physical = lanes + [lanes[0]] * max(0, MIN_BATCH - len(lanes))
            cache = store.get(physical, count)
            inputs = torch.tensor([[samples.tokens[i][-1]] for i in physical], device="cuda")
            ids = torch.tensor([samples.identities[i] for i in physical], device="cuda")
            logits, cache = arm.decode(inputs, ids, cache)
            store.update(cache, lanes, count)
            samples.accept(logits[:len(lanes)], lanes)
            alive = [i for i in lanes if samples.reasons[i] is None]
            if alive:
                queues[count + 1].extend(alive)


@torch.inference_mode()
def generate(arm: TerminalKV, items: list[dict], task: str, tokenizer, shard: int = 0, shards: int = 1):
    """Prompts prefilled in batches of up to 16 of equal length; lanes decode in batches of up to 64 of equal
    cache length. Greedy, or inverse-CDF sampling from each lane's fixed uniforms."""
    items = [x for i, x in enumerate(items) if i % shards == shard]
    samples = _Samples(items, task, tokenizer)
    started = time.monotonic()
    store, queues = _prefill_prompts(arm, items, samples)
    _decode_lanes(arm, store, queues, samples)
    results = []
    for i, x in enumerate(items):
        for s in range(samples.spec.samples):
            lane = i * samples.spec.samples + s
            raw = tokenizer.decode(samples.tokens[lane], cut_at_eos=True)
            text, _ = clean(raw, task)
            results.append(dict(task_id=x["id"], sample=s, generation=text, raw_generation=raw,
                                tokens=samples.tokens[lane], finish_reason=samples.reasons[lane]))
    return items, results, dict(seconds=time.monotonic() - started, generated_tokens=sum(map(len, samples.tokens)))


def score_generations(task: str, items: list[dict], results: list[dict]) -> tuple[list[dict], dict]:
    """Per-sample GSM8K or DROP scores and their means."""
    lookup = {x["id"]: x for x in items}
    if task == "gsm8k":
        scored = [dict(task_id=r["task_id"], **score_gsm8k(r["generation"], lookup[r["task_id"]]["raw"]))
                  for r in results]
    else:
        scored = [dict(task_id=r["task_id"], **score_drop(r["generation"], lookup[r["task_id"]])) for r in results]
    metrics = {key: sum(s[key] for s in scored) / len(scored) for key in scored[0] if key != "task_id"}
    return scored, metrics


def evaluate_mbpp(results: list[dict], items: list[dict], out: Path, tests: str | Path,
                  sandbox: list[str] | None = None) -> dict:
    """Sanitize every sample and run EvalPlus 0.3.1 base and plus tests (MBPP+ v0.2.0), optionally in a sandbox."""
    from evalplus.sanitize import sanitize

    tests = check_protocol_file(tests, "mbpp_plus")
    entry_points = {x["id"]: x["entry_point"] for x in items}
    samples = out / "samples.jsonl"
    samples.write_text("".join(json.dumps(dict(task_id=r["task_id"], solution=sanitize(
        r["generation"], entrypoint=entry_points[r["task_id"]]))) + "\n" for r in results))
    workdir = out / "sandbox"
    workdir.mkdir(exist_ok=True)
    # The paper's EvalPlus settings: 4 GiB per test process, six processes, and time limits of at least 1 s
    # and 4x the ground truth's.
    env = dict(os.environ, XDG_CACHE_HOME=str(workdir / "cache"), EVALPLUS_MAX_MEMORY_BYTES=str(4 * 1024**3),
               MBPP_OVERRIDE_PATH=str(tests))
    command = [sys.executable, "-m", "evalplus.evaluate", "--dataset", "mbpp", "--samples", str(samples),
               "--parallel", "6", "--min-time-limit", "1", "--gt-time-limit-factor", "4"]
    subprocess.run((sandbox or []) + command, env=env, cwd=workdir, check=True)
    evaluation = json.loads((out / "samples_eval_results.json").read_text())["eval"]
    if len(evaluation) != len(items):
        raise ValueError("EvalPlus did not evaluate every MBPP+ item")
    return mbpp_pass_rates(evaluation, GENERATION["mbpp"].samples)
