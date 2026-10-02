"""Prepare the Part 2 evaluation inputs from their public sources.

Downloads every source at a pinned revision (or reads it from
`--sources DIR/<name>/<file>`, `<name>` a key of SOURCES and `<file>` the last
part of its URL), checks its SHA-256, applies the paper's transformation and
checks the result against `xllm.paper_part2.eval_protocol.DATA_SHA256`. Writes
the files into `--output` and a `data.json` for `eval_paper_part2.py`.

    python release/paper-part2/prepare-eval-data.py --output /path/to/eval-data
"""

import argparse
import gzip
import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from pathlib import Path

from xllm.paper_part2.eval_protocol import CHOICE_TASKS, DATA_SHA256, GENERATION

HF = "https://huggingface.co/datasets"
# name -> (URL, SHA-256 of the downloaded bytes)
SOURCES = {
    "hellaswag": (f"{HF}/Rowan/hellaswag/resolve/218ec52e09a7e7462a5400043bb9a69a41d06b76/data/"
                  "validation-00000-of-00001.parquet",
                  "899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd"),
    "arc_easy": (f"{HF}/allenai/ai2_arc/resolve/210d026faf9955653af8916fad021475a3f00453/ARC-Easy/"
                 "test-00000-of-00001.parquet",
                 "4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177"),
    "arc_challenge": (f"{HF}/allenai/ai2_arc/resolve/210d026faf9955653af8916fad021475a3f00453/ARC-Challenge/"
                      "test-00000-of-00001.parquet",
                      "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9"),
    "lambada_openai": ("https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl",
                       "4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226"),
    "piqa": ("https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip",
             "54d32a04f59a7e354396f321723c8d7ec35cc6b08506563d8d1ffcc15ce98ddd"),
    "openbookqa": ("https://ai2-public-datasets.s3.amazonaws.com/open-book-qa/OpenBookQA-V1-Sep2018.zip",
                   "82368cf05df2e3b309c17d162e10b888b4d768fad6e171e0a041954c8553be46"),
    "sciq": (f"{HF}/allenai/sciq/resolve/2c94ad3e1aafab77146f384e23536f97a4849815/data/test-00000-of-00001.parquet",
             "3a719356a29b127fc54ef3c7f51a034db4bd105d5717215e8c85d2aa58d60667"),
    "gsm8k": (f"{HF}/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet",
              "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"),
    "wikitext": (f"{HF}/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1/"
                 "test-00000-of-00001.parquet",
                 "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"),
    "drop": ("https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip",
             "39d2278a29fd729de301b111a45f434c24834f40df8f4ff116d864589e3249d6"),
    "mbpp_plus": ("https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz",
                  "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63"),
}
SELECTION_SEED = 20260911
DROP_DEMOS = ("2358d5eb-c2cc-4784-9d45-44a766302224", "9b6bd0fc-f8db-453b-a190-046c6cd6ff26",
              "04e2ec7f-4444-4270-a007-75c0a042cff8")
MBPP_DEMOS = ("Mbpp/2", "Mbpp/3", "Mbpp/4")


def fetch(name: str, sources: Path | None) -> bytes:
    url, digest = SOURCES[name]
    # Keyed by source: several sources share a file name (test-00000-of-00001.parquet).
    local = sources / name / url.rsplit("/", 1)[1] if sources else None
    if local and local.exists():
        data = local.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=300) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"{name}: {url} does not match its pinned SHA-256")
    return data


def parquet_rows(data: bytes) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(io.BytesIO(data)).to_pylist()


def jsonl(rows, **dumps) -> bytes:
    return "".join(json.dumps(row, **dumps) + "\n" for row in rows).encode()


def compact(rows) -> bytes:
    """Sorted keys, no spaces, UTF-8: the choice tasks' serialization."""
    return jsonl(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def zip_lines(data: bytes, member: str) -> list[str]:
    return [line for line in zipfile.ZipFile(io.BytesIO(data)).read(member).decode().splitlines() if line.strip()]


def choice_tasks(sources) -> dict[str, bytes]:
    out = {name: compact(parquet_rows(fetch(name, sources))) for name in ("hellaswag", "arc_easy", "arc_challenge")}
    out["lambada_openai"] = compact(json.loads(line) for line in fetch("lambada_openai", sources).decode().splitlines()
                                    if line.strip())
    data = fetch("piqa", sources)
    labels = zipfile.ZipFile(io.BytesIO(data)).read("physicaliqa-train-dev/dev-labels.lst").decode().split()
    out["piqa"] = compact({"goal": row["goal"], "sol1": row["sol1"], "sol2": row["sol2"], "label": int(label)}
                          for row, label in zip(map(json.loads, zip_lines(data, "physicaliqa-train-dev/dev.jsonl")),
                                                labels, strict=True))
    rows = map(json.loads, zip_lines(fetch("openbookqa", sources), "OpenBookQA-V1-Sep2018/Data/Main/test.jsonl"))
    out["openbookqa"] = compact({"id": row["id"], "question_stem": row["question"]["stem"], "answerKey": row["answerKey"],
                                 "choices": {"text": [c["text"] for c in row["question"]["choices"]],
                                             "label": [c["label"] for c in row["question"]["choices"]]}}
                                for row in rows)
    out["sciq"] = jsonl(parquet_rows(fetch("sciq", sources)), ensure_ascii=False)
    return out


def gsm8k(sources) -> bytes:
    """GSM8K test with the calculator annotations <<...>> removed from the answers."""
    return jsonl({"question": row["question"], "answer": re.sub(r"<<.*?>>", "", row["answer"])}
                 for row in parquet_rows(fetch("gsm8k", sources)))


def wikitext(sources) -> bytes:
    """The test split's lines joined by newlines into one record, so the loader adds one BOS/EOS pair."""
    lines = [row["text"] for row in parquet_rows(fetch("wikitext", sources))]
    return (json.dumps({"text": "\n".join(lines)}, ensure_ascii=False) + "\n").encode()


def drop(sources) -> bytes:
    """500 sampled dev questions after three fixed demonstrations; direct answers, multi-span joined by '; '."""
    dev = json.loads(zipfile.ZipFile(io.BytesIO(fetch("drop", sources))).read("drop_dataset/drop_dataset_dev.json"))
    ids = [qa["query_id"] for passage in dev.values() for qa in passage["qa_pairs"]]
    by_id = {qa["query_id"]: (passage["passage"], qa) for passage in dev.values() for qa in passage["qa_pairs"]}

    def block(passage: str, qa: dict) -> str:
        return f"Text:\n{passage}\n\nQuestion:\n{qa['question']}\nAnswer:"

    def cleaned(text: str) -> str:  # the demonstrations' passages: single spaces, none before ',' or '.'
        return re.sub(r" (?=[,.])", "", " ".join(text.split()))

    prefix = "".join(block(cleaned(by_id[q][0]), by_id[q][1]) + " " + (
        by_id[q][1]["answer"]["number"] or "; ".join(by_id[q][1]["answer"]["spans"])) + "\n\n" for q in DROP_DEMOS)
    items = [dict(id=q, passage=by_id[q][0], qa=by_id[q][1], prompt=prefix + block(by_id[q][0].strip(), by_id[q][1]))
             for q in random.Random(SELECTION_SEED).sample(sorted(ids), GENERATION["drop"].examples)]
    return (json.dumps(items, indent=2) + "\n").encode()


def mbpp(sources) -> tuple[bytes, bytes]:
    """100 sampled MBPP+ v0.2.0 tasks after three fixed demonstrations (prompt and solution), and their tests."""
    tasks = {row["task_id"]: row for row in map(json.loads, gzip.decompress(fetch("mbpp_plus", sources)).splitlines())}
    pool = sorted(task for task in tasks if task not in MBPP_DEMOS)
    selected = sorted(random.Random(SELECTION_SEED).sample(pool, GENERATION["mbpp"].examples))
    prefix = "".join(tasks[d]["prompt"] + tasks[d]["canonical_solution"] + "\n\n" for d in MBPP_DEMOS)
    items = [dict(id=t, prompt=prefix + tasks[t]["prompt"], entry_point=tasks[t]["entry_point"]) for t in selected]
    return (json.dumps(items, indent=2) + "\n").encode(), jsonl(tasks[t] for t in selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sources", type=Path, help="already downloaded sources, as DIR/<name>/<file>")
    args = parser.parse_args()
    files = {f"{task}.jsonl": data for task, data in choice_tasks(args.sources).items()}
    files["gsm8k-test.jsonl"] = gsm8k(args.sources)
    files["wikitext-103-raw-v1-test.jsonl"] = wikitext(args.sources)
    files["drop.json"] = drop(args.sources)
    files["mbpp.json"], files["mbpp_plus.jsonl"] = mbpp(args.sources)
    keys = {**{f"{task}.jsonl": task for task in CHOICE_TASKS}, "gsm8k-test.jsonl": "gsm8k", "drop.json": "drop",
            "mbpp.json": "mbpp", "mbpp_plus.jsonl": "mbpp_plus", "wikitext-103-raw-v1-test.jsonl": "wikitext103_test"}
    wrong = [name for name, data in files.items() if hashlib.sha256(data).hexdigest() != DATA_SHA256[keys[name]]]
    if wrong:
        raise ValueError(f"prepared files differ from the protocol: {wrong}")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (args.output / name).write_bytes(data)
    out = args.output.resolve()
    config = {"wiki": str(out / "wikitext-103-raw-v1-test.jsonl"),
              "tasks": {task: str(out / f"{task}.jsonl") for task in CHOICE_TASKS},
              "gsm8k": str(out / "gsm8k-test.jsonl"), "drop": str(out / "drop.json"),
              "mbpp": str(out / "mbpp.json"), "mbpp_plus": str(out / "mbpp_plus.jsonl")}
    (args.output / "data.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"{len(files)} files match the protocol; wrote {args.output / 'data.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
