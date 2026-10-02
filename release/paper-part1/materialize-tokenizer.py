#!/usr/bin/env python3
"""Prepare the Part 1 tokenizer from a downloaded K2-Horizon tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def materialize(source_dir: Path, output_dir: Path) -> None:
    """Remove automatic BOS insertion; XLLM adds BOS when packing sequences."""
    payloads = {name: (source_dir / name).read_bytes() for name in EXPECTED_FILES}
    tokenizer = json.loads(payloads["tokenizer.json"])
    config = json.loads(payloads["tokenizer_config.json"])
    special = json.loads(payloads["special_tokens_map.json"])
    if not all(isinstance(value, dict) for value in (tokenizer, config, special)):
        raise ValueError("tokenizer files must contain JSON objects")

    processor = tokenizer.get("post_processor", {})
    processors = processor.get("processors", [])
    byte_level = {
        "type": "ByteLevel", "add_prefix_space": False,
        "trim_offsets": False, "use_regex": False,
    }
    if processor.get("type") != "Sequence" or len(processors) != 2:
        raise ValueError("expected the K2-Horizon two-stage post-processor")
    if processors[1] != byte_level:
        raise ValueError("unexpected tokenizer ByteLevel post-processor")
    tokenizer["post_processor"] = processors[1]
    if list(special) != ["bos_token", "eos_token"]:
        raise ValueError("expected bos_token and eos_token")
    if any(not isinstance(value, dict) or not isinstance(value.get("content"), str)
           for value in special.values()):
        raise ValueError("special tokens must provide string content")
    simplified = {name: value["content"] for name, value in special.items()}
    payloads["tokenizer.json"] = (
        json.dumps(tokenizer, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    payloads["special_tokens_map.json"] = json.dumps(
        simplified, ensure_ascii=False, indent=2,
    ).encode("utf-8")

    output_dir.mkdir(parents=False, exist_ok=False)
    for name, payload in payloads.items():
        with (output_dir / name).open("xb") as stream:
            stream.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source_dir, args.output_dir)
    print(f"Prepared tokenizer: {args.output_dir}")


if __name__ == "__main__":
    main()
