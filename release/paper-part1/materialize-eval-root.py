#!/usr/bin/env python3
"""Assemble prepared task JSONL files into one evaluation directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xllm.paper_part1.eval_data_contract import required_eval_files
from xllm.paper_part1.files import read_regular_bytes


def materialize(source_roots: list[Path], output_dir: Path) -> int:
    """Copy each required file from exactly one supplied source root."""
    output = output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if not source_roots:
        raise ValueError("at least one source root is required")
    roots = [root.resolve(strict=True) for root in source_roots]
    if len(set(roots)) != len(roots) or any(not root.is_dir() for root in roots):
        raise ValueError("source roots must be distinct directories")
    payloads = {}
    for relative in required_eval_files():
        matches = [root for root in roots if (root / relative).exists()
                   or (root / relative).is_symlink()]
        if len(matches) != 1:
            raise ValueError(f"expected one source for {relative}, found {len(matches)}")
        payload = read_regular_bytes(matches[0], relative)
        rows = payload.decode("utf-8").splitlines()
        if not rows or any(not isinstance(json.loads(row), dict) for row in rows):
            raise ValueError(f"expected nonempty JSONL objects: {relative}")
        payloads[relative] = payload
    # Every input is checked before the output exists; mkdir fails if it appeared meanwhile.
    output.mkdir()
    for relative, payload in payloads.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(payload)
    return len(payloads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, action="append", required=True,
                        help="directory of prepared task files; repeat for each root")
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory for the assembled task files")
    args = parser.parse_args()
    count = materialize(args.source_root, args.output)
    print(f"Prepared {count} task files: {args.output}")


if __name__ == "__main__":
    main()
