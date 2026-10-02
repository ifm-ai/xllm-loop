#!/usr/bin/env python3
"""Prepare downloaded, pinned TxT360 data as text JSONL for XLLM's data loader."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator


EFFORT_FIELDS = {"high": "think", "medium": "think_fast", "low": "think_faster"}
# File names the data loader reads from a source directory.
CHUNK_NAME = re.compile(r".*chunk\.?\d+.*\.jsonl")


def _rows(path: Path) -> Iterator[Any]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as parquet

        for batch in parquet.ParquetFile(path).iter_batches(batch_size=32):
            yield from batch.to_pylist()
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        raise ValueError("input must be .jsonl or .parquet")


def _message_texts(messages: str | list[Any], tag: str) -> list[str]:
    """Render one training text per assistant turn, with the preceding turns as context.

    This is the Part 1 release template, and it defines the training text: keep it
    byte for byte. An assistant turn opens with "<|im_start|>assistant" and no
    newline; the rendered turn holds its reasoning in <tag>...</tag>, while the
    context copy of an earlier turn keeps only its content and tool calls.
    """
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list or its JSON encoding")
    context = ""
    rendered = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {role}")
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise ValueError("message content must be text")
        if role != "assistant":
            if message.get("tools"):
                content += "\n\n<tools>\n" + "\n".join(
                    json.dumps(tool, ensure_ascii=False, sort_keys=True)
                    for tool in message["tools"]
                ) + "\n</tools>"
            context += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            continue
        # Without reasoning for the requested effort, fall back to the high-effort "think".
        reasoning = message.get(tag, message.get("think", ""))
        if not isinstance(reasoning, str):
            raise ValueError("assistant reasoning must be text")
        calls = []
        for call in message.get("tool_calls") or []:
            call = call.get("function", call)
            arguments = call["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append("<tool_call>\n" + json.dumps({
                "name": call["name"], "arguments": arguments,
            }, ensure_ascii=False, sort_keys=True) + "\n</tool_call>")
        suffix = ("\n" if content and calls else "") + "\n".join(calls)
        if not content and not calls:
            raise ValueError("assistant message needs content or tool calls")
        thinking = f"<{tag}>\n" + (reasoning + "\n" if reasoning else "") + f"</{tag}>\n"
        rendered.append(
            context + "<|im_start|>assistant" + thinking
            + content.lstrip("\n") + suffix + "<|im_end|>"
        )
        context += "<|im_start|>assistant" + content + suffix + "<|im_end|>"
    if not rendered:
        raise ValueError("messages must contain an assistant turn")
    return rendered


def prepare(
    input_path: str | Path,
    output_path: str | Path,
    *,
    effort: str | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Convert downloaded text or conversation rows into training JSONL."""
    source_path, output = Path(input_path), Path(output_path)
    if effort is not None and effort not in EFFORT_FIELDS:
        raise ValueError("effort must be high, medium or low")
    if max_rows is not None and (type(max_rows) is not int or max_rows <= 0):
        raise ValueError("max_rows must be a positive integer")
    if not CHUNK_NAME.fullmatch(output.name):
        raise ValueError(f"output must be named <name>.chunk<N>.jsonl: {output.name}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite prepared data: {output}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    before = source_path.stat()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    input_rows = output_rows = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for row in _rows(source_path):
                if max_rows is not None and input_rows == max_rows:
                    break
                input_rows += 1
                if not isinstance(row, dict):
                    raise ValueError("row must be an object")
                if isinstance(row.get("text"), str):
                    texts = [row["text"]]
                elif "messages" in row:
                    if effort is None:
                        raise ValueError("messages require an explicit effort")
                    texts = _message_texts(row["messages"], EFFORT_FIELDS[effort])
                else:
                    raise ValueError("row must contain text or supported messages")
                for text in texts:
                    stream.write(json.dumps(
                        {"text": text}, ensure_ascii=False, allow_nan=False,
                    ) + "\n")
                    output_rows += 1
            if not input_rows:
                raise ValueError("input contains no rows")
            stream.flush()
            os.fsync(stream.fileno())
        after = source_path.stat()
        identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity(before) != identity(after):
            raise ValueError("source changed during data preparation")
        summary = {
            "input_rows": input_rows,
            "output_rows": output_rows,
            "content_key": "text",
            "format": "txt360-text-jsonl-v1",
            "effort": effort,
            "message_policy": "one sample per assistant turn; prior reasoning omitted",
        }
        os.link(temporary, output)
        return summary
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--effort", choices=("high", "medium", "low"))
    args = parser.parse_args()
    print(json.dumps(prepare(
        args.input, args.output,
        effort=args.effort, max_rows=args.max_rows,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
