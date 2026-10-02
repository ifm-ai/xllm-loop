"""JSONL reader and padding helper used by the task iterator."""
import json
from typing import Any, Iterator, List


class ValidJSONLIterator:
    """Yield the decoded rows of a JSONL file that belong to one rank.

    Rank ``world_rank`` of ``world_size`` reads every ``world_size``-th line. A
    line that is not valid JSON raises an error naming the file and line.
    """

    def __init__(
        self,
        fpath: str,
        world_size: int,
        world_rank: int,
        infinite: bool,
    ):
        assert 0 <= world_rank < world_size, (world_rank, world_size)
        if infinite:
            raise ValueError("ValidJSONLIterator reads a file once; infinite is not supported")
        self.fpath = fpath
        self.world_size = world_size
        self.world_rank = world_rank

    def __iter__(self) -> Iterator[Any]:
        with open(self.fpath, "r", encoding="utf-8") as stream:
            for line_num, line in enumerate(stream, start=1):
                if (line_num - 1) % self.world_size != self.world_rank:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON on line {line_num} of {self.fpath}"
                    ) from exc
                yield row


def pad(
    tokens: List[int],
    max_length: int,
    value: int,
    padding: str = "post",
    truncating: str = "post",
):
    if len(tokens) < max_length:
        if padding == "post":
            tokens = tokens + [value] * (max_length - len(tokens))
        elif padding == "pre":
            tokens = [value] * (max_length - len(tokens)) + tokens
    if truncating == "post":
        tokens = tokens[:max_length]
    elif truncating == "pre":
        if len(tokens) > max_length:
            tokens = tokens[len(tokens) - max_length:]
    return tokens
