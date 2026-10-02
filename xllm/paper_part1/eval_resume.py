"""Strict item-level resume records for Part 1 evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from xllm.paper_part1.files import (
    atomic_write,
    canonical_json_bytes,
    require_sha256,
    sha256_json,
)


_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvalResumeHashes:
    """All identities that must match before one evaluation item is skipped."""

    model: str
    evaluator: str
    suite: str
    dataset: str
    prompt: str
    item: str
    generation: str

    def __post_init__(self) -> None:
        for field in fields(self):
            require_sha256(getattr(self, field.name), field.name)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def record_digest(self) -> str:
        return sha256_json(
            {
                "schema_version": _SCHEMA_VERSION,
                "hashes": self.to_dict(),
            }
        )


class EvalResumeStore:
    """Tamper-evident JSON cache with atomic, no-clobber item records."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def record_path(self, hashes: EvalResumeHashes) -> Path:
        digest = hashes.record_digest
        return self.root / "records" / digest[:2] / f"{digest}.json"

    def load(self, hashes: EvalResumeHashes) -> Any | None:
        path = self.record_path(hashes)
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"resume record must be a regular file: {path}")
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid resume record: {path}") from exc
        if not isinstance(record, dict) or set(record) != {
            "schema_version",
            "hashes",
            "result",
            "result_sha256",
        }:
            raise ValueError(f"invalid resume record schema: {path}")
        if record["schema_version"] != _SCHEMA_VERSION:
            raise ValueError(f"unsupported resume record schema: {path}")
        if record["hashes"] != hashes.to_dict():
            raise ValueError(f"resume record identity mismatch: {path}")
        if record["result_sha256"] != sha256_json(record["result"]):
            raise ValueError(f"resume result SHA-256 mismatch: {path}")
        return record["result"]

    def save(self, hashes: EvalResumeHashes, result: Any) -> Path:
        path = self.record_path(hashes)
        result_digest = sha256_json(result)
        if path.exists():
            existing = self.load(hashes)
            if sha256_json(existing) == result_digest:
                return path
            raise FileExistsError(
                f"resume record already contains a different result: {path}"
            )

        record = {
            "schema_version": _SCHEMA_VERSION,
            "hashes": hashes.to_dict(),
            "result": result,
            "result_sha256": result_digest,
        }
        try:
            atomic_write(path, canonical_json_bytes(record) + b"\n", replace=False)
        except FileExistsError:
            existing = self.load(hashes)
            if sha256_json(existing) != result_digest:
                raise FileExistsError(
                    "concurrent resume record contains a different result: "
                    f"{path}"
                )
        return path
