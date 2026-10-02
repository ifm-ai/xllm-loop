"""File hashing, JSON encoding and atomic writes shared by the Part 1 evaluator and artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path


HASH_CHUNK_BYTES = 8 * 1024**2
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def canonical_json_bytes(value: object) -> bytes:
    """Compact, key-sorted UTF-8 JSON: the bytes every Part 1 JSON hash covers."""
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be JSON-compatible and finite") from exc
    return encoded.encode("utf-8")


def pretty_json_bytes(value: object) -> bytes:
    """Indented, key-sorted UTF-8 JSON with a final newline, as written to result and artifact files."""
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    """Hash a JSON value independent of mapping insertion order."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash the exact bytes of a regular file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"file does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def atomic_write(path: Path, payload: bytes, *, replace: bool = True) -> None:
    """Write bytes through a synced temporary file in the destination directory.

    With ``replace`` the file is renamed over any existing one; otherwise it is
    hard-linked into place and ``FileExistsError`` reports an existing path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary_path, path)
            temporary_path = None
        else:
            os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    atomic_write(path, pretty_json_bytes(value))


def read_regular_bytes(root: Path, relative: Path) -> bytes:
    """Read a regular file below ``root`` without following any symlink on the way."""
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"evaluation input must not traverse a symlink: {relative}")
    candidate.resolve(strict=True).relative_to(root)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(candidate, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"evaluation input must be a regular file: {relative}")
        chunks = []
        while chunk := os.read(descriptor, HASH_CHUNK_BYTES):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)
