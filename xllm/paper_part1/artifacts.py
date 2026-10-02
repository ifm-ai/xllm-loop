"""Write, verify and load native XLLM Part 1 weight artifacts.

An artifact directory holds BF16 Safetensors shards
``model-NNNNN-of-NNNNN.safetensors`` with their index, ``config.json`` with the
model and tokenizer sections, the tokenizer under ``tokenizer/`` and
``artifact_manifest.json``, which lists every other file's size and SHA-256.
Released artifacts also hold a model card (``README.md``, whose SHA-256 the
manifest source records), ``LICENSE`` and ``NOTICE``.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from xllm.paper_part1.files import (
    pretty_json_bytes,
    sha256_file,
    sha256_json,
)


ARTIFACT_FORMAT = "xllm-native-bf16-safetensors"
ARTIFACT_MANIFEST = "artifact_manifest.json"
MODEL_INDEX = "model.safetensors.index.json"
MAX_SHARD_BYTES = 5 * 1024**3
_SCHEMA_VERSION = 1
_FIXED_ARTIFACT_PATHS = {"README.md", "config.json", MODEL_INDEX, "LICENSE", "NOTICE"}
_MODEL_SHARD_NAME = re.compile(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors")
# Files that `hf download --local-dir` adds next to a Hub repository's own files.
_HUB_DOWNLOAD_FILES = frozenset({".gitattributes"})
_HUB_DOWNLOAD_DIR = ".cache/huggingface/"


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return sha256_json(payload)


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_model_config(config: Mapping[str, Any]) -> None:
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("checkpoint config must contain a model object")


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {name}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / ARTIFACT_MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"artifact is missing {ARTIFACT_MANIFEST}")
    manifest = _read_json_object(manifest_path, "artifact manifest")
    digest = manifest.get("manifest_sha256")
    if not isinstance(digest, str) or digest != _manifest_digest(manifest):
        raise ValueError("artifact manifest SHA-256 mismatch")
    if (
        manifest.get("schema_version") != _SCHEMA_VERSION
        or manifest.get("release") != "paper-part1"
        or manifest.get("format") != ARTIFACT_FORMAT
    ):
        raise ValueError("unsupported Part 1 artifact manifest")
    return manifest


def _verify_source(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("artifact manifest source must be a mapping")
    _require_text(source.get("checkpoint_id"), "source.checkpoint_id")
    return source


def _verify_file_records(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, object]]:
    """Return the manifest's file records by path, each inside the closed layout."""
    file_records = manifest.get("files")
    if not isinstance(file_records, list):
        raise ValueError("artifact manifest files must be a list")
    expected: dict[str, Mapping[str, object]] = {}
    for record in file_records:
        if not isinstance(record, Mapping):
            raise ValueError("artifact file record must be a mapping")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError("artifact file record has an invalid path")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative != relative_path.as_posix()
        ):
            raise ValueError(f"artifact file path escapes its root: {relative}")
        if relative in expected or relative == ARTIFACT_MANIFEST:
            raise ValueError(f"duplicate or reserved artifact file: {relative}")
        allowed = (
            relative in _FIXED_ARTIFACT_PATHS
            or (
                len(relative_path.parts) == 1
                and _MODEL_SHARD_NAME.fullmatch(relative_path.name) is not None
            )
            or (
                len(relative_path.parts) >= 2
                and relative_path.parts[0] == "tokenizer"
            )
        )
        if not allowed:
            raise ValueError(f"artifact file is outside the closed layout: {relative}")
        expected[relative] = record
    return expected


def _verify_documents(
    expected: Mapping[str, Mapping[str, object]],
    source: Mapping[str, Any],
) -> None:
    """A model card recorded in the source must be the listed README.md.

    Released artifacts record their model card; an exported artifact has none.
    """
    if "model_card_sha256" in source:
        if "README.md" not in expected:
            raise ValueError("artifact must contain README.md")
        if expected["README.md"].get("sha256") != source["model_card_sha256"]:
            raise ValueError("model card SHA-256 does not match source provenance")
    if ("LICENSE" in expected) != ("NOTICE" in expected):
        raise ValueError("artifact LICENSE and NOTICE must be provided together")


def _verify_file_set(root: Path, expected: Mapping[str, Mapping[str, object]]) -> None:
    actual: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if relative in _HUB_DOWNLOAD_FILES or relative.startswith(_HUB_DOWNLOAD_DIR):
            continue
        if path.is_symlink():
            raise ValueError(f"artifact must not contain symlinks: {path}")
        if path.is_file() and relative != ARTIFACT_MANIFEST:
            actual.add(relative)
    if actual != set(expected):
        raise ValueError("artifact contains unlisted files or is missing listed files")


def _verify_file_hashes(
    root: Path,
    expected: Mapping[str, Mapping[str, object]],
) -> None:
    for relative, record in expected.items():
        path = root / relative
        expected_digest = record.get("sha256")
        if not isinstance(expected_digest, str) or sha256_file(path) != expected_digest:
            raise ValueError(f"artifact file SHA-256 mismatch: {relative}")
        expected_size = record.get("size_bytes")
        if type(expected_size) is not int or path.stat().st_size != expected_size:
            raise ValueError(f"artifact file size mismatch: {relative}")


def _verify_index(
    root: Path,
    manifest: Mapping[str, Any],
    expected: Mapping[str, Mapping[str, object]],
) -> None:
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("artifact manifest model must be a mapping")
    if model.get("dtype") != "bfloat16" or model.get("index") != MODEL_INDEX:
        raise ValueError("artifact model must use indexed BF16 Safetensors")
    tensor_count = _positive_int(model.get("tensor_count"), "tensor_count")
    shard_count = _positive_int(model.get("shards"), "shards")

    index = _read_json_object(root / MODEL_INDEX, "Safetensors index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != tensor_count:
        raise ValueError("Safetensors index has the wrong tensor count")
    if not all(isinstance(key, str) and key for key in weight_map):
        raise ValueError("Safetensors index has an invalid tensor name")
    shard_names = set(weight_map.values())
    if not all(
        isinstance(name, str)
        and name.endswith(".safetensors")
        and name in expected
        for name in shard_names
    ):
        raise ValueError("Safetensors index references an invalid shard")
    if len(shard_names) != shard_count:
        raise ValueError("Safetensors index has the wrong shard count")
    actual_shards = {
        relative for relative in expected if relative.endswith(".safetensors")
    }
    if actual_shards != shard_names:
        raise ValueError("artifact has an unreferenced Safetensors shard")

    metadata = index.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Safetensors index metadata must be a mapping")
    _positive_int(metadata.get("total_size"), "Safetensors total_size")


def _verify_config(
    root: Path,
    manifest: Mapping[str, Any],
    expected: Mapping[str, Mapping[str, object]],
) -> None:
    config = _read_json_object(root / "config.json", "artifact config")
    if "config_projection" in manifest:
        if manifest["config_projection"] != "model_tokenizer_only":
            raise ValueError("unsupported artifact config projection")
        if set(config) != {"model", "tokenizer"}:
            raise ValueError("public artifact config must contain model and tokenizer only")
    _validate_model_config(config)
    tokenizer = config.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValueError("artifact config must contain a tokenizer object")
    tokenizer_path = tokenizer.get("path")
    if not isinstance(tokenizer_path, str) or not tokenizer_path:
        raise ValueError("artifact tokenizer.path must be a non-empty string")
    relative_tokenizer = Path(tokenizer_path)
    if (
        relative_tokenizer.is_absolute()
        or ".." in relative_tokenizer.parts
        or tokenizer_path != relative_tokenizer.as_posix()
        or not relative_tokenizer.parts
        or relative_tokenizer.parts[0] != "tokenizer"
    ):
        raise ValueError("artifact tokenizer.path must point inside tokenizer/")
    tokenizer_files = {
        relative
        for relative in expected
        if Path(relative).parts and Path(relative).parts[0] == "tokenizer"
    }
    if not tokenizer_files:
        raise ValueError("artifact must contain at least one tokenizer payload")
    resolved_tokenizer = root / relative_tokenizer
    if not resolved_tokenizer.exists():
        raise ValueError("artifact tokenizer.path does not exist")
    if relative_tokenizer == Path("tokenizer"):
        if not resolved_tokenizer.is_dir():
            raise ValueError("artifact tokenizer.path must identify a directory")
    elif not resolved_tokenizer.is_file():
        raise ValueError("artifact tokenizer.path must identify a tokenizer file")


def verify_artifact(artifact_dir: str | Path) -> dict[str, Any]:
    """Verify the complete file set and hashes before any tensor is loaded."""
    root = Path(artifact_dir)
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(
            f"artifact directory must be a real directory, not a symlink: {root}"
        )
    manifest = _read_manifest(root)
    source = _verify_source(manifest)
    expected = _verify_file_records(manifest)
    _verify_documents(expected, source)
    _verify_file_set(root, expected)
    _verify_file_hashes(root, expected)
    _verify_index(root, manifest, expected)
    _verify_config(root, manifest, expected)
    return manifest


def load_artifact(
    artifact_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Verify once, then load the config and all weight shards on CPU.

    The config's ``tokenizer.path`` is resolved inside the artifact.
    """
    root = Path(artifact_dir)
    verify_artifact(root)
    config = _read_json_object(root / "config.json", "artifact config")
    relative = Path(config["tokenizer"]["path"])
    config["tokenizer"]["path"] = str((root / relative).resolve(strict=True))

    weight_map: dict[str, str] = _read_json_object(
        root / MODEL_INDEX, "Safetensors index"
    )["weight_map"]
    expected_by_shard: dict[str, set[str]] = {}
    for name, shard_name in weight_map.items():
        expected_by_shard.setdefault(shard_name, set()).add(name)
    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in sorted(expected_by_shard):
        shard = load_file(root / shard_name, device="cpu")
        if set(shard) != expected_by_shard[shard_name]:
            raise ValueError(f"Safetensors shard keys do not match index: {shard_name}")
        for name, tensor in shard.items():
            if name in state_dict:
                raise ValueError(f"duplicate tensor across Safetensors shards: {name}")
            if tensor.is_floating_point() and tensor.dtype is not torch.bfloat16:
                raise ValueError(f"artifact tensor is not BF16: {name}")
            state_dict[name] = tensor
    return config, state_dict


def _release_tensor(tensor: torch.Tensor) -> torch.Tensor:
    dtype = torch.bfloat16 if tensor.is_floating_point() else tensor.dtype
    return tensor.detach().to(device="cpu", dtype=dtype).contiguous()


def _release_nbytes(tensor: torch.Tensor) -> int:
    element_size = 2 if tensor.is_floating_point() else tensor.element_size()
    return tensor.numel() * element_size


def write_artifact(
    output_dir: str | Path,
    *,
    name: str,
    state_dict: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    tokenizer_dir: str | Path,
    source: Mapping[str, str],
    max_shard_bytes: int = MAX_SHARD_BYTES,
) -> dict[str, Any]:
    """Write a new artifact directory and return its verified manifest.

    Floating-point tensors are rounded to BF16. Shards take whole tensors in
    name order and stay within ``max_shard_bytes`` unless one tensor is larger.
    ``config`` holds the model and tokenizer sections; its tokenizer path is
    set to the copy of ``tokenizer_dir``. ``source`` needs ``checkpoint_id``.
    The artifact has no model card, LICENSE or NOTICE.
    """
    root = Path(output_dir)
    root.mkdir(parents=True)
    shutil.copytree(tokenizer_dir, root / "tokenizer")
    (root / "config.json").write_bytes(
        pretty_json_bytes(
            {**config, "tokenizer": {**config["tokenizer"], "path": "tokenizer"}}
        )
    )

    sizes = {key: _release_nbytes(state_dict[key]) for key in sorted(state_dict)}
    shards: list[list[str]] = [[]]
    shard_bytes = 0
    for key, size in sizes.items():
        if shards[-1] and shard_bytes + size > max_shard_bytes:
            shards.append([])
            shard_bytes = 0
        shards[-1].append(key)
        shard_bytes += size
    weight_map: dict[str, str] = {}
    for number, keys in enumerate(shards, start=1):
        shard_name = f"model-{number:05d}-of-{len(shards):05d}.safetensors"
        save_file(
            {key: _release_tensor(state_dict[key]) for key in keys},
            str(root / shard_name),
            metadata={"format": "xllm"},
        )
        weight_map.update(dict.fromkeys(keys, shard_name))
    (root / MODEL_INDEX).write_bytes(
        pretty_json_bytes(
            {"metadata": {"total_size": sum(sizes.values())}, "weight_map": weight_map}
        )
    )

    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "release": "paper-part1",
        "format": ARTIFACT_FORMAT,
        "config_projection": "model_tokenizer_only",
        "model": {
            "name": name,
            "dtype": "bfloat16",
            "index": MODEL_INDEX,
            "tensor_count": len(weight_map),
            "shards": len(shards),
        },
        "source": dict(source),
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ],
    }
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    (root / ARTIFACT_MANIFEST).write_bytes(pretty_json_bytes(manifest))
    return verify_artifact(root)
