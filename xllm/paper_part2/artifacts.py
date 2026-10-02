"""Native BF16 Safetensors artifacts of the Part 2 checkpoints.

The layout follows the Part I release: `config.json` with the model section,
`model.safetensors.index.json`, shards `model-00001-of-0000N.safetensors` of at
most 5 GiB with no tensor split across shards, and `artifact_manifest.json`
listing every file's size and SHA-256. A released artifact also holds its model
card `README.md`, `LICENSE` and `NOTICE`, and, unless it is a distilled student,
the tokenizer in `tokenizer/` (named in the config's tokenizer section). Tensors
are keyed as the model's state dict and are the training checkpoint's FP32
weights rounded to BF16, the values FSDP mixed precision computed with.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from xllm.config import ModelConf
from xllm.models.build import get_model_cls


ARTIFACT_FORMAT = "xllm-native-bf16-safetensors"
RELEASE = "paper-part2"
ARTIFACT_MANIFEST = "artifact_manifest.json"
MODEL_INDEX = "model.safetensors.index.json"
MAX_SHARD_BYTES = 5 * 1024**3
SCHEMA_VERSION = 1
DOCUMENTS = ("README.md", "LICENSE", "NOTICE")
TOKENIZER_DIR = "tokenizer"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
REQUIRED_PLACEHOLDER = "<REQUIRED:"  # an unfilled field of a document template
# What `hf download --local-dir` adds next to a Hub repository's own files.
HUB_DOWNLOAD_FILES = frozenset({".gitattributes"})
HUB_DOWNLOAD_DIRS = frozenset({".cache", ".cache/huggingface"})
HUB_DOWNLOAD_CACHE = ".cache/huggingface/"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_dcp_model_state(checkpoint_dir: str | Path) -> dict[str, torch.Tensor]:
    """Read a training checkpoint's FSDP DCP model shards into full CPU tensors, keyed as saved."""
    import torch.distributed.checkpoint as dcp

    reader = dcp.FileSystemReader(str(Path(checkpoint_dir) / "sharded_model.tp00"))
    metadata = reader.read_metadata().state_dict_metadata
    state = {key: torch.empty(entry.size, dtype=entry.properties.dtype) for key, entry in metadata.items()}
    dcp.load(state, storage_reader=reader, no_dist=True)
    return state


def release_state_dict(dcp_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Round floating-point tensors to BF16, keeping the keys."""
    return {key: (value.to(torch.bfloat16) if value.is_floating_point() else value).contiguous()
            for key, value in dcp_state.items()}


def _shards(state_dict: Mapping[str, torch.Tensor], max_shard_bytes: int) -> list[list[str]]:
    shards, current, size = [], [], 0
    for key in sorted(state_dict):
        nbytes = state_dict[key].numel() * state_dict[key].element_size()
        if current and size + nbytes > max_shard_bytes:
            shards.append(current)
            current, size = [], 0
        current.append(key)
        size += nbytes
    return shards + [current]


def _check_document(name: str, payload: bytes) -> None:
    if not payload.decode("utf-8").strip():
        raise ValueError(f"{name} is empty")
    if REQUIRED_PLACEHOLDER.encode() in payload:
        raise ValueError(f"{name} has an unfilled {REQUIRED_PLACEHOLDER} field")


def write_artifact(
    output_dir: str | Path,
    state_dict: Mapping[str, torch.Tensor],
    model_fields: Mapping[str, Any],
    source: Mapping[str, Any],
    max_shard_bytes: int = MAX_SHARD_BYTES,
    documents: Mapping[str, bytes] | None = None,
    tokenizer: str | Path | None = None,
    config_sections: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an artifact into an empty directory and return its manifest.

    `documents` maps names in `DOCUMENTS` to their bytes; `tokenizer` is a directory whose
    `TOKENIZER_FILES` are copied; `config_sections` are added to `config.json` next to `model`.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"artifact directory is not empty: {root}")
    ModelConf.from_dict(dict(model_fields))  # the config must build
    documents = dict(documents or {})
    if not set(documents) <= set(DOCUMENTS) or ("LICENSE" in documents) != ("NOTICE" in documents):
        raise ValueError(f"documents must be among {DOCUMENTS}, with LICENSE and NOTICE together")
    for name, payload in documents.items():
        _check_document(name, payload)
        (root / name).write_bytes(payload)
    config = {"model": dict(model_fields), **(config_sections or {})}
    if tokenizer is not None:
        (root / TOKENIZER_DIR).mkdir()
        for name in TOKENIZER_FILES:
            shutil.copyfile(Path(tokenizer) / name, root / TOKENIZER_DIR / name)
        config["tokenizer"] = {"type": "huggingface", "path": TOKENIZER_DIR}
    shards = _shards(state_dict, max_shard_bytes)
    weight_map = {}
    for index, keys in enumerate(shards, start=1):
        name = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        save_file({key: state_dict[key] for key in keys}, str(root / name))
        (root / name).chmod(0o644)  # Safetensors creates files readable by the owner only
        weight_map.update(dict.fromkeys(keys, name))
    total_size = sum(value.numel() * value.element_size() for value in state_dict.values())
    (root / MODEL_INDEX).write_bytes(_json_bytes({"metadata": {"total_size": total_size}, "weight_map": weight_map}))
    (root / "config.json").write_bytes(_json_bytes(config))
    files = [{"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path),
              "size_bytes": path.stat().st_size} for path in sorted(root.rglob("*")) if path.is_file()]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release": RELEASE,
        "format": ARTIFACT_FORMAT,
        "source": dict(source),
        "model": {"dtype": "bfloat16", "index": MODEL_INDEX, "tensor_count": len(state_dict),
                  "shards": len(shards)},
        "files": files,
    }
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    (root / ARTIFACT_MANIFEST).write_bytes(_json_bytes(manifest))
    return manifest


def export_checkpoint(
    checkpoint_dir: str | Path,
    model_fields: Mapping[str, Any],
    output_dir: str | Path,
    tokenizer: str | Path,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a training checkpoint's weights as an artifact with `model_fields` and the tokenizer.

    The weights are rounded to BF16 and must be exactly the parameters and buffers, with their
    names and shapes, of the model that `model_fields` builds.
    """
    state = release_state_dict(read_dcp_model_state(checkpoint_dir))
    config = ModelConf.from_dict(dict(model_fields))
    with torch.device("meta"):
        model = get_model_cls(config.arch, config)(config, None)
    shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    if {key: tuple(value.shape) for key, value in state.items()} != shapes:
        raise ValueError(f"{checkpoint_dir} does not hold the weights of this model config")
    return write_artifact(output_dir, state, model_fields, source, tokenizer=tokenizer)


def _hub_download_metadata(root: Path, path: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    return relative in HUB_DOWNLOAD_FILES or relative.startswith(HUB_DOWNLOAD_CACHE) or (
        relative in HUB_DOWNLOAD_DIRS and path.is_dir())


def verify_artifact(artifact_dir: str | Path) -> dict[str, Any]:
    """Check the manifest, the closed file set and every file's size and SHA-256.

    The metadata `hf download --local-dir` writes beside the files (`.gitattributes`
    and `.cache/huggingface/`) is ignored; anything else not in the manifest is rejected.
    """
    root = Path(artifact_dir)
    manifest = json.loads((root / ARTIFACT_MANIFEST).read_text())
    if manifest.get("manifest_sha256") != _manifest_digest(manifest):
        raise ValueError("artifact manifest SHA-256 mismatch")
    if (manifest.get("schema_version"), manifest.get("release"), manifest.get("format")) != (
            SCHEMA_VERSION, RELEASE, ARTIFACT_FORMAT):
        raise ValueError("not a Part 2 artifact manifest")
    records = {record["path"]: record for record in manifest["files"]}
    entries = [path for path in root.rglob("*") if not _hub_download_metadata(root, path)]
    present = {path.relative_to(root).as_posix() for path in entries if path.is_file()} - {ARTIFACT_MANIFEST}
    directories = {path.relative_to(root).as_posix() for path in entries if path.is_dir()}
    if any(path.is_symlink() for path in entries):
        raise ValueError(f"artifact holds symbolic links: {root}; load an `hf download --local-dir` directory, "
                         "not the Hub cache")
    if not directories <= {TOKENIZER_DIR}:
        raise ValueError(f"artifact holds unexpected directories: {sorted(directories - {TOKENIZER_DIR})}")
    if present != set(records):
        raise ValueError(f"artifact files differ from the manifest: missing {sorted(set(records) - present)}, "
                         f"unlisted {sorted(present - set(records))}")
    shards = {name for name in records if name.startswith("model-") and name.endswith(".safetensors")}
    layout = {"config.json", MODEL_INDEX, *DOCUMENTS, *(f"{TOKENIZER_DIR}/{name}" for name in TOKENIZER_FILES)}
    if not set(records) - shards <= layout or ("LICENSE" in records) != ("NOTICE" in records):
        raise ValueError("artifact files are outside the Part 2 layout")
    if (TOKENIZER_DIR in directories) != all(f"{TOKENIZER_DIR}/{name}" in records for name in TOKENIZER_FILES):
        raise ValueError("artifact tokenizer is incomplete")
    for name, record in records.items():
        path = root / name
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"artifact file does not match the manifest: {name}")
    for name in DOCUMENTS:
        if name in records:
            _check_document(name, (root / name).read_bytes())
    weight_map = json.loads((root / MODEL_INDEX).read_text())["weight_map"]
    if (len(weight_map), len(set(weight_map.values()))) != (manifest["model"]["tensor_count"],
                                                            manifest["model"]["shards"]) or \
            set(weight_map.values()) != shards:
        raise ValueError("Safetensors index does not match the manifest")
    return manifest


def artifact_config(artifact_dir: str | Path) -> dict[str, Any]:
    """Verify an artifact and return its full `config.json` (model and any other sections)."""
    verify_artifact(artifact_dir)
    return json.loads((Path(artifact_dir) / "config.json").read_text())


def open_artifact(artifact_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
    """Verify an artifact once, then return its manifest, full `config.json` and BF16 state dict on CPU."""
    root = Path(artifact_dir)
    manifest = verify_artifact(root)
    weight_map = json.loads((root / MODEL_INDEX).read_text())["weight_map"]
    state_dict = {}
    for shard in sorted(set(weight_map.values())):
        tensors = load_file(str(root / shard), device="cpu")
        if set(tensors) != {key for key, name in weight_map.items() if name == shard}:
            raise ValueError(f"shard keys do not match the index: {shard}")
        state_dict.update(tensors)
    return manifest, json.loads((root / "config.json").read_text()), state_dict


def load_artifact(artifact_dir: str | Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Verify an artifact, then return its model config fields and BF16 state dict on CPU."""
    _, config, state_dict = open_artifact(artifact_dir)
    return config["model"], state_dict


def build_model(model_fields: Mapping[str, Any], state_dict: Mapping[str, torch.Tensor], tokenizer):
    """The model of an artifact's fields and weights on the GPU in BF16, in eval mode and without gradients."""
    config = ModelConf.from_dict(dict(model_fields))
    with torch.device("cuda"):
        model = get_model_cls(config.arch, config)(config, tokenizer)
    model.load_state_dict(state_dict, strict=True)
    for parameter in model.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    return model.eval().requires_grad_(False)
