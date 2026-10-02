"""Export a Part 1 training checkpoint as a native artifact.

A checkpoint of ``train_paper_part1.py`` holds the model as FSDP DCP shards in
the current parameter layout, with ``config.json`` next to them. The artifact
uses the released format: the recipe's model fields, the run's tokenizer
section and the tensors in BF16.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed.checkpoint as dcp

from xllm.checkpointing import sharded_model_dir
from xllm.config import ModelConf
from xllm.paper_part1.artifacts import write_artifact
from xllm.paper_part1.recipes import Recipe, get_recipe, recipe_names


def read_model_state(checkpoint_dir: str | Path) -> dict[str, torch.Tensor]:
    """Read a checkpoint's DCP model shards into full CPU tensors, keyed as saved."""
    reader = dcp.FileSystemReader(str(Path(checkpoint_dir) / sharded_model_dir(0)))
    metadata = reader.read_metadata().state_dict_metadata
    state = {
        key: torch.empty(entry.size, dtype=entry.properties.dtype)
        for key, entry in metadata.items()
    }
    dcp.load(state, storage_reader=reader, no_dist=True)
    return state


def _model_config(values: Mapping[str, Any]) -> dict[str, Any]:
    """Complete a model section with the current ModelConf defaults, in JSON types."""
    return json.loads(ModelConf.from_dict(dict(values)).to_json())


def export_checkpoint(
    recipe: Recipe,
    checkpoint_dir: str | Path,
    tokenizer_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write the artifact of one checkpoint and return its verified manifest."""
    if Path(output_dir).exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output_dir}")
    checkpoint = Path(checkpoint_dir).resolve(strict=True)
    config = json.loads((checkpoint / "config.json").read_bytes())
    if config.get("model_parallel_size") != 1:
        raise ValueError("only checkpoints with model-parallel size 1 can be exported")
    trained = _model_config(config["model"])
    expected = json.loads(recipe.model_config().to_json())
    mismatched = sorted(
        key for key in trained.keys() | expected.keys()
        if trained.get(key) != expected.get(key)
    )
    if mismatched:
        raise ValueError(
            f"checkpoint model config differs from recipe {recipe.name}: {mismatched}"
        )

    return write_artifact(
        output_dir,
        name=recipe.name,
        state_dict=read_model_state(checkpoint),
        config={"model": dict(recipe.model_fields), "tokenizer": config["tokenizer"]},
        tokenizer_dir=tokenizer_dir,
        source={"checkpoint_id": checkpoint.name},
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a train_paper_part1.py checkpoint as a native Part 1 artifact.",
    )
    parser.add_argument("--recipe", required=True, choices=recipe_names(),
                        help="recipe the checkpoint was trained with")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="checkpoint directory, e.g. <dump dir>/checkpoints/checkpoint_00080000")
    parser.add_argument("--tokenizer", required=True, type=Path,
                        help="tokenizer directory the run used; it is copied into the artifact")
    parser.add_argument("--output", required=True, type=Path,
                        help="new artifact directory")
    args = parser.parse_args(argv)
    manifest = export_checkpoint(
        get_recipe(args.recipe),
        args.checkpoint,
        args.tokenizer,
        args.output,
    )
    print(json.dumps({
        "output": str(args.output),
        "manifest_sha256": manifest["manifest_sha256"],
        "tensor_count": manifest["model"]["tensor_count"],
        "shards": manifest["model"]["shards"],
    }, sort_keys=True))
    return 0
