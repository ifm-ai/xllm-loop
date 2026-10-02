"""Export a train_paper_part2.py checkpoint as an artifact for eval_paper_part2.py and distill_paper_part2.py.

The artifact holds the checkpoint's model weights in BF16 with the recipe's
model config and the tokenizer, in the layout of the released artifacts. The
checkpoint's saved model config must match the recipe.

    python export_paper_part2.py --recipe s_learned_entropy0p01 \\
        --checkpoint /path/to/run/checkpoints/checkpoint_00005120 --tokenizer /path/to/tokenizer \\
        --output /path/to/artifact
"""

import argparse
import dataclasses
import json
from pathlib import Path

from xllm.paper_part2.artifacts import export_checkpoint
from xllm.paper_part2.recipes import get_recipe, recipe_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True, choices=recipe_names())
    parser.add_argument("--checkpoint", required=True, type=Path, help="a checkpoint directory of the run")
    parser.add_argument("--tokenizer", required=True, type=Path, help="the tokenizer directory the run trained with")
    parser.add_argument("--output", required=True, type=Path, help="a new or empty directory")
    args = parser.parse_args()
    recipe = get_recipe(args.recipe)
    config = args.checkpoint / "config.json"
    differing = recipe.differing_fields(json.loads(config.read_text())["model"])
    if differing:
        parser.error(f"{args.checkpoint} was not trained with {args.recipe}: {', '.join(differing)} differ")
    source = {"recipe": args.recipe, "paper_row": recipe.table_row, "checkpoint": args.checkpoint.resolve().name}
    manifest = export_checkpoint(args.checkpoint, dataclasses.asdict(recipe.model_config()), args.output,
                                 args.tokenizer, source)
    print(f"{args.output}: {manifest['model']['tensor_count']} tensors in {manifest['model']['shards']} shard(s)")


if __name__ == "__main__":
    main()
