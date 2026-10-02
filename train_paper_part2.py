"""Train a Part 2 recipe from a JSON base config of caller-owned settings.

The base config holds data, tokenizer, logging, checkpoint retention,
evaluation and gradient-accumulation settings; the recipe supplies the model,
optimizer, schedule, the global batch of 512 sequences, the seed and the
switches the paper runs fixed (see `xllm/paper_part2/training.py`).

    ENABLE_FLASH_ATTENTION_3=true python train_paper_part2.py --recipe s_fixed_pln5 \\
        --base-config base.json --dump-dir /path/to/run
"""

import argparse
import json
from pathlib import Path

from train import main as run_training
from xllm.logger import initialize_logger
from xllm.paper_part2.recipes import get_recipe, recipe_names
from xllm.paper_part2.training import training_config
from xllm.utils import log_host, setup_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True, choices=recipe_names())
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--print-config", action="store_true", help="print the full config and exit")
    args = parser.parse_args()
    cfg = training_config(get_recipe(args.recipe), json.loads(args.base_config.read_text()), args.dump_dir)
    if args.print_config:
        print(cfg.to_json())
        return
    initialize_logger()
    setup_env()
    log_host()
    run_training(cfg)


if __name__ == "__main__":
    main()
