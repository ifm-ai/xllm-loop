"""Train a Part 1 recipe to one target from a JSON base config of caller-owned settings.

The base config holds data, tokenizer, logging, checkpoint retention, evaluation
and gradient-accumulation settings; the recipe supplies the model, optimizer, schedule, stop
step, global batch, seed and numerics (see `xllm/paper_part1/training.py`).
The LR schedule always spans the 500B horizon; `--target` only sets where
training stops (1tpp, 20tpp, 336b or 500b).

    ENABLE_FLASH_ATTENTION_3=true python train_paper_part1.py --recipe dense_split_hl_x2 --target 1tpp \\
        --base-config base.json --dump-dir /path/to/run
"""

import argparse
import json
from pathlib import Path

from train import main as run_training
from xllm.logger import initialize_logger
from xllm.paper_part1.recipes import get_recipe, recipe_names, target_names
from xllm.paper_part1.training import training_config
from xllm.utils import log_host, setup_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True, choices=recipe_names(), help="Part 1 recipe to train")
    parser.add_argument("--target", required=True, choices=target_names(), help="where training stops")
    parser.add_argument("--base-config", required=True, type=Path,
                        help="JSON object with data, tokenizer, logging, checkpoint and accumulation settings")
    parser.add_argument("--dump-dir", required=True,
                        help="run directory for checkpoints and logs; rerunning resumes from it")
    parser.add_argument("--print-config", action="store_true", help="print the full config and exit")
    args = parser.parse_args()
    base = json.loads(args.base_config.read_text())
    if not isinstance(base, dict):
        parser.error(f"--base-config must contain a JSON object: {args.base_config}")
    cfg = training_config(get_recipe(args.recipe), args.target, base, args.dump_dir)
    if args.print_config:
        print(cfg.to_json())
        return
    initialize_logger()
    setup_env()
    log_host()
    run_training(cfg)


if __name__ == "__main__":
    main()
