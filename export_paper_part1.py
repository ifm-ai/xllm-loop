"""Export a checkpoint of train_paper_part1.py as a native artifact for eval_paper_part1.py:

    ENABLE_FLASH_ATTENTION_3=true python export_paper_part1.py --recipe dense_d28 \\
        --checkpoint /path/to/run/checkpoints/checkpoint_00080000 \\
        --tokenizer /path/to/part1-tokenizer --output /path/to/artifact

The checkpoint's model config must be the recipe's. The artifact holds BF16
Safetensors shards of at most 5 GiB, config.json, the tokenizer and
artifact_manifest.json (see `xllm/paper_part1/export.py`).
"""

from xllm.paper_part1.export import main


if __name__ == "__main__":
    raise SystemExit(main())
