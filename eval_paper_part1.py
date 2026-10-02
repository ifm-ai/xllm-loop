"""Evaluate a native XLLM Part 1 artifact, e.g. on the full paper protocol:

    ENABLE_FLASH_ATTENTION_3=true python eval_paper_part1.py \\
        --artifact /path/to/artifact --tasks-root /path/to/composite-eval-root \\
        --output /path/to/evaluation-output --protocol paper-part1
"""

from xllm.paper_part1.eval_native import main


if __name__ == "__main__":
    raise SystemExit(main())
