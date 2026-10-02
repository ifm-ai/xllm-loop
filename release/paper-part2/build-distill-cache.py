"""Build the token cache of the Table 4 student trainer with the xLLM data loader.

The cache holds int32 rows of 8,208 tokens (`xllm.paper_part2.distill.TokenCache`);
row i is in shard i % N. Shard R is data-parallel rank R of N in the upstream
`MultiSourceDataLoader`, fed with the training `--data` string, tokenizer and
best-fit packing; its first `--train-rows-per-shard` rows are training rows and
the next `--valid-rows-per-shard` validation rows. Shards can be built as
separate CPU jobs and resume from their last saved row. `--finalize` checks
every shard and writes `train_manifest.json` and `valid_manifest.json`.

The trainer reserves the first 128 rows (rows 0-63 calibrate the target
energy) and reads 256 rows per update after them, so S needs 655,488 training
rows and M 2,621,568, and both use 128 validation rows. With N shards, give each
shard ceil(2,621,568 / N) training and ceil(128 / N) validation rows, which covers
both scales; rows beyond those are not read. The rows differ from those of the
paper's cache, which an earlier packer built from the same mixture.

    python release/paper-part2/build-distill-cache.py --data DATA --tokenizer TOKENIZER --output DIR \\
        --shard-count N --train-rows-per-shard TRAIN --valid-rows-per-shard VALID --shard-rank R
    python release/paper-part2/build-distill-cache.py --output DIR --shard-count N \\
        --train-rows-per-shard TRAIN --valid-rows-per-shard VALID --finalize
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from xllm.paper_part2.distill import TokenCache

ROW_TOKENS, LAYOUT = TokenCache.ROW_TOKENS, TokenCache.LAYOUT


def shard_name(rank: int, count: int) -> str:
    return f"shards/shard_{rank:05d}_of_{count:05d}.int32"


def build_shard(args) -> None:
    from xllm.config import TokenizerConf
    from xllm.data.dataloader import MultiSourceDataLoader
    from xllm.data.dataset_streamer.tokenizer import build_tokenizer

    path = args.output / shard_name(args.shard_rank, args.shard_count)
    partial, state_path = path.with_suffix(".partial"), path.with_suffix(".state.pt")
    if path.exists():
        raise FileExistsError(f"{path} is complete")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = args.train_rows_per_shard + args.valid_rows_per_shard
    loader = MultiSourceDataLoader(
        build_tokenizer(TokenizerConf(type="huggingface", path=args.tokenizer)), data_mix_str=args.data,
        seq_len=ROW_TOKENS, batch_size=args.loader_batch, num_buffered_seq=args.buffer_size,
        world_rank=args.shard_rank, world_size=args.shard_count, packing_type="bestfit",
        num_workers=args.num_workers, skip_long_docs=True)
    done = 0
    if state_path.exists():
        saved = torch.load(state_path, weights_only=False)
        done = saved["rows"]
        loader.set_state(saved["loader"])
    with open(partial, "r+b" if done else "wb") as stream:
        stream.truncate(done * ROW_TOKENS * 4)
        stream.seek(done * ROW_TOKENS * 4)
        try:
            while done < rows:
                x = next(loader).x[: rows - done]
                stream.write(np.ascontiguousarray(x, dtype="<i4").tobytes())
                done += len(x)
                if done % args.checkpoint_every < args.loader_batch or done == rows:
                    stream.flush()
                    torch.save({"rows": done, "loader": loader.get_state()}, state_path.with_suffix(".tmp"))
                    state_path.with_suffix(".tmp").replace(state_path)
        finally:
            loader.close()
    partial.replace(path)
    state_path.unlink()
    print(f"{path}: {rows} rows", flush=True)


def finalize(args) -> None:
    count, train, valid = args.shard_count, args.train_rows_per_shard, args.valid_rows_per_shard
    shards = []
    for rank in range(count):
        path = args.output / shard_name(rank, count)
        if path.stat().st_size != (train + valid) * ROW_TOKENS * 4:
            raise ValueError(f"{path} is not a complete shard")
        shards.append(path.relative_to(args.output).as_posix())
    for split, offset, rows in (("train", 0, train), ("valid", train, valid)):
        manifest = dict(split=split, dtype="int32", layout=LAYOUT, sequence_tokens=ROW_TOKENS, rows=rows * count,
                        shard_count=count, shards=[dict(path=path, row_offset=offset, rows=rows,
                                                        stored_rows=train + valid) for path in shards])
        (args.output / f"{split}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.output}: {train * count} training and {valid * count} validation rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--train-rows-per-shard", required=True, type=int)
    parser.add_argument("--valid-rows-per-shard", required=True, type=int)
    parser.add_argument("--shard-rank", type=int)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--data", help="the training data string, path:weight:json_key:source_format,...")
    parser.add_argument("--tokenizer", help="the Jais64k tokenizer directory")
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--loader-batch", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=512)
    args = parser.parse_args()
    if args.finalize:
        finalize(args)
    elif args.shard_rank is None or not args.data or not args.tokenizer:
        parser.error("building a shard needs --shard-rank, --data and --tokenizer")
    else:
        build_shard(args)


if __name__ == "__main__":
    main()
