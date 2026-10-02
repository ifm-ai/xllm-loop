# Training data

## Pretraining mixture

Every Table 2, Table 3 and D4 recipe trains on the same weighted mixture as the
xLLM Part I release: the 42 sources below, in this order and with these weights,
tokenized with the Jais64k tokenizer instead of Part I's. Source names without a
link identify prepared components of the mixture. The public TxT360 components
are `IFM/TxT360-v2` at revision `63d6dbf4d469058a8c7868909be18a5ada487be1`
(`txt360-qa`) and `IFM/TxT360-3efforts` at revision
`bfc4a082d11967cd7810fe0b773be87bf54fb32e` (`3efforts-pretrain`), both CC-BY-4.0.

Set the base config's `data` field to comma-separated
`/path/to/source:weight:text:text` entries, one per source, in the listed order
(see the [data loader guide](../../xllm/data/README.md) for the directory layout).
[base.example.json](base.example.json) shows two placeholder entries; the run
needs all 42. The paper packed 8,192-token sequences with best-fit packing and a
512-sequence buffer, and base.example.json sets the loader the same way. The
upstream loader tokenizes and packs online, so its sequences differ from the
paper's.

| Source | Weight | Public source |
| --- | ---: | --- |
| `nltk-web-high-randomized` | 0.0400 |  |
| `nemotron-cc-high` | 0.0300 | [Source](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) |
| `nemotron-cc-high-synthetic` | 0.0500 | [Source](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) |
| `web-high-medium` | 0.0430 |  |
| `nemotron-cc-high-medium` | 0.0200 | [Source](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) |
| `nemotron-cc-high-medium-synthetic` | 0.0300 | [Source](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1) |
| `nemotron-cc-2025` | 0.0300 | [Source](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1) |
| `nemotron-cc-diverse-qa` | 0.0300 | [Source](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) |
| `txt360-qa` | 0.0300 | [Source](https://huggingface.co/datasets/IFM/TxT360-v2/tree/63d6dbf4d469058a8c7868909be18a5ada487be1) |
| `opencoder` | 0.0800 | [Source](https://huggingface.co/collections/OpenCoder-LLM/opencoder-datasets-672e6db6a0fed24bd69ef1c2) |
| `nemotron-pretraining-code` | 0.0300 | [Source 1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v1), [Source 2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2) |
| `nemotron-cc-code` | 0.0190 | [Source](https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1) |
| `stack-edu` | 0.0050 | [Source](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) |
| `code_solutions_no_reasoning_8k` | 0.0350 |  |
| `code_solutions_no_reasoning_8k-hard` | 0.0270 |  |
| `code_solutions_with_reasoning_16k` | 0.0185 |  |
| `code_solutions_with_reasoning_16k-hard` | 0.0115 |  |
| `code_thinking-cleaned` | 0.0200 |  |
| `code_prompts_all_knobs_8k` | 0.0050 |  |
| `nemotron-cc-math` | 0.0120 | [Source](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) |
| `math-qwen-no-think` | 0.0400 |  |
| `math-oss-cleaned` | 0.0500 |  |
| `math-rewrite` | 0.0260 |  |
| `math-dialogue` | 0.0100 |  |
| `agentic-math-dialogue-cleaned` | 0.0050 |  |
| `nemotron-sft-no-think` | 0.0175 | [Source 1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1), [Source 2](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) |
| `instruction_following-cleaned` | 0.0040 |  |
| `nemotron-pretraining-sft` | 0.0220 | [Source 1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1), [Source 2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1) |
| `3efforts-pretrain` | 0.0050 | [Source](https://huggingface.co/datasets/IFM/TxT360-3efforts/tree/bfc4a082d11967cd7810fe0b773be87bf54fb32e) |
| `reasoning` | 0.0525 |  |
| `general` | 0.0300 |  |
| `planning` | 0.0410 |  |
| `ai` | 0.0155 |  |
| `games-8k` | 0.0070 |  |
| `other` | 0.0180 |  |
| `hq-rewrite` | 0.0200 |  |
| `codeio` | 0.0015 | [Source](https://huggingface.co/datasets/hkust-nlp/CodeIO-PyEdu-Reasoning) |
| `arabic-shuffled` | 0.0220 |  |
| `french` | 0.0150 |  |
| `hindi` | 0.0040 |  |
| `diverse-qa-multilingual` | 0.0140 | [Source](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) |
| `other-multilingual` | 0.0140 | [Source](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) |

## Tokenizer

The Jais64k tokenizer is a BPE tokenizer with a 64,000-token vocabulary that
the authors' team trained on a Jais-style data mix, with added chat and tool
special tokens (model vocabulary 64,256). Every Table 2, 3 and D4 artifact
carries it in `tokenizer/`; point the base config's `tokenizer.path` there. It
is distributed under Apache-2.0 ([NOTICE.weights](NOTICE.weights)).

## Distillation cache

The Table 4 students train on a token cache: int32 rows of 8,208 tokens in
round-robin shards, whose first 8,192 tokens the trainer reads.
[build-distill-cache.py](build-distill-cache.py) writes it from the same `data`
string with the xLLM data loader. `DATA` is the base config's `data` string,
all 42 entries:

```bash
DATA='/path/to/source-01:0.0400:text:text,...'  # the base config's full data string
SHARDS=16                                      # any number of shards; each can run as its own CPU job
TRAIN=$(( (2621568 + SHARDS - 1) / SHARDS ))   # training rows per shard, enough for S and M
VALID=$(( (128 + SHARDS - 1) / SHARDS ))       # validation rows per shard
for rank in $(seq 0 $((SHARDS - 1))); do
  python release/paper-part2/build-distill-cache.py --data "$DATA" --tokenizer /path/to/artifact/tokenizer \
    --output /path/to/cache --shard-count $SHARDS --train-rows-per-shard $TRAIN --valid-rows-per-shard $VALID \
    --shard-rank $rank
done
python release/paper-part2/build-distill-cache.py --output /path/to/cache --shard-count $SHARDS \
  --train-rows-per-shard $TRAIN --valid-rows-per-shard $VALID --finalize
```

A student reserves the first 128 rows (rows 0-63 calibrate the target energy)
and reads 256 rows per update after them: 655,488 rows at S (2,560 updates) and
2,621,568 at M (10,240 updates); both validate on 128 rows. The settings above
suffice for either scale. An interrupted shard resumes from its last saved row.
The paper's cache packed the same mixture differently, so its rows differ from
the builder's.
