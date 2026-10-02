# Training data

Download the public TxT360 components at these revisions:

| Dataset | Revision | Input |
| --- | --- | --- |
| [IFM/TxT360-v2](https://huggingface.co/datasets/IFM/TxT360-v2/tree/63d6dbf4d469058a8c7868909be18a5ada487be1) | `63d6dbf4d469058a8c7868909be18a5ada487be1` | `txt360-qa` |
| [IFM/TxT360-3efforts](https://huggingface.co/datasets/IFM/TxT360-3efforts/tree/bfc4a082d11967cd7810fe0b773be87bf54fb32e) | `bfc4a082d11967cd7810fe0b773be87bf54fb32e` | high, medium and low effort |

Both use CC-BY-4.0. Prepare the downloaded Parquet or JSONL with
[prepare-data.py](prepare-data.py):

```bash
python release/paper-part1/prepare-data.py \
  --input /path/to/txt360-qa.parquet --output /path/to/txt360-qa/txt360-qa.chunk0.jsonl

python release/paper-part1/prepare-data.py --effort high \
  --input /path/to/chat-high.parquet --output /path/to/3efforts-pretrain/chat-high.chunk0.jsonl
```

Existing `text` stays unchanged. For conversation rows, pass the downloaded
split's effort, `--effort high`, `medium` or `low`; the corresponding reasoning
field is `think`, `think_fast` or `think_faster`. The fixed template emits one
sample per assistant turn, omits earlier reasoning and preserves supplied
system and tool context. It adds no shuffle or mixture reweighting.
`--max-rows 32` limits a preparation check to 32 input rows. The output path
must be new and named `<name>.chunk<N>.jsonl`, the file pattern the data loader
reads.

## Mixture

The training recipe uses these sources in the listed order. Weights sum to one.
Each source is a directory of `<name>.chunk<N>.jsonl` files with a `text` field
([data loader guide](../../xllm/data/README.md)).
[common-base.example.json](common-base.example.json) sets the base config's
`data` field to the 42 comma-separated `/path/to/source:weight:text:text`
entries in this order; replace the paths with your prepared directories.
Source names without a link identify prepared components of the training
mixture.

The base config packs the recipe's 8,192-token sequences with `bestfit` packing
and a 512-sequence buffer, as in the paper runs, which used an older xLLM data
loader. The current loader tokenizes, mixes and packs online in its own order,
so its training batches differ from the paper's.

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

Use [IFM/K2-Horizon-3.7B](https://huggingface.co/IFM/K2-Horizon-3.7B/tree/86a26d7bf092484faa047121d5816f037d4a8ffa)
at revision `86a26d7bf092484faa047121d5816f037d4a8ffa`.
Download `tokenizer.json`, `tokenizer_config.json` and
`special_tokens_map.json`, then run [materialize-tokenizer.py](materialize-tokenizer.py)
and point the base config's `tokenizer.path` at the new directory:

```bash
python release/paper-part1/materialize-tokenizer.py \
  --source-dir /path/to/IFM-K2-Horizon-3.7B-pretrain \
  --output-dir /path/to/part1-tokenizer
```

The transform removes automatic BOS insertion, retains the ByteLevel
post-processor and converts the BOS/EOS special-token objects to strings. The
output directory must be new. The data loader adds BOS and EOS around each
document itself. Every released checkpoint carries the prepared tokenizer in
`tokenizer/`, and `tokenizer.path` may point there instead. The tokenizer uses
Apache-2.0; see [NOTICE.weights](NOTICE.weights).
