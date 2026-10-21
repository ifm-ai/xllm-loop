# Training data

## Pretraining mixture

Every Table 2, Table 3 and D4 recipe trains on the same weighted mixture as the
xLLM Part I release: the 42 sources below, in this order and with these weights,
tokenized with the Jais64k tokenizer instead of Part I's. IFM publishes 24 of
the sources in the form the paper trained on, as subsets of its K2 Horizon
datasets; the table links each subset at its pinned revision, and the
[Part I data guide](../paper-part1/data.md) lists the revisions and converts a
subset's Parquet shards into the loader's chunk files. Four sources,
`opencoder`, `arabic-shuffled`, `french` and `hindi` (12.1% of the weight),
cannot be made public at present.

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
| `nltk-web-high-randomized` | 0.0400 | [IFM/TxT360-v2 `web-high-nltk-qa`](https://huggingface.co/datasets/IFM/TxT360-v2/tree/63d6dbf4d469058a8c7868909be18a5ada487be1/web-high-nltk-qa) |
| `nemotron-cc-high` | 0.0300 | [nvidia/Nemotron-CC-v2 `High-Quality`](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2/tree/2669787c66d18601c0e91167fd520be6c1245865/High-Quality) |
| `nemotron-cc-high-synthetic` | 0.0500 | [nvidia/Nemotron-CC-v2 `High-Quality-Synthetic`](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2/tree/2669787c66d18601c0e91167fd520be6c1245865/High-Quality-Synthetic) |
| `web-high-medium` | 0.0430 | [IFM/TxT360-v2 `web-high-medium`](https://huggingface.co/datasets/IFM/TxT360-v2/tree/63d6dbf4d469058a8c7868909be18a5ada487be1/web-high-medium) |
| `nemotron-cc-high-medium` | 0.0200 | [nvidia/Nemotron-CC-v2 `Medium-High-Quality`](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2/tree/2669787c66d18601c0e91167fd520be6c1245865/Medium-High-Quality) |
| `nemotron-cc-high-medium-synthetic` | 0.0300 | [nvidia/Nemotron-CC-v2.1 `Medium-High-Quality-Synthetic`](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1/tree/ba6f2aaef7ada865bb08fc08640ca292150097db/Medium-High-Quality-Synthetic) |
| `nemotron-cc-2025` | 0.0300 | [nvidia/Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1/tree/ba6f2aaef7ada865bb08fc08640ca292150097db), seven subsets ([upstream sources](../paper-part1/data.md#upstream-sources)) |
| `nemotron-cc-diverse-qa` | 0.0300 | [nvidia/Nemotron-CC-v2 `Diverse-QA`](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2/tree/2669787c66d18601c0e91167fd520be6c1245865/Diverse-QA) |
| `txt360-qa` | 0.0300 | [IFM/TxT360-v2 `txt360-qa`](https://huggingface.co/datasets/IFM/TxT360-v2/tree/63d6dbf4d469058a8c7868909be18a5ada487be1/txt360-qa) |
| `opencoder` | 0.0800 | Not public |
| `nemotron-pretraining-code` | 0.0300 | [Upstream sources](../paper-part1/data.md#upstream-sources) |
| `nemotron-cc-code` | 0.0190 | [nvidia/Nemotron-CC-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1/tree/5c5bebce9f9a85c237d63b2afad516ee532f950f) |
| `stack-edu` | 0.0050 | [HuggingFaceTB/stack-edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu/tree/eeec5caac5cc3758a18f1d3ba4416837a9ba814c), file contents from Software Heritage ([upstream sources](../paper-part1/data.md#upstream-sources)) |
| `code_solutions_no_reasoning_8k` | 0.0350 | [IFM/Code-Reasoning `code-no-thinking`](https://huggingface.co/datasets/IFM/Code-Reasoning/tree/e971bd7801485b24bb217dcde9d46f4bad42ea00/code-no-thinking) |
| `code_solutions_no_reasoning_8k-hard` | 0.0270 | [IFM/Code-Reasoning `code-no-thinking-hard`](https://huggingface.co/datasets/IFM/Code-Reasoning/tree/e971bd7801485b24bb217dcde9d46f4bad42ea00/code-no-thinking-hard) |
| `code_solutions_with_reasoning_16k` | 0.0185 | [IFM/Code-Reasoning `code-thinking-v1`](https://huggingface.co/datasets/IFM/Code-Reasoning/tree/e971bd7801485b24bb217dcde9d46f4bad42ea00/code-thinking-v1) |
| `code_solutions_with_reasoning_16k-hard` | 0.0115 | [IFM/Code-Reasoning `code-thinking-hard`](https://huggingface.co/datasets/IFM/Code-Reasoning/tree/e971bd7801485b24bb217dcde9d46f4bad42ea00/code-thinking-hard) |
| `code_thinking-cleaned` | 0.0200 | [IFM/Code-Reasoning `code-thinking-v0`](https://huggingface.co/datasets/IFM/Code-Reasoning/tree/e971bd7801485b24bb217dcde9d46f4bad42ea00/code-thinking-v0) |
| `code_prompts_all_knobs_8k` | 0.0050 | [IFM/Code-Reasoning `code-task-synthesis`](https://huggingface.co/datasets/IFM/Code-Reasoning/tree/e971bd7801485b24bb217dcde9d46f4bad42ea00/code-task-synthesis) |
| `nemotron-cc-math` | 0.0120 | [nvidia/Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1/tree/397a2502f2028c659ba411a6c4935b464a7f03aa) `3` and `4plus` |
| `math-qwen-no-think` | 0.0400 | [IFM/Math-Reasoning `math-thinking-qwen`](https://huggingface.co/datasets/IFM/Math-Reasoning/tree/e249295774c3a66943e1483dbf53e0c811bc41dd/math-thinking-qwen) |
| `math-oss-cleaned` | 0.0500 | [IFM/Math-Reasoning `math-thinking-oss`](https://huggingface.co/datasets/IFM/Math-Reasoning/tree/e249295774c3a66943e1483dbf53e0c811bc41dd/math-thinking-oss) |
| `math-rewrite` | 0.0260 | [IFM/Math-Reasoning `math-rewrite`](https://huggingface.co/datasets/IFM/Math-Reasoning/tree/e249295774c3a66943e1483dbf53e0c811bc41dd/math-rewrite) |
| `math-dialogue` | 0.0100 | [IFM/Math-Reasoning `math-dialogue`](https://huggingface.co/datasets/IFM/Math-Reasoning/tree/e249295774c3a66943e1483dbf53e0c811bc41dd/math-dialogue) |
| `agentic-math-dialogue-cleaned` | 0.0050 | [IFM/Math-Reasoning `socratic-math-dialogue`](https://huggingface.co/datasets/IFM/Math-Reasoning/tree/e249295774c3a66943e1483dbf53e0c811bc41dd/socratic-math-dialogue) |
| `nemotron-sft-no-think` | 0.0175 | [Upstream sources](../paper-part1/data.md#upstream-sources) |
| `instruction_following-cleaned` | 0.0040 | [IFM/SFT-Reasoning `instruction-following`](https://huggingface.co/datasets/IFM/SFT-Reasoning/tree/4d4dbbe65e478a2ca179c4cce43b5e1f051aacd2/instruction-following) |
| `nemotron-pretraining-sft` | 0.0220 | [Upstream sources](../paper-part1/data.md#upstream-sources) |
| `3efforts-pretrain` | 0.0050 | [IFM/SFT-Reasoning `3efforts-pretrain`](https://huggingface.co/datasets/IFM/SFT-Reasoning/tree/4d4dbbe65e478a2ca179c4cce43b5e1f051aacd2/3efforts-pretrain) |
| `reasoning` | 0.0525 | [IFM/Pretrain-Behaviors `reasoning`](https://huggingface.co/datasets/IFM/Pretrain-Behaviors/tree/3345e13d7f3f6d0ecb5fdd67b37aed289f3191f5/reasoning) |
| `general` | 0.0300 | [IFM/Pretrain-Behaviors `general`](https://huggingface.co/datasets/IFM/Pretrain-Behaviors/tree/3345e13d7f3f6d0ecb5fdd67b37aed289f3191f5/general) |
| `planning` | 0.0410 | [IFM/Pretrain-Behaviors `planning`](https://huggingface.co/datasets/IFM/Pretrain-Behaviors/tree/3345e13d7f3f6d0ecb5fdd67b37aed289f3191f5/planning) |
| `ai` | 0.0155 | [IFM/Pretrain-Behaviors `data-science`](https://huggingface.co/datasets/IFM/Pretrain-Behaviors/tree/3345e13d7f3f6d0ecb5fdd67b37aed289f3191f5/data-science) |
| `games-8k` | 0.0070 | [IFM/Pretrain-Behaviors `games`](https://huggingface.co/datasets/IFM/Pretrain-Behaviors/tree/3345e13d7f3f6d0ecb5fdd67b37aed289f3191f5/games) |
| `other` | 0.0180 | [IFM/Pretrain-Behaviors `other`](https://huggingface.co/datasets/IFM/Pretrain-Behaviors/tree/3345e13d7f3f6d0ecb5fdd67b37aed289f3191f5/other) |
| `hq-rewrite` | 0.0200 | [IFM/Pretrain-Behaviors `format-rewrites`](https://huggingface.co/datasets/IFM/Pretrain-Behaviors/tree/3345e13d7f3f6d0ecb5fdd67b37aed289f3191f5/format-rewrites) |
| `codeio` | 0.0015 | [IFM/Code-Reasoning `codeio`](https://huggingface.co/datasets/IFM/Code-Reasoning/tree/e971bd7801485b24bb217dcde9d46f4bad42ea00/codeio) |
| `arabic-shuffled` | 0.0220 | Not public |
| `french` | 0.0150 | Not public |
| `hindi` | 0.0040 | Not public |
| `diverse-qa-multilingual` | 0.0140 | [nvidia/Nemotron-CC-v2 `Translated-Diverse-QA`](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2/tree/2669787c66d18601c0e91167fd520be6c1245865/Translated-Diverse-QA), ten languages ([upstream sources](../paper-part1/data.md#upstream-sources)) |
| `other-multilingual` | 0.0140 | [Upstream sources](../paper-part1/data.md#upstream-sources) |

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
