# Training data

IFM publishes 24 of the 42 mixture sources in the form the paper trained on,
as subsets of these datasets:

| Dataset | Revision | License |
| --- | --- | --- |
| [IFM/TxT360-v2](https://huggingface.co/datasets/IFM/TxT360-v2/tree/63d6dbf4d469058a8c7868909be18a5ada487be1) | `63d6dbf4d469058a8c7868909be18a5ada487be1` | CC-BY-4.0 |
| [IFM/Code-Reasoning](https://huggingface.co/datasets/IFM/Code-Reasoning/tree/e971bd7801485b24bb217dcde9d46f4bad42ea00) | `e971bd7801485b24bb217dcde9d46f4bad42ea00` | Apache-2.0 |
| [IFM/Math-Reasoning](https://huggingface.co/datasets/IFM/Math-Reasoning/tree/e249295774c3a66943e1483dbf53e0c811bc41dd) | `e249295774c3a66943e1483dbf53e0c811bc41dd` | Apache-2.0 |
| [IFM/SFT-Reasoning](https://huggingface.co/datasets/IFM/SFT-Reasoning/tree/4d4dbbe65e478a2ca179c4cce43b5e1f051aacd2) | `4d4dbbe65e478a2ca179c4cce43b5e1f051aacd2` | Apache-2.0 |
| [IFM/Pretrain-Behaviors](https://huggingface.co/datasets/IFM/Pretrain-Behaviors/tree/3345e13d7f3f6d0ecb5fdd67b37aed289f3191f5) | `3345e13d7f3f6d0ecb5fdd67b37aed289f3191f5` | Apache-2.0 |

The mixture table names each source's subset. A subset's rows hold the
training text in `text`, in the paper's order. Download a subset's Parquet
shards and convert each with [prepare-data.py](prepare-data.py) into a numbered
chunk file, which keeps `text` unchanged:

```bash
i=0
for shard in /path/to/Code-Reasoning/code-no-thinking/*.parquet; do
  python release/paper-part1/prepare-data.py --input "$shard" \
    --output /path/to/code_solutions_no_reasoning_8k/code_solutions_no_reasoning_8k.chunk$i.jsonl
  i=$((i + 1))
done
```

`--max-rows 32` limits a preparation check to 32 input rows. The output path
must be new and named `<name>.chunk<N>.jsonl`, the file pattern the data loader
reads.

## Upstream sources

Fourteen sources come from public datasets of other groups at these revisions.
Several NVIDIA datasets require accepting their terms on Hugging Face before
download. Unless stated otherwise, a source keeps each row's `text` unchanged;
convert its Parquet shards as above, and put the chunk files of all its subsets
in one directory.

| Source | Dataset at revision | Rows |
| --- | --- | --- |
| `nemotron-cc-high`, `nemotron-cc-high-synthetic`, `nemotron-cc-high-medium`, `nemotron-cc-diverse-qa` | nvidia/Nemotron-CC-v2 `2669787c66d18601c0e91167fd520be6c1245865` | The subset named in the mixture table |
| `nemotron-cc-high-medium-synthetic` | nvidia/Nemotron-CC-v2.1 `ba6f2aaef7ada865bb08fc08640ca292150097db` | `Medium-High-Quality-Synthetic` |
| `nemotron-cc-2025` | nvidia/Nemotron-CC-v2.1 `ba6f2aaef7ada865bb08fc08640ca292150097db` | `High-Quality`, `High-Quality-DQA`, `High-Quality-Synthetic`, `High-Quality-Translated-To-English`, `High-Quality-Translated-To-English-Synthetic`, `Medium-High-Quality` and `Medium-High-Quality-Translated-To-English` |
| `diverse-qa-multilingual` | nvidia/Nemotron-CC-v2 `2669787c66d18601c0e91167fd520be6c1245865` | `Translated-Diverse-QA` in the languages `ar`, `cn`, `de`, `es`, `fr`, `it`, `ja`, `pt`, `ru` and `zh` |
| `nemotron-cc-code` | nvidia/Nemotron-CC-Code-v1 `5c5bebce9f9a85c237d63b2afad516ee532f950f` | All |
| `nemotron-cc-math` | nvidia/Nemotron-CC-Math-v1 `397a2502f2028c659ba411a6c4935b464a7f03aa` | `3` and `4plus`, not `4plus_MIND` |
| `nemotron-pretraining-code` | nvidia/Nemotron-Pretraining-Code-v1 `01393d3bd890ddc5c15da8a2c9afb57391277659` and nvidia/Nemotron-Pretraining-Code-v2 `7b1a453d43e3c0df9749834b04f1a9510c0f5e5b` | `Synthetic-Code` of v1 and the five `Synthetic-Code` subsets of v2 |
| `nemotron-pretraining-sft` | nvidia/Nemotron-Pretraining-SFT-v1 `3f1a5b884d0b890f02ead979ce698dc95debd953` and nvidia/Nemotron-Pretraining-Specialized-v1 `9ed3718b5f2ae29074c5e34e64115432b7c4320f` | `Code`, `General` and `MATH` of SFT-v1 and all six subsets of Specialized-v1, less rows removed by a filter that is not public |
| `nemotron-sft-no-think` | nvidia/Nemotron-Post-Training-Dataset-v1 `053ba262368bf80c5864d36524731271662be115` and nvidia/Llama-Nemotron-Post-Training-Dataset `ab2a40d258a6a4d9d4c277d702aeea445081766c`, with the empty chat prompts filled from lmsys/lmsys-chat-1m `200748d9d3cddcc9d782887541057aca0b18c5da` | Conversations rendered as one text each (system prompt, then `User:` and `Assistant:` turns), English only, without Llama generations and with the `<think>` tags removed. The language and generator filters are not public, so this source cannot be rebuilt exactly |
| `stack-edu` | HuggingFaceTB/stack-edu `eeec5caac5cc3758a18f1d3ba4416837a9ba814c` | All 15 languages. The rows hold `blob_id`s; download each file's content from the Software Heritage S3 bucket as the dataset card describes, and drop files that fail to download |
| `other-multilingual` | epfml/FineWeb2-HQ `c0c06e94fd3a44ae9e802b2b0fc533817601eb5e` and HuggingFaceFW/fineweb-2 `af9c13333eb981300149d5ca60a8e9d659b276b9` | The `train` splits of `arb_Arab`, `cmn_Hani`, `deu_Latn`, `fas_Arab`, `fra_Latn`, `ind_Latn`, `ita_Latn`, `jpn_Jpan`, `por_Latn`, `rus_Cyrl`, `spa_Latn` and `tur_Latn` from FineWeb2-HQ, and of `ben_Beng`, `fil_Latn`, `mal_Mlym` and `urd_Arab` from fineweb-2 |

## Mixture

The training recipe uses these sources in the listed order. Weights sum to one.
Each source is a directory of `<name>.chunk<N>.jsonl` files with a `text` field
([data loader guide](../../xllm/data/README.md)).
[common-base.example.json](common-base.example.json) sets the base config's
`data` field to the 42 comma-separated `/path/to/source:weight:text:text`
entries in this order; replace the paths with your prepared directories.
Four sources, `opencoder`, `arabic-shuffled`, `french` and `hindi` (12.1% of
the weight), cannot be made public at present.

The base config packs the recipe's 8,192-token sequences with `bestfit` packing
and a 512-sequence buffer, as in the paper runs, which used an older xLLM data
loader. The current loader tokenizes, mixes and packs online in its own order,
so its training batches differ from the paper's.

| Source | Weight | Public source |
| --- | ---: | --- |
| `nltk-web-high-randomized` | 0.0400 | [IFM/TxT360-v2 `web-high-nltk-qa`](https://huggingface.co/datasets/IFM/TxT360-v2/tree/63d6dbf4d469058a8c7868909be18a5ada487be1/web-high-nltk-qa) |
| `nemotron-cc-high` | 0.0300 | [nvidia/Nemotron-CC-v2 `High-Quality`](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2/tree/2669787c66d18601c0e91167fd520be6c1245865/High-Quality) |
| `nemotron-cc-high-synthetic` | 0.0500 | [nvidia/Nemotron-CC-v2 `High-Quality-Synthetic`](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2/tree/2669787c66d18601c0e91167fd520be6c1245865/High-Quality-Synthetic) |
| `web-high-medium` | 0.0430 | [IFM/TxT360-v2 `web-high-medium`](https://huggingface.co/datasets/IFM/TxT360-v2/tree/63d6dbf4d469058a8c7868909be18a5ada487be1/web-high-medium) |
| `nemotron-cc-high-medium` | 0.0200 | [nvidia/Nemotron-CC-v2 `Medium-High-Quality`](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2/tree/2669787c66d18601c0e91167fd520be6c1245865/Medium-High-Quality) |
| `nemotron-cc-high-medium-synthetic` | 0.0300 | [nvidia/Nemotron-CC-v2.1 `Medium-High-Quality-Synthetic`](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1/tree/ba6f2aaef7ada865bb08fc08640ca292150097db/Medium-High-Quality-Synthetic) |
| `nemotron-cc-2025` | 0.0300 | [nvidia/Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1/tree/ba6f2aaef7ada865bb08fc08640ca292150097db), seven subsets ([upstream sources](#upstream-sources)) |
| `nemotron-cc-diverse-qa` | 0.0300 | [nvidia/Nemotron-CC-v2 `Diverse-QA`](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2/tree/2669787c66d18601c0e91167fd520be6c1245865/Diverse-QA) |
| `txt360-qa` | 0.0300 | [IFM/TxT360-v2 `txt360-qa`](https://huggingface.co/datasets/IFM/TxT360-v2/tree/63d6dbf4d469058a8c7868909be18a5ada487be1/txt360-qa) |
| `opencoder` | 0.0800 | Not public |
| `nemotron-pretraining-code` | 0.0300 | [Upstream sources](#upstream-sources) |
| `nemotron-cc-code` | 0.0190 | [nvidia/Nemotron-CC-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1/tree/5c5bebce9f9a85c237d63b2afad516ee532f950f) |
| `stack-edu` | 0.0050 | [HuggingFaceTB/stack-edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu/tree/eeec5caac5cc3758a18f1d3ba4416837a9ba814c), file contents from Software Heritage ([upstream sources](#upstream-sources)) |
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
| `nemotron-sft-no-think` | 0.0175 | [Upstream sources](#upstream-sources) |
| `instruction_following-cleaned` | 0.0040 | [IFM/SFT-Reasoning `instruction-following`](https://huggingface.co/datasets/IFM/SFT-Reasoning/tree/4d4dbbe65e478a2ca179c4cce43b5e1f051aacd2/instruction-following) |
| `nemotron-pretraining-sft` | 0.0220 | [Upstream sources](#upstream-sources) |
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
| `diverse-qa-multilingual` | 0.0140 | [nvidia/Nemotron-CC-v2 `Translated-Diverse-QA`](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2/tree/2669787c66d18601c0e91167fd520be6c1245865/Translated-Diverse-QA), ten languages ([upstream sources](#upstream-sources)) |
| `other-multilingual` | 0.0140 | [Upstream sources](#upstream-sources) |

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
