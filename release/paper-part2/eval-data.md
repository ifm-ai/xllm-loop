# Evaluation data

[prepare-eval-data.py](prepare-eval-data.py) downloads every public source at
a pinned revision, checks its SHA-256, rebuilds the paper's files and checks
each against the [protocol hashes](../../xllm/paper_part2/eval_protocol.py).
It writes the files and the `data.json` that
[eval_paper_part2.py](../../eval_paper_part2.py) reads:

```bash
python release/paper-part2/prepare-eval-data.py --output /path/to/eval-data
```

`--sources DIR` reads already downloaded files instead, from `DIR/<name>/<file>`:
`<name>` is one of `hellaswag`, `arc_easy`, `arc_challenge`, `lambada_openai`,
`piqa`, `openbookqa`, `sciq`, `gsm8k`, `wikitext`, `drop` and `mbpp_plus`, and
`<file>` is the last part of that source's URL in the script (several are named
`test-00000-of-00001.parquet`). A source missing there is downloaded. The
evaluator checks every input against the same hashes and refuses a mismatch.

## Choice tasks

The seven tasks keep their source rows and order, written as UTF-8 JSON with
sorted keys and no spaces, one row per line; the evaluator formats and scores
requests as lm-evaluation-harness v0.4.2 does
([choice_tasks.py](../../xllm/paper_part2/choice_tasks.py)).

| Task | Source | Rows |
| --- | --- | ---: |
| `lambada_openai` | [OpenAI GPT-2 LAMBADA test](https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl) | 5,153 |
| `hellaswag` | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag/tree/218ec52e09a7e7462a5400043bb9a69a41d06b76) validation, revision `218ec52e` | 10,042 |
| `piqa` | [PIQA train-dev archive](https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip): `dev.jsonl` joined with `dev-labels.lst` as `goal`, `sol1`, `sol2` and an integer `label` | 1,838 |
| `arc_easy` | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453) ARC-Easy test, revision `210d026f` | 2,376 |
| `arc_challenge` | allenai/ai2_arc ARC-Challenge test, revision `210d026f` | 1,172 |
| `openbookqa` | [OpenBookQA archive](https://ai2-public-datasets.s3.amazonaws.com/open-book-qa/OpenBookQA-V1-Sep2018.zip) `Data/Main/test.jsonl`, mapped to `id`, `question_stem`, `choices` (`text`, `label`) and `answerKey` | 500 |
| `sciq` | [allenai/sciq](https://huggingface.co/datasets/allenai/sciq/tree/2c94ad3e1aafab77146f384e23536f97a4849815) test, revision `2c94ad3e`; default JSON separators and source key order | 1,000 |

## Generation tasks

- **GSM8K.** [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866)
  main test (1,319 rows), revision `740312ad`, with the calculator annotations
  `<<...>>` removed from the answers. The evaluator builds the 8-shot prompts.
- **DROP.** The [official dataset](https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip):
  500 dev questions drawn by `random.Random(20260911).sample` from the sorted
  query ids, each prompted after three fixed dev demonstrations
  (`Text:`, `Question:`, `Answer:` blocks; demonstration passages with single
  spaces and none before `,` or `.`).
- **MBPP+.** [EvalPlus MBPP+ v0.2.0](https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0):
  100 tasks drawn the same way from the sorted task ids other than the
  demonstrations `Mbpp/2`, `Mbpp/3` and `Mbpp/4`, then sorted; the
  demonstrations' prompts and solutions precede each task's prompt, and
  `mbpp_plus.jsonl` holds the 100 tasks' tests.

GSM8K and DROP decode greedily up to 512 and 64 new tokens. MBPP+ draws 8
samples per task at temperature 0.8 up to 512 new tokens, each lane from its
own fixed uniforms, and reports pass@1 over base and plus tests with EvalPlus
0.3.1.

## Perplexity

- **WikiText.** [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3)
  `wikitext-103-raw-v1` test, revision `b08601e0`, its lines joined by newlines
  into one JSON row; the evaluator packs it into 38 sequences of 8,192 tokens.

The paper's Val PPL used a held-out validation set that is not released; the
public protocol reports Wiki PPL.
