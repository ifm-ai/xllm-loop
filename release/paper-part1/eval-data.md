# Evaluation data

Prepare each task from the sources below, then assemble the files with
[materialize-eval-root.py](materialize-eval-root.py).
The [task definitions](../../xllm/paper_part1/eval_tasks.py) select file names;
the [paper protocol](../../xllm/paper_part1/eval_protocol.py) sets prompt limits,
generation limits, metrics and expected item counts.

Every file is JSONL: one JSON object per line. The evaluator reads 129 files,
67 evaluation files and 62 few-shot files. "Output" lists them for each task
with the fields to write; the evaluator ignores `dataset`, `question_id` and
`test_setup_code`, and any other fields a row carries.

## arc_challenge

Source: [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453).
Revision: `210d026faf9955653af8916fad021475a3f00453`.

Input files: `ARC-Challenge/test-00000-of-00001.parquet`.

Output: `arc_challenge/test.jsonl`, 1,165 rows with `question`, `choices` (an
object with keys `A` to `D`) and `label` (the correct letter).

1. read the ARC-Challenge test split in source order
2. retain the 1165 rows having exactly four choices
3. map choices by source position to A through D and map the answer by its source position
4. write Python json.dumps with ensure_ascii=false and one LF per row

## hellaswag

Source: [rowanz/hellaswag](https://github.com/rowanz/hellaswag/tree/6774d74db0a963013d28bd9323c32de8dd506038).
Revision: `6774d74db0a963013d28bd9323c32de8dd506038`.

Input files: `data/hellaswag_val.jsonl`.

Output: `hellaswag/val.jsonl`, 10,042 rows; the evaluator reads
`activity_label`, `ctx_a`, `ctx_b`, `endings` (four strings) and `label`
(0 to 3).

1. copy the validation JSONL bytes unchanged to `hellaswag/val.jsonl`

## mmlu

Source: [cais/mmlu](https://huggingface.co/datasets/cais/mmlu/tree/c30699e8356da336a370243923dbaf21066bb9fe).
Revision: `c30699e8356da336a370243923dbaf21066bb9fe`.

Input files: `all/dev-00000-of-00001.parquet`, `all/test-00000-of-00001.parquet`.

Output: `mmlu/<subject>/dev.jsonl` (5 rows each, the few-shot examples in file
order) and `mmlu/<subject>/test.jsonl` (14,042 rows in total) for each of the
57 subjects in `MMLU_TASKS_DOMAINS` of
[tasks/mmlu.py](../../xllm/paper_part1/tasks/mmlu.py), with `question`,
`choices` (an object with keys `A` to `D`), `answer` (the correct letter) and
`dataset` (`<subject>_dev` or `<subject>_test`).

1. partition dev and test rows by the 57 source subject names, keeping source order
2. map the integer answer to A through D and choices to an A through D object
3. write question, answer, choices, and dataset using Python json.dumps with ensure_ascii=false and one LF per row

## mmlu_pro

Source: [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/30527804ea8854662078e457808040d872ecdf29).
Revision: `30527804ea8854662078e457808040d872ecdf29`.

Input files: `default/test-00000-of-00001.parquet`, `default/validation-00000-of-00001.parquet`.

Output: `mmlu_pro/test.jsonl` (12,032 rows) and `mmlu_pro/prompt.jsonl`
(70 rows, five per category), both with the source fields. The evaluator reads
`question`, `options` (lettered A, B, ... by position), `category`, and
`answer_index` in test rows or `answer` (a letter) in prompt rows. Each test
row takes the prompt rows of its category as few-shot examples.

1. preserve all 12032 test objects and source order
2. build five validation demonstrations for each of 14 categories and remove the fixed A-colon-space prefix from cot_content
3. write compact ASCII JSON, escape slash as backslash-slash, and terminate every row with LF

## bbh_cot

Source: [suzgunmirac/BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard/tree/9ee07bd481feebf959a6b59d61ea57bdcf30964d).
Revision: `9ee07bd481feebf959a6b59d61ea57bdcf30964d`.

Input files: `bbh/*.json and cot-prompts/*.txt`.

Output: `bbh_cot/test.jsonl` (6,511 rows) and `bbh_cot/prompt.jsonl` (81 rows,
three per category), with `input`, `target` (the exact-match answer),
`category` (the BBH task name), `description` (the task's prompt header),
`explaination` (so spelled; the chain of thought in prompt rows, empty in test
rows) and `question_id`. Each test row takes the prompt rows of its category
as few-shot examples.

1. read all 27 task files and 27 chain-of-thought prompt files
2. emit all 6511 test examples with the prompt header as description and question_id 0 through 6510
3. emit three demonstrations per category, strip the fixed Let's-think-step-by-step prefix, and assign question_id 6511 through 6591
4. preserve category order when assigning question_id; write compact ASCII JSON with escaped slashes and LF framing

## tqa

Source: [TriviaQA unfiltered plus FiD/DPR split identities](https://github.com/facebookresearch/FiD/tree/fe769f30e3714e22476910ee39ea0054dd7921de).
Revision: `FiD fe769f30e3714e22476910ee39ea0054dd7921de`.

Input files: `triviaqa-unfiltered.tar.gz`, `FiD/dataindex.tar.gz`, `DPR/trivia-train.qa.csv.gz`, `DPR/trivia-test.qa.csv.gz`.

Output: `tqa/test.jsonl` (11,313 rows) and `tqa/train.jsonl` (78,785 rows;
its first five rows are the few-shot examples), with `question`, `answers`
(the accepted answers) and `target`.

1. select source records in the exact FiD TQA.train and TQA.test index order
2. write Question as question, Answer.Aliases as answers, and Answer.Value as target
3. title-case an all-uppercase target; for the 292 still-uppercase train targets only, lowercase after title-casing
4. write Python json.dumps defaults with one LF per row

## drop

Source: [ucinlp/drop](https://huggingface.co/datasets/ucinlp/drop/tree/95cda593fae71b60b5b19f82de3fcf3298c1239c).
Revision: `95cda593fae71b60b5b19f82de3fcf3298c1239c`.

Input files: `data/validation-00000-of-00001.parquet`.

Output: `drop/test.jsonl`, 9,535 rows with the source fields; the evaluator
reads `passage`, `question` and `answers_spans.spans`. The three few-shot
examples are part of the evaluator.

1. preserve the 9535 validation rows, schema, and order in `drop/test.jsonl`
2. write compact ASCII JSON, escape slash as backslash-slash, and terminate every row with LF

## gsm8k

Source: [openai/grade-school-math](https://github.com/openai/grade-school-math/tree/b0bb162abedc65e1fdd8e93ed090fd7598ee68bc).
Revision: `b0bb162abedc65e1fdd8e93ed090fd7598ee68bc`.

Input files: `grade_school_math/data/test.jsonl`.

Output: `gsm8k/test.jsonl`, 1,319 rows with `question` and `answer` (the
solution, ending in `#### <final answer>`). The eight few-shot examples are
part of the evaluator.

1. preserve all 1319 test rows and source order
2. remove calculator annotations delimited by double angle brackets from answer
3. write Python json.dumps defaults with one LF per row

## math

Source: [HuggingFaceH4/MATH-500 and EleutherAI/hendrycks_math](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/2343f79f3640c0f795bbf4e3234396cfe9266d0f).
Revision: `MATH-500 2343f79f3640c0f795bbf4e3234396cfe9266d0f; hendrycks_math 21a5633873b6a120296cce3e2df9d5550074f4a3`.

Input files: `HuggingFaceH4/MATH-500/test.jsonl`; the train split of each of
the seven EleutherAI/hendrycks_math configs, `<config>/train-00000-of-00001.parquet`
for `algebra`, `counting_and_probability`, `geometry`, `intermediate_algebra`,
`number_theory`, `prealgebra` and `precalculus`;
[the row-order table](eval/math-train-row-order.json); the Part I tokenizer.

Output: `math/test.jsonl` (500 rows with the MATH-500 fields) and
`math/train.jsonl` (7,500 rows with `problem`, `level`, `type` and `solution`).
The evaluator reads `problem` and `solution`; the gold answer is the last
`\boxed{...}` in `solution`. Rows 6374, 3321, 6476 and 6013 (counting from 0)
of `train.jsonl` are the four few-shot examples.

1. for each config in the table's `category_order`, emit its train rows in the order of `source_row_indices_by_config[config]`; write problem, level, type and solution with Python json.dumps defaults and one LF per row (train rows 6374, 3321, 6476 and 6013 come from algebra row 1708, precalculus row 344, algebra row 1076 and algebra row 600)
2. partition MATH-500 source row indices by index modulo eight
3. within each partition stable-sort by the full four-shot prompt length under the frozen Part I tokenizer with BOS enabled and EOS disabled
4. concatenate partitions zero through seven and write JSON with sorted keys and one LF per row

## human_eval_plus

Source: [openai/human-eval plus EvalPlus](https://github.com/openai/human-eval/tree/463c980b59e818ace59f6f9803cd92c749ceae61).
Revision: `human-eval 463c980b59e818ace59f6f9803cd92c749ceae61; EvalPlus v0.3.1`.

Input files: `data/HumanEval.jsonl.gz`.

Output: `human_eval/HumanEval.jsonl`, 164 rows with `task_id`, `prompt`,
`canonical_solution`, `test` and `entry_point`.

1. gzip-decompress HumanEval.jsonl without changing the JSONL bytes
2. score with the HumanEval+ dataset included in EvalPlus 0.3.1

## mbpp_plus

Source: [google-research MBPP plus EvalPlus](https://github.com/evalplus/evalplus/tree/v0.3.1).
Revision: `google-research f82046ba5aabbbb427dbfd38a254d26bff08b533; EvalPlus v0.3.1 and MBPP+ v0.2.0`.

Input files: `google-research/mbpp/mbpp.jsonl`, `EvalPlus/MbppPlus.jsonl.gz`.

Output: `mbpp_plus/mbpp_prompting.jsonl` (10 rows; rows 1 to 3, counting from
0, are the three few-shot examples) and `mbpp_plus/mbpp_test.jsonl`
(378 rows), with `task_id`, `text` (the task statement), `code` (a reference
solution), `test_list` (assert statements shown in the prompt),
`challenge_test_list` and `test_setup_code`.

1. write the first ten Google MBPP records as compact ASCII JSON in the source key order for the prompt file
2. decompress EvalPlus MBPP+ v0.2.0 and sort lexicographically by task_id; for each row write task_id unchanged, text as prompt.strip(), code as canonical_solution.strip() plus a newline, test_list as the lines of assertion that start with assert, challenge_test_list as an empty list and test_setup_code as an empty string, using Python json.dumps with sort_keys=True and one LF per row
3. score with the MBPP+ dataset included in EvalPlus 0.3.1

## Directory layout

The assembled directory has one directory per task: `arc_challenge`,
`hellaswag`, `mmlu` (one subdirectory per subject), `mmlu_pro`, `bbh_cot`,
`tqa`, `drop`, `gsm8k`, `math`, `human_eval` (for `human_eval_plus`) and
`mbpp_plus`; [task_paths.py](../../xllm/paper_part1/tasks/task_paths.py) maps
task names to them. It contains JSONL data only. Evaluation derives cache
identities from the files it actually reads.
