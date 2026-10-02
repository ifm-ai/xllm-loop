# Part I: Topology, Input Injection, Recurrent-State Organization

Paper: *Towards Looped Models Done Right, Part I: Topology, Input Injection, Recurrent-State Organization* ([blog post](https://huskydoge.github.io/husky-blog/posts/recursive_models/towards-looped-models-done-right/)).

This directory has the code and instructions for the Part I experiments: the
seventeen Dense and MoE recipes, native inference with the released checkpoints
and the paper evaluation protocol. Some training data sources are not public
(see the [data guide](data.md)). Every command below runs from the repository
root.

## Environment

The release was checked with Python 3.12, PyTorch 2.11.0+cu128 and
FlashAttention 3.0.0; follow the [installation guide](../../README.md#installation).
Set `ENABLE_FLASH_ATTENTION_3=true` for every command. The paper itself ran on
PyTorch 2.8.0+cu128 with an earlier version of xLLM, so numerical results can
differ slightly from the paper's.

## Recipes

[recipes.py](../../xllm/paper_part1/recipes.py) defines each model, optimizer,
seed, scheduler and training geometry.

| Family | Recipe names |
|---|---|
| Dense feedforward controls | `dense_d28`, `dense_d112` |
| Dense Ouro and middle loop | `dense_ouro`, `dense_middle` |
| Dense injection variants | `dense_ouro_raw_injection`, `dense_middle_prelude_injection` |
| Dense Huginn and H/L state variants | `dense_huginn`, `dense_shared_hl`, `dense_split_hl`, `dense_split_hl_x2` |
| MoE feedforward controls | `moe_ff112`, `moe_ff112_prelude_injection` |
| MoE Ouro and middle loop | `moe_ouro`, `moe_middle`, `moe_middle_input_injection` |
| MoE Huginn and shared H/L | `moe_huginn`, `moe_shared_hl` |

`dense_split_hl` applies each six-layer H/L module once per state update;
`dense_split_hl_x2` applies it twice. Both use eight state updates, with input
injection and cross-state addition once per update. They share the same
parameter count and checkpoint tensor layout. The checkpoint config records
`huginn_split_module_repeats`; its default is `1`.

Every recipe uses width 1,536, BF16, seed 1, AdamW (0.9, 0.98, weight decay
0.1) with 200 warmup steps and cosine decay to 0.01x of the peak over 119,210
steps, and a global batch of 512 x 8,192 tokens (4,194,304 tokens per step);
recipes.py holds the learning rate.

Choose `1tpp`, `20tpp`, `336b` or `500b` as the training target. The paper uses
`336b` for Dense split H/L x2:

| Target | Optimizer stop | Training tokens |
|---|---:|---:|
| `1tpp` | `ceil(activated_parameters / 4194304)` | Stop step times 4,194,304 |
| `20tpp` | `ceil(20 * activated_parameters / 4194304)` | Stop step times 4,194,304 |
| `336b` | 80,000 | 335,544,320,000 |
| `500b` | 119,210 | 500,002,979,840 |

Every target uses the same 119,210-step scheduler. TPP divides processed tokens
by activated parameters, including embeddings, routers, shared weights and
active experts. Recurrent weights count once. The checkpoint labels are rounded
token budgets.

## Train

Prepare the training mixture and tokenizer with the [data guide](data.md). Copy
[common-base.example.json](common-base.example.json), which lists the 42 mixture
sources in order, and replace each `/absolute/path/to/` with your own paths. The
base config owns data, tokenizer, logging, checkpoint retention and gradient
accumulation; the recipe owns the model, optimizer, schedule, stop step, global
batch, seed and numerics. The launcher rejects a base config that sets a recipe
field or an unknown field. Preview the full config, then launch on your ranks:

```bash
ENABLE_FLASH_ATTENTION_3=true python train_paper_part1.py --recipe dense_d28 --target 336b \
  --base-config base.json --dump-dir /path/to/run --print-config
ENABLE_FLASH_ATTENTION_3=true torchrun --nnodes N --nproc-per-node G ... train_paper_part1.py \
  --recipe dense_d28 --target 336b --base-config base.json --dump-dir /path/to/run
```

Under Slurm, `srun` with one task per GPU also works. The global batch is 512
sequences of 8,192 tokens: the ranks, the local batch and
`gradient_accumulation_steps` (base config, default 1) multiply to 512, and the
local batch follows from the other two and must fit in GPU memory. The rank
count and the accumulation change the data order. The `20tpp` target stops at:

| Recipe | `20tpp` stop step |
|---|---:|
| `dense_d28` | 7,137 |
| `dense_d112` | 17,533 |
| `dense_ouro` | 7,137 |
| `dense_middle` | 7,137 |
| `dense_ouro_raw_injection` | 7,148 |
| `dense_middle_prelude_injection` | 7,148 |
| `dense_huginn` | 7,148 |
| `dense_shared_hl` | 7,148 |
| `dense_split_hl` | 7,148 |
| `dense_split_hl_x2` | 7,148 |
| `moe_ff112` | 18,814 |
| `moe_ff112_prelude_injection` | 18,825 |
| `moe_ouro` | 7,457 |
| `moe_middle` | 7,457 |
| `moe_middle_input_injection` | 7,469 |
| `moe_huginn` | 7,469 |
| `moe_shared_hl` | 7,469 |

Rerunning the same command on the same number of ranks resumes from the latest
checkpoint in the dump directory, including the data loader position. Training
data order follows the current xLLM data loader, while the paper runs used an
older loader, so a re-run reproduces each recipe's model, optimizer, schedule
and global batch but not the paper's exact sequence of training batches.
Released checkpoints contain BF16 weights without optimizer or data loader state
and cannot resume training.

## Checkpoints

The 25 released checkpoints are on Hugging Face, one model repository per
checkpoint, named `IFM/LoopedLM-P1-<recipe>-<endpoint>` with hyphens in the
recipe name, and collected in
[Towards Looped Models Done Right](https://huggingface.co/collections/IFM/towards-looped-models-done-right-6ab9f671bb1c97e33b27d14c).

| Recipe | Checkpoints |
|---|---|
| `dense_d28` | [`336b`](https://huggingface.co/IFM/LoopedLM-P1-dense-d28-336b) |
| `dense_d112` | [`336b`](https://huggingface.co/IFM/LoopedLM-P1-dense-d112-336b), [`500b`](https://huggingface.co/IFM/LoopedLM-P1-dense-d112-500b) |
| `dense_ouro` | [`336b`](https://huggingface.co/IFM/LoopedLM-P1-dense-ouro-336b) |
| `dense_middle` | [`336b`](https://huggingface.co/IFM/LoopedLM-P1-dense-middle-336b) |
| `dense_ouro_raw_injection` | [`336b`](https://huggingface.co/IFM/LoopedLM-P1-dense-ouro-raw-injection-336b) |
| `dense_middle_prelude_injection` | [`336b`](https://huggingface.co/IFM/LoopedLM-P1-dense-middle-prelude-injection-336b) |
| `dense_huginn` | [`336b`](https://huggingface.co/IFM/LoopedLM-P1-dense-huginn-336b), [`500b`](https://huggingface.co/IFM/LoopedLM-P1-dense-huginn-500b) |
| `dense_shared_hl` | [`336b`](https://huggingface.co/IFM/LoopedLM-P1-dense-shared-hl-336b) |
| `dense_split_hl` | [`336b`](https://huggingface.co/IFM/LoopedLM-P1-dense-split-hl-336b) |
| `dense_split_hl_x2` | [`336b`](https://huggingface.co/IFM/LoopedLM-P1-dense-split-hl-x2-336b) |
| `moe_ff112` | [`336b`](https://huggingface.co/IFM/LoopedLM-P1-moe-ff112-336b), [`500b`](https://huggingface.co/IFM/LoopedLM-P1-moe-ff112-500b) |
| `moe_ff112_prelude_injection` | [`336b`](https://huggingface.co/IFM/LoopedLM-P1-moe-ff112-prelude-injection-336b) |
| `moe_ouro` | [`336b`](https://huggingface.co/IFM/LoopedLM-P1-moe-ouro-336b), [`500b`](https://huggingface.co/IFM/LoopedLM-P1-moe-ouro-500b) |
| `moe_middle` | [`336b`](https://huggingface.co/IFM/LoopedLM-P1-moe-middle-336b), [`500b`](https://huggingface.co/IFM/LoopedLM-P1-moe-middle-500b) |
| `moe_middle_input_injection` | [`336b`](https://huggingface.co/IFM/LoopedLM-P1-moe-middle-input-injection-336b), [`500b`](https://huggingface.co/IFM/LoopedLM-P1-moe-middle-input-injection-500b) |
| `moe_huginn` | [`336b`](https://huggingface.co/IFM/LoopedLM-P1-moe-huginn-336b), [`500b`](https://huggingface.co/IFM/LoopedLM-P1-moe-huginn-500b) |
| `moe_shared_hl` | [`336b`](https://huggingface.co/IFM/LoopedLM-P1-moe-shared-hl-336b), [`500b`](https://huggingface.co/IFM/LoopedLM-P1-moe-shared-hl-500b) |

Each checkpoint holds BF16 Safetensors in shards of at most 5 GiB,
`config.json` with the checkpoint's values of the recipe's `model` fields and
the `tokenizer` settings, `artifact_manifest.json`, the tokenizer in
`tokenizer/`, a model card, `LICENSE` and `NOTICE`. Download a checkpoint into
its own directory:

```bash
hf download IFM/LoopedLM-P1-dense-huginn-336b --local-dir /path/to/artifact
```

```python
from xllm.paper_part1.native_inference import generate_native, load_native_model

model, tokenizer, config = load_native_model("/path/to/artifact")
tokens = generate_native(model, tokenizer, ["The capital of France is"],
                         max_gen_len=32, use_sampling=False)
print(tokenizer.decode(tokens[0]))
```

The loader checks every file against the manifest and loads tensor names,
shapes and dtypes strictly. Native generation uses full-prefix recomputation without a KV cache.

### Export a trained checkpoint

[export_paper_part1.py](../../export_paper_part1.py) converts a checkpoint written by
`train_paper_part1.py` into this format, which the loader and `eval_paper_part1.py`
accept. Pass the tokenizer directory the run used:

```bash
ENABLE_FLASH_ATTENTION_3=true python export_paper_part1.py --recipe dense_d28 \
  --checkpoint /path/to/run/checkpoints/checkpoint_00080000 \
  --tokenizer /path/to/part1-tokenizer --output /path/to/artifact
```

The command checks that the checkpoint's model config is the recipe's, stores the
tensors as BF16 and copies the tokenizer. It runs on the
CPU and holds the checkpoint's FP32 weights in host memory. An exported
checkpoint has no model card, `LICENSE` or `NOTICE`.

## Evaluate

The [evaluation data guide](eval-data.md) pins the public datasets, revisions
and transformations. MATH's [row-order mapping](eval/math-train-row-order.json)
fixes its few-shot examples; MATH-500 ordering also uses the Part I tokenizer.

Prepare the task files as the guide describes, in one or more directories, and
assemble them into a new directory:

```bash
python release/paper-part1/materialize-eval-root.py \
  --source-root /path/to/prepared-tasks \
  --output /path/to/composite-eval-root
```

The command copies each required task file from exactly one source root; pass
`--source-root` once per directory. A file found in two roots is an error, as
are missing files, malformed JSONL and an existing output directory.

Run the full paper protocol:

```bash
ENABLE_FLASH_ATTENTION_3=true python eval_paper_part1.py \
  --artifact /path/to/artifact \
  --tasks-root /path/to/composite-eval-root \
  --output /path/to/evaluation-output \
  --protocol paper-part1
```

The [evaluation protocol](../../xllm/paper_part1/eval_protocol.py) defines task
names, metrics, prompts, generation limits and item counts, and decodes
greedily. To evaluate a subset, replace `--protocol paper-part1` with task names,
for example `--tasks gsm8k mmlu`, where `mmlu` stands for its 57 subjects. A
subset uses the same prompts, limits and scoring, checks each task's item count
and reports its paper metric (MMLU's needs all subjects), but reads the task
files in place. Only `--tasks` accepts the sampling options `--use-sampling`,
`--temperature`, `--top-k` and `--top-p`. BBH uses the first-lowercase-answer
scoring rule. HumanEval+ and MBPP+ are scored with EvalPlus 0.3.1, which
executes the generated code; run them in an isolated environment. The evaluator
needs only EvalPlus's scoring dependencies:

```bash
python -m pip install --no-deps evalplus==0.3.1
python -m pip install wget tempdir appdirs termcolor rich tree-sitter tree-sitter-python datasets psutil
```

Results are written to `results.native.json`. Evaluation resumes only when
model, evaluator, suite, dataset, prompt, item and generation hashes match. With
`--protocol`, the evaluator copies the task files into a temporary read-only
snapshot, whose hashes key the cache, and removes it on completion; changed
inputs invalidate their cache entries.

## License

Weights and the tokenizer use [Apache-2.0](LICENSE.weights);
[NOTICE.weights](NOTICE.weights) records attribution and modifications.
