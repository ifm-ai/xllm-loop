# Part II: Rethinking at Fixed Points

Paper: *Towards Looped Models Done Right, Part II: Rethinking at Fixed Points* ([arXiv](https://arxiv.org/abs/2610.06833), [PDF](../../papers/part2.pdf)).

This directory provides the code, recipes and evaluation protocol behind the
main-text experiments of the paper: the depth priors of Table 2, the
input-injection variants of Table 3, and the distilled prefill of Table 4 with
its D4 baseline, together with the 24 checkpoints ([Artifacts](#artifacts)).
Part of the training mixture (the sources without a link in the
[data guide](data.md)) and the paper's distillation token cache are not public,
and the upstream loader packs sequences differently from the paper's runs, so
retraining follows the recipes on the data you supply. Every command below runs from the repository root.

## Environment

The release was checked with Python 3.12, PyTorch 2.11.0+cu128 and
FlashAttention 3.0.0; follow the [installation guide](../../README.md#installation).
Set `ENABLE_FLASH_ATTENTION_3=true` for every command. The paper itself ran on
PyTorch 2.8.0+cu128 with FlashAttention 2.8.3, so numerical results can differ
slightly from the paper's.

The MBPP+ stage scores with EvalPlus 0.3.1, installed without its dependencies
as for Part I:

```bash
python -m pip install --no-deps evalplus==0.3.1
python -m pip install wget tempdir appdirs termcolor rich tree-sitter tree-sitter-python datasets psutil
```

The evaluator points EvalPlus at the hash-pinned `mbpp_plus.jsonl`, and
EvalPlus computes the expected outputs once in the sandbox's cache, so scoring
needs no network.

## Recipes

[recipes.py](../../xllm/paper_part2/recipes.py) defines every training row.
All Huginn rows use 1 prelude, 2 recurrent and 1 coda Llama block with diagonal
input injection unless stated, unfused blocks, 5 recurrences at evaluation, the
Jais64k vocabulary (64,256), global batch 512 x 8,192 tokens, AdamW (0.9, 0.98,
weight decay 0.1) and a warmup-stable-decay schedule (5% warmup, cosine decay to
0.1x the peak over the last 10%). Sampled depths are drawn i.i.d., one per
training forward and shared by every rank.

| Recipe | Paper row |
| --- | --- |
| `s_fixed_r5`, `m_fixed_r5`, `l_fixed_r5` | Table 2, Fixed R=5 |
| `s_fixed_pln5`, `m_fixed_pln5`, `l_fixed_pln5` | Table 2, Fixed PLN-5 |
| `s_learned_entropy0`, `m_learned_entropy0`, `l_learned_entropy0` | Table 2, Learned lambda_H=0 |
| `s_learned_entropy0p01`, `m_learned_entropy0p01`, `l_learned_entropy0p01` | Table 2, Learned lambda_H=0.01 |
| `s_huginn_linear`, `m_huginn_linear` | Table 3, Huginn-Linear |
| `s_parcae`, `m_parcae` | Table 3, Parcae-Decay |
| `s_orthoinj`, `m_orthoinj` | Table 3, OrthoInj |
| `s_d4`, `m_d4` | Table 4, D4 (dense, 4 blocks) |

S, M and L have widths 1,536, 3,072 and 6,144 and train for 5,120, 20,480 and
81,920 updates (21.47B, 85.90B and 343.6B tokens). PLN-5 draws capped
Poisson-lognormal depths with mean 5 and backpropagates the last 10
recurrences; the learned prior starts from PLN-5, stays fixed for the first 32
updates while its reward baselines settle, and then updates by REINFORCE after
every step.

## Train

Copy [base.example.json](base.example.json), replace its two placeholder `data`
entries with the full 42-entry mixture string and set the tokenizer path
([data guide](data.md)). The base config owns data, tokenizer, logging,
checkpoint retention and gradient accumulation; the recipe owns the model,
optimizer, schedule, global batch and seed. Preview the full config, then
launch:

```bash
ENABLE_FLASH_ATTENTION_3=true python train_paper_part2.py --recipe s_learned_entropy0p01 \
  --base-config base.json --dump-dir /path/to/run --print-config
ENABLE_FLASH_ATTENTION_3=true torchrun --nnodes N --nproc-per-node G ... train_paper_part2.py \
  --recipe s_learned_entropy0p01 --base-config base.json --dump-dir /path/to/run
```

Every recipe trains on a global batch of 512 sequences: the data-parallel ranks,
the local batch and `gradient_accumulation_steps` (base config, default 1)
multiply to 512, and the local batch follows from the other two. The rank count
and the accumulation change the data order. With sampled depths each
accumulation step draws its own depth, so the accumulation changes how many
depths an update averages over. This can change results slightly, but is not
expected to change the ranking of the models. Rerunning the same command resumes from the
latest checkpoint, including the depth-draw counter and the learned prior.
`stop_step` in the base config stops early while keeping the schedule.

To evaluate a run or distill from it, export a checkpoint as an artifact with
the recipe's model config and the run's tokenizer
([export_paper_part2.py](../../export_paper_part2.py)); the script refuses a
checkpoint whose saved model config differs from the recipe:

```bash
python export_paper_part2.py --recipe s_learned_entropy0p01 \
  --checkpoint /path/to/run/checkpoints/checkpoint_00005120 --tokenizer /path/to/tokenizer \
  --output /path/to/artifact
```

The artifact has the released layout without the model card, `LICENSE` and
`NOTICE`; `eval_paper_part2.py --artifact` and `distill_paper_part2.py --teacher`
read it.

## Distilled prefill

A student learns the pre-coda state that the Learned lambda_H=0.01 teacher
reaches after five recurrences, from the teacher's prelude output, with a
hidden-state MSE and a KL term through the teacher's coda
([distill.py](../../xllm/paper_part2/distill.py)). Build the token cache
([data guide](data.md#distillation-cache)), then train:

```bash
ENABLE_FLASH_ATTENTION_3=true torchrun --nnodes N --nproc-per-node G ... distill_paper_part2.py \
  --recipe s_distill_teacher_init --teacher /path/to/huginn-s-learned-entropy0p01 \
  --train /path/to/cache/train_manifest.json --valid /path/to/cache/valid_manifest.json --out /path/to/student
```

| Recipe | Initialization | Updates |
| --- | --- | ---: |
| `s_distill_random_init`, `m_distill_random_init` | random | 2,560, 10,240 |
| `s_distill_teacher_init`, `m_distill_teacher_init` | the teacher's injection and recurrent blocks | 2,560, 10,240 |

Every update reads 256 rows: the ranks, `--local-batch` and the accumulation
multiply to 256, and without `--local-batch` each rank reads its share in one
micro-batch. The learning rate warms up for 8 updates, then decays by cosine to
0.1x its peak at update 2,048 and stays there. The run checkpoints every 128
updates, resumes by rerunning, and writes `student_final.pt`. The paper
initialized teacher-init students from the teacher's FP32 weights; from a
released BF16 teacher they start from its BF16-rounded weights.

## Evaluate

[eval_paper_part2.py](../../eval_paper_part2.py) runs the paper's evaluation:
every Huginn model at R=5 with terminal KV sharing and the recurrent state
seeded by example identity. The [evaluation data guide](eval-data.md) lists the
inputs of `data.json`; each is checked against its SHA-256. Each stage except
`summary` takes one GPU; shard the long stages across jobs.

```bash
A=/path/to/huginn-s-learned-entropy0p01
EVAL=/path/to/eval-data   # the --output of prepare-eval-data.py
run() { ENABLE_FLASH_ATTENTION_3=true python eval_paper_part2.py --artifact $A --data $EVAL/data.json --out out "$@"; }
run ppl
for shard in 0 1 2 3; do run tasks $shard 4; done   # each shard can be its own GPU job
for shard in 0 1; do run gen gsm8k $shard 2; done
run gen drop 0 1
run gen mbpp 0 1 --mbpp-sandbox "bwrap ..."
run summary
```

`summary` needs every shard of the stages it reports and writes the table row
to `out/summary.json`. The paper's Val PPL used a held-out validation set that
is not released; the public protocol reports Wiki PPL. Wiki PPL prefills the
first 4,096 tokens of each 8,192-token sequence and scores the tokens after the
prompt that continue a document begun inside it, with attention inside
documents (`continued` in the output). MBPP+ runs EvalPlus 0.3.1 on generated
code; run it in a sandbox.

Table 4 swaps the prefill path while the teacher decodes at R=5:
`--prefill r1` prefills with one teacher recurrence, `--student
/path/to/distilled-s-teacher-init` with a distilled student, and a D4 artifact
decodes itself with its full KV cache. The `latency` stage times the path and
the teacher's own prefill on 8 x 8,192 tokens: one warm-up call, then six
rotating repetitions. Latencies depend on the software environment: Table 4
was measured with PyTorch 2.8.0 and FlashAttention 2.8.3, one GPU per process.

## Artifacts

The 24 checkpoints are on the Hugging Face Hub as `IFM/LoopedLM-P2-<name>`:
the 12 Table 2 models `huginn-{s,m,l}-{fixed-r5, fixed-pln5, learned-entropy0,
learned-entropy0p01}`, the 6 Table 3 models
`huginn-{s,m}-{huginn-linear,parcae,orthoinj}`, `d4-{s,m}` and the students
`distilled-{s,m}-{random,teacher}-init`. Each holds BF16 Safetensors (the
checkpoint's weights rounded to BF16), `config.json`, `artifact_manifest.json`,
a model card, `LICENSE`, `NOTICE` and, except for students, the tokenizer. A
student's config names its teacher artifact and that artifact's manifest SHA-256.

```bash
hf download IFM/LoopedLM-P2-huginn-s-learned-entropy0p01 --local-dir huginn-s-learned-entropy0p01
```

```python
from xllm.paper_part2.artifacts import load_artifact
fields, state_dict = load_artifact("huginn-s-learned-entropy0p01")
```

`load_artifact` checks every file against the manifest. It ignores the
`.gitattributes` and `.cache/huggingface/` that `hf download --local-dir` adds
and rejects symbolic links, so load a `--local-dir` download, not the Hub cache.

## Citation

To be added with the arXiv link.

## License

Weights and the tokenizer use [Apache-2.0](LICENSE.weights);
[NOTICE.weights](NOTICE.weights) records attribution and preparation.
