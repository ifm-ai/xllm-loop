<div align="center">

<h1>xLLM-Loop</h1>

<p><strong>Official code for <em>Towards Looped Models Done Right</em></strong></p>

<p>
  <a href="https://huskydoge.github.io/husky-blog/posts/recursive_models/towards-looped-models-done-right/"><img src="https://img.shields.io/badge/Paper-Part_I-B31B1B" alt="Part I paper"></a>
  <a href="papers/part2.pdf"><img src="https://img.shields.io/badge/Paper-Part_II-B31B1B" alt="Part II paper (PDF)"></a>
  <a href="https://pytorch.org/get-started/locally/"><img src="https://img.shields.io/badge/PyTorch-2.11%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.11+"></a>
  <img src="https://img.shields.io/badge/CUDA-12.8%2B-76B900?logo=nvidia&logoColor=white" alt="CUDA 12.8+">
  <a href="https://huggingface.co/collections/IFM/towards-looped-models-done-right-6ab9f671bb1c97e33b27d14c"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoints-49-FFD21E" alt="49 checkpoints on Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0 License"></a>
</p>

<p>
  <a href="#papers">Papers</a> &nbsp;|&nbsp;
  <a href="#highlights">Highlights</a> &nbsp;|&nbsp;
  <a href="#installation">Installation</a> &nbsp;|&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;|&nbsp;
  <a href="#reproducing-the-papers">Reproducing the Papers</a> &nbsp;|&nbsp;
  <a href="#checkpoints">Checkpoints</a> &nbsp;|&nbsp;
  <a href="https://github.com/ifm-ai/xllm-loop/issues">Issues</a>
</p>

</div>

xLLM-Loop is the code release of our work on looped language models, which
reuse a block of layers several times per token. It adds looped Transformer
and Huginn architectures with depth-controlled training to the
[xLLM](https://github.com/ifm-ai/xllm) training framework, together with the
training, evaluation and checkpoint code of two papers.

## Papers

| Paper | Scope | Code | Guide |
| --- | --- | --- | --- |
| **Part I**: Topology, Input Injection, Recurrent-State Organization<br>[Blog post](https://huskydoge.github.io/husky-blog/posts/recursive_models/towards-looped-models-done-right/) · *(PDF and arXiv link to be added)* | 17 Dense and MoE recipes; 25 checkpoints | [xllm/paper_part1](xllm/paper_part1/) | [release/paper-part1](release/paper-part1/README.md) |
| **Part II**: Rethinking at Fixed Points<br>[PDF](papers/part2.pdf) · *(arXiv link to be added)* | Depth priors, input-injection variants and distilled prefill; 24 checkpoints | [xllm/paper_part2](xllm/paper_part2/) | [release/paper-part2](release/paper-part2/README.md) |

## Highlights

| Area | Capabilities |
| --- | --- |
| **Looped architectures** | `LoopedTransformer` iterates a range of layers; `Huginn` runs a prelude, a recurrent block and a coda, with a single or a hierarchical (H/L) recurrent state. See [looped.py](xllm/models/looped.py). |
| **Depth-controlled training** | `DepthControlledHuginn` trains at a fixed, sampled (Poisson-lognormal) or learned depth with truncated backpropagation, and logs training FLOPs in expectation over the depths, or at the drawn depths with the learned prior. |
| **Input injection** | Diagonal, linear and orthogonal input injection, with optional normalization of the injected input or the recurrent state. |
| **Training loop** | Gradient accumulation, early stopping on an unchanged schedule (`stop_step`), and resume that can restore per-rank RNG states. |
| **Paper protocols** | Recipe launchers, evaluators and data preparation scripts for the experiments of both papers. |
| **Released checkpoints** | 49 checkpoints in native xLLM format on [Hugging Face](https://huggingface.co/collections/IFM/towards-looped-models-done-right-6ab9f671bb1c97e33b27d14c); the loaders verify every file against its manifest. |

## Installation

xLLM-Loop installs like xLLM. Start with **PyTorch >= 2.11** and
**CUDA >= 12.8**; the release was checked with Python 3.12, PyTorch
2.11.0+cu128 and FlashAttention 3.0.0. The papers' experiments ran on NVIDIA
H200 GPUs.

### 1. Install xLLM-Loop

```bash
git clone https://github.com/ifm-ai/xllm-loop.git
cd xllm-loop

python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation --config-settings editable_mode=compat
python -m pip install "transformers[torch]" pyarrow
```

### 2. Set Up FlashAttention 3

The paper code runs with FlashAttention 3. Follow the upstream
[installation instructions](https://github.com/Dao-AILab/flash-attention/tree/main?tab=readme-ov-file#flashattention-3-beta-release),
then select it for every command:

```bash
export ENABLE_FLASH_ATTENTION_3=true
```

The [xLLM installation guide](https://github.com/ifm-ai/xllm#installation)
covers the native extensions and the other attention backends.

## Quick Start

Download a released checkpoint and generate from it:

```bash
hf download IFM/LoopedLM-P1-dense-huginn-336b --local-dir checkpoints/dense-huginn-336b
```

```python
from xllm.paper_part1.native_inference import generate_native, load_native_model

model, tokenizer, config = load_native_model("checkpoints/dense-huginn-336b")
tokens = generate_native(model, tokenizer, ["The capital of France is"],
                         max_gen_len=32, use_sampling=False)
print(tokenizer.decode(tokens[0]))
```

Preview the full training config of a paper recipe. Copy
[common-base.example.json](release/paper-part1/common-base.example.json) to
`base.json` and fill in the data and tokenizer paths first:

```bash
python train_paper_part1.py --recipe dense_huginn --target 336b \
  --base-config base.json --dump-dir runs/dense-huginn --print-config
```

## Reproducing the Papers

| I want to... | Start here |
| --- | --- |
| **Train a Part I recipe** | [Part I guide](release/paper-part1/README.md#train); entry point: [train_paper_part1.py](train_paper_part1.py). |
| **Export a trained Part I checkpoint for evaluation** | [Part I guide](release/paper-part1/README.md#export-a-trained-checkpoint); entry point: [export_paper_part1.py](export_paper_part1.py). |
| **Evaluate a Part I checkpoint** | [Part I guide](release/paper-part1/README.md#evaluate); entry point: [eval_paper_part1.py](eval_paper_part1.py). |
| **Train a Part II recipe** | [Part II guide](release/paper-part2/README.md#train); entry point: [train_paper_part2.py](train_paper_part2.py). |
| **Export a trained Part II checkpoint for evaluation or distillation** | [Part II guide](release/paper-part2/README.md#train); entry point: [export_paper_part2.py](export_paper_part2.py). |
| **Distill a Part II prefill student** | [Part II guide](release/paper-part2/README.md#distilled-prefill); entry point: [distill_paper_part2.py](distill_paper_part2.py). |
| **Evaluate a Part II checkpoint** | [Part II guide](release/paper-part2/README.md#evaluate); entry point: [eval_paper_part2.py](eval_paper_part2.py). |
| **Prepare the data** | Part I [training](release/paper-part1/data.md) and [evaluation](release/paper-part1/eval-data.md) data; Part II [training](release/paper-part2/data.md) and [evaluation](release/paper-part2/eval-data.md) data. |

## Checkpoints

All 49 checkpoints are in the Hugging Face collection
[Towards Looped Models Done Right](https://huggingface.co/collections/IFM/towards-looped-models-done-right-6ab9f671bb1c97e33b27d14c).
They use the native xLLM format and load with this code, not with Hugging Face
Transformers.

| Paper | Repositories | Contents |
| --- | --- | --- |
| Part I | `IFM/LoopedLM-P1-<recipe>-<336b\|500b>` | 25 checkpoints of the 17 recipes; [list](release/paper-part1/README.md#checkpoints) |
| Part II | `IFM/LoopedLM-P2-<name>` | 24 checkpoints: depth priors, injection variants, D4 and distilled students; [list](release/paper-part2/README.md#artifacts) |

## Repository Layout

xLLM-Loop adds these paths to xLLM:

```text
xllm/models/
  looped.py              Looped Transformer, Huginn and DepthControlledHuginn
  looped_depth.py        Recurrent-depth distributions and draws
  looped_depth_prior.py  Learned depth prior
  looped_flops.py        Training FLOPs of DepthControlledHuginn
xllm/paper_part1/        Part I recipes, training, inference, checkpoints and evaluation
xllm/paper_part2/        Part II recipes, training, distillation, checkpoints and evaluation
release/                 Reproduction guides, data preparation and weight licenses
papers/                  Paper PDFs
train_paper_part1.py     Part I entry points (with eval_paper_part1.py, export_paper_part1.py)
train_paper_part2.py     Part II entry points (with eval_paper_part2.py, distill_paper_part2.py,
                         export_paper_part2.py)
```

## Built on xLLM

The rest of the repository is [xLLM](https://github.com/ifm-ai/xllm): data
loading, configuration, distributed training, evaluation and checkpointing.
The [xLLM README](https://github.com/ifm-ai/xllm#readme) covers its general
usage, including data preparation, training configuration, evaluation and
checkpoint conversion. xLLM-Loop's additions to the shared training loop are
opt-in: xLLM models train as they do in xLLM, and checkpoints additionally
record per-rank RNG states.

## Citation

```bibtex
@misc{huang2026loopedmodels,
  title  = {Towards Looped Models Done Right. Part I: Topology, Input Injection, Recurrent-State Organization},
  author = {Benhao Huang and Chufan Shi and Junlin Chen and Shicheng Wen and Zhengzhong Liu and Eric Xing and Xuezhe Ma},
  year   = {2026},
  url    = {https://huskydoge.github.io/husky-blog/posts/recursive_models/towards-looped-models-done-right/}
}

@misc{huang2026fixedpoints,
  title  = {Towards Looped Models Done Right. Part II: Rethinking at Fixed Points},
  author = {Benhao Huang and Chufan Shi and Junlin Chen and Shicheng Wen and Zhengzhong Liu and Eric Xing and Xuezhe Ma},
  year   = {2026},
  url    = {https://github.com/ifm-ai/xllm-loop/blob/main/papers/part2.pdf}
}
```

## License

xLLM-Loop is released under the [Apache 2.0 License](LICENSE). Third-party code
in the evaluation modules is listed with its licenses in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The released weights carry
their own licenses: [Part I](release/paper-part1/LICENSE.weights) and
[Part II](release/paper-part2/LICENSE.weights).
