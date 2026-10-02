"""Single-GPU native loading and generation for Part 1 artifacts."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch import nn

from xllm.config import ModelConf, TokenizerConf
from xllm.data.dataset_streamer.tokenizer import Tokenizer, build_tokenizer
from xllm.generation import sample_top_k, sample_top_p
from xllm.models.build import get_model_cls
from xllm.modules.model_parallel import gather_from_model_parallel_region
from xllm.paper_part1.artifacts import load_artifact


def configure_native_runtime() -> None:
    """Apply the numerical policy of Part 1 evaluation: TF32 matmuls."""
    torch.backends.cuda.matmul.allow_tf32 = True


def model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("model has no parameters") from exc


@contextmanager
def _construction_defaults(
    device: torch.device,
    dtype: torch.dtype,
) -> Iterator[None]:
    original_device = torch.get_default_device()
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(original_dtype)
        torch.set_default_device(original_device)


def load_native_model(
    artifact_dir: str | Path,
    *,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, Tokenizer, ModelConf]:
    """Verify and strictly load one self-contained Part 1 artifact."""
    configure_native_runtime()
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("native Part 1 inference requires CUDA")
        target_device = torch.device("cuda", torch.cuda.current_device())
    else:
        target_device = torch.device(device)

    config, state_dict = load_artifact(artifact_dir)
    tokenizer_config = TokenizerConf.from_dict(config["tokenizer"])
    tokenizer = build_tokenizer(tokenizer_config)
    model_config = ModelConf.from_dict(config["model"])
    # The artifact supplies the weights, so skip initialization; the fused blocks and
    # fused output layer are training implementations, so use the plain modules.
    model_config.init_mode = "none"
    model_config.fused_block = False
    model_config.fused_output_layer = False
    if model_config.vocab_size == -1:
        model_config.vocab_size = tokenizer.vocab_size
    if model_config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "artifact model/tokenizer vocabulary mismatch: "
            f"{model_config.vocab_size} != {tokenizer.vocab_size}"
        )

    with _construction_defaults(target_device, torch.bfloat16):
        model = get_model_cls(model_config.arch, model_config)(
            model_config,
            tokenizer,
        )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"strict state-dict load failed: {exc}") from exc
    model.eval()
    return model, tokenizer, model_config


@torch.inference_mode()
def generate_native(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompts: Sequence[str],
    *,
    max_prompt_len: int = 256,
    max_gen_len: int = 256,
    use_sampling: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    remove_prompts: bool = True,
) -> list[list[int]]:
    """Generate with exact full-prefix recomputation and no logical KV cache.

    A prompt longer than ``max_prompt_len`` tokens (BOS included) keeps only
    its last ``max_prompt_len`` tokens. The Part 1 evaluator passes each task's
    ``max_text_len``; raise the limit for longer prompts of your own.
    """
    if not prompts:
        raise ValueError("prompts must not be empty")
    if max_prompt_len <= 0 or max_gen_len <= 0:
        raise ValueError("max_prompt_len and max_gen_len must be positive")
    if use_sampling and temperature <= 0:
        raise ValueError("temperature must be positive when sampling")
    if top_k < 0 or not 0 <= top_p < 1:
        raise ValueError("invalid top-k/top-p sampling parameters")

    model.eval()
    device = model_device(model)

    prompt_tokens = [
        tokenizer.encode(prompt, bos=True, eos=False)
        for prompt in prompts
    ]
    prompt_tokens = [
        tokens[-max_prompt_len:]
        if len(tokens) > max_prompt_len
        else tokens
        for tokens in prompt_tokens
    ]
    if any(not tokens for tokens in prompt_tokens):
        raise ValueError("encoded prompts must not be empty")

    start_pos = min(len(tokens) for tokens in prompt_tokens)
    end_pos = max(len(tokens) for tokens in prompt_tokens) + max_gen_len
    tokens = torch.full(
        (len(prompt_tokens), end_pos),
        tokenizer.pad_id,
        device=device,
        dtype=torch.long,
    )
    for row, encoded in enumerate(prompt_tokens):
        tokens[row, : len(encoded)] = torch.tensor(
            encoded,
            device=device,
            dtype=torch.long,
        )
    prompt_mask = tokens != tokenizer.pad_id

    for current_pos in range(start_pos, end_pos):
        logits, _, cache = model(
            tokens[:, :current_pos],
            multi_segments=False,
            cache=None,
        )
        if cache is not None:
            raise RuntimeError(
                "native no-cache generation returned an unexpected cache"
            )
        next_logits = gather_from_model_parallel_region(
            logits[:, -1].contiguous()
        )
        if use_sampling:
            probabilities = torch.softmax(next_logits / temperature, dim=-1)
            if top_p > 0:
                next_token = sample_top_p(probabilities, top_p)
            elif top_k > 0:
                next_token = sample_top_k(probabilities, top_k)
            else:
                next_token = torch.multinomial(probabilities, num_samples=1)
            next_token = next_token.reshape(-1)
        else:
            next_token = torch.argmax(next_logits, dim=-1)
        tokens[:, current_pos] = torch.where(
            prompt_mask[:, current_pos],
            tokens[:, current_pos],
            next_token,
        )

    if remove_prompts:
        return [
            row[
                len(prompt_tokens[index]) :
                len(prompt_tokens[index]) + max_gen_len
            ].tolist()
            for index, row in enumerate(tokens)
        ]
    return [
        row[: len(prompt_tokens[index]) + max_gen_len].tolist()
        for index, row in enumerate(tokens)
    ]
