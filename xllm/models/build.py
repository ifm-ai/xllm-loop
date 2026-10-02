from typing import Optional
from logging import getLogger

import torch
from torch import nn

from xllm.config import ModelConf
from xllm.distributed import FullyShardedDataParallel
from xllm.distributed import (
    get_context_parallel_world_size,
    get_data_parallel_world_size,
    get_model_parallel_group,
    get_model_parallel_world_size,
)
from xllm.distributed.utils import get_device_mesh
from xllm.utils import (
    get_torch_dtype,
    create_on_gpu,
)
from xllm.data.dataset_streamer.tokenizer import Tokenizer
from xllm.models.xllm import XLLModel
from xllm.models.gekko import Gekko
from xllm.models.looped import Huginn, LoopedTransformer, DepthControlledHuginn
from xllm.models.transformer import Transformer

logger = getLogger()



def get_model_cls(arch: str, model_cfg: Optional[ModelConf] = None):
    """Return the model class of `arch`; looped models need `model_cfg` to pick their class."""
    if arch == "huginn":
        if model_cfg is None:
            raise ValueError("arch='huginn' needs model_cfg to pick the Huginn class")
        return DepthControlledHuginn if model_cfg.huginn_depth_control else Huginn
    if arch == "transformer" and model_cfg is not None and (
        model_cfg.loop_times != 1
        or model_cfg.loop_input_injection != "none"
        or model_cfg.dense_prelude_input_injection != "none"
    ):
        return LoopedTransformer
    return {
        "transformer": Transformer,
        "gekko": Gekko,
    }[arch]


def build_model(
    model_cfg: ModelConf,
    dtype: str,
    fully_sharded_size: Optional[int],
    fp32_reduce_scatter: bool,
    reshard_after_forward: bool,
    forward_prefetch: bool,
    tokenizer: Tokenizer,
) -> nn.Module:
    if model_cfg.ddp_backend == 'fsdp1':
        return build_model_fsdp1(
            model_cfg, dtype, fully_sharded_size, fp32_reduce_scatter,
            reshard_after_forward, forward_prefetch, tokenizer
        )
    elif model_cfg.ddp_backend == 'fsdp2':
        return build_model_fsdp2(
            model_cfg, dtype, fully_sharded_size, fp32_reduce_scatter,
            reshard_after_forward, forward_prefetch, tokenizer
        )
    else:
        raise ValueError(f"Unknown DDP backend: {model_cfg.ddp_backend}")


def build_model_fsdp1(
    model_cfg: ModelConf,
    dtype: str,
    fully_sharded_size: Optional[int],
    fp32_reduce_scatter: bool,
    reshard_after_forward: bool,
    forward_prefetch: bool,
    tokenizer: Tokenizer,
) -> nn.Module:
    from torch.distributed.fsdp.wrap import enable_wrap, wrap
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

    data_parallel_size = get_data_parallel_world_size()
    context_parallel_size = get_context_parallel_world_size()
    fully_sharded_size = data_parallel_size if fully_sharded_size is None else fully_sharded_size
    assert data_parallel_size % fully_sharded_size == 0
    assert fully_sharded_size > 1 or context_parallel_size == 1

    replicated_size = data_parallel_size // fully_sharded_size * context_parallel_size
    if replicated_size == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD if reshard_after_forward else ShardingStrategy.SHARD_GRAD_OP
    else:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD if reshard_after_forward else ShardingStrategy._HYBRID_SHARD_ZERO2

    device_mesh = get_device_mesh(fully_sharded_size, replicated_size, get_model_parallel_world_size())
    compute_dtype = get_torch_dtype(dtype)
    assert compute_dtype is not None, f"Unknown dtype: {dtype}"
    mixed_precision = MixedPrecision(param_dtype=compute_dtype,
                                     reduce_dtype=torch.float32 if fp32_reduce_scatter else compute_dtype,
                                     buffer_dtype=torch.float32)
    fsdp_cfg = {
        "model_parallel_process_group": get_model_parallel_group(),
        "sharding_strategy": sharding_strategy,
        "mixed_precision": mixed_precision,
        "forward_prefetch": forward_prefetch,
        "sync_module_states": False,
        "use_orig_params": False,  # flatten parameters
        "device_mesh": device_mesh,
        "device_id": torch.cuda.current_device()
    }
    model_cls = get_model_cls(model_cfg.arch, model_cfg)
    with create_on_gpu():
        with enable_wrap(wrapper_cls=FullyShardedDataParallel, **fsdp_cfg):
            model = model_cls(model_cfg, tokenizer)
            model = wrap(model.cuda())
            model.train()

        return model


def build_model_fsdp2(
    model_cfg: ModelConf,
    dtype: str,
    fully_sharded_size: Optional[int],
    fp32_reduce_scatter: bool,
    reshard_after_forward: bool,
    forward_prefetch: bool,
    tokenizer: Tokenizer,
) -> nn.Module:
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from xllm.distributed.wrap import enable_wrap, wrap

    if forward_prefetch:
        logger.warning("forward prefetch is not supported yet for FSDP2.")

    data_parallel_size = get_data_parallel_world_size()
    context_parallel_size = get_context_parallel_world_size()
    fully_sharded_size = data_parallel_size if fully_sharded_size is None else fully_sharded_size
    assert data_parallel_size % fully_sharded_size == 0
    assert fully_sharded_size > 1 or context_parallel_size == 1

    replicated_size = data_parallel_size // fully_sharded_size * context_parallel_size

    device_mesh = get_device_mesh(fully_sharded_size, replicated_size, get_model_parallel_world_size())
    compute_dtype = get_torch_dtype(dtype)
    assert compute_dtype is not None, f"Unknown dtype: {dtype}"
    mixed_precision = MixedPrecisionPolicy(
        param_dtype=compute_dtype, reduce_dtype=torch.float32 if fp32_reduce_scatter else compute_dtype
    )
    fsdp_cfg = {
        "mesh": device_mesh,
        "reshard_after_forward": reshard_after_forward,
        "mp_policy": mixed_precision,
    }
    model_cls = get_model_cls(model_cfg.arch, model_cfg)
    with create_on_gpu():
        with enable_wrap(**fsdp_cfg):
            model = model_cls(model_cfg, tokenizer)
            model = wrap(model.cuda())
            model.train()

        return model


def init_cache(model: nn.Module):
    if isinstance(model, FullyShardedDataParallel):
        model = model.module

    if model.arch == 'transformer':
        return init_transformer_cache(model)
    elif model.arch == 'gekko':
        return init_gekko_cache(model)
    else:
        raise ValueError(f"Unknown model architecture: {model.arch}")


def init_transformer_cache(model):
    cache = [(None, None, 0) for _ in range(model.num_layers)]
    return cache


def init_gekko_cache(model):
    cache_layers = [((None, None, 0), (None, None, None, None), (None, None, None), (None, None, None)) for _ in range(model.num_layers)]
    cache_norm = (None, None, None)
    cache_segment = (None, None, None)
    cache = (cache_layers, cache_norm, cache_segment)
    return cache


def truncate_cache(cache, truncated_cache):
    cache_layers, cache_output = cache
    truncated_layers, truncated_output = truncated_cache
    n_layers = len(cache_layers)
    for i in range(n_layers):
        cache_attn, cache_norm, hx = cache_layers[i]
        cache_layers[i] = (cache_attn, truncated_layers[i][1], hx)

    cache = (cache_layers, truncated_output)
    return cache
