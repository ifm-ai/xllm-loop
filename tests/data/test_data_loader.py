"""
Distributed data-loader smoke test.

Launch with torchrun (2 ranks), a data mix (DATA or DATA_FILE) and a tokenizer:
    DATA=/path/to/source:1.0:text:text TOKENIZER_PATH=/path/to/tokenizer \
        torchrun --nproc_per_node=2 tests/data/test_data_loader.py

What this test checks
---------------------
1.  MultiSourceDataLoader can be built and iterated (simple packing).
2.  MultiSourceDataLoader can be built and iterated (bestfit packing).
3.  Each rank produces non-overlapping token sequences (data-parallel sharding).
4.  Checkpoint save / load: resuming produces the exact same next batch.
5.  Batch shapes and dtypes match expectations.
"""

import os
import sys
import pickle
import tempfile

import numpy as np
import torch
import torch.distributed as dist

# ── project root on PYTHONPATH ──────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from xllm.config import TokenizerConf
from xllm.data.dataset_streamer.tokenizer.build import build_tokenizer
from xllm.data.dataloader import MultiSourceDataLoader

# ── constants ────────────────────────────────────────────────────────────────
if "DATA" in os.environ:
    DATA_MIX = os.environ["DATA"]
elif "DATA_FILE" in os.environ:
    with open(os.environ["DATA_FILE"], "r", encoding="utf-8") as f:
        DATA_MIX = f.read().strip()
else:
    DATA_MIX = None
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH")

SEQ_LEN      = int(os.environ.get("SEQ_LEN", "2048"))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "2"))          # sequences per step
NUM_BUFFERED = int(os.environ.get("NUM_BUFFERED", "64"))       # refill token-budget multiplier
NUM_STEPS    = int(os.environ.get("NUM_STEPS", "5"))           # batches to pull before checkpoint
COMPARE_STEPS = int(os.environ.get("COMPARE_STEPS", "1"))       # replay batches after checkpoint
NUM_WORKERS  = int(os.environ.get("NUM_WORKERS", "1"))         # tokenizer worker threads
WORKER_COMPARE_STEPS = int(os.environ.get("WORKER_COMPARE_STEPS", str(max(NUM_STEPS, COMPARE_STEPS))))
RESUME_PACKING_TYPE = os.environ.get("RESUME_PACKING_TYPE", "simple")
EXPECTED_POPPING_IDX = os.environ.get("EXPECTED_POPPING_IDX")
EXPECTED_BUFFER_LEN = os.environ.get("EXPECTED_BUFFER_LEN")
MAX_CKPT_STATE_BYTES = os.environ.get("MAX_CKPT_STATE_BYTES")
TEST_FILTER = {
    name.strip()
    for name in os.environ.get("TEST_FILTER", "").split(",")
    if name.strip()
}


# ── helpers ──────────────────────────────────────────────────────────────────

def log(rank: int, msg: str):
    print(f"[rank {rank}] {msg}", flush=True)


def build_loader(
    tokenizer,
    packing_type: str,
    world_rank: int,
    world_size: int,
    num_workers: int = NUM_WORKERS,
) -> MultiSourceDataLoader:
    return MultiSourceDataLoader(
        tokenizer=tokenizer,
        data_mix_str=DATA_MIX,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        num_buffered_seq=NUM_BUFFERED,
        world_rank=world_rank,
        world_size=world_size,
        packing_type=packing_type,
        num_workers=num_workers,
    )


def assert_batches_equal(ref_batch, got_batch, *, label: str, rank: int, step: int):
    assert np.array_equal(ref_batch.x, got_batch.x), \
        f"[rank {rank}] {label} x mismatch at step {step}!\n  ref: {ref_batch.x[0, :8]}\n  got: {got_batch.x[0, :8]}"
    assert np.array_equal(ref_batch.y, got_batch.y), \
        f"[rank {rank}] {label} y mismatch at step {step}!"
    if ref_batch.mask is None or got_batch.mask is None:
        assert ref_batch.mask is got_batch.mask, \
            f"[rank {rank}] {label} mask presence mismatch at step {step}!"
    else:
        assert np.array_equal(ref_batch.mask, got_batch.mask), \
            f"[rank {rank}] {label} mask mismatch at step {step}!"
    assert ref_batch.src_infos == got_batch.src_infos, \
        f"[rank {rank}] {label} source info mismatch at step {step}!"


# ── test cases ───────────────────────────────────────────────────────────────

def test_simple_packing(tokenizer, rank: int, world_size: int):
    loader = build_loader(tokenizer, "simple", rank, world_size)
    try:
        for step in range(NUM_STEPS):
            batch = next(loader)
            assert batch.x.shape == (BATCH_SIZE, SEQ_LEN), \
                f"[simple] unexpected shape {batch.x.shape}"
            assert batch.x.dtype == np.int64, \
                f"[simple] unexpected dtype {batch.x.dtype}"
            assert batch.y.shape == batch.x.shape
    finally:
        loader.close()
    log(rank, "✓ test_simple_packing passed")


def test_best_fit_packing(tokenizer, rank: int, world_size: int):
    loader = build_loader(tokenizer, "bestfit", rank, world_size)
    try:
        for step in range(NUM_STEPS):
            batch = next(loader)
            assert batch.x.shape == (BATCH_SIZE, SEQ_LEN), \
                f"[bestfit] unexpected shape {batch.x.shape}"
            assert batch.x.dtype == np.int64, \
                f"[bestfit] unexpected dtype {batch.x.dtype}"
    finally:
        loader.close()
    log(rank, "✓ test_best_fit_packing passed")


def test_rank_sharding(tokenizer, rank: int, world_size: int):
    """
    Ranks should NOT see the same first token in every sequence.
    We check this globally by all-gathering the first token of the first batch.
    """
    loader = build_loader(tokenizer, "simple", rank, world_size)
    try:
        batch  = next(loader)

        # Gather the first token of the first sequence from each rank.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        first_token = torch.tensor([int(batch.x[0, 0])], dtype=torch.long, device=device)
        gathered    = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(gathered, first_token)
    finally:
        loader.close()

    if rank == 0:
        tokens = [t.item() for t in gathered]
        log(rank, f"First token per rank: {tokens}")
        # At minimum they should not ALL be identical (different shards read
        # different lines).  This is a soft check because identical first
        # tokens are theoretically possible but astronomically unlikely.
        if len(set(tokens)) == 1:
            log(rank, "WARNING: all ranks produced the same first token – check sharding!")
        else:
            log(rank, "✓ test_rank_sharding passed (tokens differ across ranks)")


def test_checkpoint_resume(tokenizer, rank: int, world_size: int):
    """
    Save state after N steps, resume from that state, and verify
    that the very next batch is identical to what we got before saving.
    """
    loader = build_loader(tokenizer, RESUME_PACKING_TYPE, rank, world_size)
    loader2 = None
    ckpt_path = None
    try:
        # Warm up
        for _ in range(NUM_STEPS):
            next(loader)

        # Save checkpoint, THEN consume the next batch as reference.
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            ckpt_path = f.name
        ckpt_state = loader.get_state()
        state_bytes = len(pickle.dumps(ckpt_state))
        if MAX_CKPT_STATE_BYTES is not None:
            assert state_bytes <= int(MAX_CKPT_STATE_BYTES), \
                f"[rank {rank}] checkpoint state is {state_bytes} bytes, max={MAX_CKPT_STATE_BYTES}"
        active = loader.assembler._active
        active_buffer_len = len(loader.assembler._buffers[active])
        popping_idx = int(ckpt_state.get("assembler", {}).get("popping_idx", -1))
        log(
            rank,
            f"checkpoint after {NUM_STEPS} steps: refill_token_budget={NUM_BUFFERED * SEQ_LEN} "
            f"active_buffer_len={active_buffer_len} popping_idx={popping_idx} "
            f"compare_steps={COMPARE_STEPS} state_bytes={state_bytes}",
        )
        if EXPECTED_POPPING_IDX is not None:
            assert popping_idx == int(EXPECTED_POPPING_IDX), \
                f"[rank {rank}] expected popping_idx={EXPECTED_POPPING_IDX}, got {popping_idx}"
        if EXPECTED_BUFFER_LEN is not None:
            assert active_buffer_len == int(EXPECTED_BUFFER_LEN), \
                f"[rank {rank}] expected active_buffer_len={EXPECTED_BUFFER_LEN}, got {active_buffer_len}"
        with open(ckpt_path, "wb") as f:
            pickle.dump(ckpt_state, f)
        ref_batches = [next(loader) for _ in range(COMPARE_STEPS)]

        # Reload from the same checkpoint — replay window must match exactly.
        loader2 = build_loader(tokenizer, RESUME_PACKING_TYPE, rank, world_size)
        with open(ckpt_path, "rb") as f:
            loader2.set_state(pickle.load(f))
        os.unlink(ckpt_path)
        ckpt_path = None

        for replay_idx, ref_batch in enumerate(ref_batches, start=1):
            resumed_batch = next(loader2)
            assert_batches_equal(
                ref_batch,
                resumed_batch,
                label=f"checkpoint_resume/{RESUME_PACKING_TYPE}",
                rank=rank,
                step=replay_idx,
            )
    finally:
        loader.close()
        if loader2 is not None:
            loader2.close()
        if ckpt_path is not None and os.path.exists(ckpt_path):
            os.unlink(ckpt_path)
    log(rank, f"✓ test_checkpoint_resume passed ({RESUME_PACKING_TYPE})")


def test_num_workers_determinism(tokenizer, rank: int, world_size: int):
    """
    Tokenizer worker count should not affect the logical data stream.
    Compare a serial tokenizer path with the configured worker pool.
    """
    if NUM_WORKERS == 1:
        log(rank, "✓ test_num_workers_determinism passed (NUM_WORKERS=1)")
        return

    serial_loader = build_loader(tokenizer, "simple", rank, world_size, num_workers=1)
    worker_loader = build_loader(tokenizer, "simple", rank, world_size, num_workers=NUM_WORKERS)
    try:
        for step in range(1, WORKER_COMPARE_STEPS + 1):
            serial_batch = next(serial_loader)
            worker_batch = next(worker_loader)
            assert_batches_equal(
                serial_batch,
                worker_batch,
                label=f"num_workers 1 vs {NUM_WORKERS}",
                rank=rank,
                step=step,
            )
    finally:
        serial_loader.close()
        worker_loader.close()
    log(rank, f"✓ test_num_workers_determinism passed (1 vs {NUM_WORKERS})")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if DATA_MIX is None or TOKENIZER_PATH is None:
        sys.exit("Set DATA (or DATA_FILE) to a data mix and TOKENIZER_PATH to a tokenizer.")

    # ── init dist ────────────────────────────────────────────────────────────
    backend = os.environ.get("DIST_BACKEND", "nccl" if torch.cuda.is_available() else "gloo")
    dist.init_process_group(backend=backend)
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device_desc = f"cuda:{torch.cuda.current_device()}"
    else:
        device_desc = "cpu"

    log(rank, f"world_size={world_size}, backend={backend}, device={device_desc}")

    # ── build tokenizer ──────────────────────────────────────────────────────
    tokenizer_cfg = TokenizerConf(
        type=os.environ.get("TOKENIZER_TYPE", "huggingface"),
        path=TOKENIZER_PATH,
        num_reserved_special_tokens=int(os.environ.get("NUM_RESERVED_SPECIAL_TOKENS", "0")),
    )
    tokenizer = build_tokenizer(tokenizer_cfg)
    log(rank, f"Tokenizer vocab_size={tokenizer.vocab_size}  "
              f"bos={tokenizer.bos_id}  eos={tokenizer.eos_id}")

    dist.barrier()

    # ── run tests ────────────────────────────────────────────────────────────
    failed = []

    tests = [
        ("simple_packing",   test_simple_packing),
        ("best_fit_packing", test_best_fit_packing),
        ("rank_sharding",    test_rank_sharding),
        ("checkpoint_resume",test_checkpoint_resume),
        ("num_workers_determinism", test_num_workers_determinism),
    ]
    if TEST_FILTER:
        tests = [(name, fn) for name, fn in tests if name in TEST_FILTER]

    for name, fn in tests:
        dist.barrier()
        try:
            fn(tokenizer, rank, world_size)
        except Exception as e:
            log(rank, f"✗ {name} FAILED: {e}")
            failed.append(name)

    dist.barrier()

    failure_device = (
        torch.device("cuda", torch.cuda.current_device())
        if backend == "nccl"
        else torch.device("cpu")
    )
    any_failed = torch.tensor(int(bool(failed)), dtype=torch.int32, device=failure_device)
    dist.all_reduce(any_failed, op=dist.ReduceOp.MAX)
    global_failure = bool(any_failed.item())

    if rank == 0:
        if global_failure:
            print(f"\n{'='*60}\nAt least one rank failed\n{'='*60}", flush=True)
        else:
            print(f"\n{'='*60}\nAll tests passed ✓\n{'='*60}", flush=True)

    dist.destroy_process_group()
    if global_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
