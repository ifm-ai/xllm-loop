#!/usr/bin/env bash
#SBATCH --job-name=k2m-resume-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --open-mode=append
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --mem=128G

set -euo pipefail

# Activate the xLLM environment and set the site paths, then run both phases
# with the same unique tag and compare:
#   export REPO=/path/to/xllm SAVE_ROOT=/path/to/runs TOKENIZER=/path/to/tokenizer \
#     DATA=/path/to/source:1.0:text:text VALID_DATA=/path/to/valid.jsonl TASK_ROOT=/path/to/eval-tasks
#   PHASE=baseline RUN_TAG=my-test sbatch --export=ALL,PHASE,RUN_TAG "$0"
#   PHASE=resume RUN_TAG=my-test sbatch --export=ALL,PHASE,RUN_TAG "$0"
#   python tests/data/compare_train_resume.py \
#     "$SAVE_ROOT/k2mobile900m_resume_eval_my-test"

PHASE="${PHASE:-}"
RUN_TAG="${RUN_TAG:-}"
if [[ "${PHASE}" != "baseline" && "${PHASE}" != "resume" ]]; then
  echo "Set PHASE=baseline or PHASE=resume." >&2
  exit 2
fi
if [[ -z "${RUN_TAG}" ]]; then
  echo "Set RUN_TAG to a unique name shared by both phases." >&2
  exit 2
fi

REPO="${REPO:?Set REPO to the xLLM checkout}"
SAVE_ROOT="${SAVE_ROOT:?Set SAVE_ROOT to the directory for run outputs}"
DATA="${DATA:?Set DATA to the training data mix (path:weight:text:text,...)}"
TOKENIZER="${TOKENIZER:?Set TOKENIZER to a Hugging Face tokenizer directory}"
VALID_DATA="${VALID_DATA:?Set VALID_DATA to a validation JSONL file}"
TASK_ROOT="${TASK_ROOT:?Set TASK_ROOT to the evaluation task root (with copa)}"
RUN_ROOT="${SAVE_ROOT}/k2mobile900m_resume_eval_${RUN_TAG}"
BASELINE_SAVE="${RUN_ROOT}/baseline"
RESUME_SAVE="${RUN_ROOT}/resume"
BUFFER_SIZE="${BUFFER_SIZE:-128}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-200}"
STEPS="${STEPS:-300}"
DUMP_FREQ="${DUMP_FREQ:-100}"
CHECKPOINT_NAME="$(printf 'checkpoint_%07d' "${CHECKPOINT_STEP}")"

if [[ "${PHASE}" == "baseline" ]]; then
  SAVE="${BASELINE_SAVE}"
  if [[ -e "${SAVE}/checkpoints" || -s "${SAVE}/metrics.optim.jsonl" ]]; then
    echo "Baseline output already exists: ${SAVE}" >&2
    exit 2
  fi
  export XLLM_DISABLE_EVAL=1
else
  SAVE="${RESUME_SAVE}"
  BASELINE_CHECKPOINT="${BASELINE_SAVE}/checkpoints/${CHECKPOINT_NAME}"
  if [[ ! -f "${BASELINE_CHECKPOINT}/config.json" ]]; then
    echo "Complete the baseline phase first; checkpoint is missing: ${BASELINE_CHECKPOINT}" >&2
    exit 2
  fi
  if [[ -e "${SAVE}/checkpoints/checkpoint_$(printf '%07d' "${STEPS}")" ]]; then
    echo "Resume output already completed: ${SAVE}" >&2
    exit 2
  fi
  if [[ -s "${SAVE}/metrics.optim.jsonl" ]]; then
    echo "Resume output contains partial optimizer metrics: ${SAVE}" >&2
    exit 2
  fi
  mkdir -p "${SAVE}/checkpoints"
  if [[ ! -e "${SAVE}/checkpoints/${CHECKPOINT_NAME}" ]]; then
    ln -s "${BASELINE_CHECKPOINT}" "${SAVE}/checkpoints/${CHECKPOINT_NAME}"
  fi
  unset XLLM_DISABLE_EVAL
fi

mkdir -p "${SAVE}/restarts"
cp "$0" "${SAVE}/run_${PHASE}.sbatch"

export NCCL_DEBUG=WARN
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export ENABLE_FLASH_ATTENTION_3=True
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export XLLM_LOG_DATA_SOURCES=1
export XLLM_LOG_DATA_SOURCES_FREQ=1
export XLLM_LOG_DATA_SOURCES_MAX_SEQS=0
export XLLM_LOG_DATA_SOURCES_DIR="${SAVE}/data_source_logs"
unset XLLM_LOG_DATALOADER_TIMING
unset XLLM_DATALOADER_TIMING_DIR

cd "${REPO}"
echo "PHASE=${PHASE} RUN_TAG=${RUN_TAG} SAVE=${SAVE}"

srun --cpu-bind=none -u --label python -u train.py \
  --context_parallel_size 1 \
  --model_parallel_size 1 \
  --slurm.partition main \
  --slurm.mem_gb 128 \
  --model k2-horizon-0.9B \
  --model.efficient_attn flash \
  --model.chunk_size 1024 \
  --model.fused_block true \
  --model.recompute_v true \
  --model.recompute_attention false \
  --model.recompute_fc1_out true \
  --model.recompute_fc3_out true \
  --model.recompute_logits false \
  --model.init_mode gaussian \
  --model.init_std 0.02 \
  --model.rope_head_dim 64 \
  --model.rope_base 500000 \
  --data "${DATA}" \
  --dataloader.buffer_size "${BUFFER_SIZE}" \
  --dataloader.packing_type bestfit \
  --dataloader.skip_long_docs false \
  --dataloader.num_workers 2 \
  --tokenizer.path "${TOKENIZER}" \
  --tokenizer.type huggingface \
  --steps "${STEPS}" \
  --seed 42 \
  --dtype bf16 \
  --batch_size 1 \
  --seq_len 1024 \
  --multi_segments true \
  --deterministic true \
  --forward_prefetch true \
  --nccl_timeout 18000 \
  --cluster_check_level -1 \
  --sync_check_freq 1000000 \
  --optim.lr 8e-4 \
  --optim.scheduler cosine \
  --optim.lr_init_ratio 1e-4 \
  --optim.lr_end_ratio 0.1 \
  --optim.beta1 0.9 \
  --optim.beta2 0.95 \
  --optim.clip 1.0 \
  --optim.warmup 3000 \
  --optim.weight_decay 0.1 \
  --valid.ppl_files_str "${VALID_DATA}" \
  --valid.tasks_root_dir "${TASK_ROOT}" \
  --valid.tasks_str copa \
  --valid.batch_size 8 \
  --valid.seq_len 1024 \
  --valid.n_batches 1 \
  --async_eval_ngpus -1 \
  --async_checkpointing false \
  --log_freq 1 \
  --dump_freq "${DUMP_FREQ}" \
  --eval_freq "${STEPS}" \
  --keep_eval_checkpoints true \
  --keep_n_last_checkpoints 2 \
  --log_wandb false \
  --disable_workers_print false \
  --dump_dir "${SAVE}"
