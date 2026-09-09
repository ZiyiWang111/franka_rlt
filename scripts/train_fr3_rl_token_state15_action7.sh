#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/workspace/wangziyi/projects/franka_rlt
PYTHON_BIN=/workspace/wangziyi/miniconda3/envs/evo-rlt/bin/python
GPU_INDEX="${GPU_INDEX:-4}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-2000}"
LOG_FREQUENCY="${LOG_FREQUENCY:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/fr3_rl_token_state15_action7_bs${TRAIN_BATCH_SIZE}_${TRAIN_STEPS}steps}"

if [[ -n "${RUN_LOG:-}" ]]; then
  exec > >(tee -a "${RUN_LOG}") 2>&1
fi

DATASET_ROOT=${PROJECT_ROOT}/datasets/franka_m1_manual_demo_state15_action7
VLA_CHECKPOINT=${PROJECT_ROOT}/outputs/fr3_pi05_sft_60ep_state15_action7_bs4_30k/checkpoints/030000/pretrained_model
TOKENIZER_SNAPSHOT=/workspace/wangziyi/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c

for required_path in \
  "${DATASET_ROOT}/meta/info.json" \
  "${VLA_CHECKPOINT}/model.safetensors" \
  "${VLA_CHECKPOINT}/policy_preprocessor.json" \
  "${TOKENIZER_SNAPSHOT}/tokenizer.json"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing required file: ${required_path}" >&2
    exit 1
  fi
done

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output directory: ${OUTPUT_DIR}" >&2
  exit 1
fi

IFS=',' read -r gpu_memory_mib gpu_utilization_pct < <(
  nvidia-smi \
    --id="${GPU_INDEX}" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits
)
gpu_memory_mib="${gpu_memory_mib//[[:space:]]/}"
gpu_utilization_pct="${gpu_utilization_pct//[[:space:]]/}"
if (( gpu_memory_mib > 1024 || gpu_utilization_pct > 10 )); then
  echo "Refusing to start: GPU ${GPU_INDEX} is busy (${gpu_memory_mib} MiB, ${gpu_utilization_pct}%)." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export HF_HOME=/workspace/wangziyi/.cache/huggingface
export HF_DATASETS_CACHE=/tmp/fr3-rlt-token-state15-action7-hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

cd "${PROJECT_ROOT}"

exec "${PYTHON_BIN}" \
  -c 'from evo_rlt.adapters.lerobot import register; register(); from lerobot.scripts.lerobot_train import main; main()' \
  --dataset.repo_id=franka_m1_manual_demo_state15_action7 \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=rlt_token \
  --policy.push_to_hub=false \
  --policy.input_features=null \
  --policy.vla_pretrained_path="${VLA_CHECKPOINT}" \
  --policy.vla_dtype=bfloat16 \
  --policy.vla_ft_weight=0 \
  --policy.rl_token_num_rl_tokens=1 \
  --policy.tokenizer_path="${TOKENIZER_SNAPSHOT}" \
  --policy.token_pool_size=0 \
  --policy.image_only=false \
  --policy.camera_keys='[wrist,front]' \
  --policy.proprio_dim=15 \
  --policy.action_dim=7 \
  --policy.device=cuda \
  --batch_size="${TRAIN_BATCH_SIZE}" \
  --steps="${TRAIN_STEPS}" \
  --save_freq="${SAVE_FREQUENCY}" \
  --log_freq="${LOG_FREQUENCY}" \
  --eval_freq=0 \
  --tolerance_s=0.04 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=fr3_rl_token_state15_action7
