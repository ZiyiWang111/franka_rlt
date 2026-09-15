#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX=1
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
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export HF_HOME=/workspace/wangziyi/.cache/huggingface
export HF_DATASETS_CACHE=/tmp/fr3-pi05-state15-action7-train-hf-cache
export HF_TOKEN="$(</workspace/wangziyi/.cache/huggingface/token)"

cd /workspace/wangziyi

exec /workspace/wangziyi/miniconda3/envs/evo-rlt/bin/python \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=franka_m1_manual_demo_state15_action7 \
  --dataset.root=/workspace/wangziyi/projects/franka_rlt/datasets/franka_m1_manual_demo_state15_action7 \
  --policy.path=/workspace/wangziyi/models/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false \
  --policy.input_features=null \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_decay_lr=5e-6 \
  --batch_size=16 \
  --steps=30000 \
  --save_freq=5000 \
  --eval_freq=0 \
  --tolerance_s=0.04 \
  --output_dir=/workspace/wangziyi/projects/franka_rlt/outputs/fr3_pi05_sft_60ep_state15_action7_bs16_chunk10_lr2x_30k \
  --job_name=fr3_pi05_sft_60ep_state15_action7_bs16_chunk10_lr2x_30k
