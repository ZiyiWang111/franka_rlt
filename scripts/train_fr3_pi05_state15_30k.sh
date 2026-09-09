#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export HF_HOME=/workspace/wangziyi/.cache/huggingface
export HF_DATASETS_CACHE=/tmp/fr3-pi05-state15-train-hf-cache
export HF_TOKEN="$(</workspace/wangziyi/.cache/huggingface/token)"

cd /workspace/wangziyi

exec /workspace/wangziyi/miniconda3/envs/evo-rlt/bin/python \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=franka_m1_manual_demo_state15 \
  --dataset.root=/workspace/wangziyi/projects/franka_rlt/datasets/franka_m1_manual_demo_state15 \
  --policy.path=/workspace/wangziyi/models/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false \
  --policy.input_features=null \
  --batch_size=4 \
  --steps=30000 \
  --save_freq=5000 \
  --eval_freq=0 \
  --tolerance_s=0.04 \
  --output_dir=/workspace/wangziyi/projects/franka_rlt/outputs/fr3_pi05_sft_60ep_state15_bs4_30k \
  --job_name=fr3_pi05_sft_60ep_state15_bs4_30k
