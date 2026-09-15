#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/workspace/wangziyi/miniconda3/envs/evo-rlt/bin/python}"
BASE_MODEL="${BASE_MODEL:-/workspace/wangziyi/models/pi05_base}"
GPU_INDEX="${GPU_INDEX:-2}"
cd "$PROJECT_ROOT"
test -f datasets/fa1_s15/meta/info.json
test -f "$BASE_MODEL/model.safetensors"
if [[ -e outputs/fa1_rgb ]]; then
  echo "Output already exists: outputs/fa1_rgb" >&2
  exit 1
fi
IFS=',' read -r used util < <(nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
if (( used > 1024 || util > 10 )); then
  echo "GPU $GPU_INDEX is busy: $used MiB, $util%" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-/workspace/wangziyi/.cache/huggingface}"
export HF_DATASETS_CACHE=/tmp/fa1-rgb-hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$PYTHON_BIN" -m evo_rlt.cli.train_pi05_rgb \
  --discrete-state-input=false \
  --dataset.repo_id=fa1_s15 \
  --dataset.root="$PROJECT_ROOT/datasets/fa1_s15" \
  --policy.path="$BASE_MODEL" \
  --policy.device=cuda --policy.dtype=bfloat16 \
  --policy.push_to_hub=false --policy.input_features=null \
  --policy.use_relative_actions=false \
  --policy.chunk_size=10 --policy.n_action_steps=10 \
  --policy.optimizer_lr=5e-5 --policy.scheduler_decay_lr=5e-6 \
  --policy.gradient_checkpointing=true \
  --batch_size=16 --steps=30000 --save_freq=5000 --log_freq=50 \
  --eval_freq=0 --tolerance_s=0.04 --seed=1000 \
  --output_dir="$PROJECT_ROOT/outputs/fa1_rgb" --job_name=fa1_rgb
