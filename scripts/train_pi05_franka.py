"""Train PI05 on joint7/TCP6 demonstrations with a fixed external gripper."""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def validate_dataset(root):
    info = json.loads((root / "meta/info.json").read_text())
    stats = json.loads((root / "meta/stats.json").read_text())
    expected = {
        "observation.state": [f"joint_{i}" for i in range(7)],
        "action": ["dx", "dy", "dz", "drx", "dry", "drz"],
    }
    for key, names in expected.items():
        feature = info["features"][key]
        if feature["shape"] != [len(names)] or feature["names"] != names:
            raise ValueError(f"Expected {key} {names}, got {feature}")
        low, high = stats[key]["q01"], stats[key]["q99"]
        if len(low) != len(names) or len(high) != len(names):
            raise ValueError(f"Invalid quantile dimensions for {key}")
        if any(not math.isfinite(a) or not math.isfinite(b) or b <= a for a, b in zip(low, high)):
            raise ValueError(f"Nonfinite/degenerate quantiles in {key}; prepare exact statistics first")
    if not any(f["dtype"] == "video" for f in info["features"].values()):
        raise ValueError("Dataset must contain a camera")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "datasets/pi05_dataset_928_state7_action6")
    parser.add_argument("--base-model", type=Path, required=True, help="Local PI05 base checkpoint directory")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/pi05_dataset_928_state7_action6")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--save-freq", type=int, default=10000)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--train-expert-only", action="store_true",
                        help="Optional VLM-freezing experiment; full fine-tuning is the default")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset and print command; does not load weights")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.chunk_size, args.n_action_steps, args.save_freq, args.log_freq) <= 0:
        parser.error("Steps, batch size, horizons and logging/saving intervals must be positive")
    if args.gpu < 0 or args.workers < 0 or args.warmup_steps < 0:
        parser.error("GPU, workers and warmup must be non-negative")
    if args.n_action_steps > args.chunk_size:
        parser.error("n-action-steps cannot exceed chunk-size")
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.min_lr_ratio <= 1:
        parser.error("Invalid learning rate or min-lr-ratio")
    root, base, output = (p.expanduser().resolve() for p in (args.root, args.base_model, args.output))
    if output.exists():
        parser.error(f"Output already exists: {output}")
    info = validate_dataset(root)
    options = {
        "dataset.repo_id": root.name, "dataset.root": root,
        "dataset.video_backend": "pyav", "dataset.image_transforms.enable": False,
        "policy.path": base, "policy.device": "cuda", "policy.dtype": "bfloat16",
        "policy.push_to_hub": False, "policy.input_features": "null",
        "policy.use_relative_actions": False,
        "policy.normalization_mapping": json.dumps({"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"}),
        "policy.chunk_size": args.chunk_size, "policy.n_action_steps": args.n_action_steps,
        "policy.optimizer_lr": args.lr,
        "policy.scheduler_warmup_steps": min(args.warmup_steps, args.steps // 10),
        "policy.scheduler_decay_steps": args.steps,
        "policy.scheduler_decay_lr": args.lr * args.min_lr_ratio,
        "policy.gradient_checkpointing": True,
        "policy.freeze_vision_encoder": args.train_expert_only,
        "policy.train_expert_only": args.train_expert_only,
        "use_policy_training_preset": True,
        "batch_size": args.batch_size, "steps": args.steps, "num_workers": args.workers,
        "save_checkpoint": True, "save_freq": min(args.save_freq, args.steps),
        "log_freq": min(args.log_freq, args.steps), "eval_freq": 0,
        "tolerance_s": 1e-4, "seed": args.seed, "wandb.enable": False,
        "output_dir": output, "job_name": "pi05_franka_state7_action6",
    }
    command = [sys.executable, "-u", "-m", "evo_rlt.cli.train_pi05_masked"]
    command += [f"--{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in options.items()]
    print(f"Dataset: {root} ({info['total_episodes']} episodes, {info['total_frames']} frames)", flush=True)
    print(f"GPU: {args.gpu}; state7 + camera + task -> TCP6; gripper held by deployment controller", flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        if not (base / "config.json").is_file():
            print(f"Dry-run only: base checkpoint not available locally at {base}", file=sys.stderr)
        return
    if not (base / "config.json").is_file() or not (base / "model.safetensors").is_file():
        parser.error(f"Missing local base checkpoint files: {base}")
    if json.loads((base / "config.json").read_text()).get("type") != "pi05":
        parser.error("Base checkpoint must be pi05")
    query = ["nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    used, util = [int(v.strip()) for v in subprocess.check_output(query, text=True).strip().split(",")]
    processes = subprocess.check_output(
        ["nvidia-smi", f"--id={args.gpu}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    ).strip()
    if used > 1024 or util > 10 or processes:
        parser.error(f"GPU {args.gpu} is busy: {used} MiB, {util}%, processes={processes!r}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), env.get("PYTHONPATH")]))
    env["PYTHONUNBUFFERED"] = "1"
    raise SystemExit(subprocess.call(command, cwd=PROJECT_ROOT, env=env))


if __name__ == "__main__":
    main()
