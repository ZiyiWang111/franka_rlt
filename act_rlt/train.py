#!/usr/bin/env python3
"""Train ACT with the active Python environment's LeRobot installation."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


PRESETS = {
    "standard": (20000, 8),
    "lowmem": (20000, 2),
    "smoke": (100, 2),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "datasets/act_rlt_001",
                        help="Exact dataset directory containing meta/info.json")
    parser.add_argument("--repo-id", help="Defaults to the dataset directory name")
    parser.add_argument("--output", type=Path, default=Path("runs/act_rlt_001"))
    parser.add_argument("--preset", choices=PRESETS, default="standard",
                        help="standard: 20000 steps/batch 8; lowmem: 20000/2; smoke: 100/2")
    parser.add_argument("--steps", type=int, help="Override preset steps")
    parser.add_argument("--batch-size", type=int, help="Override preset batch size")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--n-action-steps", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=10000)
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine",
                        help="Warmup + cosine decay (default), or a constant learning rate")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Warmup steps, capped at 10%% of the run")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1,
                        help="Final/initial learning-rate ratio for both parameter groups")
    parser.add_argument("--log-freq", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without training")
    args = parser.parse_args()

    steps, batch = PRESETS[args.preset]
    steps = args.steps if args.steps is not None else steps
    batch = args.batch_size if args.batch_size is not None else batch
    if min(steps, batch, args.chunk_size, args.n_action_steps, args.save_freq, args.log_freq) <= 0:
        parser.error("steps, batch size, chunk sizes and save/log frequencies must be positive")
    if args.workers < 0 or not 0 < args.lr < float("inf"):
        parser.error("workers must be non-negative and lr must be finite and positive")
    if args.n_action_steps > args.chunk_size:
        parser.error("n-action-steps must be <= chunk-size")
    if args.warmup_steps < 0 or not 0 <= args.min_lr_ratio <= 1:
        parser.error("warmup-steps must be non-negative and min-lr-ratio must be in [0, 1]")

    root = args.root.expanduser().resolve()
    if not (root / "meta/info.json").is_file():
        parser.error(f"Missing {root / 'meta/info.json'}; --root must be the dataset itself")
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}; choose a new --output directory")
    info = json.loads((root / "meta/info.json").read_text())
    print(f"Dataset: {root} ({info['total_episodes']} episodes, {info['total_frames']} frames)", flush=True)

    # Same ACT architecture as RoboLab. Lowmem/smoke only change batch/steps.
    options = {
        "dataset.repo_id": args.repo_id or root.name,
        "dataset.root": root,
        "dataset.video_backend": "pyav",
        "dataset.image_transforms.enable": args.augment,
        "output_dir": output,
        "policy.type": "act",
        "policy.device": args.device,
        "policy.push_to_hub": False,
        "policy.chunk_size": args.chunk_size,
        "policy.n_action_steps": args.n_action_steps,
        "policy.dim_model": 768,
        "policy.dim_feedforward": 3200,
        "policy.n_encoder_layers": 4,
        "policy.n_decoder_layers": 1,
        "policy.vision_backbone": "resnet18",
        "policy.use_vae": False,
        "policy.optimizer_lr": args.lr,
        "policy.optimizer_lr_backbone": 1e-5,
        "use_policy_training_preset": True,
        "steps": steps,
        "batch_size": batch,
        "num_workers": args.workers,
        "save_checkpoint": True,
        "save_freq": min(args.save_freq, steps),
        "log_freq": min(args.log_freq, steps),
        "seed": args.seed,
        "wandb.enable": args.wandb,
    }
    if args.scheduler == "cosine":
        options.update({
            "scheduler.type": "cosine_decay_with_warmup",
            "scheduler.num_warmup_steps": min(args.warmup_steps, steps // 10),
            "scheduler.num_decay_steps": steps,
            "scheduler.peak_lr": args.lr,
            "scheduler.decay_lr": args.lr * args.min_lr_ratio,
        })
        # LeRobot's ACT preset otherwise replaces an explicit scheduler with None.
        command = [sys.executable, "-u", str(Path(__file__).with_name("train_scheduled.py"))]
    else:
        command = [sys.executable, "-u", "-m", "lerobot.scripts.lerobot_train"]
    command += [f"--{key}={str(value).lower() if isinstance(value, bool) else value}"
                for key, value in options.items()]
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
