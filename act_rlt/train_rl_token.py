#!/usr/bin/env python3
"""Train Stage-1 RL Token reconstruction on a frozen ACT encoder."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from act_rlt.infer import resolve_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets/act_rlt_001_state7",
        help="7D-state LeRobot dataset directory containing meta/info.json",
    )
    parser.add_argument("--repo-id", help="Defaults to the dataset directory name")
    parser.add_argument(
        "--act-checkpoint",
        type=Path,
        default=Path("outputs/act_rlt_001_state7_act"),
        help="ACT run, checkpoint, or pretrained_model directory",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/act_rlt_001_stage1_pos"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=2_000)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--verify-extractor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare encoder-only extraction with an ACT.forward hook on the first batch",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from OUTPUT/checkpoints/last, including optimizer/scheduler/RNG state",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if min(args.steps, args.batch_size, args.save_freq, args.log_freq) <= 0:
        parser.error("steps, batch size and save/log frequencies must be positive")
    if args.workers < 0:
        parser.error("workers must be non-negative")

    output = args.output.expanduser().resolve()
    if args.resume:
        train_config = output / "checkpoints/last/pretrained_model/train_config.json"
        if not train_config.is_file():
            parser.error(f"Cannot resume: missing {train_config}")
        command = [
            sys.executable,
            "-u",
            "-c",
            "from act_rlt import register; register(); "
            "from lerobot.scripts.lerobot_train import main; main()",
            f"--config_path={train_config}",
            "--resume=true",
        ]
        print(f"Resuming: {output}", flush=True)
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return
        _run(command)
        return

    root = args.root.expanduser().resolve()
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        parser.error(f"Missing {info_path}; --root must be the dataset itself")
    info = json.loads(info_path.read_text())
    state_feature = info["features"].get("observation.state", {})
    if state_feature.get("shape") != [7]:
        parser.error(
            f"Stage 1 requires the 7D joint-state dataset, got {state_feature.get('shape')} at {root}"
        )

    try:
        act_checkpoint = resolve_checkpoint(args.act_checkpoint)
    except FileNotFoundError as error:
        parser.error(str(error))

    if output.exists():
        parser.error(f"Output already exists: {output}; choose a new --output directory")

    options = {
        "dataset.repo_id": args.repo_id or root.name,
        "dataset.root": root,
        "dataset.video_backend": "pyav",
        "dataset.image_transforms.enable": False,
        "policy.type": "act_rlt_token",
        "policy.input_features": "null",
        "policy.act_pretrained_path": act_checkpoint,
        "policy.verify_encoder_equivalence": args.verify_extractor,
        "policy.device": args.device,
        "policy.use_amp": False,
        "policy.push_to_hub": False,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "num_workers": args.workers,
        "save_checkpoint": True,
        "save_freq": min(args.save_freq, args.steps),
        "log_freq": min(args.log_freq, args.steps),
        "eval_freq": 0,
        "output_dir": output,
        "job_name": "act_rlt_stage1",
    }
    command = [
        sys.executable,
        "-u",
        "-c",
        "from act_rlt import register; register(); "
        "from lerobot.scripts.lerobot_train import main; main()",
    ]
    command += [
        f"--{key}={str(value).lower() if isinstance(value, bool) else value}"
        for key, value in options.items()
    ]

    print(
        f"Dataset: {root} ({info['total_episodes']} episodes, {info['total_frames']} frames)",
        flush=True,
    )
    print(f"Frozen ACT: {act_checkpoint}", flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return

    _run(command)


def _run(command: list[str]) -> None:
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    source_root = str(project_root / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_root, str(project_root), env.get("PYTHONPATH")) if part
    )
    raise SystemExit(subprocess.call(command, cwd=project_root, env=env))


if __name__ == "__main__":
    main()
