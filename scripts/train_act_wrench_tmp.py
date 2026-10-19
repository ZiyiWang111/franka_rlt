#!/usr/bin/env python3
"""Train a fresh RGB + external-wrench ACT with six TCP delta outputs.

The state feature contains only fx/fy/fz/mx/my/mz, never joints, TCP pose or
gripper state. --batch-size is GLOBAL batch size, split under torchrun DDP.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil

import draccus
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig


WRENCH_NAMES = ["fx", "fy", "fz", "mx", "my", "mz"]
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz"]


@dataclass
class WrenchTrainConfig(TrainPipelineConfig):
    def validate(self):
        scheduler = self.scheduler
        super().validate()
        if scheduler is not None:
            self.scheduler = scheduler
        if not isinstance(self.policy, ACTConfig):
            raise ValueError("An ACT reference checkpoint is required")
        state = self.policy.robot_state_feature
        if state is None or tuple(state.shape) != (6,) or self.policy.env_state_feature:
            raise ValueError("The only nonvisual model input must be six-dimensional wrench")
        if not self.policy.image_features or self.policy.use_vae:
            raise ValueError("This run requires RGB input and use_vae=False")
        if tuple(self.policy.action_feature.shape) != (6,):
            raise ValueError("This run must output six TCP deltas")


def resolve_reference(path):
    for candidate in (path, path / "pretrained_model", path / "checkpoints/last/pretrained_model"):
        if (candidate / "train_config.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Reference train_config.json not found under {path}")


def build_config(args):
    reference = resolve_reference(args.reference_checkpoint.expanduser())
    data = deepcopy(json.loads((reference / "train_config.json").read_text()))
    root = args.root.expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text())
    if data["policy"]["type"] != "act":
        raise ValueError("Reference must be ACT")
    state = info["features"].get("observation.state", {})
    action = info["features"].get("action", {})
    if state.get("shape") != [6] or state.get("names") != WRENCH_NAMES:
        raise ValueError("Prepare the dataset with --wrench-as-state first")
    if action.get("shape") != [6] or action.get("names") != ACTION_NAMES:
        raise ValueError("Expected six TCP delta labels")
    visual = {key: value for key, value in data["policy"]["input_features"].items()
              if value["type"] == "VISUAL"}
    if not visual:
        raise ValueError("Reference must include RGB input")
    for key, value in visual.items():
        shape = value["shape"]
        if info["features"].get(key, {}).get("shape") != [shape[1], shape[2], shape[0]]:
            raise ValueError(f"Camera shape differs from reference: {key}")
    data["policy"]["input_features"] = {
        **visual, "observation.state": {"type": "STATE", "shape": [6]},
    }
    data["policy"]["output_features"] = {"action": {"type": "ACTION", "shape": [6]}}
    data["policy"].update(pretrained_path=None, device=args.device, push_to_hub=False)
    data["policy"]["normalization_mapping"]["STATE"] = "MEAN_STD"
    data["dataset"].update(root=str(root), repo_id=root.name)
    data.update(output_dir=str(args.output.expanduser().resolve()), resume=False,
                checkpoint_path=None, job_name="act_rgb_wrench_action6")
    data["wandb"]["enable"] = False
    for key in ("steps", "batch_size", "num_workers", "save_freq", "log_freq"):
        value = getattr(args, key, None)
        if value is not None:
            data[key] = value
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1 or data["batch_size"] <= 0 or data["batch_size"] % world_size:
        raise ValueError("Global batch size must be positive and divisible by WORLD_SIZE")
    data["batch_size"] //= world_size
    if args.steps is not None and data.get("scheduler") is not None:
        data["scheduler"]["num_decay_steps"] = args.steps
        data["scheduler"]["num_warmup_steps"] = min(data["scheduler"]["num_warmup_steps"], max(1, args.steps // 10))
    cfg = draccus.decode(WrenchTrainConfig, data)
    cfg.validate()
    return cfg, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, help="global batch size; defaults to reference batch size")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--save-freq", type=int)
    parser.add_argument("--log-freq", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cfg, reference = build_config(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps({
            "reference": str(reference), "dataset": str(cfg.dataset.root), "output": str(cfg.output_dir),
            "visual_inputs": list(cfg.policy.image_features), "state_names": WRENCH_NAMES,
            "action_names": ACTION_NAMES, "steps": cfg.steps,
            "processes": int(os.environ.get("WORLD_SIZE", "1")), "batch_per_process": cfg.batch_size,
            "global_batch": cfg.batch_size * int(os.environ.get("WORLD_SIZE", "1")),
            "seed": cfg.seed, "scheduler": draccus.encode(cfg.scheduler),
        }, indent=2), flush=True)
    if args.validate_only:
        print("Validated RGB + wrench ACT, action6; no joint/TCP/gripper inputs.", flush=True)
        return
    from lerobot.scripts.lerobot_train import train

    train.__wrapped__(cfg)
    if int(os.environ.get("RANK", "0")) == 0:
        shutil.copy2(__file__, Path(cfg.output_dir) / Path(__file__).name)


if __name__ == "__main__":
    main()
