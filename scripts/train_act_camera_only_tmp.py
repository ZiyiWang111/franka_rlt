#!/usr/bin/env python3
"""Train a fresh camera-only ACT using a reference run's exact training settings.

Use prepare_act_camera_only_action6_tmp.py first. The six TCP delta labels are
unchanged; proprio and the gripper target are excluded. Under torchrun,
--batch-size means GLOBAL batch size and is divided between processes.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
import textwrap

import draccus
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig


@dataclass
class CameraOnlyTrainConfig(TrainPipelineConfig):
    def validate(self):
        scheduler = self.scheduler
        super().validate()
        if scheduler is not None:
            self.scheduler = scheduler  # ACT's preset would otherwise erase cosine scheduling.
        if not isinstance(self.policy, ACTConfig):
            raise ValueError("An ACT reference checkpoint is required")
        if self.policy.robot_state_feature or self.policy.env_state_feature:
            raise ValueError("Camera-only training must have no proprio/environment state feature")
        if not self.policy.image_features or self.policy.use_vae:
            raise ValueError("This temporary entry point requires visual inputs and use_vae=False")
        if tuple(self.policy.action_feature.shape) != (6,):
            raise ValueError("This run must output only six-dimensional TCP deltas")


def install_camera_only_forward():
    """Fix LeRobot's device lookup without adding a dummy proprio input.

    Some ACT versions support omitting the state token but still read its device
    when allocating the zero latent. Use the image device instead. No installed
    package files are changed. Call this helper when loading this model for
    image-only inference with affected LeRobot versions as well.
    """
    from lerobot.policies.act import modeling_act

    if getattr(modeling_act.ACT.forward, "_camera_only_device_fix", False):
        return
    source = textwrap.dedent(inspect.getsource(modeling_act.ACT.forward))
    source = source.replace("batch[OBS_STATE].device", "batch[OBS_IMAGES][0].device")
    namespace = dict(vars(modeling_act))
    exec(compile(source, "<camera_only_act_forward>", "exec"), namespace)
    original_forward = namespace["forward"]

    def forward(self, batch):
        if not self.config.robot_state_feature and not self.config.env_state_feature:
            # The dataset may contain state, but none of it enters the model.
            batch = {
                key: value for key, value in batch.items()
                if not key.startswith("observation.")
                or key == modeling_act.OBS_IMAGES or key in self.config.image_features
            }
        return original_forward(self, batch)

    forward._camera_only_device_fix = True
    modeling_act.ACT.forward = forward


def resolve_reference(path):
    for candidate in (path, path / "pretrained_model", path / "checkpoints/last/pretrained_model"):
        if (candidate / "train_config.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Reference train_config.json not found under {path}")


def build_config(args):
    reference = resolve_reference(args.reference_checkpoint.expanduser())
    original = json.loads((reference / "train_config.json").read_text())
    data = deepcopy(original)
    root = args.root.expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text())
    if data["policy"]["type"] != "act":
        raise ValueError("Reference policy is not ACT")
    visual = {
        key: feature for key, feature in data["policy"]["input_features"].items()
        if feature["type"] == "VISUAL"
    }
    if not visual:
        raise ValueError("Reference must have at least one camera")
    for key, feature in visual.items():
        recorded = info["features"].get(key, {})
        shape = feature["shape"]
        if recorded.get("shape") != [shape[1], shape[2], shape[0]]:
            raise ValueError(f"Dataset camera does not match reference: {key}")
    if info["features"]["action"]["shape"] != [6] or info["features"]["action"]["names"] != [
        "dx", "dy", "dz", "drx", "dry", "drz",
    ]:
        raise ValueError("Prepare a separate action6 dataset with prepare_act_camera_only_action6_tmp.py first")
    if "observation.state" in info["features"]:
        raise ValueError("The prepared camera-only dataset must not include observation.state")
    data["policy"]["input_features"] = visual
    data["policy"]["output_features"]["action"]["shape"] = [6]
    data["policy"]["pretrained_path"] = None  # New ACT weights, not a state7 checkpoint resume.
    data["policy"]["device"] = args.device
    data["policy"]["push_to_hub"] = False
    data["dataset"].update(root=str(root), repo_id=root.name)
    data.update(output_dir=str(args.output.expanduser().resolve()), resume=False,
                checkpoint_path=None, job_name="act_camera_only")
    data["wandb"]["enable"] = False
    for name in ("steps", "batch_size", "num_workers", "save_freq", "log_freq"):
        value = getattr(args, name, None)
        if value is not None:
            data[name] = value
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch = data["batch_size"]
    if world_size < 1 or global_batch <= 0 or global_batch % world_size:
        raise ValueError("Global batch size must be positive and divisible by WORLD_SIZE")
    data["batch_size"] = global_batch // world_size
    if args.steps is not None and data.get("scheduler") is not None:
        data["scheduler"]["num_decay_steps"] = args.steps
        data["scheduler"]["num_warmup_steps"] = min(
            data["scheduler"]["num_warmup_steps"], max(1, args.steps // 10)
        )
    cfg = draccus.decode(CameraOnlyTrainConfig, data)
    cfg.validate()
    return cfg, reference, original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, help="GLOBAL batch size (default: reference batch size)")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--save-freq", type=int)
    parser.add_argument("--log-freq", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cfg, reference, original = build_config(args)
    print(json.dumps({
        "reference": str(reference), "dataset": str(cfg.dataset.root),
        "output": str(cfg.output_dir), "input_features": list(cfg.policy.input_features),
        "action": {"shape": [6], "names": ["dx", "dy", "dz", "drx", "dry", "drz"]},
        "steps": cfg.steps, "batch_size_per_process": cfg.batch_size,
        "processes": int(os.environ.get("WORLD_SIZE", "1")),
        "global_batch_size": cfg.batch_size * int(os.environ.get("WORLD_SIZE", "1")),
        "seed": cfg.seed, "scheduler": draccus.encode(cfg.scheduler),
    }, indent=2), flush=True)
    if args.validate_only:
        print("Validated: only camera features enter ACT; source dataset is unchanged.", flush=True)
        return
    install_camera_only_forward()
    from lerobot.scripts.lerobot_train import train

    train.__wrapped__(cfg)
    # Store a standalone compatibility helper next to the completed run.
    import shutil
    if int(os.environ.get("RANK", "0")) == 0:
        shutil.copy2(__file__, Path(cfg.output_dir) / Path(__file__).name)


if __name__ == "__main__":
    main()
