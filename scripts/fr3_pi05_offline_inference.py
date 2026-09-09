#!/usr/bin/env python3
"""Run one π0.5 action-chunk inference from a real FR3 dataset sample.

This script is deliberately offline with respect to the robot: it only imports
LeRobot dataset/policy utilities, reads a local dataset and checkpoint, and
prints the predicted action chunk.  It never creates a robot, connects to a
control server, or sends an action.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets/franka_m1_manual_demo_state15_action7"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/fr3_pi05_sft_60ep_state15_action7_bs4_30k"
    / "checkpoints/005000/pretrained_model"
)

# Hugging Face datasets creates lock files even when all dataset files are local.
# Keep those transient files outside the repository and prohibit network access.
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/fr3_pi05_offline_inference_hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.feature_utils import dataset_to_policy_features  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.utils import prepare_observation_for_inference  # noqa: E402


ACTION_KEY = "action"
STATE_KEY = "observation.state"
GRIPPER_ACTION_NAME = "gripper_target_width"
REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--dataset-repo-id",
        default="franka_m1_manual_demo_state15_action7",
        help="Local LeRobotDataset repo_id (no Hub access is performed).",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-print-actions", type=int, default=5)
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Read a real sample and validate schemas without loading the model or using CUDA.",
    )
    return parser.parse_args()


def feature_shape(feature: Any) -> tuple[int, ...]:
    """Read a feature shape from either a PolicyFeature or a JSON dictionary."""
    shape = feature.shape if hasattr(feature, "shape") else feature["shape"]
    return tuple(int(value) for value in shape)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Schema / processor mismatch: {message}")


def load_processor_json(checkpoint: Path, filename: str) -> dict[str, Any]:
    with (checkpoint / filename).open() as stream:
        return json.load(stream)


def processor_feature_shapes(processor_json: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Collect feature shapes stored in normalizer/unnormalizer processor steps."""
    shapes: dict[str, tuple[int, ...]] = {}
    for step in processor_json["steps"]:
        features = step.get("config", {}).get("features", {})
        for key, feature in features.items():
            shapes[key] = tuple(int(value) for value in feature["shape"])
    return shapes


def check_schema(
    *,
    checkpoint: Path,
    dataset: LeRobotDataset,
    sample: dict[str, Any],
    policy_config: PreTrainedConfig,
) -> tuple[list[str], list[str], int]:
    require(policy_config.type == "pi05", f"expected policy type pi05, got {policy_config.type!r}")

    dataset_policy_features = dataset_to_policy_features(dataset.meta.features)
    configured_input_keys = list(policy_config.input_features)
    configured_camera_keys = [key for key in configured_input_keys if key.startswith("observation.images.")]
    require(configured_camera_keys, "checkpoint declares no camera inputs")

    for key, configured_feature in policy_config.input_features.items():
        require(key in dataset_policy_features, f"checkpoint input {key!r} is absent from the dataset")
        dataset_shape = feature_shape(dataset_policy_features[key])
        configured_shape = feature_shape(configured_feature)
        require(
            dataset_shape == configured_shape,
            f"{key} dataset shape {dataset_shape} != checkpoint shape {configured_shape}",
        )
        require(key in sample, f"real sample does not contain {key!r}")
        require(
            tuple(sample[key].shape) == configured_shape,
            f"{key} sample shape {tuple(sample[key].shape)} != checkpoint shape {configured_shape}",
        )

    require(ACTION_KEY in policy_config.output_features, "checkpoint has no action output")
    action_dim = feature_shape(policy_config.output_features[ACTION_KEY])[0]
    dataset_action_shape = feature_shape(dataset_policy_features[ACTION_KEY])
    require(dataset_action_shape == (action_dim,), f"dataset action shape {dataset_action_shape} != ({action_dim},)")
    require(tuple(sample[ACTION_KEY].shape) == (action_dim,), "sample action shape differs from checkpoint")

    action_names = list(dataset.meta.features[ACTION_KEY].get("names") or [])
    configured_action_names = list(getattr(policy_config, "action_feature_names", None) or [])
    require(len(action_names) == action_dim, f"dataset has {len(action_names)} action names for {action_dim}D action")
    require(
        configured_action_names == action_names,
        f"checkpoint action names {configured_action_names} != dataset action names {action_names}",
    )
    if action_dim == 7:
        require(
            action_names[6] == GRIPPER_ACTION_NAME,
            f"7th action is {action_names[6]!r}, expected {GRIPPER_ACTION_NAME!r}",
        )

    pre_shapes = processor_feature_shapes(load_processor_json(checkpoint, "policy_preprocessor.json"))
    post_shapes = processor_feature_shapes(load_processor_json(checkpoint, "policy_postprocessor.json"))
    for key, configured_feature in policy_config.input_features.items():
        require(
            pre_shapes.get(key) == feature_shape(configured_feature),
            f"preprocessor shape for {key} is {pre_shapes.get(key)}, expected {feature_shape(configured_feature)}",
        )
    require(pre_shapes.get(ACTION_KEY) == (action_dim,), "preprocessor action shape is inconsistent")
    require(post_shapes.get(ACTION_KEY) == (action_dim,), "postprocessor action shape is inconsistent")

    return configured_camera_keys, action_names, action_dim


def dataset_sample_to_raw_observation(
    sample: dict[str, Any], camera_keys: list[str]
) -> dict[str, np.ndarray]:
    """Convert LeRobot's CHW float images back to the raw HWC uint8 inference form."""
    observation: dict[str, np.ndarray] = {
        STATE_KEY: sample[STATE_KEY].detach().cpu().numpy().astype(np.float32, copy=True)
    }
    for key in camera_keys:
        image = sample[key].detach().cpu()
        require(image.ndim == 3 and image.shape[0] == 3, f"{key} is not a CHW RGB image: {tuple(image.shape)}")
        observation[key] = (
            image.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
        )
    return observation


def predict_full_action_chunk(
    *,
    raw_observation: dict[str, np.ndarray],
    task: str,
    robot_type: str,
    policy: torch.nn.Module,
    preprocessor: Any,
    postprocessor: Any,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Run the same inference path as predict_action(), retaining all chunk steps."""
    prepared = prepare_observation_for_inference(
        dict(raw_observation), device=device, task=task, robot_type=robot_type
    )
    with torch.inference_mode():
        processed = preprocessor(prepared)
        normalized_chunk = policy.predict_action_chunk(processed)
        action_chunk = postprocessor(normalized_chunk)
    return processed, action_chunk


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()

    missing_files = [name for name in REQUIRED_CHECKPOINT_FILES if not (checkpoint / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Incomplete checkpoint {checkpoint}; missing: {missing_files}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    print(f"Checkpoint: {checkpoint}")
    print(f"Dataset:    {dataset_root}")
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=dataset_root,
        video_backend="torchcodec",
    )
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"sample-index {args.sample_index} is outside [0, {len(dataset)})")
    sample = dataset[args.sample_index]
    task = sample.get("task")
    require(isinstance(task, str) and bool(task.strip()), f"sample task must be non-empty text, got {task!r}")

    policy_config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    camera_keys, action_names, action_dim = check_schema(
        checkpoint=checkpoint,
        dataset=dataset,
        sample=sample,
        policy_config=policy_config,
    )

    print(f"Sample index: {args.sample_index}")
    for key in camera_keys:
        print(f"Input {key} shape: {tuple(sample[key].shape)}")
    print(f"Input {STATE_KEY} shape: {tuple(sample[STATE_KEY].shape)}")
    print(f"Task text: {task!r}")
    print(f"Action schema ({action_dim}D): {action_names}")
    current_gripper_width: float | None = None
    if action_dim == 7:
        print(f"7th action: {action_names[6]} (metres)")
        state_names = list(dataset.meta.features[STATE_KEY].get("names") or [])
        require("gripper_width" in state_names, "15D state has no gripper_width name")
        gripper_state_index = state_names.index("gripper_width")
        current_gripper_width = float(sample[STATE_KEY][gripper_state_index])
        print(f"Current sample gripper_width: {current_gripper_width:.6f} m")
    print("Schema / processor check: PASS")

    if args.schema_only:
        print("Schema-only mode: model loading and inference skipped")
        return

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This 4.1B π0.5 checkpoint dry run is configured for CUDA; pass --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process")

    policy_config.device = str(device)
    policy_config.pretrained_path = str(checkpoint)
    print(f"Loading checkpoint on {device} ...")
    policy = make_policy(policy_config, ds_meta=dataset.meta)
    policy.eval()
    policy.reset()
    print("Checkpoint load: PASS")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    raw_observation = dataset_sample_to_raw_observation(sample, camera_keys)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    processed, batched_action_chunk = predict_full_action_chunk(
        raw_observation=raw_observation,
        task=task,
        robot_type=dataset.meta.robot_type,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
    )

    require(
        batched_action_chunk.ndim == 3 and batched_action_chunk.shape[0] == 1,
        f"model returned unexpected batched chunk shape {tuple(batched_action_chunk.shape)}",
    )
    action_chunk = batched_action_chunk[0].detach().cpu().to(torch.float32)
    expected_shape = (int(policy_config.chunk_size), action_dim)
    require(
        tuple(action_chunk.shape) == expected_shape,
        f"output chunk shape {tuple(action_chunk.shape)} != expected {expected_shape}",
    )
    require(bool(torch.isfinite(action_chunk).all()), "output action chunk contains NaN or Inf")

    print(f"Processed {STATE_KEY} shape: {tuple(processed[STATE_KEY].shape)}")
    for key in camera_keys:
        print(f"Processed {key} shape: {tuple(processed[key].shape)}")
    print(f"Output action chunk shape: {tuple(action_chunk.shape)}")
    rows_to_print = min(max(args.num_print_actions, 0), action_chunk.shape[0])
    np.set_printoptions(precision=6, suppress=True, linewidth=160)
    print(f"First {rows_to_print} action rows ({', '.join(action_names)}):")
    print(action_chunk[:rows_to_print].numpy())
    if action_dim == 7:
        gripper = action_chunk[:, 6]
        print(
            "Predicted gripper_target_width range: "
            f"{gripper.min().item():.6f} .. {gripper.max().item():.6f} m"
        )
        assert current_gripper_width is not None
        close_margin_m = 0.005
        close_tendency = bool(gripper.min().item() < current_gripper_width - close_margin_m)
        print(
            f"Chunk contains >= {close_margin_m:.3f} m closing tendency: "
            f"{'YES' if close_tendency else 'NO'}"
        )
    print("Inference: PASS")
    print("No robot/control-server code was imported or called.")


if __name__ == "__main__":
    main()
