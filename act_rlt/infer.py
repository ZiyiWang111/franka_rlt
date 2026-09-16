"""Single-episode ACT inference: two cameras, seven joint angles, 15 Hz."""

import argparse
import time
from pathlib import Path

import numpy as np


ACTION_KEYS = ("dx", "dy", "dz", "drx", "dry", "drz")


def observation_frame(observation):
    import torch

    state = np.array([observation[f"joint_{i}"] for i in range(7)], dtype=np.float32)
    if not np.isfinite(state).all():
        raise ValueError("Non-finite joint state")
    frame = {"observation.state": torch.from_numpy(state)}
    for camera in ("wrist", "front"):
        pixels = np.asarray(observation[camera])
        if pixels.shape != (480, 640, 3) or pixels.dtype != np.uint8:
            raise ValueError(f"{camera}: expected uint8 RGB (480, 640, 3)")
        frame[f"observation.images.{camera}"] = (
            torch.from_numpy(pixels.copy()).permute(2, 0, 1).float().div(255)
        )
    return frame


def resolve_checkpoint(path):
    """Accept pretrained_model itself, a checkpoint directory, or a run directory."""
    path = Path(path).expanduser().resolve()
    candidates = (path, path / "pretrained_model", path / "checkpoints/last/pretrained_model")
    for candidate in candidates:
        required = (
            "config.json", "model.safetensors", "policy_preprocessor.json",
            "policy_postprocessor.json",
        )
        if all((candidate / name).is_file() for name in required):
            return candidate
    raise FileNotFoundError(
        f"No complete local pretrained_model found under {path}. "
        "Pass the downloaded run directory or checkpoints/last/pretrained_model."
    )


def bounded_action(values, max_translation, max_rotation):
    values = np.asarray(values, dtype=float).reshape(-1).copy()
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("Expected a finite 7D action")
    for start, limit in ((0, max_translation), (3, max_rotation)):
        norm = np.linalg.norm(values[start:start + 3])
        if norm > limit:
            values[start:start + 3] *= limit / norm
    # The gripper remains held; do not send repetitive gripper commands.
    return dict(zip(ACTION_KEYS, values[:6], strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--n-action-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--wrist-camera", default="349622072679")
    parser.add_argument("--front-camera", default="233522075778")
    parser.add_argument("--workspace-min", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--workspace-max", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--max-step-m", type=float, default=0.002)
    parser.add_argument("--max-step-rad", type=float, default=0.02)
    args = parser.parse_args()
    if not all(np.isfinite(x) and x > 0 for x in (args.duration, args.max_step_m, args.max_step_rad)):
        parser.error("duration and step limits must be finite and positive")
    if not args.dry_run and (args.workspace_min is None or args.workspace_max is None):
        parser.error("Motion requires --workspace-min X Y Z and --workspace-max X Y Z (base-frame metres)")
    if (args.workspace_min is None) != (args.workspace_max is None):
        parser.error("Supply both workspace bounds")
    if args.workspace_min is not None:
        bounds = np.array([args.workspace_min, args.workspace_max])
        if not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
            parser.error("Workspace bounds must be finite, with min < max")

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from evo_rlt.adapters.lerobot.franka_robot import FrankaRobot, FrankaRobotConfig

    checkpoint = str(resolve_checkpoint(args.checkpoint))
    print(f"Loading checkpoint: {checkpoint}", flush=True)
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if config.type != "act":
        raise ValueError(f"Expected an ACT checkpoint, got {config.type!r}")
    config.device = args.device
    expected = {"observation.state": (7,), "observation.images.wrist": (3, 480, 640),
                "observation.images.front": (3, 480, 640)}
    actual = {k: tuple(v.shape) for k, v in config.input_features.items()}
    if actual != expected or tuple(config.output_features["action"].shape) != (7,):
        raise ValueError(f"Checkpoint must use 7D state, two VGA cameras and 7D action: {actual}")
    if args.n_action_steps is not None:
        config.n_action_steps = args.n_action_steps
    if not 1 <= config.n_action_steps <= config.chunk_size:
        parser.error("n-action-steps must be between 1 and checkpoint chunk_size")
    if config.temporal_ensemble_coeff is not None and config.n_action_steps != 1:
        parser.error("Temporal ensembling requires n-action-steps=1")
    policy = ACTPolicy.from_pretrained(
        checkpoint, config=config, local_files_only=True
    ).to(args.device).eval()
    pre, post = make_pre_post_processors(
        config, pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    robot = FrankaRobot(FrankaRobotConfig(
        robot_ip=args.robot_ip, camera_serial=args.wrist_camera,
        front_camera_serial=args.front_camera, front_camera_width=640, front_camera_height=480,
        enable_fci_keepalive=False, workspace_min_xyz=args.workspace_min,
        workspace_max_xyz=args.workspace_max,
    ))
    try:
        robot.connect()
        # Warm up kernels using real observations before allowing motion.
        with torch.inference_mode():
            post(policy.select_action(pre(observation_frame(robot.get_observation()))))
        input(f"{'DRY RUN' if args.dry_run else 'MOTION'}: Enter to start {args.duration:g}s; Ctrl+C to stop: ")
        policy.reset()
        start = time.monotonic()
        ticks = 0
        with torch.inference_mode():
            while time.monotonic() - start < args.duration:
                tick = time.monotonic()
                observation = robot.get_observation()
                values = post(policy.select_action(pre(observation_frame(observation))))
                values = values.detach().cpu().numpy().reshape(-1)
                action = bounded_action(values, args.max_step_m, args.max_step_rad)
                elapsed = time.monotonic() - tick
                if elapsed > 0.5:
                    message = f"Observation/inference took {elapsed:.2f}s"
                    if not args.dry_run:
                        raise RuntimeError(message + "; stopping")
                    print("WARNING: " + message + "; dry-run continues without real-time pacing")
                if time.monotonic() - start >= args.duration:
                    break
                if not args.dry_run:
                    # Anchor each increment to current measured pose, matching training labels.
                    robot.resync_command_pose()
                    robot.send_action(action)
                if ticks % 15 == 0:
                    print(f"t={time.monotonic()-start:.1f}s delta={list(action.values())} "
                          f"gripper_prediction={values[6]:.4f}m (held)", flush=True)
                ticks += 1
                time.sleep(max(0, 1 / 15 - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        print("Stopped by operator.")
    finally:
        try:
            if robot.is_connected and not args.dry_run:
                robot.robot.stop_move()
        finally:
            policy.reset()
            if robot.is_connected:
                robot.disconnect()


if __name__ == "__main__":
    main()
