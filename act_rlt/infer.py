"""Interactive ACT inference: sampled reset poses and repeatable 15 Hz episodes."""

import argparse
import time
from pathlib import Path

import numpy as np


ACTION_KEYS = ("dx", "dy", "dz", "drx", "dry", "drz")
TRAINING_TCP_ORIENTATION_MEAN_RAD = np.array(
    [-2.19841545, 2.22703212, -0.03082950], dtype=float
)
DEFAULT_RESET_XY_PADDING_M = 0.03
DEFAULT_RESET_SPEED_M_S = 0.02


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


def bounded_action_with_triggers(values, max_translation, max_rotation):
    values = np.asarray(values, dtype=float).reshape(-1).copy()
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("Expected a finite 7D action")
    triggered = []
    for start, limit in ((0, max_translation), (3, max_rotation)):
        norm = np.linalg.norm(values[start:start + 3])
        triggered.append(bool(norm > limit))
        if norm > limit:
            values[start:start + 3] *= limit / norm
    # The gripper remains held; do not send repetitive gripper commands.
    return dict(zip(ACTION_KEYS, values[:6], strict=True)), triggered[0], triggered[1]


def bounded_action(values, max_translation, max_rotation):
    action, _, _ = bounded_action_with_triggers(values, max_translation, max_rotation)
    return action


def sample_workspace_pose(workspace_min, workspace_max, orientation, xy_padding, rng):
    """Sample XYZ inside the workspace, with padding on X/Y and fixed rotvec orientation."""
    lower = np.asarray(workspace_min, dtype=float)
    upper = np.asarray(workspace_max, dtype=float)
    orientation = np.asarray(orientation, dtype=float)
    if lower.shape != (3,) or upper.shape != (3,) or orientation.shape != (3,):
        raise ValueError("workspace bounds and reset orientation must be 3D")
    if not np.isfinite(np.concatenate([lower, upper, orientation])).all():
        raise ValueError("workspace bounds and reset orientation must be finite")
    if np.any(lower >= upper):
        raise ValueError("workspace min must be smaller than workspace max")
    if not np.isfinite(xy_padding) or xy_padding < 0:
        raise ValueError("reset XY padding must be finite and non-negative")
    sample_lower = lower.copy()
    sample_upper = upper.copy()
    sample_lower[:2] += xy_padding
    sample_upper[:2] -= xy_padding
    if np.any(sample_lower[:2] >= sample_upper[:2]):
        raise ValueError(
            f"workspace X/Y spans must each exceed {2 * xy_padding:.3f} m "
            f"for {xy_padding:.3f} m reset padding"
        )
    return np.concatenate([rng.uniform(sample_lower, sample_upper), orientation])


def format_pose(pose):
    return " ".join(f"{float(value):+.6f}" for value in pose)


def move_to_pose(robot, target, speed, *, label):
    target = np.asarray(target, dtype=float)
    print(f"{label}: {format_pose(target)}", flush=True)
    measured = np.asarray(
        robot.robot.move_tool(target.tolist(), speed=float(speed)), dtype=float
    )
    if measured.shape != (6,) or not np.isfinite(measured).all():
        raise RuntimeError(f"invalid measured pose after {label}: {measured}")
    robot.resync_command_pose()
    print(f"Reached {label}: {format_pose(measured)}", flush=True)
    return measured


def prompt_choice(prompt, choices):
    while True:
        choice = input(prompt).strip().lower()
        if choice in choices:
            return choices[choice]
        print("Invalid choice; please use one of the keys shown.", flush=True)


def build_parser():
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
    parser.add_argument(
        "--reset-orientation",
        type=float,
        nargs=3,
        metavar=("RX", "RY", "RZ"),
        default=TRAINING_TCP_ORIENTATION_MEAN_RAD.tolist(),
        help="Fixed reset rotation vector in radians (default: training-data TCP mean)",
    )
    parser.add_argument("--reset-xy-padding", type=float, default=DEFAULT_RESET_XY_PADDING_M)
    parser.add_argument("--reset-speed", type=float, default=DEFAULT_RESET_SPEED_M_S)
    parser.add_argument("--sampling-seed", type=int)
    return parser


def validate_args(parser, args):
    if not all(np.isfinite(x) and x > 0 for x in (args.duration, args.max_step_m, args.max_step_rad)):
        parser.error("duration and step limits must be finite and positive")
    if not np.isfinite(args.reset_speed) or args.reset_speed <= 0:
        parser.error("reset-speed must be finite and positive")
    if not np.isfinite(args.reset_xy_padding) or args.reset_xy_padding < 0:
        parser.error("reset-xy-padding must be finite and non-negative")
    orientation = np.asarray(args.reset_orientation, dtype=float)
    if orientation.shape != (3,) or not np.isfinite(orientation).all():
        parser.error("reset-orientation must contain three finite rotation-vector values")
    if not args.dry_run and (args.workspace_min is None or args.workspace_max is None):
        parser.error("Motion requires --workspace-min X Y Z and --workspace-max X Y Z (base-frame metres)")
    if (args.workspace_min is None) != (args.workspace_max is None):
        parser.error("Supply both workspace bounds")
    if args.workspace_min is not None:
        bounds = np.array([args.workspace_min, args.workspace_max])
        if not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
            parser.error("Workspace bounds must be finite, with min < max")
        if np.any(bounds[1, :2] - bounds[0, :2] <= 2 * args.reset_xy_padding):
            parser.error(
                "Workspace X/Y spans must exceed twice reset-xy-padding "
                f"({2 * args.reset_xy_padding:.3f} m)"
            )


def run_inference_episode(robot, policy, pre, post, args, torch):
    policy.reset()
    start = time.monotonic()
    ticks = 0
    translation_triggers = 0
    rotation_triggers = 0
    either_triggers = 0
    both_triggers = 0
    try:
        with torch.inference_mode():
            while time.monotonic() - start < args.duration:
                tick = time.monotonic()
                observation = robot.get_observation()
                values = post(policy.select_action(pre(observation_frame(observation))))
                values = values.detach().cpu().numpy().reshape(-1)
                action, translation_triggered, rotation_triggered = bounded_action_with_triggers(
                    values, args.max_step_m, args.max_step_rad
                )
                elapsed = time.monotonic() - tick
                if elapsed > 0.5:
                    message = f"Observation/inference took {elapsed:.2f}s"
                    if not args.dry_run:
                        raise RuntimeError(message + "; stopping")
                    print("WARNING: " + message + "; dry-run continues without real-time pacing")
                if time.monotonic() - start >= args.duration:
                    break
                translation_triggers += int(translation_triggered)
                rotation_triggers += int(rotation_triggered)
                either_triggers += int(translation_triggered or rotation_triggered)
                both_triggers += int(translation_triggered and rotation_triggered)
                step_index = ticks
                ticks += 1
                if not args.dry_run:
                    # Anchor each increment to current measured pose, matching training labels.
                    robot.resync_command_pose()
                    robot.send_action(action)
                if step_index % 15 == 0:
                    print(f"t={time.monotonic()-start:.1f}s delta={list(action.values())} "
                          f"gripper_prediction={values[6]:.4f}m (held)", flush=True)
                time.sleep(max(0, 1 / 15 - (time.monotonic() - tick)))
    finally:
        denominator = max(ticks, 1)
        print(
            "Action-limit report: "
            f"steps={ticks}, "
            f"max_step_m_triggered={translation_triggers} "
            f"({100 * translation_triggers / denominator:.1f}%), "
            f"max_step_rad_triggered={rotation_triggers} "
            f"({100 * rotation_triggers / denominator:.1f}%), "
            f"either={either_triggers} ({100 * either_triggers / denominator:.1f}%), "
            f"both={both_triggers} ({100 * both_triggers / denominator:.1f}%)",
            flush=True,
        )
        policy.reset()
        if robot.is_connected and not args.dry_run:
            if not robot.robot.stop_servo():
                raise RuntimeError("control server did not acknowledge stop_servo after inference")
            robot.resync_command_pose()


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

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
    rng = np.random.default_rng(args.sampling_seed)
    try:
        robot.connect()
        # Warm up kernels using real observations before allowing motion.
        with torch.inference_mode():
            post(policy.select_action(pre(observation_frame(robot.get_observation()))))
        policy.reset()
        if args.dry_run:
            while True:
                action = prompt_choice(
                    f"DRY RUN: [Enter/s]=start {args.duration:g}s inference, q=quit: ",
                    {"": "start", "s": "start", "start": "start", "q": "quit", "quit": "quit"},
                )
                if action == "quit":
                    break
                run_inference_episode(robot, policy, pre, post, args, torch)
            return

        workspace_min = np.asarray(args.workspace_min, dtype=float)
        workspace_max = np.asarray(args.workspace_max, dtype=float)
        orientation = np.asarray(args.reset_orientation, dtype=float)
        print(
            "Reset orientation (rotation vector, rad; training-data mean by default): "
            f"{format_pose(orientation)}",
            flush=True,
        )
        episode_start = None
        need_new_sample = True
        first_move = True
        episode_index = 0
        while True:
            if need_new_sample:
                target = sample_workspace_pose(
                    workspace_min,
                    workspace_max,
                    orientation,
                    args.reset_xy_padding,
                    rng,
                )
                if first_move:
                    decision = prompt_choice(
                        "First sampled reset target is "
                        f"{format_pose(target)}. [Enter/y]=move, n=new sample, q=quit: ",
                        {
                            "": "move", "y": "move", "yes": "move",
                            "n": "new", "new": "new",
                            "q": "quit", "quit": "quit",
                        },
                    )
                    if decision == "quit":
                        break
                    if decision == "new":
                        continue
                    first_move = False
                episode_start = move_to_pose(
                    robot, target, args.reset_speed, label="sampled inference start"
                )
                need_new_sample = False

            decision = prompt_choice(
                "At inference start "
                f"{format_pose(episode_start)}. [Enter/s]=start inference, "
                "n=move to next sample, q=quit: ",
                {
                    "": "start", "s": "start", "start": "start",
                    "n": "new", "new": "new",
                    "q": "quit", "quit": "quit",
                },
            )
            if decision == "quit":
                break
            if decision == "new":
                need_new_sample = True
                continue

            episode_index += 1
            print(f"Starting inference episode {episode_index} for {args.duration:g}s.", flush=True)
            run_inference_episode(robot, policy, pre, post, args, torch)
            print(f"Inference episode {episode_index} finished.", flush=True)
            decision = prompt_choice(
                "Next action: r=return to this episode's start, "
                "n=move to a new sampled point, q=quit: ",
                {
                    "r": "return", "return": "return",
                    "n": "new", "new": "new",
                    "q": "quit", "quit": "quit",
                },
            )
            if decision == "quit":
                break
            if decision == "new":
                need_new_sample = True
                continue
            episode_start = move_to_pose(
                robot, episode_start, args.reset_speed, label="previous inference start"
            )
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
