"""Run a trained ACT-RLT Stage-2 Actor on the local FR3 without learning."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from act_rlt.infer import checkpoint_camera_shapes, robot_camera_kwargs
from act_rlt.infer_stage2_terminal import InferenceTerminal
from act_rlt.stage2 import ACTStage2Config, ACTStage2Policy, resolve_pretrained_model
from act_rlt.train_stage2 import (
    FrankaInsertionStage2Env,
    RESET_XY_MARGIN_M,
    capture_centered_workspace,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Stage-2 run directory or checkpoints/latest.pt")
    parser.add_argument("--stage1-checkpoint", type=Path,
                        help="override the Stage-1 path stored in the Stage-2 checkpoint")
    parser.add_argument("--act-checkpoint", type=Path,
                        help="override the ACT path stored in the Stage-2 checkpoint")
    parser.add_argument("--device", default=None, help="default: checkpoint device")
    parser.add_argument("--dry-run", action="store_true", help="load models and processors only")
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--wrist-camera", default="349622072679")
    parser.add_argument("--front-camera", default="233522075778")
    parser.add_argument("--z-insertion-mode", action="store_true")
    parser.add_argument(
        "--randomize-reset-xy", action="store_true",
        help="with --z-insertion-mode, randomize reset X/Y +/-1 cm around p1 (default: select p1/p2/p3)",
    )
    parser.add_argument("--workspace-min", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--workspace-max", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--sample-min", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--sample-max", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument(
        "--camera-boundary-mode", choices=("timestamp", "legacy"), default="timestamp",
        help="timestamp: align image/state when available; legacy: use ordinary observations",
    )
    parser.add_argument(
        "--observation-selection", choices=("latest", "boundary"), default="latest",
        help="latest: use newer cached observations like ACT infer; boundary: require post-command observations",
    )
    parser.add_argument("--episode-time", type=float, default=5.0)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.randomize_reset_xy and not args.z_insertion_mode:
        parser.error("--randomize-reset-xy requires --z-insertion-mode")
    if args.episodes <= 0 or not math.isfinite(args.episode_time) or args.episode_time <= 0:
        parser.error("--episodes and --episode-time must be positive")
    if args.fps is None:
        args.fps = 30.0 if args.z_insertion_mode else 15.0
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive and finite")
    if not args.dry_run and not args.allow_motion:
        parser.error("robot inference requires --allow-motion")
    if (args.workspace_min is None) != (args.workspace_max is None):
        parser.error("supply both workspace bounds")
    if (args.sample_min is None) != (args.sample_max is None):
        parser.error("supply both sample bounds")
    if args.z_insertion_mode:
        if args.workspace_min is not None or args.sample_min is not None:
            parser.error("Z insertion captures its workspace at startup; omit workspace/sample bounds")
    elif not args.dry_run and args.workspace_min is None:
        parser.error("motion requires --workspace-min X Y Z and --workspace-max X Y Z")
    if args.workspace_min is not None:
        bounds = np.asarray([args.workspace_min, args.workspace_max])
        if not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
            parser.error("workspace bounds must be finite with min < max")
        if np.any(bounds[1, :2] - bounds[0, :2] <= 2 * RESET_XY_MARGIN_M):
            parser.error("workspace X/Y spans must each exceed 4 cm")
        if args.sample_min is not None:
            sample = np.asarray([args.sample_min, args.sample_max])
            if (not np.isfinite(sample).all() or np.any(sample[0] >= sample[1])
                    or np.any(sample[0] < bounds[0]) or np.any(sample[1] > bounds[1])):
                parser.error("sample bounds must be finite, ordered, and inside the workspace")


def resolve_stage2_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "checkpoints/latest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Stage-2 checkpoint not found: {path}")
    return path


def load_frozen_policy(
    checkpoint: Path, *, device: str | None = None,
    stage1_checkpoint: Path | None = None, act_checkpoint: Path | None = None,
) -> tuple[ACTStage2Policy, ACTStage2Config]:
    # Stage-2 checkpoints contain Python replay objects; load only trusted local runs.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    learner = saved["learner"]
    if learner.get("learner_version") != 3 or not learner.get("initialized"):
        raise ValueError("Stage-2 inference requires an initialized v3 Actor checkpoint")
    config_data = dict(saved["config"])
    config_data["device"] = device or config_data["device"]
    if device is None and str(config_data["device"]).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.", flush=True)
        config_data["device"] = "cpu"
    for key, override in (("stage1_checkpoint", stage1_checkpoint),
                          ("act_checkpoint", act_checkpoint)):
        config_data[key] = str(resolve_pretrained_model(
            override if override is not None else config_data[key]
        ))
    config = ACTStage2Config(**config_data)
    policy = ACTStage2Policy(config).to(config.device)
    policy.actor.load_state_dict(learner["actor"], strict=True)
    policy.eval()
    policy.requires_grad_(False)
    return policy, config


def run_episodes(env: FrankaInsertionStage2Env, policy: ACTStage2Policy, episodes: int) -> None:
    # A duration bound is applied inside the servo worker. The generous step
    # budget is only a fallback if wall-clock accounting is delayed.
    step_budget = max(1000, math.ceil(env.episode_time_s / env.period_s) + policy.config.chunk_length)
    terminal = getattr(env, "inference_terminal", None) or InferenceTerminal(episodes, env.episode_time_s)
    stopped = False
    try:
        for episode_id in range(episodes):
            batch = env.reset(episode_id=episode_id, warmup=False)
            started = time.monotonic()
            steps = clipped = 0
            boundary_times = []
            while True:
                execution = env.execute_policy_chunk(
                    policy, batch, warmup=False,
                    remaining_steps=step_budget - steps, deterministic=True,
                )
                steps += execution.actual_steps
                clipped += execution.info.get("safety_clip_steps", 0)
                if "boundary_inference_ms" in execution.info:
                    boundary_times.append(execution.info["boundary_inference_ms"])
                elapsed = execution.info.get("episode_elapsed_s", time.monotonic() - started)
                batch = execution.next_batch
                if execution.stop_requested or execution.done or execution.info.get("collector_paused"):
                    terminal.finish_episode(episode_id, execution, elapsed, steps, clipped, boundary_times)
                    stopped = execution.stop_requested
                    break
                terminal.status(
                    episode_id, elapsed, steps, clipped,
                    execution.info.get("boundary_inference_ms", 0.0),
                )
            if stopped:
                break
    except BaseException as error:
        terminal.session_summary("interrupted" if isinstance(error, KeyboardInterrupt) else "aborted")
        raise
    terminal.session_summary("stopped" if stopped else "complete")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    terminal = InferenceTerminal(args.episodes, args.episode_time)
    checkpoint = resolve_stage2_checkpoint(args.checkpoint)
    terminal.write(f"Loading Stage-2 checkpoint: {checkpoint}")
    policy, config = load_frozen_policy(
        checkpoint, device=args.device,
        stage1_checkpoint=args.stage1_checkpoint, act_checkpoint=args.act_checkpoint,
    )
    from lerobot.policies.factory import make_pre_post_processors

    act_path = resolve_pretrained_model(config.act_checkpoint)
    camera_shapes = checkpoint_camera_shapes(policy.stage1._act.config)
    pre, post = make_pre_post_processors(
        policy.stage1._act.config, pretrained_path=str(act_path),
        preprocessor_overrides={"device_processor": {"device": config.device}},
    )
    terminal.configuration(checkpoint, act_path, config, camera_shapes, args)
    if args.dry_run:
        terminal.write("[READY] Model check complete. Robot was not connected.")
        return 0

    from evo_rlt.adapters.lerobot.franka_robot import FrankaRobot, FrankaRobotConfig

    robot = FrankaRobot(FrankaRobotConfig(
        robot_ip=args.robot_ip, **robot_camera_kwargs(camera_shapes, args),
        include_gripper_action=True, enable_fci_keepalive=False,
        workspace_min_xyz=None if args.z_insertion_mode else tuple(args.workspace_min),
        workspace_max_xyz=None if args.z_insertion_mode else tuple(args.workspace_max),
    ))
    if args.camera_boundary_mode == "timestamp":
        robot.enable_timestamped_observations()
    env = None
    try:
        terminal.write(f"\nConnecting to robot {args.robot_ip} ...")
        robot.connect()
        terminal.write("[READY] Robot and cameras connected.")
        reference_pose = None
        if args.z_insertion_mode:
            reference_pose, workspace_min, workspace_max = capture_centered_workspace(
                robot, randomize_xy=args.randomize_reset_xy, terminal=terminal,
            )
            robot.config.workspace_min_xyz = tuple(workspace_min)
            robot.config.workspace_max_xyz = tuple(workspace_max)
        env = FrankaInsertionStage2Env(
            robot=robot, preprocessor=pre, postprocessor=post, config=config,
            fps=args.fps, episode_time_s=args.episode_time,
            max_step_m=config.max_step_m, max_step_rad=config.max_step_rad,
            camera_shapes=camera_shapes, z_insertion_mode=args.z_insertion_mode,
            reference_pose=reference_pose,
            randomize_reset_xy=args.randomize_reset_xy,
            sample_min=None if args.sample_min is None else tuple(args.sample_min),
            sample_max=None if args.sample_max is None else tuple(args.sample_max),
            observation_selection=args.observation_selection,
            inference_terminal=terminal,
        )
        with torch.inference_mode():
            run_episodes(env, policy, args.episodes)
    except KeyboardInterrupt:
        terminal.write("[STOP] Stopped by operator.")
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if robot.is_connected:
                try:
                    robot.robot.stop_move()
                finally:
                    robot.disconnect()
                    terminal.write("[CLOSED] Robot disconnected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
