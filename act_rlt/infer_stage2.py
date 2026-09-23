"""Run a trained ACT-RLT Stage-2 Actor on the local FR3 without learning."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from act_rlt.infer import checkpoint_camera_shapes, robot_camera_kwargs
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
    parser.add_argument("--workspace-min", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--workspace-max", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--sample-min", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--sample-max", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--episode-time", type=float, default=5.0)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
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
        print("CUDA is unavailable; loading on CPU for checkpoint validation.", flush=True)
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
    for episode_id in range(episodes):
        batch = env.reset(episode_id=episode_id, warmup=False)
        steps = 0
        while True:
            execution = env.execute_policy_chunk(
                policy, batch, warmup=False,
                remaining_steps=step_budget - steps, deterministic=True,
            )
            steps += execution.actual_steps
            batch = execution.next_batch
            print(
                f"Episode {episode_id}: steps={steps} "
                f"success={execution.info.get('success', False)} "
                f"clipped={execution.info.get('safety_clip_steps', 0)} "
                f"boundary_inference_ms={execution.info.get('boundary_inference_ms', 0):.1f}",
                flush=True,
            )
            if execution.stop_requested:
                return
            if execution.done or execution.info.get("collector_paused"):
                break


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    checkpoint = resolve_stage2_checkpoint(args.checkpoint)
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
    print(f"Frozen Stage-2 Actor loaded: {checkpoint}", flush=True)
    print(f"ACT: {act_path}; Stage 1: {config.stage1_checkpoint}", flush=True)
    print(f"Cameras: {camera_shapes}; device: {config.device}", flush=True)
    if args.dry_run:
        print("Dry run complete; robot not connected.", flush=True)
        return 0

    from evo_rlt.adapters.lerobot.franka_robot import FrankaRobot, FrankaRobotConfig

    robot = FrankaRobot(FrankaRobotConfig(
        robot_ip=args.robot_ip, **robot_camera_kwargs(camera_shapes, args),
        include_gripper_action=True, enable_fci_keepalive=False,
        workspace_min_xyz=None if args.z_insertion_mode else tuple(args.workspace_min),
        workspace_max_xyz=None if args.z_insertion_mode else tuple(args.workspace_max),
    ))
    env = None
    try:
        robot.connect()
        reference_pose = None
        if args.z_insertion_mode:
            reference_pose, workspace_min, workspace_max = capture_centered_workspace(robot)
            robot.config.workspace_min_xyz = tuple(workspace_min)
            robot.config.workspace_max_xyz = tuple(workspace_max)
        env = FrankaInsertionStage2Env(
            robot=robot, preprocessor=pre, postprocessor=post, config=config,
            fps=args.fps, episode_time_s=args.episode_time,
            max_step_m=config.max_step_m, max_step_rad=config.max_step_rad,
            camera_shapes=camera_shapes, z_insertion_mode=args.z_insertion_mode,
            reference_pose=reference_pose,
            sample_min=None if args.sample_min is None else tuple(args.sample_min),
            sample_max=None if args.sample_max is None else tuple(args.sample_max),
        )
        with torch.inference_mode():
            run_episodes(env, policy, args.episodes)
    except KeyboardInterrupt:
        print("Stopped by operator.", flush=True)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
