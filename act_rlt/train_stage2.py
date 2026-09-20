#!/usr/bin/env python3
"""Run ACT-RLT Stage-2 online training on the local FR3.

The first implementation is deliberately critical-phase only. Episode starts
are sampled and reached automatically; the operator labels success/failure
with the keyboard. Human action intervention is not enabled.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import select
import sys
import termios
import time
import tty
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from act_rlt.infer import bounded_action, observation_frame
from act_rlt.stage2 import (
    ACTStage2Config,
    ACTStage2Policy,
    ChunkExecution,
    OnlineStage2Metrics,
    Stage2Learner,
    resolve_pretrained_model,
    run_online_stage2,
)
from evo_rlt.core.replay_buffer import ReplayBuffer
from lerobot.processor import NormalizerProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.types import TransitionKey


RESET_RETREAT_Y_M = 0.01
RESET_MOVE_SPEED_M_S = 0.02
RESET_XY_MARGIN_M = 0.02


def retreat_pose_along_positive_y(
    measured_pose: np.ndarray,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
) -> np.ndarray:
    """Return a 1 cm base-frame +Y retreat, refusing an out-of-bounds target."""
    pose = np.asarray(measured_pose, dtype=float).copy()
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError(f"invalid measured TCP pose: {pose}")
    pose[1] += RESET_RETREAT_Y_M
    if np.any(pose[:3] < workspace_min) or np.any(pose[:3] > workspace_max):
        raise RuntimeError(
            "+Y reset retreat would leave workspace: "
            f"target={pose[:3]}, min={workspace_min}, max={workspace_max}"
        )
    return pose


def sample_workspace_pose(
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    orientation: np.ndarray,
    *,
    sample_min: np.ndarray | None = None,
    sample_max: np.ndarray | None = None,
) -> np.ndarray:
    """Sample a reset XYZ and retain a fixed validated orientation.

    Without an explicit sample box, the safe workspace is used with a 2 cm
    X/Y inset.  An explicit sample box is used exactly as supplied; the safe
    workspace continues to constrain all robot commands independently.
    """
    workspace_min = np.asarray(workspace_min, dtype=float)
    workspace_max = np.asarray(workspace_max, dtype=float)
    orientation = np.asarray(orientation, dtype=float)
    if (
        workspace_min.shape != (3,)
        or workspace_max.shape != (3,)
        or not np.isfinite(workspace_min).all()
        or not np.isfinite(workspace_max).all()
    ):
        raise ValueError("invalid reset workspace bounds")
    if orientation.shape != (3,) or not np.isfinite(orientation).all():
        raise ValueError(f"invalid reset orientation: {orientation}")
    if (sample_min is None) != (sample_max is None):
        raise ValueError("supply both sample_min and sample_max")
    if sample_min is None:
        sample_min = workspace_min.copy()
        sample_max = workspace_max.copy()
        sample_min[:2] += RESET_XY_MARGIN_M
        sample_max[:2] -= RESET_XY_MARGIN_M
        if np.any(sample_min[:2] >= sample_max[:2]):
            raise ValueError(
                "workspace X/Y spans must each exceed 4 cm for the 2 cm reset margin"
            )
    else:
        sample_min = np.asarray(sample_min, dtype=float)
        sample_max = np.asarray(sample_max, dtype=float)
        if (
            sample_min.shape != (3,)
            or sample_max.shape != (3,)
            or not np.isfinite(sample_min).all()
            or not np.isfinite(sample_max).all()
            or np.any(sample_min >= sample_max)
        ):
            raise ValueError("invalid reset sample bounds")
        if np.any(sample_min < workspace_min) or np.any(sample_max > workspace_max):
            raise ValueError("reset sample bounds must lie inside the safe workspace")
    return np.concatenate([np.random.uniform(sample_min, sample_max), orientation])


class OutcomeMonitor:
    """Non-blocking single-key episode outcome monitor for an interactive TTY."""

    def __init__(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("Stage-2 outcome labeling requires an interactive terminal")
        self.fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def poll(self) -> str | None:
        readable, _, _ = select.select([self.fd], [], [], 0)
        if not readable:
            return None
        return os.read(self.fd, 1).decode(errors="ignore").lower()

    def close(self) -> None:
        if self._old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)
            self._old = None


class FrankaInsertionStage2Env:
    """Minimal normalized-chunk environment around the existing Franka adapter."""

    def __init__(
        self,
        *,
        robot,
        preprocessor,
        postprocessor,
        config: ACTStage2Config,
        fps: float,
        episode_time_s: float,
        max_step_m: float,
        max_step_rad: float,
        sample_min: tuple[float, float, float] | None = None,
        sample_max: tuple[float, float, float] | None = None,
    ) -> None:
        self.robot = robot
        self.pre = preprocessor
        self.post = postprocessor
        try:
            self.action_normalizer = next(
                step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)
            )
        except StopIteration as error:
            raise ValueError("ACT preprocessor has no NormalizerProcessorStep") from error
        self.config = config
        self.period_s = 1.0 / fps
        self.episode_time_s = episode_time_s
        self.max_step_m = max_step_m
        self.max_step_rad = max_step_rad
        self.workspace_min = np.asarray(robot.config.workspace_min_xyz, dtype=float)
        self.workspace_max = np.asarray(robot.config.workspace_max_xyz, dtype=float)
        self.sample_min = (
            None if sample_min is None else np.asarray(sample_min, dtype=float)
        )
        self.sample_max = (
            None if sample_max is None else np.asarray(sample_max, dtype=float)
        )
        if (
            self.workspace_min.shape != (3,)
            or self.workspace_max.shape != (3,)
            or not np.isfinite(self.workspace_min).all()
            or not np.isfinite(self.workspace_max).all()
            or np.any(self.workspace_min >= self.workspace_max)
        ):
            raise ValueError("automatic reset requires valid 3D workspace bounds")
        self._monitor: OutcomeMonitor | None = None
        self._episode_start = 0.0
        self._reset_orientation: np.ndarray | None = None
        # Every reset, including Episode 0, moves to a sampled point. The first
        # move retains a one-time operator authorization but requires no teaching.
        self._needs_automatic_reset = True
        self._first_reset_move = True

    def _processed_observation(self) -> dict[str, torch.Tensor]:
        return self.pre(observation_frame(self.robot.get_observation()))

    def _close_monitor(self) -> None:
        if self._monitor is not None:
            self._monitor.close()
            self._monitor = None

    @staticmethod
    def _format_pose(pose: np.ndarray) -> str:
        return " ".join(f"{value:+.6f}" for value in pose)

    def _move_reset_pose(self, target: np.ndarray, *, label: str) -> None:
        print(f"{label}: {self._format_pose(target)}", flush=True)
        measured = np.asarray(
            self.robot.robot.move_tool(target.tolist(), speed=RESET_MOVE_SPEED_M_S),
            dtype=float,
        )
        if measured.shape != (6,) or not np.isfinite(measured).all():
            raise RuntimeError(f"invalid measured pose after {label}: {measured}")
        self.robot.resync_command_pose()
        print(f"Reached {label}: {self._format_pose(measured)}", flush=True)

    def _retreat_after_outcome(self) -> None:
        measured = np.asarray(self.robot.robot.get_tool_pose(), dtype=float)
        retreat = retreat_pose_along_positive_y(
            measured,
            self.workspace_min,
            self.workspace_max,
        )
        self._move_reset_pose(retreat, label="+Y 1 cm retreat")

    def _move_to_accepted_workspace_sample(self) -> None:
        if self._reset_orientation is None:
            measured = np.asarray(self.robot.robot.get_tool_pose(), dtype=float)
            if measured.shape != (6,) or not np.isfinite(measured).all():
                raise RuntimeError(f"invalid initial TCP pose: {measured}")
            self._reset_orientation = measured[3:].copy()
            print(
                "Using current TCP reset orientation: "
                f"{self._format_pose(self._reset_orientation)}",
                flush=True,
            )
        while True:
            target = sample_workspace_pose(
                self.workspace_min,
                self.workspace_max,
                self._reset_orientation,
                sample_min=self.sample_min,
                sample_max=self.sample_max,
            )
            if self._first_reset_move:
                choice = input(
                    "First sampled reset target is "
                    f"{self._format_pose(target)}. Press Enter/y to move, q to stop: "
                ).strip().lower()
                if choice not in {"", "y", "yes"}:
                    if choice in {"q", "quit"}:
                        raise KeyboardInterrupt
                    print("First automatic reset not authorized; stopping.", flush=True)
                    raise KeyboardInterrupt
                self._first_reset_move = False
            self._move_reset_pose(target, label="sampled workspace reset pose")
            choice = input(
                "Sample point OK? [Enter/y]=start episode, r=move to a new sample, "
                "q=stop training: "
            ).strip().lower()
            if choice in {"", "y", "yes"}:
                return
            if choice in {"r", "retry", "n", "no"}:
                continue
            if choice in {"q", "quit"}:
                raise KeyboardInterrupt
            print("Please enter y, r, or q.", flush=True)

    @staticmethod
    def _prompt_outcome_after_timeout() -> str:
        while True:
            choice = input(
                "Episode inference time elapsed. Confirm outcome: "
                "s=success, f=failure, q=stop training: "
            ).strip().lower()
            if choice in {"s", "f", "q"}:
                return choice
            print("Please enter s, f, or q.", flush=True)

    def reset(self, *, episode_id: int, warmup: bool) -> dict[str, torch.Tensor]:
        self._close_monitor()
        phase = "ACT WARMUP" if warmup else "RLT ACTOR"
        print(f"\nPreparing Episode {episode_id} [{phase}] with automatic reset.", flush=True)
        self._move_to_accepted_workspace_sample()
        self._needs_automatic_reset = False
        self.robot.resync_command_pose()
        batch = self._processed_observation()
        self._episode_start = time.monotonic()
        self._monitor = OutcomeMonitor()
        print("Episode active: s=success, f=failure, q=stop training", flush=True)
        return batch

    def _normalize_executed_action(self, physical_full: torch.Tensor) -> torch.Tensor:
        """Map one safety-bounded physical 7D action back to ACT model space."""
        transition = create_transition(
            action=physical_full.detach().reshape(1, -1).to(self.config.device)
        )
        normalized = self.action_normalizer(transition)[TransitionKey.ACTION]
        if normalized.ndim != 2 or normalized.shape[0] != 1:
            raise RuntimeError(f"unexpected normalized action shape: {tuple(normalized.shape)}")
        return normalized[0, : self.config.action_dim].detach().cpu()

    def execute_chunk(self, action_chunk: torch.Tensor, full_reference: torch.Tensor) -> ChunkExecution:
        if action_chunk.shape != (1, self.config.chunk_length, self.config.action_dim):
            raise RuntimeError(f"unexpected policy chunk shape: {tuple(action_chunk.shape)}")
        if (
            full_reference.ndim != 3
            or full_reference.shape[0] != 1
            or full_reference.shape[1] < self.config.chunk_length
            or full_reference.shape[2] != self.config.action_dim + 1
        ):
            raise RuntimeError(f"unexpected ACT reference shape: {tuple(full_reference.shape)}")

        executed = torch.zeros(self.config.chunk_length, self.config.action_dim)
        rewards = torch.zeros(self.config.chunk_length)
        actual_steps = 0
        done = terminated = truncated = stop_requested = False
        success = False
        workspace_violation = False
        workspace_error: str | None = None
        last_batch: dict[str, torch.Tensor] | None = None
        clipping_count = 0

        for index in range(self.config.chunk_length):
            tick = time.monotonic()
            normalized_full = full_reference[:, index, :].clone()
            normalized_full[:, : self.config.action_dim] = action_chunk[:, index, :]
            physical_full = self.post(normalized_full).detach().cpu().reshape(-1)
            if physical_full.numel() != full_reference.shape[-1]:
                raise RuntimeError(f"unexpected postprocessed action shape: {tuple(physical_full.shape)}")

            command = bounded_action(
                physical_full.numpy(), self.max_step_m, self.max_step_rad
            )
            bounded_six = torch.tensor(list(command.values()), dtype=physical_full.dtype)
            if not torch.allclose(bounded_six, physical_full[: self.config.action_dim]):
                clipping_count += 1
            physical_executed = physical_full.clone()
            physical_executed[: self.config.action_dim] = bounded_six
            normalized_executed = self._normalize_executed_action(physical_executed)

            self.robot.resync_command_pose()
            try:
                self.robot.send_action(command)
            except RuntimeError as error:
                # A strict workspace refusal is a recoverable episode boundary,
                # not a training-process failure. Do not swallow other robot or
                # control-server faults.
                if "refusing TCP target outside workspace" not in str(error):
                    raise
                workspace_violation = True
                workspace_error = str(error)
                truncated = done = True
                self._close_monitor()
                self.robot.robot.stop_move()
                self.robot.resync_command_pose()
                last_batch = self._processed_observation()
                print(
                    f"Workspace boundary reached; truncating Episode and resetting: {error}",
                    flush=True,
                )
                break
            executed[index] = normalized_executed
            actual_steps += 1
            time.sleep(max(0.0, self.period_s - (time.monotonic() - tick)))
            last_batch = self._processed_observation()

            key = self._monitor.poll() if self._monitor is not None else None
            if key == "s":
                rewards[index] = 1.0
                success = True
                terminated = done = True
            elif key == "f":
                terminated = done = True
            elif key == "q":
                truncated = done = stop_requested = True
            elif time.monotonic() - self._episode_start >= self.episode_time_s:
                # Stop inference at the wall-clock limit, but keep outcomes
                # human-labelled in both warmup and online RL.
                self._close_monitor()
                key = self._prompt_outcome_after_timeout()
                if key == "s":
                    rewards[index] = 1.0
                    success = True
                    terminated = done = True
                elif key == "f":
                    terminated = done = True
                else:
                    truncated = done = stop_requested = True
            if done:
                self._close_monitor()
                if not stop_requested and not workspace_violation:
                    # Retreat immediately after the human outcome key, before
                    # replay updates/checkpointing can delay extraction.
                    self._retreat_after_outcome()
                if not stop_requested:
                    self._needs_automatic_reset = True
                break

        if last_batch is None:
            raise RuntimeError("chunk execution produced no next observation")
        return ChunkExecution(
            next_batch=last_batch,
            exec_chunk=executed,
            reward_seq=rewards,
            actual_steps=actual_steps,
            done=done,
            terminated=terminated,
            truncated=truncated,
            stop_requested=stop_requested,
            info={
                "success": success,
                "safety_clip_steps": clipping_count,
                "workspace_violation": workspace_violation,
                "workspace_error": workspace_error,
            },
        )

    def close(self) -> None:
        self._close_monitor()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--act-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/act_rlt_001_stage2"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-motion", action="store_true")

    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--wrist-camera", default="349622072679")
    parser.add_argument("--front-camera", default="233522075778")
    parser.add_argument("--workspace-min", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--workspace-max", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--sample-min",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="optional reset-pose sampling lower bound inside the safe workspace",
    )
    parser.add_argument(
        "--sample-max",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="optional reset-pose sampling upper bound inside the safe workspace",
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--episode-time", type=float, default=5.0)
    parser.add_argument("--max-step-m", type=float, default=0.002)
    parser.add_argument("--max-step-rad", type=float, default=0.02)
    parser.add_argument("--warmup-steps", type=int, default=4_000)
    parser.add_argument(
        "--total-env-steps",
        type=int,
        default=10_000,
        help="cumulative online-step target after warmup; use 20000 to add 10000 after 10000",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--utd-ratio", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--save-every-env-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.dry_run and not args.allow_motion:
        parser.error("real Stage-2 collection requires --allow-motion")
    if args.allow_motion and (args.workspace_min is None or args.workspace_max is None):
        parser.error("motion requires --workspace-min X Y Z and --workspace-max X Y Z")
    if (args.workspace_min is None) != (args.workspace_max is None):
        parser.error("supply both workspace bounds")
    if (args.sample_min is None) != (args.sample_max is None):
        parser.error("supply both --sample-min and --sample-max")
    if args.workspace_min is not None:
        bounds = np.asarray([args.workspace_min, args.workspace_max], dtype=float)
        if not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
            parser.error("workspace bounds must be finite with min < max")
        if np.any(bounds[1, :2] - bounds[0, :2] <= 2 * RESET_XY_MARGIN_M):
            parser.error("workspace X/Y spans must each exceed 4 cm for 2 cm margins")
        if args.sample_min is not None:
            sample_bounds = np.asarray([args.sample_min, args.sample_max], dtype=float)
            if (
                not np.isfinite(sample_bounds).all()
                or np.any(sample_bounds[0] >= sample_bounds[1])
            ):
                parser.error("sample bounds must be finite with min < max")
            if np.any(sample_bounds[0] < bounds[0]) or np.any(sample_bounds[1] > bounds[1]):
                parser.error("sample bounds must lie inside the safe workspace")
    elif args.sample_min is not None:
        parser.error("sample bounds require workspace bounds")
    positive = {
        "fps": args.fps,
        "episode-time": args.episode_time,
        "max-step-m": args.max_step_m,
        "max-step-rad": args.max_step_rad,
        "warmup-steps": args.warmup_steps,
        "total-env-steps": args.total_env_steps,
        "batch-size": args.batch_size,
        "utd-ratio": args.utd_ratio,
        "save-every-env-steps": args.save_every_env_steps,
    }
    invalid = [name for name, value in positive.items() if not np.isfinite(value) or value <= 0]
    if invalid:
        parser.error(f"these settings must be finite and positive: {invalid}")


def save_checkpoint(
    output: Path,
    config: ACTStage2Config,
    learner: Stage2Learner,
    replay: ReplayBuffer,
    metrics: OnlineStage2Metrics,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "config": config.to_dict(),
        "learner": learner.state_dict(),
        "replay": list(replay.buffer),
        "metrics": {
            key: value
            for key, value in asdict(metrics).items()
            if key != "last_learner"
        },
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    temporary = checkpoint_dir / "latest.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(checkpoint_dir / "latest.pt")


def load_checkpoint(
    path: Path,
    config: ACTStage2Config,
    learner: Stage2Learner,
    replay: ReplayBuffer,
) -> OnlineStage2Metrics:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    saved_config = saved["config"]
    current_config = config.to_dict()
    schedule_keys = {"warmup_steps", "total_env_steps"}
    incompatible = {
        key: (saved_config.get(key), current_config.get(key))
        for key in set(saved_config) | set(current_config)
        if key not in schedule_keys and saved_config.get(key) != current_config.get(key)
    }
    if incompatible:
        raise ValueError(
            "resume config differs outside the adjustable schedule fields: "
            f"{incompatible}"
        )
    saved_metrics = saved["metrics"]
    warmup_progress = int(saved_metrics.get("warmup_env_steps", 0))
    online_progress = int(saved_metrics.get("online_env_steps", 0))
    if config.warmup_steps > warmup_progress and online_progress > 0:
        raise ValueError(
            "cannot raise warmup_steps above completed warmup progress after online training "
            f"has started: warmup_progress={warmup_progress}, online_progress={online_progress}"
        )
    if config.total_env_steps < online_progress:
        raise ValueError(
            "total_env_steps is a cumulative target and cannot be below current online progress: "
            f"target={config.total_env_steps}, progress={online_progress}"
        )
    schedule_changes = {
        key: (saved_config.get(key), current_config[key])
        for key in schedule_keys
        if saved_config.get(key) != current_config[key]
    }
    if schedule_changes:
        print(f"Applying resume schedule overrides: {schedule_changes}", flush=True)
    learner.load_state_dict(saved["learner"])
    for transition in saved["replay"]:
        replay.add(transition)
    torch.set_rng_state(saved["torch_rng"])
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])
    if torch.cuda.is_available() and "cuda_rng" in saved:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return OnlineStage2Metrics(**saved_metrics)


def append_metrics(output: Path, metrics: OnlineStage2Metrics, execution: ChunkExecution) -> None:
    learner = metrics.last_learner
    record: dict[str, Any] = {
        "time": time.time(),
        "env_steps": metrics.env_steps,
        "warmup_env_steps": metrics.warmup_env_steps,
        "online_env_steps": metrics.online_env_steps,
        "chunks": metrics.chunks,
        "episodes": metrics.episodes,
        "successes": metrics.successes,
        "critic_updates": metrics.critic_updates,
        "actor_updates": metrics.actor_updates,
        "actual_steps": execution.actual_steps,
        "done": execution.done,
        "terminated": execution.terminated,
        "truncated": execution.truncated,
        **execution.info,
    }
    if learner is not None:
        record.update(
            critic_loss=learner.critic_loss,
            actor_loss=learner.actor_loss,
            critic_grad_norm=learner.critic_grad_norm,
            actor_grad_norm=learner.actor_grad_norm,
        )
    with (output / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    stage1_path = resolve_pretrained_model(args.stage1_checkpoint)
    act_path = resolve_pretrained_model(args.act_checkpoint)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.resume:
        parser.error(f"output already exists: {output}; use --resume or a new directory")
    if args.resume and not (output / "checkpoints/latest.pt").is_file():
        parser.error(f"resume checkpoint is missing: {output / 'checkpoints/latest.pt'}")

    config = ACTStage2Config(
        stage1_checkpoint=str(stage1_path),
        act_checkpoint=str(act_path),
        device=args.device,
        warmup_steps=args.warmup_steps,
        total_env_steps=args.total_env_steps,
        batch_size=args.batch_size,
        utd_ratio=args.utd_ratio,
        beta=args.beta,
        seed=args.seed,
    )
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True), flush=True)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTStage2Policy(config).to(config.device)
    pre, post = make_pre_post_processors(
        policy.stage1._act.config,
        pretrained_path=str(act_path),
        preprocessor_overrides={"device_processor": {"device": config.device}},
    )
    learner = Stage2Learner(policy, config)
    replay = ReplayBuffer(config.replay_capacity)
    metrics = OnlineStage2Metrics()
    if args.resume:
        metrics = load_checkpoint(output / "checkpoints/latest.pt", config, learner, replay)
    if args.dry_run:
        print(
            "Dry run complete: checkpoints and saved processors loaded; robot not connected.",
            flush=True,
        )
        return 0

    from evo_rlt.adapters.lerobot.franka_robot import FrankaRobot, FrankaRobotConfig

    robot = FrankaRobot(
        FrankaRobotConfig(
            robot_ip=args.robot_ip,
            camera_serial=args.wrist_camera,
            front_camera_serial=args.front_camera,
            front_camera_width=640,
            front_camera_height=480,
            include_gripper_action=True,
            enable_fci_keepalive=False,
            workspace_min_xyz=tuple(args.workspace_min),
            workspace_max_xyz=tuple(args.workspace_max),
        )
    )
    env = FrankaInsertionStage2Env(
        robot=robot,
        preprocessor=pre,
        postprocessor=post,
        config=config,
        fps=args.fps,
        episode_time_s=args.episode_time,
        max_step_m=args.max_step_m,
        max_step_rad=args.max_step_rad,
        sample_min=None if args.sample_min is None else tuple(args.sample_min),
        sample_max=None if args.sample_max is None else tuple(args.sample_max),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")
    last_save_step = metrics.env_steps

    def on_chunk(current: OnlineStage2Metrics, execution: ChunkExecution) -> None:
        nonlocal last_save_step
        append_metrics(output, current, execution)
        if execution.done or current.env_steps - last_save_step >= args.save_every_env_steps:
            save_checkpoint(output, config, learner, replay, current)
            last_save_step = current.env_steps

    try:
        robot.connect()
        metrics = run_online_stage2(
            policy,
            learner,
            env,
            replay,
            config,
            on_chunk=on_chunk,
            initial_metrics=metrics,
        )
    except KeyboardInterrupt:
        print("Interrupted by operator; saving recoverable state.", flush=True)
    finally:
        env.close()
        if robot.is_connected:
            try:
                robot.robot.stop_move()
            finally:
                robot.disconnect()
        save_checkpoint(output, config, learner, replay, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
