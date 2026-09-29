#!/usr/bin/env python3
"""Run ACT-RLT Stage-2 online training on the local FR3.

Critical-phase episodes begin at automatically sampled reset poses. Optional
keyboard takeover switches control only after a complete action chunk.
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

from act_rlt.data_collection.sampling import (
    SAMPLE_EULER_XYZ_DEG,
    SAMPLE_ROTATION_VECTOR,
    SAMPLE_Z_M,
)
from act_rlt.infer import (
    bounded_action,
    checkpoint_camera_shapes,
    observation_frame,
    robot_camera_kwargs,
)
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
RESET_RETREAT_Z_M = 0.01
RESET_MOVE_SPEED_M_S = 0.02
RESET_XY_MARGIN_M = 0.02
STARTUP_WORKSPACE_HALF_RANGE_M = 0.03
Z_SAMPLE_XY_HALF_RANGE_M = 0.01


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


def retreat_pose_along_positive_z(
    measured_pose: np.ndarray,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
) -> np.ndarray:
    """Return a 1 cm base-frame +Z retreat within the configured workspace."""
    pose = np.asarray(measured_pose, dtype=float).copy()
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError(f"invalid measured TCP pose: {pose}")
    pose[2] += RESET_RETREAT_Z_M
    if np.any(pose[:3] < workspace_min) or np.any(pose[:3] > workspace_max):
        raise RuntimeError(
            "+Z reset retreat would leave workspace: "
            f"target={pose[:3]}, min={workspace_min}, max={workspace_max}"
        )
    return pose


def sample_z_insertion_workspace_pose(
    reference: np.ndarray,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    *,
    randomize_xy: bool = False,
) -> np.ndarray:
    """Use p0 XY, optionally randomized ±1 cm, with fixed Z and orientation."""
    reference = np.asarray(reference, dtype=float)
    lower = np.asarray(workspace_min, dtype=float)
    upper = np.asarray(workspace_max, dtype=float)
    if reference.shape != (6,) or not np.isfinite(reference).all():
        raise ValueError(f"invalid Z insertion reference pose: {reference}")
    if (
        lower.shape != (3,) or upper.shape != (3,)
        or not np.isfinite(lower).all() or not np.isfinite(upper).all()
        or np.any(lower >= upper)
    ):
        raise ValueError("invalid Z insertion workspace bounds")
    xy_half_range = Z_SAMPLE_XY_HALF_RANGE_M if randomize_xy else 0.0
    sample_min = reference[:3] + [-xy_half_range, -xy_half_range, 0]
    sample_max = reference[:3] + [xy_half_range, xy_half_range, 0]
    sample_min[2] = sample_max[2] = SAMPLE_Z_M
    if np.any(sample_min < lower) or np.any(sample_max > upper):
        raise ValueError("Z insertion sample volume extends outside the safe workspace")
    if reference[2] - RESET_RETREAT_Z_M < lower[2]:
        raise ValueError("Z insertion endpoint extends below the safe workspace")
    position = np.random.uniform(sample_min, sample_max) if randomize_xy else sample_min
    return np.concatenate([position, SAMPLE_ROTATION_VECTOR])


def capture_centered_workspace(robot, *, randomize_xy: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Capture the current TCP pose on Enter, then bound XYZ to ±3 cm."""
    while True:
        choice = input(
            "Place TCP at Z insertion p0; press Enter to capture its pose "
            "and set XYZ workspace to +/-3 cm (q to stop): "
        ).strip().lower()
        if not choice:
            break
        if choice in {"q", "quit"}:
            raise KeyboardInterrupt
        print("Press Enter to capture or q to stop.", flush=True)
    pose = np.asarray(robot.robot.get_tool_pose(), dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise RuntimeError(f"invalid TCP pose at workspace capture: {pose}")
    lower = pose[:3] - STARTUP_WORKSPACE_HALF_RANGE_M
    upper = pose[:3] + STARTUP_WORKSPACE_HALF_RANGE_M
    print(
        f"Captured p0: {pose.tolist()}; workspace XYZ: "
        f"min={lower.tolist()}, max={upper.tolist()}",
        flush=True,
    )
    xy_description = (
        f"XY randomized +/-{Z_SAMPLE_XY_HALF_RANGE_M:.3f} m around p0"
        if randomize_xy else "XY fixed at p0"
    )
    print(
        f"Sample resets: {xy_description}, "
        f"fixed absolute Z={SAMPLE_Z_M:.3f} m, "
        f"intrinsic XYZ Euler={SAMPLE_EULER_XYZ_DEG} deg.",
        flush=True,
    )
    return pose, lower, upper


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
        camera_shapes: dict[str, tuple[int, int, int]] | None = None,
        z_insertion_mode: bool = False,
        randomize_reset_xy: bool = False,
        reference_pose: np.ndarray | None = None,
        enable_human_intervention: bool = False,
        teleop_input_backend: str = "evdev",
        teleop_keyboard: str | None = None,
        teleop_speed_m_s: float = 0.01,
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
        self.camera_shapes = camera_shapes
        self.period_s = 1.0 / fps
        self.episode_time_s = episode_time_s
        self.max_step_m = max_step_m
        self.max_step_rad = max_step_rad
        self.enable_human_intervention = enable_human_intervention
        self.teleop_input_backend = teleop_input_backend
        self.teleop_keyboard = teleop_keyboard
        self.teleop_speed_m_s = teleop_speed_m_s
        self._control_mode = "policy"
        self.workspace_min = np.asarray(robot.config.workspace_min_xyz, dtype=float)
        self.workspace_max = np.asarray(robot.config.workspace_max_xyz, dtype=float)
        self.sample_min = (
            None if sample_min is None else np.asarray(sample_min, dtype=float)
        )
        self.sample_max = (
            None if sample_max is None else np.asarray(sample_max, dtype=float)
        )
        self.z_insertion_mode = z_insertion_mode
        self.randomize_reset_xy = randomize_reset_xy
        self._reference_pose = (
            None if reference_pose is None else np.asarray(reference_pose, dtype=float).copy()
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
        self._runtime = None
        self._episode_sampled_tcp_command_sum = np.zeros(6, dtype=float)
        self._episode_reference_tcp_command_sum = np.zeros(6, dtype=float)
        self._episode_mean_tcp_command_sum = np.zeros(6, dtype=float)
        self._episode_tcp_target_comparison_steps = 0
        self._episode_human_steps = 0

    def _processed_observation(self) -> dict[str, torch.Tensor]:
        return self.pre(observation_frame(self.robot.get_observation(), self.camera_shapes))

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
        if self.z_insertion_mode:
            retreat = retreat_pose_along_positive_z(
                measured, self.workspace_min, self.workspace_max
            )
            label = "+Z 1 cm retreat"
        else:
            retreat = retreat_pose_along_positive_y(
                measured, self.workspace_min, self.workspace_max
            )
            label = "+Y 1 cm retreat"
        self._move_reset_pose(retreat, label=label)

    def _move_to_accepted_workspace_sample(self) -> None:
        if self.z_insertion_mode and self._reference_pose is None:
            measured = np.asarray(self.robot.robot.get_tool_pose(), dtype=float)
            if measured.shape != (6,) or not np.isfinite(measured).all():
                raise RuntimeError(f"invalid initial TCP reference pose: {measured}")
            choice = input(
                "Use current TCP pose as fixed Z insertion p0 "
                f"{self._format_pose(measured)}? Press Enter/y to accept, q to stop: "
            ).strip().lower()
            if choice not in {"", "y", "yes"}:
                raise KeyboardInterrupt
            self._reference_pose = measured.copy()
        if self._reset_orientation is None:
            measured = (
                self._reference_pose if self.z_insertion_mode
                else np.asarray(self.robot.robot.get_tool_pose(), dtype=float)
            )
            if measured.shape != (6,) or not np.isfinite(measured).all():
                raise RuntimeError(f"invalid initial TCP pose: {measured}")
            self._reset_orientation = (
                np.asarray(SAMPLE_ROTATION_VECTOR, dtype=float).copy()
                if self.z_insertion_mode else measured[3:].copy()
            )
            orientation_label = (
                "fixed collection" if self.z_insertion_mode else "current TCP"
            )
            print(
                f"Using {orientation_label} reset orientation: "
                f"{self._format_pose(self._reset_orientation)}",
                flush=True,
            )
        while True:
            if self.z_insertion_mode:
                target = sample_z_insertion_workspace_pose(
                    self._reference_pose, self.workspace_min, self.workspace_max,
                    randomize_xy=self.randomize_reset_xy,
                )
            else:
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
                "Episode maximum time elapsed. Confirm outcome: "
                "s=success, f=failure, q=stop training: "
            ).strip().lower()
            if choice in {"s", "f", "q"}:
                return choice
            print("Please enter s, f, or q.", flush=True)

    def reset(self, *, episode_id: int, warmup: bool) -> dict[str, torch.Tensor]:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        self._close_monitor()
        self._control_mode = "policy"
        phase = "ACT WARMUP" if warmup else "RLT ACTOR"
        print(f"\nPreparing Episode {episode_id} [{phase}] with automatic reset.", flush=True)
        self._move_to_accepted_workspace_sample()
        self._needs_automatic_reset = False
        self.robot.resync_command_pose()
        self._episode_sampled_tcp_command_sum.fill(0.0)
        self._episode_reference_tcp_command_sum.fill(0.0)
        self._episode_mean_tcp_command_sum.fill(0.0)
        self._episode_tcp_target_comparison_steps = 0
        self._episode_human_steps = 0
        batch = self._processed_observation()
        if self.enable_human_intervention:
            from act_rlt.human_input import HumanInputMonitor

            self._monitor = HumanInputMonitor(self.teleop_input_backend, self.teleop_keyboard)
            print(
                "Episode active: Space=take over/return at 4-step boundary; "
                "W/X/A/D/P/L=human XYZ; S/F=finish after this 4-step chunk; Esc=stop training",
                flush=True,
            )
        else:
            self._monitor = OutcomeMonitor()
            print("Episode active: s/f=finish after this 4-step chunk; q=stop training", flush=True)
        self._episode_start = time.monotonic()
        return batch

    def _record_episode_action_comparison(self, result: ChunkExecution) -> None:
        """Add executed command targets and emit an episode-end comparison.

        A reference rollout is intentionally not executed on the robot.  The
        reported endpoint difference is therefore the difference between the
        *integrated, safety-bounded TCP command deltas* from the same episode
        start, not a second measured physical rollout.
        """
        # Keep this helper usable by lightweight test environments which are
        # intentionally constructed without running the hardware __init__.
        for name in (
            "_episode_sampled_tcp_command_sum",
            "_episode_reference_tcp_command_sum",
            "_episode_mean_tcp_command_sum",
        ):
            if not hasattr(self, name):
                setattr(self, name, np.zeros(6, dtype=float))
        if not hasattr(self, "_episode_tcp_target_comparison_steps"):
            self._episode_tcp_target_comparison_steps = 0
        if not hasattr(self, "_episode_human_steps"):
            self._episode_human_steps = 0
        sums = (
            ("_sampled_tcp_command_sum", self._episode_sampled_tcp_command_sum),
            ("_reference_tcp_command_sum", self._episode_reference_tcp_command_sum),
            ("_mean_tcp_command_sum", self._episode_mean_tcp_command_sum),
        )
        for name, destination in sums:
            value = np.asarray(result.info.pop(name, np.zeros(6)), dtype=float)
            if value.shape != (6,) or not np.isfinite(value).all():
                raise RuntimeError(f"invalid Stage-2 TCP command summary {name}: {value}")
            if not result.intervention:
                destination += value
        if result.intervention:
            self._episode_human_steps += result.actual_steps
        else:
            self._episode_tcp_target_comparison_steps += result.actual_steps
        if not result.done:
            return

        def add_delta(prefix: str, delta: np.ndarray) -> None:
            result.info[f"{prefix}_xyz_m"] = delta[:3].tolist()
            result.info[f"{prefix}_translation_norm_mm"] = float(np.linalg.norm(delta[:3]) * 1000)
            result.info[f"{prefix}_rotvec_rad"] = delta[3:].tolist()
            result.info[f"{prefix}_rotation_norm_rad"] = float(np.linalg.norm(delta[3:]))

        # This is the quantity that measures the net effect of sigma after
        # physical de-normalization and safety clipping.
        add_delta(
            "episode_exploration_vs_actor_mean_tcp_target",
            self._episode_sampled_tcp_command_sum - self._episode_mean_tcp_command_sum,
        )
        # Keep the learned-policy deviation separate from random exploration.
        add_delta(
            "episode_actor_mean_vs_reference_tcp_target",
            self._episode_mean_tcp_command_sum - self._episode_reference_tcp_command_sum,
        )
        add_delta(
            "episode_sampled_vs_reference_tcp_target",
            self._episode_sampled_tcp_command_sum - self._episode_reference_tcp_command_sum,
        )
        result.info["episode_tcp_target_comparison_steps"] = self._episode_tcp_target_comparison_steps
        result.info["episode_human_steps"] = self._episode_human_steps

    def execute_policy_chunk(
        self, policy: ACTStage2Policy, initial_batch: dict[str, torch.Tensor],
        *, warmup: bool, remaining_steps: int, deterministic: bool = False,
    ) -> ChunkExecution:
        """Consume one result from the persistent fixed-chunk collector."""
        from act_rlt.stage2_runtime import FixedChunkRuntime

        if self._runtime is None:
            self._runtime = FixedChunkRuntime(
                self, policy, initial_batch, warmup=warmup, step_budget=remaining_steps,
                deterministic=deterministic,
            )
        elif self._runtime.warmup != warmup or self._runtime.deterministic != deterministic:
            raise RuntimeError("collector must pause at the warmup boundary")
        publish_ms = self._runtime.publish_actor(policy.actor)
        result = self._runtime.next_result()
        result.info["actor_publish_wall_ms"] = publish_ms
        self._record_episode_action_comparison(result)
        if result.info["collector_paused"]:
            self._runtime.close()
            self._runtime = None

        if result.info.get("timed_out"):
            self._close_monitor()
            key = self._prompt_outcome_after_timeout()
            if key == "s":
                result.reward_seq[result.actual_steps - 1] = 1.0
                result.info["success"] = True
                result.terminated = True
            elif key == "f":
                result.terminated = True
            else:
                result.truncated = result.stop_requested = True
        if result.done:
            self._close_monitor()
            if not result.stop_requested and not result.info["workspace_violation"]:
                self._retreat_after_outcome()
            if not result.stop_requested:
                self._needs_automatic_reset = True
        return result

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
        pending_outcome = None
        workspace_violation = False
        workspace_error: str | None = None
        last_batch = None
        clipping_count = 0

        for index in range(self.config.chunk_length):
            tick = time.monotonic()
            normalized_full = full_reference[:, index, :].clone()
            normalized_full[:, : self.config.action_dim] = action_chunk[:, index, :]
            physical_full = self.post(normalized_full).detach().cpu().reshape(-1)
            if physical_full.numel() != self.config.action_dim + 1:
                raise RuntimeError(
                    f"unexpected Stage-2 physical action shape: {tuple(physical_full.shape)}"
                )
            command = bounded_action(physical_full.numpy(), self.max_step_m, self.max_step_rad)
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
            if key in {"s", "f"} and pending_outcome is None:
                pending_outcome = key
            if key == "q":
                truncated = done = stop_requested = True
            elif pending_outcome is None and time.monotonic() - self._episode_start >= self.episode_time_s:
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
                    self._retreat_after_outcome()
                if not stop_requested:
                    self._needs_automatic_reset = True
                break
        if pending_outcome is not None and not done:
            rewards[actual_steps - 1] = float(pending_outcome == "s")
            success = pending_outcome == "s"
            terminated = done = True
            self._close_monitor()
            self._retreat_after_outcome()
            self._needs_automatic_reset = True
        if last_batch is None:
            raise RuntimeError("action chunk executed zero steps without a next observation")
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

    def exclude_learning_time(self, seconds: float) -> None:
        self._episode_start += seconds

    def close(self) -> None:
        try:
            if self._runtime is not None:
                self._runtime.close()
                self._runtime = None
        finally:
            self._close_monitor()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--act-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/act_rlt_001_stage2"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replay-from", type=Path, help="reuse compatible replay with fresh Actor/Critic; new output required")
    parser.add_argument("--prepare-only", action="store_true", help="initialize from replay and save v3 checkpoint without connecting robot")
    parser.add_argument("--bc-init-steps", type=int, default=1000)
    parser.add_argument("--critic-init-steps", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--enable-human-intervention", action="store_true")
    parser.add_argument(
        "--teleop-input-backend",
        choices=("auto", "evdev", "pynput"),
        default="auto",
        help="auto uses pynput in an X11 session, or evdev on a headless host",
    )
    parser.add_argument("--teleop-keyboard", help="evdev input device, e.g. /dev/input/event4")
    parser.add_argument("--teleop-speed-m-s", type=float, default=0.01)

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
    parser.add_argument(
        "--z-insertion-mode", action="store_true",
        help="reset at p0 XY with fixed Z=0.402 m and collection orientation; retreat +Z 1 cm",
    )
    parser.add_argument(
        "--randomize-reset-xy", action="store_true",
        help="with --z-insertion-mode, randomize reset X/Y +/-1 cm around p0 (default: fixed p0 XY)",
    )
    parser.add_argument("--fps", type=float, default=None,
                        help="control rate in Hz (default: 30 for Z insertion, 15 otherwise)")
    parser.add_argument(
        "--camera-boundary-mode", choices=("timestamp", "legacy"), default="timestamp",
        help="timestamp: use verified exposure timing, falling back to legacy waits when unavailable",
    )
    parser.add_argument("--episode-time", type=float, default=5.0)
    parser.add_argument("--max-step-m", type=float, default=0.002)
    parser.add_argument("--max-step-rad", type=float, default=0.02)
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=0.01,
        metavar="M",
        help="legacy checkpoint setting; Stage-2 always executes fixed chunks without temporal ensemble",
    )
    parser.add_argument("--warmup-steps", type=int, default=4_000)
    parser.add_argument(
        "--total-env-steps",
        type=int,
        default=10_000,
        help="cumulative online-step target after warmup; use 20000 to add 10000 after 10000",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fusion-dim", type=int, default=128,
                        help="width of each token, joint, and action input branch (default: 128)")
    parser.add_argument("--utd-ratio", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.5,
                        help="BC regularization coefficient in -Q + beta * BC (default: 0.5)")
    parser.add_argument("--exploration-sigma", type=float, default=0.2,
                        help="fixed normalized action standard deviation during collection (default: 0.2)")
    parser.add_argument("--save-every-env-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.randomize_reset_xy and not args.z_insertion_mode:
        parser.error("--randomize-reset-xy requires --z-insertion-mode")
    if args.fps is None:
        args.fps = 30.0 if args.z_insertion_mode else 15.0
    if args.teleop_input_backend == "auto":
        args.teleop_input_backend = (
            "evdev" if args.teleop_keyboard or not os.environ.get("DISPLAY") else "pynput"
        )
    if args.teleop_keyboard and args.teleop_input_backend != "evdev":
        parser.error("--teleop-keyboard applies only to --teleop-input-backend evdev")
    if args.enable_human_intervention:
        from act_rlt.human_input import validate_teleop_speed

        try:
            validate_teleop_speed(args.teleop_speed_m_s, args.fps, args.max_step_m)
        except ValueError as error:
            parser.error(str(error))
    if args.resume and args.replay_from:
        parser.error("--resume and --replay-from are mutually exclusive")
    if args.prepare_only and not (args.replay_from or args.resume):
        parser.error("--prepare-only requires --replay-from or --resume")
    if not args.dry_run and not args.prepare_only and not args.allow_motion:
        parser.error("real Stage-2 collection requires --allow-motion")
    if args.allow_motion and not args.z_insertion_mode and (
        args.workspace_min is None or args.workspace_max is None
    ):
        parser.error("motion requires --workspace-min X Y Z and --workspace-max X Y Z")
    if (args.workspace_min is None) != (args.workspace_max is None):
        parser.error("supply both workspace bounds")
    if args.z_insertion_mode and args.workspace_min is not None:
        parser.error("--z-insertion-mode captures the +/-3 cm workspace at startup; omit workspace bounds")
    if (args.sample_min is None) != (args.sample_max is None):
        parser.error("supply both --sample-min and --sample-max")
    if args.z_insertion_mode and args.sample_min is not None:
        parser.error("--sample-min/--sample-max cannot be combined with --z-insertion-mode")
    if args.workspace_min is not None:
        bounds = np.asarray([args.workspace_min, args.workspace_max], dtype=float)
        if not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
            parser.error("workspace bounds must be finite with min < max")
        if not args.z_insertion_mode and np.any(bounds[1, :2] - bounds[0, :2] <= 2 * RESET_XY_MARGIN_M):
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
    if (not np.isfinite(args.temporal_ensemble_coeff)
            or args.temporal_ensemble_coeff < 0):
        parser.error("temporal-ensemble-coeff must be finite and non-negative")


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
    if saved["learner"].get("learner_version") != 3:
        raise ValueError(
            "old Stage-2 network cannot resume with balanced input fusion; "
            "use --replay-from with a new output directory"
        )
    saved_config = saved["config"]
    current_config = config.to_dict()
    # Checkpoints created before the real-time temporal collector existed have
    # no value for this inference-only setting. Adopt the requested runtime
    # value without invalidating their learner/replay state.
    saved_config.setdefault(
        "temporal_ensemble_coeff", current_config["temporal_ensemble_coeff"]
    )
    # Beta changes only the Actor objective on future updates; it does not
    # invalidate the learned network, optimizer state, or replay contents.
    schedule_keys = {"warmup_steps", "total_env_steps", "device", "beta"}
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


def import_replay(path: Path, config: ACTStage2Config, replay: ReplayBuffer) -> OnlineStage2Metrics:
    """Reuse data only. Learned weights, optimizers and old online counts are discarded."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("stage1_checkpoint", "act_checkpoint"):
        if resolve_pretrained_model(saved["config"][key]) != resolve_pretrained_model(getattr(config, key)):
            raise ValueError(f"replay {key} differs; latent/normalization spaces must match")
    for key in ("chunk_length", "action_dim", "proprio_dim"):
        if saved["config"][key] != getattr(config, key):
            raise ValueError(f"replay {key} mismatch")
    for transition in saved["replay"]:
        expected = {"state_vec": (config.state_dim,), "next_state_vec": (config.state_dim,),
                    "exec_chunk": (config.chunk_length, config.action_dim),
                    "ref_chunk": (config.chunk_length, config.action_dim),
                    "next_ref_chunk": (config.chunk_length, config.action_dim),
                    "reward_seq": (config.chunk_length,)}
        for key, shape in expected.items():
            value = getattr(transition, key)
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid replay {key}")
        bc_target = getattr(transition, "bc_target_chunk", None)
        if bc_target is not None and (
            tuple(bc_target.shape) != expected["ref_chunk"] or not torch.isfinite(bc_target).all()
        ):
            raise ValueError("invalid replay bc_target_chunk")
        k = int(transition.actual_steps)
        rewards = transition.reward_seq
        if (not 1 <= k <= config.chunk_length or float(transition.done) not in (0, 1)
                or not ((rewards == 0) | (rewards == 1)).all()
                or rewards.sum() > 1 or rewards[k:].any()
                or (rewards.any() and (not bool(transition.done) or rewards[k-1] != 1))):
            raise ValueError("replay violates sparse terminal-success reward contract")
        replay.add(transition)
    if len(replay) < config.batch_size:
        raise ValueError("insufficient replay for initialization")
    # Imported data replaces collection warmup; new online budget starts at zero.
    print(f"Imported {len(replay)} transitions; resetting Actor/Critic and online counters", flush=True)
    return OnlineStage2Metrics(warmup_env_steps=config.warmup_steps, env_steps=config.warmup_steps)


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
        "human_chunks": metrics.human_chunks,
        "human_env_steps": metrics.human_env_steps,
        "critic_updates": metrics.critic_updates,
        "actor_updates": metrics.actor_updates,
        "actual_steps": execution.actual_steps,
        "intervention": execution.intervention,
        "done": execution.done,
        "terminated": execution.terminated,
        "truncated": execution.truncated,
        **execution.info,
    }
    if learner is not None:
        record.update(learner.diagnostics)
        record.update(
            critic_loss=learner.critic_loss,
            actor_loss=learner.actor_loss,
            critic_grad_norm=learner.critic_grad_norm,
            actor_grad_norm=learner.actor_grad_norm,
        )
    with (output / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(_format_terminal_metrics(record), flush=True)


def _format_terminal_metrics(record: dict[str, Any]) -> str:
    """Return the small live status line; JSONL retains every metric."""
    def number(name: str, digits: int = 3) -> str | None:
        value = record.get(name)
        return None if value is None else f"{float(value):.{digits}f}"

    phase = "WARMUP" if record.get("warmup") else "ONLINE"
    parts = [
        f"[{phase}]",
        f"env={record['env_steps']}",
        f"warm={record['warmup_env_steps']}",
        f"online={record['online_env_steps']}",
        f"ep={record['episodes']}",
        f"succ={record['successes']}",
        f"k={record['actual_steps']}",
    ]
    if record.get("workspace_violation"):
        parts.append("WORKSPACE")
    if record.get("intervention"):
        parts.append("HUMAN")
    if record.get("safety_clip_steps"):
        parts.append(f"clip={record['safety_clip_steps']}")
    if record.get("done"):
        outcome = "success" if record.get("success") else "stop" if record.get("stop_requested") else "end"
        parts.append(f"DONE={outcome}")

    losses = [
        f"critic={value}" for value in [number("critic_loss")] if value is not None
    ] + [f"actor={value}" for value in [number("actor_loss")] if value is not None]
    if losses:
        parts.append("loss(" + " ".join(losses) + ")")
    q_values = []
    for label, name in (("q1", "q1_mean"), ("q2", "q2_mean"),
                        ("ref", "q_reference_mean"), ("deploy", "q_actor_deploy_mean")):
        value = number(name)
        if value is not None:
            q_values.append(f"{label}={value}")
    if q_values:
        parts.append("Q(" + " ".join(q_values) + ")")
    if record.get("done"):
        exploration = number("episode_exploration_vs_actor_mean_tcp_target_translation_norm_mm", 2)
        sampled_ref = number("episode_sampled_vs_reference_tcp_target_translation_norm_mm", 2)
        if exploration is not None and sampled_ref is not None:
            parts.append(f"Δtcp_mm(explore={exploration} sampled-ref={sampled_ref})")
    return " ".join(parts)


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
        fusion_dim=args.fusion_dim,
        utd_ratio=args.utd_ratio,
        beta=args.beta,
        actor_fixed_std=args.exploration_sigma,
        seed=args.seed,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        bc_init_steps=args.bc_init_steps,
        critic_init_steps=args.critic_init_steps,
        max_step_m=args.max_step_m,
        max_step_rad=args.max_step_rad,
    )
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True), flush=True)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTStage2Policy(config).to(config.device)
    camera_shapes = checkpoint_camera_shapes(policy.stage1._act.config)
    pre, post = make_pre_post_processors(
        policy.stage1._act.config,
        pretrained_path=str(act_path),
        preprocessor_overrides={"device_processor": {"device": config.device}},
    )
    learner = Stage2Learner(policy, config)
    learner.configure_action_projection(post)
    replay = ReplayBuffer(config.replay_capacity)
    metrics = OnlineStage2Metrics()
    if args.resume:
        metrics = load_checkpoint(output / "checkpoints/latest.pt", config, learner, replay)
    elif args.replay_from:
        metrics = import_replay(args.replay_from, config, replay)
    if args.dry_run:
        print(
            "Dry run complete: checkpoints and saved processors loaded; robot not connected.",
            flush=True,
        )
        return 0

    if args.prepare_only:
        learner.initialize(replay)
        metrics.critic_updates = learner.critic_updates
        metrics.actor_updates = learner.actor_updates
        save_checkpoint(output, config, learner, replay, metrics)
        (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
        print(f"Prepared v3 checkpoint: {output / 'checkpoints/latest.pt'}", flush=True)
        return 0

    if args.enable_human_intervention:
        from act_rlt.human_input import validate_human_input

        validate_human_input(args.teleop_input_backend, args.teleop_keyboard)

    from evo_rlt.adapters.lerobot.franka_robot import FrankaRobot, FrankaRobotConfig

    robot = FrankaRobot(
        FrankaRobotConfig(
            robot_ip=args.robot_ip,
            **robot_camera_kwargs(camera_shapes, args),
            include_gripper_action=True,
            enable_fci_keepalive=False,
            workspace_min_xyz=None if args.z_insertion_mode else tuple(args.workspace_min),
            workspace_max_xyz=None if args.z_insertion_mode else tuple(args.workspace_max),
        )
    )
    if args.camera_boundary_mode == "timestamp":
        robot.enable_timestamped_observations()
    env = None
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
        reference_pose = None
        if args.z_insertion_mode:
            reference_pose, workspace_min, workspace_max = capture_centered_workspace(
                robot, randomize_xy=args.randomize_reset_xy
            )
            robot.config.workspace_min_xyz = tuple(workspace_min)
            robot.config.workspace_max_xyz = tuple(workspace_max)
        env = FrankaInsertionStage2Env(
            robot=robot,
            preprocessor=pre,
            postprocessor=post,
            config=config,
            fps=args.fps,
            episode_time_s=args.episode_time,
            max_step_m=args.max_step_m,
            max_step_rad=args.max_step_rad,
            camera_shapes=camera_shapes,
            z_insertion_mode=args.z_insertion_mode,
            randomize_reset_xy=args.randomize_reset_xy,
            reference_pose=reference_pose,
            sample_min=None if args.sample_min is None else tuple(args.sample_min),
            sample_max=None if args.sample_max is None else tuple(args.sample_max),
            enable_human_intervention=args.enable_human_intervention,
            teleop_input_backend=args.teleop_input_backend,
            teleop_keyboard=args.teleop_keyboard,
            teleop_speed_m_s=args.teleop_speed_m_s,
        )
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
        try:
            if env is not None:
                env.close()
        finally:
            try:
                if robot.is_connected:
                    try:
                        robot.robot.stop_move()
                    finally:
                        robot.disconnect()
            finally:
                save_checkpoint(output, config, learner, replay, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
