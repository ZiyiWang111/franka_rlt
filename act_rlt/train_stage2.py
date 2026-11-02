#!/usr/bin/env python3
"""Run ACT-RLT Stage-2 online training on the local FR3.

Critical-phase episodes begin at automatically sampled reset poses. Optional
SpaceMouse takeover switches control only after a complete action chunk.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import select
import sys
import termios
import tempfile
import time
import tty
from collections import deque
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
from act_rlt.exploration_bias import DOWNWARD_BIAS_STEP_M, EpisodeDownwardBias, EpisodeExplorationBias
from act_rlt.infer import (
    bounded_action,
    checkpoint_camera_shapes,
    checkpoint_state_mode,
    observation_frame,
    robot_camera_kwargs,
)
from act_rlt.stage2 import (
    strip_removed_force_feedback_config,
    ACTStage2Config,
    ACTStage2Policy,
    ChunkExecution,
    OnlineStage2Metrics,
    Stage2Learner,
    resolve_pretrained_model,
    run_online_stage2,
)
from act_rlt.stage2_logging import Stage2MetricsWriter
from evo_rlt.core.replay_buffer import ReplayBuffer
from lerobot.processor import NormalizerProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.types import TransitionKey


RESET_RETREAT_Y_M = 0.01
RESET_RETREAT_Z_M = 0.01
RESET_MOVE_SPEED_M_S = 0.02
RESET_XY_MARGIN_M = 0.02
STARTUP_WORKSPACE_HALF_RANGE_M = 0.05
Z_SAMPLE_XY_HALF_RANGE_M = 0.0
RESET_XY_RANGE_STEP_M = 0.005
RESET_XY_MAX_HALF_RANGE_M = 0.05


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
    if workspace_min is not None and workspace_max is not None and (
        np.any(pose[:3] < workspace_min) or np.any(pose[:3] > workspace_max)
    ):
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
    if workspace_min is not None and workspace_max is not None and (
        np.any(pose[:3] < workspace_min) or np.any(pose[:3] > workspace_max)
    ):
        raise RuntimeError(
            "+Z reset retreat would leave workspace: "
            f"target={pose[:3]}, min={workspace_min}, max={workspace_max}"
        )
    return pose


def sample_z_insertion_workspace_pose(
    reference: np.ndarray,
    workspace_min: np.ndarray | None,
    workspace_max: np.ndarray | None,
    *,
    randomize_xy: bool = True,
    xy_half_range: float = Z_SAMPLE_XY_HALF_RANGE_M,
    use_reference_pose: bool = False,
) -> np.ndarray:
    """Sample reference XY; optionally retain reference Z and orientation."""
    reference = np.asarray(reference, dtype=float)
    lower = None if workspace_min is None else np.asarray(workspace_min, dtype=float)
    upper = None if workspace_max is None else np.asarray(workspace_max, dtype=float)
    if reference.shape != (6,) or not np.isfinite(reference).all():
        raise ValueError(f"invalid Z insertion reference pose: {reference}")
    if (lower is None) != (upper is None):
        raise ValueError("supply both workspace bounds or neither")
    if lower is not None and (
        lower.shape != (3,) or upper.shape != (3,)
        or not np.isfinite(lower).all() or not np.isfinite(upper).all()
        or np.any(lower >= upper)
    ):
        raise ValueError("invalid Z insertion workspace bounds")
    if not np.isfinite(xy_half_range) or not 0 <= xy_half_range <= RESET_XY_MAX_HALF_RANGE_M:
        raise ValueError("reset XY half range must be between 0 and 0.05 m")
    xy_half_range = xy_half_range if randomize_xy else 0.0
    sample_min = reference[:3] + [-xy_half_range, -xy_half_range, 0]
    sample_max = reference[:3] + [xy_half_range, xy_half_range, 0]
    sample_min[2] = sample_max[2] = reference[2] if use_reference_pose else SAMPLE_Z_M
    if lower is not None and (np.any(sample_min < lower) or np.any(sample_max > upper)):
        raise ValueError("Z insertion sample volume extends outside the safe workspace")
    if lower is not None and reference[2] - RESET_RETREAT_Z_M < lower[2]:
        raise ValueError("Z insertion endpoint extends below the safe workspace")
    position = np.random.uniform(sample_min, sample_max) if randomize_xy else sample_min
    orientation = reference[3:6] if use_reference_pose else SAMPLE_ROTATION_VECTOR
    return np.concatenate([position, orientation])


def read_reset_point_choice(prompt: str) -> str:
    """Read a single terminal key so digits move immediately and Enter starts."""
    if not sys.stdin.isatty():
        return input(prompt).strip().lower()
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    print(prompt, end="", flush=True)
    try:
        tty.setcbreak(fd)
        key = os.read(fd, 1)
        if not key:
            raise EOFError("terminal closed during reset point selection")
        choice = key.decode(errors="ignore").lower()
        if choice == "\x03":
            raise KeyboardInterrupt
        return "" if choice in {"\r", "\n"} else choice
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        print(flush=True)


def capture_centered_workspace(
    robot, *, randomize_xy: bool = True, terminal=None,
    workspace_y_half_range_m: float = STARTUP_WORKSPACE_HALF_RANGE_M,
    confirm_capture: bool = True,
    enforce_workspace: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Capture TCP, optionally generating bounds when workspace is enabled."""
    if not np.isfinite(workspace_y_half_range_m) or workspace_y_half_range_m <= 0:
        raise ValueError("workspace Y half range must be finite and positive")
    while confirm_capture:
        prompt = (
            "Place TCP at Z insertion p1; press Enter to capture its pose "
            f"and set workspace X/Z +/-5 cm, Y +/-{workspace_y_half_range_m * 100:g} cm (q to stop): "
        )
        if not enforce_workspace:
            prompt = "Place TCP at reference; press Enter to capture its pose (q to stop): "
        if terminal is not None:
            label = "reference and workspace" if enforce_workspace else "reference"
            prompt = terminal.prompt(f"Place TCP at p1. Enter: capture {label} | q: exit")
        choice = input(prompt).strip().lower()
        if not choice:
            break
        if choice in {"q", "quit"}:
            raise KeyboardInterrupt
        print("Press Enter to capture or q to stop.", flush=True)
    pose = np.asarray(robot.robot.get_tool_pose(), dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise RuntimeError(f"invalid TCP pose at workspace capture: {pose}")
    if not enforce_workspace:
        print(f"Captured reference: {pose.tolist()}; workspace disabled.", flush=True)
        return pose, None, None
    lower = pose[:3] - STARTUP_WORKSPACE_HALF_RANGE_M
    upper = pose[:3] + STARTUP_WORKSPACE_HALF_RANGE_M
    lower[1] = pose[1] - workspace_y_half_range_m
    upper[1] = pose[1] + workspace_y_half_range_m
    if terminal is not None:
        terminal.workspace(pose, lower, upper, None)
        return pose, lower, upper
    print(
        f"Captured p1 reference: {pose.tolist()}; workspace XYZ: "
        f"min={lower.tolist()}, max={upper.tolist()}",
        flush=True,
    )
    xy_description = f"XY randomized +/-{Z_SAMPLE_XY_HALF_RANGE_M:.3f} m around initial reference"
    print(
        f"Sample resets: {xy_description}, "
        f"fixed absolute Z={SAMPLE_Z_M:.3f} m, "
        f"intrinsic XYZ Euler={SAMPLE_EULER_XYZ_DEG} deg.",
        flush=True,
    )
    return pose, lower, upper


def restore_training_reference(
    saved: dict, *, enforce_workspace: bool = True, use_current_reference: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Validate the base-frame pose and workspace recorded with this checkpoint."""
    if not use_current_reference:
        saved = saved.get("initial_reference", saved)
    if (saved.get("version") != 1 or saved.get("mode") != "z_insertion"
            or saved.get("frame") != "base" or saved.get("position_unit") != "m"
            or saved.get("orientation") != "rotation_vector_rad"):
        raise ValueError("unsupported saved training reference metadata")
    pose = np.asarray(saved["reference_pose"], dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("invalid saved training reference pose")
    if not enforce_workspace:
        return pose, None, None
    lower = np.asarray(saved["workspace_min_xyz"], dtype=float)
    upper = np.asarray(saved["workspace_max_xyz"], dtype=float)
    if (pose.shape != (6,) or lower.shape != (3,) or upper.shape != (3,)
            or not all(np.isfinite(value).all() for value in (pose, lower, upper))
            or np.any(lower >= upper) or np.any(pose[:3] < lower)
            or np.any(pose[:3] > upper)):
        raise ValueError("invalid saved training reference pose or workspace")
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
        self._pending_outcomes = deque()
        self._downward_bias_pending = False
        tty.setcbreak(self.fd)

    def _read_available(self):
        while select.select([self.fd], [], [], 0)[0]:
            value = os.read(self.fd, 1)
            if not value:
                break
            key = value.decode(errors="ignore").lower()
            if key == "b":
                self._downward_bias_pending = True
            else:
                self._pending_outcomes.append(key)

    def poll(self) -> str | None:
        self._read_available()
        return self._pending_outcomes.popleft() if self._pending_outcomes else None

    def consume_downward_bias(self) -> bool:
        self._read_available()
        pending = self._downward_bias_pending
        self._downward_bias_pending = False
        return pending

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
        camera_shapes: dict[str, tuple[int, int, int]] | None = None,
        reference_pose: np.ndarray | None = None,
        initial_reference_pose: np.ndarray | None = None,
        reset_xy_half_range_m: float = Z_SAMPLE_XY_HALF_RANGE_M,
        use_reference_pose: bool = False,
        save_snapshot=None,
        enable_human_intervention: bool = False,
        review_episode_replay: bool = False,
        teleop_input_backend: str = "evdev",
        teleop_keyboard: str | None = None,
        teleop_speed_m_s: float = 0.0135,
        spacemouse=None,
        observation_selection: str = "boundary",
        inference_terminal=None,
        enforce_workspace: bool = True,
        enable_random_reset: bool = True,
        enable_exploration_bias: bool = False,
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
        self.inference_terminal = inference_terminal
        self._exploration_bias = EpisodeExplorationBias() if enable_exploration_bias else None
        self._downward_bias = EpisodeDownwardBias() if enable_exploration_bias else None
        self.save_snapshot = save_snapshot
        self.camera_shapes = camera_shapes
        self.period_s = 1.0 / fps
        self.episode_time_s = episode_time_s
        self.max_step_m = max_step_m
        self.max_step_rad = max_step_rad
        self.enable_human_intervention = enable_human_intervention
        self.review_episode_replay = review_episode_replay
        self.teleop_input_backend = teleop_input_backend
        self.teleop_keyboard = teleop_keyboard
        self.teleop_speed_m_s = teleop_speed_m_s
        self._spacemouse = spacemouse
        if observation_selection not in {"latest", "boundary"}:
            raise ValueError("observation_selection must be latest or boundary")
        self.observation_selection = observation_selection
        self._control_mode = "policy"
        self.enforce_workspace = enforce_workspace
        self.workspace_min = (np.asarray(robot.config.workspace_min_xyz, dtype=float)
                              if enforce_workspace else None)
        self.workspace_max = (np.asarray(robot.config.workspace_max_xyz, dtype=float)
                              if enforce_workspace else None)
        self.enable_random_reset = enable_random_reset
        self._inference_start_pose = None
        if (not np.isfinite(reset_xy_half_range_m)
                or not 0 <= reset_xy_half_range_m <= RESET_XY_MAX_HALF_RANGE_M):
            raise ValueError("reset XY half range must be between 0 and 0.05 m")
        self.reset_xy_half_range_m = float(reset_xy_half_range_m)
        self.use_reference_pose = use_reference_pose
        self._reference_pose = (
            None if reference_pose is None else np.asarray(reference_pose, dtype=float).copy()
        )
        self._initial_reference_pose = (
            None if initial_reference_pose is None
            else np.asarray(initial_reference_pose, dtype=float).copy()
        )
        if self._initial_reference_pose is not None:
            self._reference_pose = self._initial_reference_pose.copy()
        if enforce_workspace and (
            self.workspace_min.shape != (3,)
            or self.workspace_max.shape != (3,)
            or not np.isfinite(self.workspace_min).all()
            or not np.isfinite(self.workspace_max).all()
            or np.any(self.workspace_min >= self.workspace_max)
        ):
            raise ValueError("automatic reset requires valid 3D workspace bounds")
        if self._initial_reference_pose is not None:
            pose = self._initial_reference_pose
            if (pose.shape != (6,) or not np.isfinite(pose).all()
                    or (enforce_workspace and (np.any(pose[:3] < self.workspace_min)
                                              or np.any(pose[:3] > self.workspace_max)))):
                raise ValueError("initial training reference pose must be finite and inside workspace")
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
        observation = self._capture_auxiliary_state(self.robot.get_observation())
        return self.pre(self._observation_frame(observation))

    def _capture_auxiliary_state(self, observation):
        if getattr(getattr(self, "config", None), "proprio_dim", 7) == 6:
            observation = dict(observation)
            observation["external_wrench"] = self.robot.get_external_wrench_base().copy()
        return observation

    def _observation_frame(self, observation):
        dim = getattr(getattr(self, "config", None), "proprio_dim", 7)
        mode = {0: "none", 1: "tcp_z", 6: "wrench", 7: "joints"}[dim]
        return observation_frame(observation, self.camera_shapes, state_mode=mode)

    def _close_monitor(self) -> None:
        if self._monitor is not None:
            self._monitor.close()
            self._monitor = None

    @staticmethod
    def _format_pose(pose: np.ndarray) -> str:
        return " ".join(f"{value:+.6f}" for value in pose)

    def _move_reset_pose(self, target: np.ndarray, *, label: str) -> None:
        terminal = getattr(self, "inference_terminal", None)
        if terminal is not None:
            terminal.moving(target, label)
        else:
            print(f"{label}: {self._format_pose(target)}", flush=True)
        measured = np.asarray(
            self.robot.robot.move_tool(target.tolist(), speed=RESET_MOVE_SPEED_M_S),
            dtype=float,
        )
        if measured.shape != (6,) or not np.isfinite(measured).all():
            raise RuntimeError(f"invalid measured pose after {label}: {measured}")
        self.robot.resync_command_pose()
        if terminal is not None:
            terminal.arrived(measured)
        else:
            print(f"Reached {label}: {self._format_pose(measured)}", flush=True)

    def _retreat_after_outcome(self) -> None:
        measured = np.asarray(self.robot.robot.get_tool_pose(), dtype=float)
        retreat = retreat_pose_along_positive_z(
            measured, self.workspace_min, self.workspace_max
        )
        self._move_reset_pose(retreat, label="+Z 1 cm retreat")

    def _read_idle_reset_choice(self, prompt: str) -> str:
        """Allow held-key XY servo motion while waiting for a reset choice."""
        from act_rlt.idle_xy import read_idle_xy_choice

        return read_idle_xy_choice(self, prompt)

    def _update_reference_xy_from_idle(self) -> None:
        """Use measured XY after a manual idle jog, retaining reference Z/rotation."""
        if not self.enable_random_reset or self._reference_pose is None:
            return
        pose = np.asarray(self.robot.robot.get_tool_pose(), dtype=float)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise RuntimeError("invalid TCP pose after idle XY adjustment")
        self._reference_pose[:2] = pose[:2]
        self._reference_xy_revision = getattr(self, "_reference_xy_revision", 0) + 1
        print(f"Updated reset reference XY: {pose[0]:+.6f} {pose[1]:+.6f} m", flush=True)

    def _prepare_fixed_inference_start(self) -> None:
        if self._inference_start_pose is None:
            self._inference_start_pose = (
                np.asarray(self.robot.robot.get_tool_pose(), dtype=float).copy()
                if self._reference_pose is None else self._reference_pose.copy()
            )
        if self._first_reset_move:
            choice = self._read_idle_reset_choice("Enter/y: move to reference | q: exit")
            if choice not in {"", "y", "yes"}:
                raise KeyboardInterrupt
            self._first_reset_move = False
        self._move_reset_pose(self._inference_start_pose.copy(), label="inference start pose")
        while True:
            choice = self._read_idle_reset_choice("Enter/y: start from current TCP | q: exit")
            if choice in {"", "y", "yes"}:
                pose = np.asarray(self.robot.robot.get_tool_pose(), dtype=float)
                if pose.shape != (6,) or not np.isfinite(pose).all():
                    raise RuntimeError(f"invalid inference start pose: {pose}")
                self._inference_start_pose = pose.copy()
                return
            if choice in {"q", "quit"}:
                raise KeyboardInterrupt

    def _move_to_accepted_workspace_sample(self) -> None:
        if not getattr(self, "enable_random_reset", True):
            self._prepare_fixed_inference_start()
            return
        terminal = getattr(self, "inference_terminal", None)
        if self._reference_pose is None:
            measured = np.asarray(self.robot.robot.get_tool_pose(), dtype=float)
            if measured.shape != (6,) or not np.isfinite(measured).all():
                raise RuntimeError(f"invalid initial TCP reference pose: {measured}")
            choice = input(
                "Use current TCP pose as fixed Z insertion p1 "
                f"{self._format_pose(measured)}? Press Enter/y to accept, q to stop: "
            ).strip().lower()
            if choice not in {"", "y", "yes"}:
                raise KeyboardInterrupt
            self._reference_pose = measured.copy()
        if self._reset_orientation is None:
            measured = self._reference_pose
            if measured.shape != (6,) or not np.isfinite(measured).all():
                raise RuntimeError(f"invalid initial TCP pose: {measured}")
            self._reset_orientation = np.asarray(
                measured[3:6] if self.use_reference_pose else SAMPLE_ROTATION_VECTOR,
                dtype=float,
            ).copy()
            if terminal is None:
                print(
                    ("Using reference reset Z/orientation: " if self.use_reference_pose else
                     "Using fixed collection reset orientation: ") +
                    (f"Z={measured[2]:.6f} m; " if self.use_reference_pose else "") +
                    f"{self._format_pose(self._reset_orientation)}",
                    flush=True,
                )
        while True:
            at_reference = False
            target = sample_z_insertion_workspace_pose(
                self._reference_pose, self.workspace_min, self.workspace_max,
                xy_half_range=self.reset_xy_half_range_m,
                use_reference_pose=self.use_reference_pose,
            )
            label = "sampled workspace reset pose"
            if self._first_reset_move:
                prompt = (
                    "First sampled reset target is "
                    f"{self._format_pose(target)}. Press Enter/y to move, q to stop: "
                )
                if terminal is not None:
                    prompt = terminal.prompt(
                        f"First target: {terminal.pose(target)}\n  Enter/y: move | q: exit"
                    )
                prompt += f" XY +/-{self.reset_xy_half_range_m * 100:g} cm | =/-: adjust next reset range"
                if self._initial_reference_pose is not None:
                    prompt += " | 4: first training reference"
                if self.save_snapshot is not None:
                    prompt += " | p: save episode snapshot"
                revision = getattr(self, "_reference_xy_revision", 0)
                choice = self._read_idle_reset_choice(prompt)
                if getattr(self, "_reference_xy_revision", 0) != revision:
                    # The first target was sampled before the idle jog. Keep
                    # the adjusted XY instead of moving back to the old target.
                    target[:2] = self._reference_pose[:2]
                if choice == "p" and self.save_snapshot is not None:
                    self.save_snapshot()
                    continue
                if choice in {"=", "-"}:
                    self._adjust_reset_xy_range(choice)
                    continue
                if choice == "4" and self._initial_reference_pose is not None:
                    target = self._initial_reference_pose.copy()
                    label = "first training reference pose"
                    at_reference = True
                elif choice not in {"", "y", "yes"}:
                    if choice in {"q", "quit"}:
                        raise KeyboardInterrupt
                    print("First automatic reset not authorized; stopping.", flush=True)
                    raise KeyboardInterrupt
                self._first_reset_move = False
            self._move_reset_pose(target, label=label)
            while True:
                reference_hint = ("4=first training reference, "
                                  if self._initial_reference_pose is not None else "")
                location = "First training reference" if at_reference else "Sample point"
                prompt = (f"{location} OK? [Enter/y]=start episode, r=move to a new sample, "
                          f"{reference_hint}q=stop: ")
                if terminal is not None:
                    prompt = terminal.prompt(
                        f"{location} ready | Enter/y: start | r: new sample | {reference_hint}q: exit"
                    )
                prompt += f" XY +/-{self.reset_xy_half_range_m * 100:g} cm | =/-: adjust next reset range: "
                if self.save_snapshot is not None:
                    prompt += " p: save episode snapshot: "
                choice = self._read_idle_reset_choice(prompt)
                if choice == "p" and self.save_snapshot is not None:
                    self.save_snapshot()
                    continue
                if choice in {"=", "-"}:
                    self._adjust_reset_xy_range(choice)
                    continue
                if choice in {"", "y", "yes"}:
                    return
                if choice == "4" and self._initial_reference_pose is not None:
                    self._move_reset_pose(self._initial_reference_pose.copy(),
                                          label="first training reference pose")
                    at_reference = True
                    continue
                if choice in {"r", "retry", "n", "no"}:
                    break
                if choice in {"q", "quit"}:
                    raise KeyboardInterrupt
                print(f"Please use Enter/y, r, {reference_hint}or q.", flush=True)

    def _adjust_reset_xy_range(self, key: str) -> None:
        steps = round(self.reset_xy_half_range_m / RESET_XY_RANGE_STEP_M)
        steps = max(0, min(round(RESET_XY_MAX_HALF_RANGE_M / RESET_XY_RANGE_STEP_M),
                           steps + (1 if key == "=" else -1)))
        self.reset_xy_half_range_m = steps * RESET_XY_RANGE_STEP_M
        message = (f"Next reset: XY +/-{self.reset_xy_half_range_m * 100:g} cm "
                   f"around initial reference {self._reference_pose[:2].tolist()}; "
                   "current pose unchanged. Press r to sample now.")
        if self.inference_terminal is not None:
            self.inference_terminal.write(message)
        else:
            print(message, flush=True)

    def _prompt_outcome_after_timeout(self) -> str:
        while True:
            choice = input(
                "Episode maximum time elapsed. Confirm outcome: " +
                ("s=success, f=failure, d=delete episode, q=stop training: "
                 if getattr(self, "review_episode_replay", False) else "s=success, f=failure, q=stop training: ")
            ).strip().lower()
            if choice in {"s", "f", "q"} or (choice == "d" and getattr(self, "review_episode_replay", False)):
                return choice
            print("Please enter s, f, or q.", flush=True)

    def reset(self, *, episode_id: int, warmup: bool) -> dict[str, torch.Tensor]:
        self._episode_warmup = warmup
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        self._close_monitor()
        self._control_mode = "policy"
        downward_bias = getattr(self, "_downward_bias", None)
        if downward_bias is not None:
            downward_bias.prepare()
        bias = getattr(self, "_exploration_bias", None)
        if bias is not None:
            bias.prepare(warmup=warmup)
        phase = "ACT WARMUP" if warmup else "RLT ACTOR"
        terminal = getattr(self, "inference_terminal", None)
        if terminal is not None:
            terminal.prepare_episode(episode_id)
        else:
            print(f"\nPreparing Episode {episode_id} [{phase}] with automatic reset.", flush=True)
        self._move_to_accepted_workspace_sample()
        if bias is not None:
            bias.begin()
        self._needs_automatic_reset = False
        self.robot.resync_command_pose()
        self._episode_sampled_tcp_command_sum.fill(0.0)
        self._episode_reference_tcp_command_sum.fill(0.0)
        self._episode_mean_tcp_command_sum.fill(0.0)
        self._episode_tcp_target_comparison_steps = 0
        self._episode_human_steps = 0
        batch = self._processed_observation()
        if self.enable_human_intervention:
            from act_rlt.spacemouse_input import SpaceMouseHumanInputMonitor

            if self._spacemouse is None:
                raise RuntimeError('Human intervention requires a SpaceMouse input source')
            self._monitor = SpaceMouseHumanInputMonitor(
                self.teleop_input_backend, self.teleop_keyboard, self._spacemouse,
            )
            print(
                "Episode active: Space=take over/return at 4-step boundary; "
                "SpaceMouse=human XYZ/rotation (release before returning); "
                "S/F=finish and keep; D=finish and delete episode replay; Esc=stop training",
                flush=True,
            )
        else:
            self._monitor = OutcomeMonitor()
            if terminal is not None:
                terminal.active()
            else:
                print("Episode active: s/f=finish and keep; d=finish and delete episode replay; q=stop training", flush=True)
        if downward_bias is not None:
            print(f"B: enable -Z {abs(DOWNWARD_BIAS_STEP_M) * 1000:g} mm per remaining policy step (warmup/rollout).", flush=True)
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

        # This includes random noise and any directional exploration bias after
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
        *, warmup: bool, remaining_steps: int | None, deterministic: bool = False,
    ) -> ChunkExecution:
        """Consume one result; a None budget collects until the episode ends."""
        from act_rlt.stage2_runtime import FixedChunkRuntime

        self._replay_policy = policy
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
        if getattr(self, "inference_terminal", None) is not None:
            result.info["episode_elapsed_s"] = time.monotonic() - self._episode_start
        self._record_episode_action_comparison(result)
        bias = getattr(self, "_exploration_bias", None)
        if bias is not None:
            bias.record(result)
        downward_bias = getattr(self, "_downward_bias", None)
        if downward_bias is not None:
            downward_bias.record(result)
        if result.info["collector_paused"]:
            self._runtime.close()
            self._runtime = None

        if result.info.get("timed_out"):
            self._close_monitor()
            terminal = getattr(self, "inference_terminal", None)
            if terminal is None:
                key = self._prompt_outcome_after_timeout()
            else:
                terminal.write("  TIME LIMIT | motion stopped; waiting for your outcome")
                while True:
                    key = input(terminal.prompt("s: success | f: failure | q: exit")).strip().lower()
                    if key in {"s", "f", "q"}:
                        break
                    terminal.write("  Choose s, f, or q.")
            if key == "s":
                result.reward_seq[result.actual_steps - 1] = 1.0
                result.info["success"] = True
                result.terminated = True
            elif key == "f" or (key == "d" and getattr(self, "review_episode_replay", False)):
                result.terminated = True
                result.info["discard_episode_replay"] = key == "d"
            else:
                result.truncated = result.stop_requested = True
        if result.done:
            self._close_monitor()
            if not result.stop_requested and not result.info["workspace_violation"]:
                self._retreat_after_outcome()
            if not result.stop_requested:
                self._needs_automatic_reset = True
        return result

    def encode_replay_observation(self, observation: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a saved CPU frame only after collection/motion has stopped."""
        if self._runtime is not None:
            raise RuntimeError("overlap encoding requires a stopped collector")
        frame = self._observation_frame(observation)
        with torch.inference_mode():
            state, reference, _ = self._replay_policy.encode_and_reference(self.pre(frame))
        return state[0].detach().float().cpu(), reference[0].detach().float().cpu()

    def _normalize_executed_action(self, physical_full: torch.Tensor) -> torch.Tensor:
        """Map one bounded physical ACT action back to its saved model space."""
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
            or full_reference.shape[2] not in (self.config.action_dim, self.config.action_dim + 1)
        ):
            raise RuntimeError(f"unexpected ACT reference shape: {tuple(full_reference.shape)}")

        executed = torch.zeros(self.config.chunk_length, self.config.action_dim)
        rewards = torch.zeros(self.config.chunk_length)
        actual_steps = 0
        done = terminated = truncated = stop_requested = False
        success = False
        discard_episode_replay = False
        pending_outcome = None
        workspace_violation = False
        workspace_error: str | None = None
        last_batch = None
        clipping_count = 0
        bias = getattr(self, "_exploration_bias", None)
        bias_step = np.zeros(2) if bias is None else bias.step_m.copy()
        bias_yaw_step = 0.0 if bias is None else bias.step_yaw_rad
        bias_applied = np.zeros(2)
        bias_yaw_applied = 0.0
        bias_requested = np.zeros(2)
        bias_yaw_requested = 0.0
        downward_bias = getattr(self, "_downward_bias", None)
        downward_requested_z = 0.0
        downward_applied_xyz = np.zeros(3)
        downward_steps = 0

        for index in range(self.config.chunk_length):
            tick = time.monotonic()
            warmup = getattr(self, "_episode_warmup", False)
            consume = getattr(self._monitor, "consume_downward_bias", None)
            if consume is not None and consume():
                if downward_bias is not None:
                    downward_bias.enable()
            bias_step = np.zeros(2) if bias is None or warmup else bias.step_m.copy()
            bias_yaw_step = 0.0 if bias is None or warmup else bias.step_yaw_rad
            normalized_full = full_reference[:, index, :].clone()
            normalized_full[:, : self.config.action_dim] = action_chunk[:, index, :]
            physical_full = self.post(normalized_full).detach().cpu().reshape(-1)
            if physical_full.numel() != full_reference.shape[2]:
                raise RuntimeError(
                    f"unexpected Stage-2 physical action shape: {tuple(physical_full.shape)}"
                )
            unbiased_command = bounded_action(physical_full.numpy(), self.max_step_m, self.max_step_rad)
            physical_full[:2] += physical_full.new_tensor(bias_step)
            physical_full[5] += bias_yaw_step
            baseline_command = bounded_action(physical_full.numpy(), self.max_step_m, self.max_step_rad)
            downward_step = 0.0
            if downward_bias is not None:
                downward_step = downward_bias.step_m
                physical_full[2] += downward_step
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
            bias_requested += bias_step
            bias_yaw_requested += bias_yaw_step
            bias_applied += np.asarray(list(baseline_command.values()))[:2] - np.asarray(list(unbiased_command.values()))[:2]
            bias_yaw_applied += baseline_command["drz"] - unbiased_command["drz"]
            if downward_step:
                downward_requested_z += downward_step
                downward_applied_xyz += (np.asarray(list(command.values()))[:3]
                                         - np.asarray(list(baseline_command.values()))[:3])
                downward_steps += 1
            actual_steps += 1

            time.sleep(max(0.0, self.period_s - (time.monotonic() - tick)))
            last_batch = self._processed_observation()

            key = self._monitor.poll() if self._monitor is not None else None
            if (key in {"s", "f"} or (key == "d" and getattr(self, "review_episode_replay", False))) and pending_outcome is None:
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
                elif key == "f" or (key == "d" and getattr(self, "review_episode_replay", False)):
                    terminated = done = True
                    discard_episode_replay = key == "d"
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
            discard_episode_replay = pending_outcome == "d"
            terminated = done = True
            self._close_monitor()
            self._retreat_after_outcome()
            self._needs_automatic_reset = True
        if last_batch is None:
            raise RuntimeError("action chunk executed zero steps without a next observation")
        result = ChunkExecution(
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
                "discard_episode_replay": discard_episode_replay,
                "safety_clip_steps": clipping_count,
                "workspace_violation": workspace_violation,
                "workspace_error": workspace_error,
                "exploration_bias_requested_xy_m": bias_requested.tolist(),
                "exploration_bias_applied_xy_m": bias_applied.tolist(),
                "exploration_bias_requested_yaw_rad": bias_yaw_requested,
                "exploration_bias_applied_yaw_rad": bias_yaw_applied,
                "downward_bias_requested_z_m": downward_requested_z,
                "downward_bias_applied_xyz_m": downward_applied_xyz.tolist(),
                "downward_bias_steps": downward_steps,
            },
        )
        if bias is not None:
            bias.record(result)
        if downward_bias is not None:
            downward_bias.record(result)
        return result

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
    parser.add_argument("--output", type=Path, default=Path("outputs/ethernet_rlt/act_rlt_001_stage2"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replay-from", type=Path, help="reuse compatible replay with fresh Actor/Critic; new output required")
    parser.add_argument(
        "--replay-stride", type=int, choices=(2, 4), default=4,
        help="2: cache intermediate frames and add overlapping replay after each episode; 4: original replay",
    )
    parser.add_argument("--prepare-only", action="store_true", help="initialize from replay and save v3 checkpoint without connecting robot")
    parser.add_argument("--bc-init-steps", type=int, default=1000)
    parser.add_argument("--critic-init-steps", type=int, default=1000)
    parser.add_argument(
        "--three-chunk-td", action="store_true",
        help="use 3-chunk TD (up to 12 control steps); default: 1-chunk TD",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument(
        "--use-reference-pose", action="store_true",
        help="use captured/restored reference Z and orientation for resets; retain the XY range rules",
    )
    parser.add_argument("--enable-human-intervention", action="store_true")
    parser.add_argument(
        "--teleop-input-backend",
        choices=("auto", "evdev", "pynput"),
        default="auto",
        help="auto uses pynput in an X11 session, or evdev on a headless host",
    )
    parser.add_argument("--teleop-keyboard", help="evdev input device, e.g. /dev/input/event4")
    parser.add_argument("--teleop-speed-m-s", type=float, default=0.0135,
                        help="SpaceMouse translation speed per axis at full deflection (m/s)")
    parser.add_argument("--teleop-angular-speed-rad-s", type=float, default=0.09,
                        help="SpaceMouse roll/pitch speed; yaw uses 1.2 times this speed")
    parser.add_argument("--spacemouse-deadzone", type=float, default=0.05)

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
    parser.add_argument("--fps", type=float, default=30.0,
                        help="control rate in Hz (default: 30)")
    parser.add_argument(
        "--camera-boundary-mode", choices=("timestamp", "legacy"), default="timestamp",
        help="timestamp: align image/state when available; legacy: use ordinary observations",
    )
    parser.add_argument(
        "--observation-selection", choices=("latest", "boundary"), default="latest",
        help="latest: use newer cached observations like ACT infer; boundary: require post-command observations",
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
    parser.add_argument(
        "--warmup-steps", type=int, default=4_000,
        help="minimum ACT collection steps; finish the current episode before initialization",
    )
    parser.add_argument(
        "--total-env-steps",
        type=int,
        default=10_000,
        help="cumulative online-step target after warmup; use 20000 to add 10000 after 10000",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fusion-dim", type=int, default=128,
                        help="projection width for each RL token, auxiliary-state and action branch (default: 128)")
    parser.add_argument("--utd-ratio", type=int, default=5)
    parser.add_argument("--beta", type=float, default=2.0,
                        help="BC regularization coefficient in -Q + beta * BC (default: 2.0)")
    parser.add_argument("--exploration-sigma", type=float, default=0.3,
                        help="fixed normalized action standard deviation during collection (default: 0.3)")
    parser.add_argument(
        "--target-q-clipping", action=argparse.BooleanOptionalAction, default=True,
        help="clip bootstrap next-Q to [0,1] (default: enabled; --no-target-q-clipping disables it)",
    )
    parser.add_argument("--save-every-env-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.teleop_input_backend == "auto":
        args.teleop_input_backend = (
            "evdev" if args.teleop_keyboard or not os.environ.get("DISPLAY") else "pynput"
        )
    if args.teleop_keyboard and args.teleop_input_backend != "evdev":
        parser.error("--teleop-keyboard applies only to --teleop-input-backend evdev")
    if args.enable_human_intervention:
        from act_rlt.human_input import validate_teleop_speed
        from evo_franka.spacemouse_bridge import YAW_SPEED_MULTIPLIER

        try:
            validate_teleop_speed(args.teleop_speed_m_s, args.fps, args.max_step_m)
            # SpaceMouse allows three fully deflected axes simultaneously.
            validate_teleop_speed(np.sqrt(3) * args.teleop_speed_m_s, args.fps, args.max_step_m)
            if (not np.isfinite(args.teleop_angular_speed_rad_s)
                    or args.teleop_angular_speed_rad_s <= 0
                    or (np.isfinite(args.fps) and args.fps > 0 and
                        np.sqrt(2 + YAW_SPEED_MULTIPLIER ** 2)
                        * args.teleop_angular_speed_rad_s / args.fps > args.max_step_rad)):
                raise ValueError('SpaceMouse angular speed must be positive and fit --max-step-rad per tick')
            if not np.isfinite(args.spacemouse_deadzone) or not 0 <= args.spacemouse_deadzone < 1:
                raise ValueError('--spacemouse-deadzone must be in [0, 1)')
        except ValueError as error:
            parser.error(str(error))
    if args.resume and args.replay_from:
        parser.error("--resume and --replay-from are mutually exclusive")
    if args.prepare_only and not (args.replay_from or args.resume):
        parser.error("--prepare-only requires --replay-from or --resume")
    if not args.dry_run and not args.prepare_only and not args.allow_motion:
        parser.error("real Stage-2 collection requires --allow-motion")
    if args.workspace_min is not None or args.workspace_max is not None:
        parser.error("training workspace is disabled; omit --workspace-min/--workspace-max")
    if args.sample_min is not None or args.sample_max is not None:
        parser.error("training uses random XY reset with fixed Z=0.402 m; omit --sample-min/--sample-max")
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
    *, startup_reference: dict | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "config": config.to_dict(),
        "startup_reference": startup_reference,
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
    *, startup_metadata: dict | None = None,
) -> OnlineStage2Metrics:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if startup_metadata is not None:
        startup_metadata["startup_reference"] = saved.get("startup_reference")
    if saved["learner"].get("learner_version") != 3:
        raise ValueError(
            "old Stage-2 network cannot resume with this Actor/Critic architecture; "
            "use --replay-from with a new output directory"
        )
    saved_config = strip_removed_force_feedback_config(saved["config"])
    saved_config.setdefault("input_fusion", "balanced")
    if saved_config["input_fusion"] != config.input_fusion:
        raise ValueError(
            f"Stage-2 input fusion changed ({saved_config['input_fusion']} -> {config.input_fusion}); "
            "use --replay-from with a new output directory to reuse replay and initialize fresh networks"
        )
    current_config = config.to_dict()
    # Compare model identities after relocation, rather than saved path spelling.
    from evo_rlt.artifact_paths import resolve_artifact_path

    for key in ("stage1_checkpoint", "act_checkpoint"):
        saved_config[key] = str(resolve_artifact_path(saved_config[key]))
        current_config[key] = str(resolve_artifact_path(current_config[key]))
    # Checkpoints created before the real-time temporal collector existed have
    # no value for this inference-only setting. Adopt the requested runtime
    # value without invalidating their learner/replay state.
    saved_config.setdefault(
        "temporal_ensemble_coeff", current_config["temporal_ensemble_coeff"]
    )
    saved_config.setdefault("target_q_clipping", True)
    saved_config.setdefault("replay_stride", saved_config.get("chunk_length", 4))
    saved_config.setdefault("td_chunks", 1)
    # These settings affect future updates/collection without changing the
    # network, optimizer state, or the semantics of existing replay entries.
    schedule_keys = {"warmup_steps", "total_env_steps", "device", "beta", "actor_fixed_std", "target_q_clipping", "replay_stride", "td_chunks", "actor_update_interval"}
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
    strip_removed_force_feedback_config(saved["config"])
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


def append_metrics(writer: Stage2MetricsWriter, metrics: OnlineStage2Metrics,
                   execution: ChunkExecution) -> None:
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
        "replay_overlap_chunks": metrics.replay_overlap_chunks,
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
    writer.append(record)


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
    if record.get("replay_overlap_added"):
        parts.append(f"replay+{record['replay_overlap_added']}")
    if record.get("episode_replay_discarded"):
        parts.append("REPLAY=DELETED")
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

    from lerobot.configs.policies import PreTrainedConfig

    act_config = PreTrainedConfig.from_pretrained(act_path, local_files_only=True)
    state_mode = checkpoint_state_mode(act_config)
    config = ACTStage2Config(
        stage1_checkpoint=str(stage1_path),
        act_checkpoint=str(act_path),
        proprio_dim={"none": 0, "tcp_z": 1, "wrench": 6, "joints": 7}[state_mode],
        device=args.device,
        warmup_steps=args.warmup_steps,
        total_env_steps=args.total_env_steps,
        batch_size=args.batch_size,
        fusion_dim=args.fusion_dim,
        utd_ratio=args.utd_ratio,
        beta=args.beta,
        actor_fixed_std=args.exploration_sigma,
        target_q_clipping=args.target_q_clipping,
        seed=args.seed,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        bc_init_steps=args.bc_init_steps,
        critic_init_steps=args.critic_init_steps,
        max_step_m=args.max_step_m,
        max_step_rad=args.max_step_rad,
        replay_stride=args.replay_stride,
        td_chunks=3 if args.three_chunk_td else 1,
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
    startup_metadata = {"startup_reference": None}
    if args.resume:
        metrics = load_checkpoint(output / "checkpoints/latest.pt", config, learner, replay,
                                  startup_metadata=startup_metadata)
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
        save_checkpoint(output, config, learner, replay, metrics,
                        startup_reference=startup_metadata["startup_reference"])
        (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
        print(f"Prepared v3 checkpoint: {output / 'checkpoints/latest.pt'}", flush=True)
        return 0

    from act_rlt.human_input import validate_human_input

    validate_human_input(args.teleop_input_backend, args.teleop_keyboard)

    from evo_rlt.adapters.lerobot.franka_robot import FrankaRobot, FrankaRobotConfig

    robot = FrankaRobot(
        FrankaRobotConfig(
            robot_ip=args.robot_ip,
            **robot_camera_kwargs(camera_shapes, args),
            include_gripper_action=policy.act_action_dim == 7,
            enable_fci_keepalive=False,
            workspace_min_xyz=None,
            workspace_max_xyz=None,
        )
    )
    if args.camera_boundary_mode == "timestamp":
        robot.enable_timestamped_observations()
    env = None
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")
    last_save_step = metrics.env_steps
    metrics_writer = None
    spacemouse = None

    def current_reference() -> dict | None:
        reference = startup_metadata["startup_reference"]
        if reference is not None and env is not None:
            reference["reset_xy_half_range_m"] = env.reset_xy_half_range_m
            if env._reference_pose is not None:
                reference["reference_pose"] = env._reference_pose.tolist()
            temporary = output / "startup_reference.json.tmp"
            temporary.write_text(json.dumps(reference, indent=2) + "\n")
            temporary.replace(output / "startup_reference.json")
        return reference

    def save_episode_snapshot() -> None:
        if metrics.episodes == 0:
            print("No completed episode to snapshot yet.", flush=True)
            return
        snapshot_output = output.with_name(f"{output.name}_step{metrics.env_steps}")
        if snapshot_output.exists():
            print(f"Snapshot already exists; kept unchanged: {snapshot_output}", flush=True)
            return
        reference = current_reference()
        try:
            # Publish a complete sibling output only after all files are written.
            with tempfile.TemporaryDirectory(prefix=f".{snapshot_output.name}_", dir=output.parent) as temporary:
                staged = Path(temporary)
                save_checkpoint(staged, config, learner, replay, metrics,
                                startup_reference=reference)
                (staged / "config.json").write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")
                if reference is not None:
                    (staged / "startup_reference.json").write_text(json.dumps(reference, indent=2) + "\n")
                (staged / "snapshot.json").write_text(json.dumps({
                    "source_output": str(output), "env_steps": metrics.env_steps,
                    "episodes": metrics.episodes, "created_at_unix_s": time.time(),
                    "actor_initialized": bool(learner.initialized),
                    "reset_xy_half_range_m": env.reset_xy_half_range_m if env is not None else None,
                }, indent=2) + "\n")
                staged.rename(snapshot_output)
        except Exception as exc:
            print(f"Snapshot save failed; training can continue: {exc}", flush=True)
            return
        print(f"Episode snapshot saved: {snapshot_output}\n"
              "Continue training here; resume or infer can load the snapshot output separately.", flush=True)

    def on_chunk(current: OnlineStage2Metrics, execution: ChunkExecution) -> None:
        nonlocal last_save_step
        assert metrics_writer is not None
        append_metrics(metrics_writer, current, execution)
        if execution.done or current.env_steps - last_save_step >= args.save_every_env_steps:
            save_checkpoint(output, config, learner, replay, current,
                            startup_reference=current_reference())
            last_save_step = current.env_steps

    try:
        if args.enable_human_intervention:
            from act_rlt.spacemouse_input import SpaceMouseInput

            # Verify live input before connecting or moving the robot. Keep the
            # input process across episodes; it never creates a robot client.
            spacemouse = SpaceMouseInput(
                args.teleop_speed_m_s, args.teleop_angular_speed_rad_s, args.spacemouse_deadzone,
            )
        metrics_writer = Stage2MetricsWriter(
            output / "metrics.jsonl", format_terminal=_format_terminal_metrics,
        )
        robot.connect()
        saved_reference = startup_metadata["startup_reference"]
        if saved_reference is not None:
            reference_pose, workspace_min, workspace_max = restore_training_reference(
                saved_reference, enforce_workspace=False, use_current_reference=True,
            )
            print(f"Restored training reference: {reference_pose.tolist()}", flush=True)
        else:
            reference_pose, workspace_min, workspace_max = capture_centered_workspace(robot, enforce_workspace=False)
        robot.config.workspace_min_xyz = None
        robot.config.workspace_max_xyz = None
        startup_reference = {
            "version": 1,
            "mode": "z_insertion",
            "frame": "base",
            "position_unit": "m",
            "orientation": "rotation_vector_rad",
            "captured_at_unix_s": time.time(),
            "reference_pose": reference_pose.tolist(),
            "workspace_min_xyz": None,
            "workspace_max_xyz": None,
            "workspace_enabled": False,
            "use_reference_pose": args.use_reference_pose,
            "reset_z_m": float(reference_pose[2]) if args.use_reference_pose else SAMPLE_Z_M,
            "reset_rotation_vector": np.asarray(
                reference_pose[3:6] if args.use_reference_pose else SAMPLE_ROTATION_VECTOR
            ).tolist(),
            "randomize_reset_xy": True,
            "reset_xy_half_range_m": (saved_reference.get("reset_xy_half_range_m", Z_SAMPLE_XY_HALF_RANGE_M)
                                      if saved_reference is not None else Z_SAMPLE_XY_HALF_RANGE_M),
        }
        previous_reference = startup_metadata["startup_reference"]
        startup_reference["initial_reference"] = (
            previous_reference.get("initial_reference", previous_reference)
            if previous_reference is not None else dict(startup_reference)
        )
        startup_metadata["startup_reference"] = startup_reference
        reference_file = output / "startup_reference.json"
        temporary_reference = output / "startup_reference.json.tmp"
        temporary_reference.write_text(json.dumps(startup_reference, indent=2) + "\n")
        temporary_reference.replace(reference_file)
        save_checkpoint(output, config, learner, replay, metrics,
                        startup_reference=startup_reference)
        print(f"Saved training reference: {reference_file}; workspace disabled.", flush=True)
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
            reference_pose=reference_pose,
            use_reference_pose=args.use_reference_pose,
            reset_xy_half_range_m=(startup_metadata["startup_reference"].get("reset_xy_half_range_m", Z_SAMPLE_XY_HALF_RANGE_M)
                                   if startup_metadata["startup_reference"] is not None else Z_SAMPLE_XY_HALF_RANGE_M),
            save_snapshot=save_episode_snapshot,
            enable_human_intervention=args.enable_human_intervention,
            review_episode_replay=True,
            teleop_input_backend=args.teleop_input_backend,
            teleop_keyboard=args.teleop_keyboard,
            teleop_speed_m_s=args.teleop_speed_m_s,
            spacemouse=spacemouse,
            observation_selection=args.observation_selection,
            enable_exploration_bias=True,
            enforce_workspace=False,
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
                try:
                    if spacemouse is not None:
                        spacemouse.close()
                finally:
                    try:
                        save_checkpoint(output, config, learner, replay, metrics,
                                        startup_reference=current_reference())
                    finally:
                        if metrics_writer is not None:
                            metrics_writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
