#!/usr/bin/env python3
"""Record FR3 sample-space episodes as a local LeRobot dataset.

The operator teaches a TCP reference position and orientation. Each saved
episode starts at a uniformly sampled pose with a fixed orientation, returns
to the full taught reference pose, and then moves along the robot-base -Z axis
while keeping the taught orientation. Repositioning is deliberately performed
outside the episode, so the dataset contains only the task trajectory.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep

from evo_rlt.adapters.lerobot.franka_robot import FrankaRobot, FrankaRobotConfig
from evo_rlt.adapters.lerobot.record.annotations import COLLECTOR_HUMAN, PHASE_PREFIX
from evo_rlt.adapters.lerobot.record.backend import (
    _add_collector_policy_id_feature,
    _ensure_human_inloop_compatible_features,
    _write_schema_metadata,
)
from evo_rlt.adapters.lerobot.record.loop import (
    _build_manual_demo_action,
    manual_demo_health_tick,
    new_health_state,
)
from act_rlt.data_collection.sampling import (
    DEFAULT_INSERT_MINUS_Z_M,
    SAMPLE_EULER_XYZ_DEG,
    insertion_pose_minus_z,
    sample_z_insertion_pose,
)


COMMAND_ACTION_KEY = "complementary_info.command_action"
COMMAND_ACTION_NAMES = [
    "target_x",
    "target_y",
    "target_z",
    "target_rx",
    "target_ry",
    "target_rz",
    "speed_m_s",
]
SUPPORTED_FPS = (15, 30)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="LeRobot dataset repo/name")
    parser.add_argument("--root", type=Path, required=True, help="Local dataset directory")
    parser.add_argument(
        "--fps",
        type=int,
        choices=SUPPORTED_FPS,
        default=15,
        help="Recording frequency in Hz (default: 15).",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of newly saved episodes")
    parser.add_argument("--episode-time", type=float, default=60.0, help="Per-attempt timeout")
    parser.add_argument(
        "--task",
        default=None,
    )
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--wrist-camera", default="349622072679")
    parser.add_argument("--front-camera", default="233522075778")
    cameras = parser.add_mutually_exclusive_group()
    cameras.add_argument(
        "--wrist_only", "--wrist-only", dest="wrist_only", action="store_true",
        help="Record only wrist at 1280x720; do not open the front camera.",
    )
    cameras.add_argument(
        "--front_only", "--front-only", dest="front_only", action="store_true",
        help="Record only front at 640x480; do not open the wrist camera.",
    )
    parser.add_argument("--move-speed", type=float, default=0.02, help="TCP speed in m/s")
    parser.add_argument(
        "--insertion-speed",
        type=float,
        default=0.01,
        help="Reference-to-final TCP speed in m/s",
    )
    parser.add_argument(
        "--pre-episode-sleep",
        type=float,
        default=1.0,
        help="Pause after reaching a sample and before recording",
    )
    parser.add_argument(
        "--post-episode-sleep",
        type=float,
        default=1.0,
        help="Pause after saving an episode and before retreating",
    )
    parser.add_argument(
        "--insert-minus-z",
        type=float,
        default=DEFAULT_INSERT_MINUS_Z_M,
        help="Recorded -Z advance from the reference pose (default: 0.01 m).",
    )
    parser.add_argument("--abort-key", default="x", help="Single-key active-episode abort")
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic sampler seed")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--vcodec", default="h264")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "fps": args.fps,
        "episodes": args.episodes,
        "episode-time": args.episode_time,
        "move-speed": args.move_speed,
        "insertion-speed": args.insertion_speed,
        "pre-episode-sleep": args.pre_episode_sleep,
        "post-episode-sleep": args.post_episode_sleep,
        "insert-minus-z": args.insert_minus_z,
    }
    invalid = [
        name
        for name, value in positive.items()
        if not math.isfinite(float(value)) or value <= 0
    ]
    if invalid:
        raise ValueError(f"these arguments must be positive: {', '.join(invalid)}")
    if not (args.wrist_only or args.front_only) and args.wrist_camera == args.front_camera:
        raise ValueError("wrist and front camera serials must be different")
    if len(args.abort_key) != 1:
        raise ValueError("abort-key must be exactly one character")


def measured_pose(robot: FrankaRobot) -> list[float]:
    pose = [float(value) for value in robot.robot.get_tool_pose()]
    if len(pose) != 6 or not np.isfinite(pose).all():
        raise RuntimeError(f"invalid measured TCP pose: {pose}")
    return pose


def format_pose(pose: list[float]) -> str:
    return " ".join(f"{value:+.6f}" for value in pose)


class EpisodeAborted(RuntimeError):
    """Raised after the operator presses the active-episode abort key."""


class EpisodeAbortMonitor:
    """Poll one abort key without requiring ENTER; Ctrl+C remains functional."""

    def __init__(self, key: str) -> None:
        self.key = key.lower()
        self.fd: int | None = None
        self.saved_terminal: list[Any] | None = None

    def __enter__(self) -> EpisodeAbortMonitor:
        if not sys.stdin.isatty():
            raise RuntimeError("episode abort control requires an interactive terminal")
        self.fd = sys.stdin.fileno()
        self.saved_terminal = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        print(f"Episode active: press '{self.key}' to STOP and DISCARD it.", flush=True)
        return self

    def pressed(self) -> bool:
        if self.fd is None:
            return False
        while select.select([sys.stdin], [], [], 0.0)[0]:
            char = os.read(self.fd, 1).decode(errors="ignore")
            if char.lower() == self.key:
                return True
        return False

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.fd is not None and self.saved_terminal is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved_terminal)


def robot_is_executable(robot: FrankaRobot) -> bool:
    try:
        mode = str(robot.robot.robot_mode_nowait())
    except Exception as exc:
        print(f"Cannot verify robot mode: {exc}", flush=True)
        return False
    if "Guiding" in mode or "UserStopped" in mode:
        print(
            f"Robot is {mode}. Switch Franka Desk to Execution and release user-stop.",
            flush=True,
        )
        return False
    return True


def nonrecorded_arm_move(
    robot: FrankaRobot,
    target: list[float],
    *,
    speed: float,
    label: str,
) -> bool:
    """Move outside an episode; recoverable Desk refusals keep the recorder alive."""
    print(f"\nMoving outside episode to {label}: {format_pose(target)}", flush=True)
    robot._stop_keepalive()
    try:
        robot.robot.move_tool(target, speed=speed)
    except RuntimeError as exc:
        probe = robot.probe()
        if not probe.get("ok", False):
            raise RuntimeError(
                f"control server disconnected during non-recorded move: {probe.get('error')}"
            ) from exc
        try:
            mode = str(robot.robot.robot_mode_nowait())
        except Exception:
            mode = "unknown"
        print(f"Move not completed (robot_mode={mode}): {exc}", flush=True)
        print("Correct the robot state and retry; no frame was recorded.", flush=True)
        return False
    finally:
        robot.resync_command_pose()
        if robot.is_connected and robot.config.enable_fci_keepalive:
            robot._start_keepalive()
    print(f"Reached {label}.", flush=True)
    return True


def read_key(robot: FrankaRobot, message: str, *, accepted: set[str]) -> str:
    """Read a single key while periodically checking control-server health."""
    if not sys.stdin.isatty():
        raise RuntimeError("operator controls require an interactive terminal")
    fd = sys.stdin.fileno()
    saved_terminal = termios.tcgetattr(fd)
    print(message, end="", flush=True)
    try:
        tty.setcbreak(fd)
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 1.0)
            if not readable:
                probe = robot.probe()
                if not probe.get("ok", False):
                    raise RuntimeError(
                        "control server disconnected while waiting for input: "
                        f"{probe.get('error')}"
                    )
                continue
            key = os.read(fd, 1).decode(errors="ignore").lower()
            if key in {"\r", "\n"}:
                key = "enter"
            if key in accepted:
                print(flush=True)
                return key
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved_terminal)


def teach_reference(robot: FrankaRobot) -> list[float]:
    while True:
        key = read_key(
            robot,
            "\nGuide the arm to reference p0, then press ENTER to store it (Q quit): ",
            accepted={"enter", "q"},
        )
        if key == "q":
            raise KeyboardInterrupt
        return measured_pose(robot)


def read_key_during_delay(seconds: float) -> str | None:
    """Sleep for ``seconds`` while accepting optional non-episode hotkeys."""
    if not sys.stdin.isatty():
        precise_sleep(seconds)
        return None
    fd = sys.stdin.fileno()
    saved_terminal = termios.tcgetattr(fd)
    deadline = time.perf_counter() + seconds
    try:
        tty.setcbreak(fd)
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return None
            readable, _, _ = select.select([sys.stdin], [], [], remaining)
            if not readable:
                return None
            key = os.read(fd, 1).decode(errors="ignore").lower()
            if key in {"1", "2", "q"}:
                return key
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved_terminal)


def wait_for_resume(
    robot: FrankaRobot,
    *,
    reference: list[float],
    sample: list[float],
    move_speed: float,
) -> bool:
    """Pause automatic motion after an abort or an explicit numeric hotkey."""
    prompt = "\nAUTOMATION PAUSED. ENTER resume | 1 reference | 2 sample | Q quit: "
    while True:
        key = read_key(robot, prompt, accepted={"enter", "1", "2", "q"})
        if key == "q":
            return False
        if key == "1":
            nonrecorded_arm_move(robot, reference, speed=move_speed, label="REFERENCE p0")
            continue
        if key == "2":
            nonrecorded_arm_move(robot, sample, speed=move_speed, label="CURRENT SAMPLE")
            continue
        if not robot_is_executable(robot):
            continue
        return True


def handle_delay_hotkey(
    key: str | None,
    robot: FrankaRobot,
    *,
    reference: list[float],
    sample: list[float],
    move_speed: float,
) -> bool:
    """Handle an optional transition-window key; return False to quit."""
    if key is None:
        return True
    if key == "q":
        return False
    target, label = (reference, "REFERENCE p0") if key == "1" else (sample, "CURRENT SAMPLE")
    nonrecorded_arm_move(robot, target, speed=move_speed, label=label)
    return wait_for_resume(
        robot,
        reference=reference,
        sample=sample,
        move_speed=move_speed,
    )


def normalize_lerobot_scalar_buffer(dataset: LeRobotDataset) -> None:
    """Work around LeRobot 0.5.1 scalar encoding with NumPy 2.x.

    LeRobot validates a custom ``shape=(1,)`` feature as a one-element ndarray
    when a frame is added, but Hugging Face maps the same feature to a scalar at
    save time. NumPy 2.x no longer permits ``float(array([value]))``. Convert
    only buffered singleton features to NumPy scalars immediately before save;
    vector actions and images are untouched.
    """
    episode_buffer = dataset.writer.episode_buffer
    for key, feature in dataset.features.items():
        if tuple(feature["shape"]) != (1,):
            continue
        values = episode_buffer.get(key)
        if not isinstance(values, list) or not values:
            continue
        dtype = np.dtype(feature["dtype"])
        episode_buffer[key] = [np.asarray(value, dtype=dtype).reshape(-1)[0] for value in values]


class EpisodeRecorder:
    """One-frame-delayed LeRobot writer using measured TCP deltas as actions."""

    def __init__(
        self,
        *,
        robot: FrankaRobot,
        dataset: LeRobotDataset,
        observation_processor: Any,
        action_names: list[str],
        task: str,
        fps: int,
        episode_time: float,
    ) -> None:
        self.robot = robot
        self.dataset = dataset
        self.observation_processor = observation_processor
        self.action_names = action_names
        self.task = task
        self.fps = fps
        self.episode_time = episode_time
        self.health = new_health_state()
        self.pending: tuple[dict[str, Any], list[float], float, dict[str, float]] | None = None
        self.command_action = {name: 0.0 for name in COMMAND_ACTION_NAMES}
        self.started_at = 0.0
        self.last_sample_started_at: float | None = None
        self.abort_monitor: EpisodeAbortMonitor | None = None
        self.frames = 0

    def start(self) -> None:
        self.pending = None
        self.command_action = {name: 0.0 for name in COMMAND_ACTION_NAMES}
        self.started_at = time.perf_counter()
        self.last_sample_started_at = None
        self.frames = 0

    def set_command(self, target: list[float], speed: float) -> None:
        """Set the exact high-level move_tool command associated with future motion."""
        values = [*target, float(speed)]
        self.command_action = dict(zip(COMMAND_ACTION_NAMES, values, strict=True))
        # The pending observation is the state immediately before this command
        # was issued, so align the command with that frame's following motion.
        if self.pending is not None:
            observation_frame, pose, width, _ = self.pending
            self.pending = (observation_frame, pose, width, self.command_action.copy())

    def clear_command(self) -> None:
        """Mark the terminal frame as having no following motion command."""
        self.command_action = {name: 0.0 for name in COMMAND_ACTION_NAMES}
        if self.pending is not None:
            observation_frame, pose, width, _ = self.pending
            self.pending = (observation_frame, pose, width, self.command_action.copy())

    def check_deadline(self) -> None:
        if self.abort_monitor is not None and self.abort_monitor.pressed():
            raise EpisodeAborted("operator requested active-episode abort")
        if time.perf_counter() - self.started_at > self.episode_time:
            raise TimeoutError(f"episode exceeded {self.episode_time:g} seconds")

    def capture(self) -> None:
        self.check_deadline()
        if self.last_sample_started_at is not None:
            precise_sleep(
                max(self.last_sample_started_at + 1.0 / self.fps - time.perf_counter(), 0.0)
            )
        observation_started = time.perf_counter()
        self.last_sample_started_at = observation_started
        observation = self.robot.get_observation()
        self.check_deadline()
        self.health["last_obs_ms"] = (time.perf_counter() - observation_started) * 1000.0
        manual_demo_health_tick(
            robot=self.robot,
            health=self.health,
            recording=True,
            play_sounds=False,
        )
        if self.health["server_down"] or self.health["camera_down"]:
            raise RuntimeError(
                f"recording health failure: server_down={self.health['server_down']} "
                f"camera_down={self.health['camera_down']}"
            )

        pose = [
            float(observation[f"ee_{axis}"])
            for axis in ("x", "y", "z", "rx", "ry", "rz")
        ]
        gripper_width = float(observation["gripper_width"])
        observation_frame = build_dataset_frame(
            self.dataset.features,
            self.observation_processor(observation),
            prefix=OBS_STR,
        )
        if self.pending is not None:
            previous_frame, previous_pose, _, previous_command = self.pending
            action = _build_manual_demo_action(
                self.action_names,
                previous_pose,
                pose,
                gripper_width,
            )
            self._add_frame(previous_frame, action, previous_command)
        self.pending = (
            observation_frame,
            pose,
            gripper_width,
            self.command_action.copy(),
        )

    def _add_frame(
        self,
        observation_frame: dict[str, Any],
        actual_action: dict[str, float],
        command_action: dict[str, float],
    ) -> None:
        action_frame = build_dataset_frame(self.dataset.features, actual_action, prefix=ACTION)
        command_action_frame = build_dataset_frame(
            self.dataset.features,
            command_action,
            prefix=COMMAND_ACTION_KEY,
        )
        zero_action = {name: 0.0 for name in self.action_names}
        policy_action = build_dataset_frame(
            self.dataset.features,
            zero_action,
            prefix="complementary_info.policy_action",
        )
        frame = {
            **observation_frame,
            **action_frame,
            **command_action_frame,
            **policy_action,
            "task": self.task,
        }
        frame["complementary_info.is_intervention"] = np.array([0.0], dtype=np.float32)
        frame["complementary_info.state"] = np.array([0.0], dtype=np.float32)
        frame["complementary_info.phase"] = np.array([PHASE_PREFIX], dtype=np.float32)
        frame["complementary_info.collector_policy_id"] = np.array(
            [COLLECTOR_HUMAN], dtype=np.int64
        )
        self.dataset.add_frame(frame)
        self.frames += 1

    def finish(self) -> int:
        if self.pending is not None:
            observation_frame, pose, gripper_width, command_action = self.pending
            final_action = _build_manual_demo_action(
                self.action_names,
                pose,
                None,
                gripper_width,
            )
            self._add_frame(observation_frame, final_action, command_action)
            self.pending = None
        return self.frames


def run_recorded_move(
    recorder: EpisodeRecorder,
    target: list[float],
    *,
    speed: float,
    label: str,
) -> None:
    """Run a server-managed asynchronous trajectory while capturing frames."""
    robot = recorder.robot
    logging.info("Starting %s -> %s", label, format_pose(target))
    recorder.set_command(target, speed)
    robot._stop_keepalive()
    try:
        # Use the control server's tracked async trajectory path.  An async
        # ``move_tool`` used to run through the legacy helper: that helper
        # returned immediately after issuing franky's motion, marked the worker
        # idle, and never called finish_move()/join_motion().  The explicit
        # trajectory RPC stays busy until the server has reaped the motion.
        waypoint = [*target, float(speed), 0.0, 0.0]
        robot.robot.move_tool_traj([waypoint], is_async=True)
        while True:
            recorder.capture()
            if not robot.robot.is_running():
                result = robot.robot._wait_idle(min(0.2, 1.0 / recorder.fps))
                if result.get("success"):
                    break
                if str(result.get("error", "")).startswith("timeout after"):
                    continue
                raise RuntimeError(f"{label} failed: {result.get('error', 'unknown error')}")
        recorder.capture()
    except BaseException:
        try:
            robot.robot.stop_move()
        except Exception:
            logging.exception("failed to stop arm after %s error", label)
        raise
    finally:
        robot.resync_command_pose()
        if robot.is_connected and robot.config.enable_fci_keepalive:
            robot._start_keepalive()


def build_dataset_and_robot(args: argparse.Namespace):
    config = FrankaRobotConfig(
        robot_ip=args.robot_ip,
        camera_serial=args.wrist_camera,
        camera_width=1280 if args.wrist_only else 640,
        camera_height=720 if args.wrist_only else 480,
        enable_wrist_camera=not args.front_only,
        front_camera_serial=args.front_camera,
        front_camera_width=640,
        front_camera_height=480,
        enable_front_camera=not args.wrist_only,
        include_gripper_action=True,
    )
    robot = FrankaRobot(config)
    teleop_processor, _, observation_processor = make_default_processors()
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )
    action_names = list(features[ACTION]["names"] or robot.action_features)
    _ensure_human_inloop_compatible_features(features, action_feature_names=action_names)
    _add_collector_policy_id_feature(features)
    features[COMMAND_ACTION_KEY] = {
        "dtype": "float32",
        "shape": (len(COMMAND_ACTION_NAMES),),
        "names": COMMAND_ACTION_NAMES,
    }

    if args.resume:
        dataset = LeRobotDataset.resume(
            args.dataset,
            root=args.root,
            batch_encoding_size=1,
            vcodec=args.vcodec,
            image_writer_processes=0,
            image_writer_threads=8,
            streaming_encoding=True,
            encoder_queue_maxsize=30,
        )
        if dataset.fps != args.fps:
            raise ValueError(
                f"cannot resume at a different fps: dataset={dataset.fps}, requested={args.fps}"
            )
    else:
        dataset = LeRobotDataset.create(
            args.dataset,
            args.fps,
            root=args.root,
            robot_type=robot.name,
            features=features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=8,
            batch_encoding_size=1,
            vcodec=args.vcodec,
            streaming_encoding=True,
            encoder_queue_maxsize=30,
        )
    if COMMAND_ACTION_KEY not in dataset.features:
        raise ValueError(
            f"dataset is missing {COMMAND_ACTION_KEY}; create a new ACT-RLT dataset instead of "
            "resuming a dataset recorded with the old schema"
        )
    _write_schema_metadata(
        dataset,
        collector_policy_id_codebook={str(COLLECTOR_HUMAN): "human-taught-script"},
        include_rlt_episode_metadata=False,
    )
    return dataset, robot, observation_processor, action_names


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    if args.task is None:
        args.task = "Move from a sampled workspace pose to the reference point and advance along -Z."
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(args.seed)
    dataset, robot, observation_processor, action_names = build_dataset_and_robot(args)
    encoding_manager_used = False
    try:
        robot.connect()
        robot.get_observation()  # Warm the gripper cache before recording.

        reference = teach_reference(robot)
        insertion_distance = args.insert_minus_z
        final_pose = insertion_pose_minus_z(reference, insertion_distance)
        print(f"REFERENCE p0: {format_pose(reference)}")
        print(f"FINAL (-Z {insertion_distance * 1000:.1f} mm): {format_pose(final_pose)}")
        print(
            "Sample bounds (robot base frame): "
            "x=x0+/-0.0200 m, y=[y0+0.0150, y0+0.0250] m, z=0.4020 m. "
            f"Sample orientation is fixed XYZ Euler {SAMPLE_EULER_XYZ_DEG} deg. "
            "Reference and insertion retain the taught orientation."
        )

        recorder = EpisodeRecorder(
            robot=robot,
            dataset=dataset,
            observation_processor=observation_processor,
            action_names=action_names,
            task=args.task,
            fps=args.fps,
            episode_time=args.episode_time,
        )
        saved = 0
        stop_requested = False
        encoding_manager_used = True
        with VideoEncodingManager(dataset):
            while saved < args.episodes:
                current_sample = sample_z_insertion_pose(reference, rng)
                print(f"CURRENT SAMPLE: {format_pose(current_sample)}")
                reteach_reference = False

                # Position, settle, and record continuously. A numeric hotkey
                # deliberately pauses automation and requires ENTER to resume.
                while True:
                    if not robot_is_executable(robot):
                        reteach_reference = True
                        break
                    if not nonrecorded_arm_move(
                        robot,
                        current_sample,
                        speed=args.move_speed,
                        label="CURRENT SAMPLE",
                    ):
                        if not wait_for_resume(
                            robot,
                            reference=reference,
                            sample=current_sample,
                            move_speed=args.move_speed,
                        ):
                            stop_requested = True
                            break
                        continue
                    print(
                        f"Sample reached; sleeping {args.pre_episode_sleep:g} s before episode "
                        "(1 reference | 2 sample | Q quit).",
                        flush=True,
                    )
                    delay_key = read_key_during_delay(args.pre_episode_sleep)
                    if delay_key is None:
                        break
                    if not handle_delay_hotkey(
                        delay_key,
                        robot,
                        reference=reference,
                        sample=current_sample,
                        move_speed=args.move_speed,
                    ):
                        stop_requested = True
                        break
                    # The hotkey may have moved away from the sampled start.
                    # Reposition and perform the complete settling delay again.
                if reteach_reference:
                    reference = teach_reference(robot)
                    final_pose = insertion_pose_minus_z(reference, insertion_distance)
                    print(f"REFERENCE p0: {format_pose(reference)}")
                    print(f"FINAL (-Z {insertion_distance * 1000:.1f} mm): {format_pose(final_pose)}")
                    continue
                if stop_requested:
                    break

                recorder.start()
                try:
                    with EpisodeAbortMonitor(args.abort_key) as abort_monitor:
                        recorder.abort_monitor = abort_monitor
                        recorder.capture()
                        run_recorded_move(
                            recorder,
                            reference,
                            speed=args.move_speed,
                            label="sample -> reference",
                        )
                        run_recorded_move(
                            recorder,
                            final_pose,
                            speed=args.insertion_speed,
                            label="reference -> final -Z",
                        )
                        recorder.clear_command()
                        frame_count = recorder.finish()
                except EpisodeAborted:
                    dataset.clear_episode_buffer()
                    print(
                        "\nEpisode stopped and discarded. Automatic motion is paused for safety."
                    )
                    if not wait_for_resume(
                        robot,
                        reference=reference,
                        sample=current_sample,
                        move_speed=args.move_speed,
                    ):
                        stop_requested = True
                    if stop_requested:
                        break
                    continue
                except BaseException:
                    dataset.clear_episode_buffer()
                    raise
                finally:
                    recorder.abort_monitor = None

                normalize_lerobot_scalar_buffer(dataset)
                dataset.save_episode()
                saved += 1
                print(f"Saved episode {saved}/{args.episodes} ({frame_count} frames).")
                print(
                    f"Episode saved; sleeping {args.post_episode_sleep:g} s before retreat "
                    "(1 reference | 2 sample | Q quit).",
                    flush=True,
                )
                delay_key = read_key_during_delay(args.post_episode_sleep)
                if not handle_delay_hotkey(
                    delay_key,
                    robot,
                    reference=reference,
                    sample=current_sample,
                    move_speed=args.move_speed,
                ):
                    stop_requested = True

                # The final recorded pose is one centimetre along the negative
                # insertion axis. Returning to p0 is the matching +axis retreat
                # and is not recorded.
                if not nonrecorded_arm_move(
                    robot,
                    reference,
                    speed=args.move_speed,
                    label="REFERENCE p0 (retreat +Z)",
                ):
                    print("Retreat failed; automatic collection is paused.")
                    if not stop_requested and not wait_for_resume(
                        robot,
                        reference=reference,
                        sample=current_sample,
                        move_speed=args.move_speed,
                    ):
                        stop_requested = True
                if saved >= args.episodes or stop_requested:
                    break
        return 0
    except KeyboardInterrupt:
        print("\nStopping sample-space collection.")
        return 130
    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
        finally:
            if not encoding_manager_used:
                dataset.finalize()


if __name__ == "__main__":
    raise SystemExit(main())
