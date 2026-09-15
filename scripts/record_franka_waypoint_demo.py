#!/usr/bin/env python3
"""Record repeated FR3 episodes from two taught TCP waypoints."""

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

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset repo/name")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-time", type=float, default=300.0)
    parser.add_argument("--task", default="Move between two taught waypoints and insert along -Y.")
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--wrist-camera", default="349622072679")
    parser.add_argument("--front-camera", default="233522075778")
    parser.add_argument("--move-speed", type=float, default=0.05, help="TCP waypoint speed in m/s")
    parser.add_argument(
        "--final-move-speed",
        type=float,
        default=0.05,
        help="Independent speed for the point-2 to final -Y segment in m/s",
    )
    parser.add_argument("--final-minus-y", type=float, default=0.01, help="Final -Y distance in metres")
    parser.add_argument(
        "--start-square-side",
        type=float,
        default=0.15,
        help="Side length of the XY start-pose sampling square in metres",
    )
    parser.add_argument(
        "--grasp-square-side",
        type=float,
        default=0.10,
        help="Side length of the XY grasp-point sampling square in metres",
    )
    parser.add_argument(
        "--start-yaw-range-deg",
        type=float,
        default=15.0,
        help="Maximum absolute base-Z yaw perturbation for sampled starts (0-15 degrees)",
    )
    parser.add_argument("--gripper-width", type=float, default=0.0, help="Grasp target width in metres")
    parser.add_argument("--gripper-force", type=float, default=40.0, help="Grasp force in newtons")
    parser.add_argument("--abort-key", default="x", help="Single key that aborts the active episode")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--vcodec", default="h264")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0 or args.episodes <= 0 or args.episode_time <= 0:
        raise ValueError("fps, episodes and episode-time must be positive")
    if (
        args.move_speed <= 0
        or args.final_move_speed <= 0
        or args.final_minus_y <= 0
        or args.start_square_side <= 0
        or args.grasp_square_side <= 0
    ):
        raise ValueError(
            "move-speed, final-move-speed, final-minus-y, start-square-side and "
            "grasp-square-side must be positive"
        )
    if not 0.0 <= args.gripper_width <= 0.08:
        raise ValueError("gripper-width must be within the Franka Hand range [0, 0.08] m")
    if not 0.0 <= args.start_yaw_range_deg <= 15.0:
        raise ValueError("start-yaw-range-deg must be within [0, 15] degrees")
    if not 20.0 <= args.gripper_force <= 70.0:
        raise ValueError("gripper-force must be within the continuous-use range [20, 70] N")
    if args.wrist_camera == args.front_camera:
        raise ValueError("wrist and front camera serials must be different")
    if len(args.abort_key) != 1:
        raise ValueError("abort-key must be exactly one character")


class EpisodeAborted(RuntimeError):
    """Raised after the operator presses the per-episode emergency abort key."""


class EpisodeAbortMonitor:
    """Read a single abort key without ENTER while preserving Ctrl+C handling."""

    def __init__(self, key: str) -> None:
        self.key = key.lower()
        self.fd: int | None = None
        self.saved_terminal = None

    def __enter__(self) -> "EpisodeAbortMonitor":
        if not sys.stdin.isatty():
            raise RuntimeError("episode abort key requires an interactive terminal")
        self.fd = sys.stdin.fileno()
        self.saved_terminal = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        print(f"Episode running: press '{self.key}' to stop and discard it.", flush=True)
        return self

    def pressed(self) -> bool:
        while select.select([sys.stdin], [], [], 0.0)[0]:
            char = os.read(self.fd, 1).decode(errors="ignore")
            if char.lower() == self.key:
                return True
        return False

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.fd is not None and self.saved_terminal is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved_terminal)


def nonrecorded_arm_move(
    robot: FrankaRobot,
    target: list[float],
    *,
    speed: float,
    label: str,
) -> bool:
    """Move outside an episode; a Desk-mode refusal returns to the key prompt."""
    print(f"\nMoving to {label} ...", flush=True)
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
        print(f"Motion was not started/completed (robot_mode={mode}): {exc}", flush=True)
        if "Guiding" in mode or "UserStopped" in mode or "programming" in str(exc).lower():
            print(
                "Please switch Franka Desk from Programming/Guiding to Execution, "
                "release user-stop/brakes, then press the same number to retry. "
                "The recorder is still running.",
                flush=True,
            )
        else:
            print("The recorder is still running; correct the arm state and retry.", flush=True)
        return False
    finally:
        robot.resync_command_pose()
        if robot.is_connected and robot.config.enable_fci_keepalive:
            robot._start_keepalive()
    print(f"Reached {label}.", flush=True)
    return True


def point_key_help(points: dict[str, tuple[str, list[float] | None]]) -> str:
    return " | ".join(
        f"{key} {label}{'' if pose is not None else '=NULL'}"
        for key, (label, pose) in points.items()
    )


def read_operator_key(
    robot: FrankaRobot,
    message: str,
    *,
    points: dict[str, tuple[str, list[float] | None]],
    move_speed: float,
    gripper_width: float,
    gripper_force: float,
    accepted: set[str],
    action_help: str,
    require_execution_on_enter: bool = False,
) -> str:
    """Interactive non-recording prompt with gripper and stored-pose controls."""
    if not sys.stdin.isatty():
        raise RuntimeError("operator controls require an interactive terminal")
    fd = sys.stdin.fileno()
    saved_terminal = termios.tcgetattr(fd)
    prompt = (
        f"\n{message}\n"
        f"{action_help} | C close | O open | {point_key_help(points)}: "
    )
    print(prompt, end="", flush=True)
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
                if key == "enter" and require_execution_on_enter:
                    try:
                        mode = str(robot.robot.robot_mode_nowait())
                    except Exception as exc:
                        print(f"\nCannot verify robot mode yet: {exc}\n" + prompt, end="", flush=True)
                        continue
                    if "Guiding" in mode or "UserStopped" in mode:
                        print(
                            f"\nRobot is {mode}. Switch Franka Desk to Execution, release "
                            "Guiding/user-stop, then press ENTER again. Nothing was recorded.\n"
                            + prompt,
                            end="",
                            flush=True,
                        )
                        continue
                print()
                return key
            if key in points:
                label, target = points[key]
                if target is None:
                    print(f"\n{label}: NULL (not recorded/sampled yet).\n" + prompt, end="", flush=True)
                    continue
                nonrecorded_arm_move(robot, target, speed=move_speed, label=label)
                print("\n" + prompt, end="", flush=True)
            elif key == "c":
                print(
                    f"\nClosing gripper: width={gripper_width:.4f} m, "
                    f"force={gripper_force:.1f} N ...",
                    flush=True,
                )
                robot.close_gripper(width=gripper_width, force=gripper_force)
                print("Gripper close complete.\n" + prompt, end="", flush=True)
            elif key == "o":
                print("\nOpening gripper ...", flush=True)
                robot.open_gripper()
                print("Gripper open complete.\n" + prompt, end="", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved_terminal)


def prompt_enter(
    robot: FrankaRobot,
    message: str,
    *,
    gripper_width: float,
    gripper_force: float,
    points: dict[str, tuple[str, list[float] | None]],
    move_speed: float,
    require_execution: bool = False,
) -> None:
    """Wait for pose confirmation with all non-recording controls active."""
    key = read_operator_key(
        robot,
        message,
        points=points,
        move_speed=move_speed,
        gripper_width=gripper_width,
        gripper_force=gripper_force,
        accepted={"enter", "q"},
        action_help="ENTER confirm/start | Q quit",
        require_execution_on_enter=require_execution,
    )
    if key == "q":
        raise KeyboardInterrupt


def measured_pose(robot: FrankaRobot) -> list[float]:
    pose = [float(value) for value in robot.robot.get_tool_pose()]
    if len(pose) != 6 or not np.isfinite(pose).all():
        raise RuntimeError(f"invalid measured TCP pose: {pose}")
    return pose


class EpisodeRecorder:
    """One-frame-delayed writer matching the manual-demo action convention."""

    def __init__(
        self,
        *,
        robot: FrankaRobot,
        dataset: LeRobotDataset,
        observation_processor,
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
        self.pending: tuple[dict, list[float], float] | None = None
        self.started_at = 0.0
        self.last_sample_started_at: float | None = None
        self.abort_monitor: EpisodeAbortMonitor | None = None
        self.frames = 0

    def start(self) -> None:
        self.pending = None
        self.started_at = time.perf_counter()
        self.last_sample_started_at = None
        self.frames = 0

    def check_deadline(self) -> None:
        if self.abort_monitor is not None and self.abort_monitor.pressed():
            raise EpisodeAborted("operator requested episode abort")
        if time.perf_counter() - self.started_at > self.episode_time:
            raise TimeoutError(f"episode exceeded {self.episode_time:g} s")

    def capture(self) -> None:
        self.check_deadline()
        if self.last_sample_started_at is not None:
            precise_sleep(
                max(
                    self.last_sample_started_at + 1.0 / self.fps - time.perf_counter(),
                    0.0,
                )
            )
        observation_started = time.perf_counter()
        self.last_sample_started_at = observation_started
        obs = self.robot.get_observation()
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
                f"health failure: server_down={self.health['server_down']} "
                f"camera_down={self.health['camera_down']}"
            )

        pose = [float(obs[f"ee_{axis}"]) for axis in ("x", "y", "z", "rx", "ry", "rz")]
        width = float(obs["gripper_width"])
        obs_frame = build_dataset_frame(
            self.dataset.features,
            self.observation_processor(obs),
            prefix=OBS_STR,
        )
        if self.pending is not None:
            prev_frame, prev_pose, _ = self.pending
            action = _build_manual_demo_action(self.action_names, prev_pose, pose, width)
            self._add_frame(prev_frame, action)
        self.pending = (obs_frame, pose, width)

    def _add_frame(self, observation_frame: dict, action: dict[str, float]) -> None:
        action_frame = build_dataset_frame(self.dataset.features, action, prefix=ACTION)
        zero_action = {name: 0.0 for name in self.action_names}
        policy_action = build_dataset_frame(
            self.dataset.features,
            zero_action,
            prefix="complementary_info.policy_action",
        )
        frame = {**observation_frame, **action_frame, **policy_action, "task": self.task}
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
            obs_frame, pose, width = self.pending
            action = _build_manual_demo_action(self.action_names, pose, None, width)
            self._add_frame(obs_frame, action)
            self.pending = None
        return self.frames

def run_arm_move(
    recorder: EpisodeRecorder,
    target: list[float],
    *,
    speed: float,
    label: str,
) -> None:
    robot = recorder.robot
    logging.info("Starting %s -> %s", label, np.array2string(np.asarray(target), precision=5))
    robot._stop_keepalive()
    try:
        robot.robot.move_tool(target, speed=speed, is_async=True)
        while True:
            recorder.capture()
            if not robot.robot.is_running():
                # is_running can briefly be false before Franky engages. The
                # server's worker status is authoritative and also carries a
                # deferred async-motion error.
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


def run_gripper_close(recorder: EpisodeRecorder, target_width: float, force: float) -> None:
    robot = recorder.robot
    logging.info("Closing gripper to target width %.4f m with %.1f N", target_width, force)
    started = robot.start_gripper_transition("close", width=target_width, force=force)
    command_id = started.get("command_id")
    try:
        while True:
            recorder.capture()
            state = robot.get_gripper_transition_state()
            if state.get("command_id") == command_id and state.get("status") != "running":
                if state.get("status") == "error":
                    raise RuntimeError(f"gripper close failed: {state.get('error')}")
                if not state.get("result", False):
                    logging.warning(
                        "Gripper close completed but grasp result is false; recording continues "
                        "because save/discard is decided after the trajectory."
                    )
                break
        recorder.capture()
    except BaseException:
        try:
            state = robot.get_gripper_transition_state()
            if state.get("status") == "running":
                robot.robot.stop_gripper()
        except Exception:
            logging.exception("failed to stop gripper after close-phase error")
        raise
    finally:
        robot.finish_gripper_transition()


def build_dataset_and_robot(args: argparse.Namespace):
    config = FrankaRobotConfig(
        robot_ip=args.robot_ip,
        camera_serial=args.wrist_camera,
        front_camera_serial=args.front_camera,
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
                f"cannot resume dataset at a different fps: dataset={dataset.fps}, requested={args.fps}"
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
    _write_schema_metadata(
        dataset,
        collector_policy_id_codebook={str(COLLECTOR_HUMAN): "human-taught-script"},
        include_rlt_episode_metadata=False,
    )
    return dataset, robot, observation_processor, action_names


def pose_controls(
    point1: list[float] | None,
    point2: list[float] | None,
    initial_start: list[float] | None,
    next_start: list[float] | None,
) -> dict[str, tuple[str, list[float] | None]]:
    return {
        "1": ("POINT 1", point1),
        "2": ("POINT 2", point2),
        "3": ("INITIAL START", initial_start),
        "4": ("NEXT START", next_start),
    }


def sample_next_start(
    initial_start: list[float],
    square_side: float,
    yaw_range_deg: float = 15.0,
) -> list[float]:
    """Sample base XY and base-Z yaw around the immutable initial start pose."""
    rng = np.random.default_rng()
    target = initial_start.copy()
    target[:2] = (
        np.asarray(initial_start[:2], dtype=float)
        + rng.uniform(-square_side / 2.0, square_side / 2.0, size=2)
    ).tolist()
    yaw_rad = math.radians(float(rng.uniform(-yaw_range_deg, yaw_range_deg)))
    initial_rotation = Rotation.from_rotvec(np.asarray(initial_start[3:6], dtype=float))
    yaw_rotation = Rotation.from_rotvec([0.0, 0.0, yaw_rad])
    # Left multiplication makes this a yaw about the robot base Z axis, not
    # about an arbitrarily tilted tool-local axis.
    target[3:6] = (yaw_rotation * initial_rotation).as_rotvec().tolist()
    return target


def sample_grasp_point(initial_grasp: list[float], square_side: float = 0.10) -> list[float]:
    """Sample grasp XY around its immutable taught centre; preserve Z and orientation."""
    target = initial_grasp.copy()
    target[:2] = (
        np.asarray(initial_grasp[:2], dtype=float)
        + np.random.default_rng().uniform(-square_side / 2.0, square_side / 2.0, size=2)
    ).tolist()
    return target


def choose_episode(
    robot: FrankaRobot,
    *,
    points: dict[str, tuple[str, list[float] | None]],
    move_speed: float,
    gripper_width: float,
    gripper_force: float,
) -> str:
    while True:
        choice = read_operator_key(
            robot,
            "Trajectory complete. POINT 1 and NEXT START now belong to the next attempt.",
            points=points,
            move_speed=move_speed,
            gripper_width=gripper_width,
            gripper_force=gripper_force,
            accepted={"enter", "s", "d", "q"},
            action_help="S save | D discard | Q discard+quit",
        )
        if choice != "enter":
            return choice
        print(
            "Previous episode is still awaiting a decision. "
            "Press S to save, D to discard, or Q to discard and quit.",
            flush=True,
        )


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dataset, robot, observation_processor, action_names = build_dataset_and_robot(args)
    encoding_manager_used = False
    try:
        robot.connect()
        # Populate the gripper cache before any asynchronous command begins.
        robot.get_observation()

        prompt_enter(
            robot,
            "Guide the arm to POINT 1 (the grasp point).",
            gripper_width=args.gripper_width,
            gripper_force=args.gripper_force,
            points=pose_controls(None, None, None, None),
            move_speed=args.move_speed,
        )
        initial_grasp_point = measured_pose(robot)
        point1 = initial_grasp_point.copy()
        print("INITIAL GRASP / POINT 1:", " ".join(f"{value:+.6f}" for value in point1))

        prompt_enter(
            robot,
            "Guide the arm to POINT 2 (the placement/pre-insertion point).",
            gripper_width=args.gripper_width,
            gripper_force=args.gripper_force,
            points=pose_controls(point1, None, None, None),
            move_speed=args.move_speed,
        )
        point2 = measured_pose(robot)
        final_point = point2.copy()
        final_point[1] -= args.final_minus_y
        if not all(math.isfinite(value) for value in final_point):
            raise RuntimeError(f"invalid final point: {final_point}")
        print("POINT 2:", " ".join(f"{value:+.6f}" for value in point2))
        print(
            f"FINAL (-Y {args.final_minus_y * 1000:.1f} mm):",
            " ".join(f"{value:+.6f}" for value in final_point),
        )
        prompt_enter(
            robot,
            "Guide the arm to INITIAL START. This third ENTER records the immutable sampling centre.",
            gripper_width=args.gripper_width,
            gripper_force=args.gripper_force,
            points=pose_controls(point1, point2, None, None),
            move_speed=args.move_speed,
        )
        initial_start = measured_pose(robot)
        next_start = initial_start.copy()
        print("INITIAL START:", " ".join(f"{value:+.6f}" for value in initial_start))
        print(
            "The initial grasp centre, POINT 2 and INITIAL START are fixed for this process. "
            f"Later POINT 1 poses are sampled in an XY square of side "
            f"{args.grasp_square_side:.3f} m around the initial grasp centre. "
            f"Later starts are sampled in an XY square of side {args.start_square_side:.3f} m "
            f"with base-Z yaw within +/-{args.start_yaw_range_deg:.1f} deg, "
            "both relative to INITIAL START."
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
        encoding_manager_used = True
        with VideoEncodingManager(dataset):
            while saved < args.episodes:
                prompt_enter(
                    robot,
                    f"Prepare attempt for saved episode {saved + 1}/{args.episodes}. "
                    "Press 4 to move to NEXT START, then ENTER to record from the current pose.",
                    gripper_width=args.gripper_width,
                    gripper_force=args.gripper_force,
                    points=pose_controls(point1, point2, initial_start, next_start),
                    move_speed=args.move_speed,
                    require_execution=True,
                )
                recorder.start()
                try:
                    with EpisodeAbortMonitor(args.abort_key) as abort_monitor:
                        recorder.abort_monitor = abort_monitor
                        recorder.capture()
                        run_arm_move(recorder, point1, speed=args.move_speed, label="move to POINT 1")
                        run_gripper_close(recorder, args.gripper_width, args.gripper_force)
                        run_arm_move(recorder, point2, speed=args.move_speed, label="move to POINT 2")
                        run_arm_move(
                            recorder,
                            final_point,
                            speed=args.final_move_speed,
                            label="final -Y move",
                        )
                        frame_count = recorder.finish()
                except EpisodeAborted:
                    dataset.clear_episode_buffer()
                    point1 = sample_grasp_point(
                        initial_grasp_point,
                        args.grasp_square_side,
                    )
                    next_start = sample_next_start(
                        initial_start,
                        args.start_square_side,
                        args.start_yaw_range_deg,
                    )
                    print(
                        "\nEpisode interrupted and discarded. "
                        "New POINT 1 and NEXT START poses were sampled from their fixed centres."
                    )
                    print("POINT 1:", " ".join(f"{value:+.6f}" for value in point1))
                    print("NEXT START:", " ".join(f"{value:+.6f}" for value in next_start))
                    continue
                except BaseException:
                    dataset.clear_episode_buffer()
                    raise
                finally:
                    recorder.abort_monitor = None

                point1 = sample_grasp_point(
                    initial_grasp_point,
                    args.grasp_square_side,
                )
                next_start = sample_next_start(
                    initial_start,
                    args.start_square_side,
                    args.start_yaw_range_deg,
                )
                print("POINT 1:", " ".join(f"{value:+.6f}" for value in point1))
                print("NEXT START:", " ".join(f"{value:+.6f}" for value in next_start))
                # Reset outside the just-finished episode. C/O remain available
                # in the outcome prompt if the operator wants another state.
                robot.open_gripper()
                choice = choose_episode(
                    robot,
                    points=pose_controls(point1, point2, initial_start, next_start),
                    move_speed=args.move_speed,
                    gripper_width=args.gripper_width,
                    gripper_force=args.gripper_force,
                )
                if choice == "s":
                    dataset.save_episode()
                    saved += 1
                    print(f"Saved episode {saved}/{args.episodes} ({frame_count} frames).")
                else:
                    dataset.clear_episode_buffer()
                    print("Episode discarded.")
                if choice == "q":
                    break
        return 0
    except KeyboardInterrupt:
        print("\nStopping collection.")
        return 130
    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
        finally:
            if not encoding_manager_used:
                dataset.finalize()


if __name__ == "__main__":
    sys.exit(main())
