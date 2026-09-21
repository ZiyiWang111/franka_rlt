#!/usr/bin/env python3
"""Low-latency 15 Hz Cartesian keyboard teleoperation for an FR3.

Uses Linux evdev for a locally attached keyboard, or pynput for keyboard events
in an X11 desktop such as ToDesk. Key-up events are real events, so releasing
all motion keys reliably stops the arm and closes a measurement segment.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import select
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


KEY_DIRECTIONS = {
    "KEY_W": np.array([1.0, 0.0, 0.0]),
    "KEY_S": np.array([-1.0, 0.0, 0.0]),
    "KEY_A": np.array([0.0, 1.0, 0.0]),
    "KEY_D": np.array([0.0, -1.0, 0.0]),
    "KEY_P": np.array([0.0, 0.0, 1.0]),
    "KEY_L": np.array([0.0, 0.0, -1.0]),
}
GRIPPER_KEYS = {"KEY_C": "close", "KEY_O": "open"}
WAYPOINT_KEYS = {f"KEY_{slot}": str(slot) for slot in range(1, 10)}
RECORD_WAYPOINT_KEY = "KEY_R"
DEFAULT_WAYPOINTS_FILE = Path.home() / ".config" / "evo-rlt" / "fr3_keyboard_waypoints.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-motion", action="store_true", help="Required acknowledgement")
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument(
        "--input-backend",
        choices=("evdev", "pynput"),
        default="evdev",
        help="pynput reads ToDesk/X11 events; evdev reads a physical host keyboard",
    )
    parser.add_argument("--keyboard", help="evdev path, e.g. /dev/input/event4")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--speed-m-s", type=float, default=0.03)
    parser.add_argument("--max-distance-from-start-m", type=float, default=0.15)
    parser.add_argument("--max-command-lead-m", type=float, default=0.010)
    parser.add_argument("--workspace-min", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--workspace-max", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--motion-detect-m", type=float, default=0.0002)
    parser.add_argument("--gripper-width-m", type=float, default=0.0)
    parser.add_argument("--gripper-force-n", type=float, default=55.0)
    parser.add_argument("--waypoints-file", type=Path, default=DEFAULT_WAYPOINTS_FILE)
    parser.add_argument("--waypoint-speed-m-s", type=float, default=0.03)
    parser.add_argument(
        "--pynput-release-debounce-ms",
        type=float,
        default=60.0,
        help="Treat a ToDesk/X11 release quickly followed by press as key auto-repeat",
    )
    parser.add_argument("--log", type=Path, help="Append one JSON object per segment")
    parser.add_argument(
        "--no-grab",
        action="store_true",
        help="Do not exclusively grab control keys while the teleop is running",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("fps", "speed_m_s", "max_distance_from_start_m", "max_command_lead_m"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.motion_detect_m) or args.motion_detect_m <= 0:
        raise ValueError("--motion-detect-m must be finite and positive")
    if not math.isfinite(args.pynput_release_debounce_ms) or args.pynput_release_debounce_ms < 0:
        raise ValueError("--pynput-release-debounce-ms must be finite and non-negative")
    if not math.isfinite(args.gripper_width_m) or not 0.0 <= args.gripper_width_m <= 0.08:
        raise ValueError("--gripper-width-m must be in [0, 0.08]")
    if not math.isfinite(args.gripper_force_n) or not 20.0 <= args.gripper_force_n <= 70.0:
        raise ValueError("--gripper-force-n must be in [20, 70]")
    if not math.isfinite(args.waypoint_speed_m_s) or args.waypoint_speed_m_s <= 0:
        raise ValueError("--waypoint-speed-m-s must be finite and positive")
    if (args.workspace_min is None) != (args.workspace_max is None):
        raise ValueError("--workspace-min and --workspace-max must be supplied together")
    if args.workspace_min is not None:
        lower, upper = np.asarray(args.workspace_min), np.asarray(args.workspace_max)
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower >= upper):
            raise ValueError("workspace bounds must be finite and min < max on every axis")


def percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values, dtype=float), q))


def optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


class WaypointStore:
    """Persistent named 6D TCP waypoints, deliberately separate from training data."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.data = {"version": 1, "names": {}, "waypoints": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"cannot read waypoint file {self.path}: {exc}") from exc
            if not isinstance(loaded, dict):
                raise RuntimeError(f"invalid waypoint file {self.path}: root must be an object")
            names = loaded.get("names") or {}
            waypoints = loaded.get("waypoints") or {}
            if not isinstance(names, dict) or not isinstance(waypoints, dict):
                raise RuntimeError(f"invalid waypoint file {self.path}: names and waypoints must be objects")
            self.data["names"] = dict(names)
            self.data["waypoints"] = dict(waypoints)
        for slot, entry in self.data["waypoints"].items():
            pose = np.asarray(entry.get("pose") if isinstance(entry, dict) else None, dtype=float)
            valid_slot = slot in {str(value) for value in range(1, 10)}
            if not valid_slot or pose.shape != (6,) or not np.isfinite(pose).all():
                raise RuntimeError(f"invalid waypoint {slot!r} in {self.path}")

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def name(self, slot: str) -> str:
        return str(self.data["names"].get(slot) or f"waypoint-{slot}")

    def rename(self, slot: str, name: str) -> None:
        self.data["names"][slot] = name.strip()
        self._write()

    def save_pose(self, slot: str, pose: np.ndarray) -> None:
        self.data["waypoints"][slot] = {
            "pose": np.asarray(pose, dtype=float).tolist(),
            "saved_at_unix_s": time.time(),
        }
        self._write()

    def pose(self, slot: str) -> np.ndarray | None:
        entry = self.data["waypoints"].get(slot)
        return None if entry is None else np.asarray(entry["pose"], dtype=float)


@dataclass
class SegmentStats:
    index: int
    start_time: float
    start_pose: np.ndarray
    first_key_time: float
    command_times: list[float] = field(default_factory=list)
    dispatch_ms: list[float] = field(default_factory=list)
    schedule_lateness_ms: list[float] = field(default_factory=list)
    input_to_command_ms: list[float] = field(default_factory=list)
    pending_input_times: list[float] = field(default_factory=list)
    commanded_path_m: float = 0.0
    first_motion_time: float | None = None

    def record_command(self, sent_at: float, dispatch_ms: float, lateness_ms: float) -> None:
        self.command_times.append(sent_at)
        self.dispatch_ms.append(dispatch_ms)
        self.schedule_lateness_ms.append(max(0.0, lateness_ms))
        self.input_to_command_ms.extend(
            max(0.0, (sent_at - event_time) * 1000.0)
            for event_time in self.pending_input_times
        )
        self.pending_input_times.clear()

    def finish(
        self,
        *,
        released_at: float,
        stopped_at: float,
        end_pose: np.ndarray,
        target_hz: float,
    ) -> dict[str, Any]:
        interval_ms = (np.diff(np.asarray(self.command_times)) * 1000.0).tolist()
        actual_hz = None
        if len(interval_ms) > 0 and sum(interval_ms) > 0:
            actual_hz = 1000.0 * len(interval_ms) / sum(interval_ms)
        expected_ms = 1000.0 / target_hz
        jitter_ms = [abs(interval - expected_ms) for interval in interval_ms]
        duration = max(0.0, released_at - self.start_time)
        return {
            "segment": self.index,
            "duration_s": duration,
            "commands": len(self.command_times),
            "target_hz": target_hz,
            "actual_hz": actual_hz,
            "command_rate_over_segment_hz": len(self.command_times) / duration if duration else None,
            "period_ms_mean": statistics.fmean(interval_ms) if interval_ms else None,
            "period_ms_p95": percentile(interval_ms, 95),
            "period_jitter_ms_p95": percentile(jitter_ms, 95),
            "deadline_lateness_ms_p95": percentile(self.schedule_lateness_ms, 95),
            "deadline_lateness_ms_max": max(self.schedule_lateness_ms, default=None),
            "input_to_command_ms_mean": (
                statistics.fmean(self.input_to_command_ms) if self.input_to_command_ms else None
            ),
            "input_to_command_ms_p95": percentile(self.input_to_command_ms, 95),
            "input_to_command_ms_max": max(self.input_to_command_ms, default=None),
            "first_key_to_measured_motion_ms": (
                None if self.first_motion_time is None
                else (self.first_motion_time - self.first_key_time) * 1000.0
            ),
            "release_to_stop_ack_ms": max(0.0, (stopped_at - released_at) * 1000.0),
            "dispatch_ms_mean": statistics.fmean(self.dispatch_ms) if self.dispatch_ms else None,
            "dispatch_ms_p95": percentile(self.dispatch_ms, 95),
            "dispatch_ms_max": max(self.dispatch_ms, default=None),
            "commanded_path_m": self.commanded_path_m,
            "measured_displacement_m": float(np.linalg.norm(end_pose[:3] - self.start_pose[:3])),
            "start_xyz": self.start_pose[:3].tolist(),
            "end_xyz": end_pose[:3].tolist(),
        }


def print_summary(summary: dict[str, Any]) -> None:
    print(
        f"\n--- segment {summary['segment']} ---\n"
        f"duration={summary['duration_s']:.3f}s commands={summary['commands']} "
        f"actual={optional(summary['actual_hz'])} Hz (target={summary['target_hz']:g})\n"
        f"period mean/p95={optional(summary['period_ms_mean'])}/"
        f"{optional(summary['period_ms_p95'])} ms; jitter p95="
        f"{optional(summary['period_jitter_ms_p95'])} ms\n"
        f"input->command mean/p95/max={optional(summary['input_to_command_ms_mean'])}/"
        f"{optional(summary['input_to_command_ms_p95'])}/"
        f"{optional(summary['input_to_command_ms_max'])} ms\n"
        f"first key->measured motion={optional(summary['first_key_to_measured_motion_ms'])} ms; "
        f"release->stop ACK={optional(summary['release_to_stop_ack_ms'])} ms\n"
        f"dispatch mean/p95/max={optional(summary['dispatch_ms_mean'])}/"
        f"{optional(summary['dispatch_ms_p95'])}/{optional(summary['dispatch_ms_max'])} ms\n"
        f"commanded path={summary['commanded_path_m']:.4f}m; "
        f"measured net displacement={summary['measured_displacement_m']:.4f}m",
        flush=True,
    )


def discover_keyboard(requested: str | None):
    try:
        from evdev import InputDevice, ecodes, list_devices
    except ImportError as exc:
        raise RuntimeError("evdev backend requires `pip install evdev`") from exc
    if requested:
        try:
            return InputDevice(requested)
        except PermissionError as exc:
            raise RuntimeError(f"permission denied opening {requested}; add user to group 'input'") from exc

    required = {getattr(ecodes, name) for name in KEY_DIRECTIONS}
    candidates = []
    for path in list_devices():
        try:
            device = InputDevice(path)
            if required.issubset(set(device.capabilities().get(ecodes.EV_KEY, []))):
                candidates.append(device)
            else:
                device.close()
        except PermissionError:
            continue
    if len(candidates) == 1:
        return candidates[0]
    choices = ", ".join(f"{dev.path} ({dev.name})" for dev in candidates)
    for device in candidates:
        device.close()
    if not choices:
        raise RuntimeError("no readable keyboard with W/S/A/D/P/L found; pass --keyboard")
    raise RuntimeError(f"multiple keyboards found; choose one with --keyboard: {choices}")


class Teleop:
    def __init__(self, robot: Any, args: argparse.Namespace, waypoints: WaypointStore) -> None:
        self.robot = robot
        self.args = args
        self.waypoints = waypoints
        self.period = 1.0 / args.fps
        self.pressed: set[str] = set()
        self.segment: SegmentStats | None = None
        self.segment_count = 0
        self.next_tick: float | None = None
        self.target_pose: np.ndarray | None = None
        self.record_waypoint_armed = False
        self.session_start_pose = self._pose()
        self.lower = None if args.workspace_min is None else np.asarray(args.workspace_min, dtype=float)
        self.upper = None if args.workspace_max is None else np.asarray(args.workspace_max, dtype=float)

    def _pose(self) -> np.ndarray:
        pose = np.asarray(self.robot.get_tool_pose(), dtype=float)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise RuntimeError(f"invalid measured TCP pose: {pose}")
        return pose

    def _direction(self) -> np.ndarray:
        direction = sum((KEY_DIRECTIONS[key] for key in self.pressed), start=np.zeros(3))
        norm = float(np.linalg.norm(direction))
        return direction if norm == 0 else direction / norm

    def key_event(self, key: str, down: bool, event_time: float) -> None:
        before = set(self.pressed)
        if down:
            self.pressed.add(key)
        else:
            self.pressed.discard(key)
        if before == self.pressed:
            return
        if not before and self.pressed:
            measured = self._pose()
            self.target_pose = measured.copy()
            self.segment_count += 1
            self.segment = SegmentStats(
                index=self.segment_count,
                start_time=event_time,
                start_pose=measured,
                first_key_time=event_time,
                pending_input_times=[event_time],
            )
            self.next_tick = event_time
        elif self.segment is not None and self.pressed:
            self.segment.pending_input_times.append(event_time)
        elif before and not self.pressed:
            self.finish_segment(event_time)

    def tick(self, deadline: float) -> None:
        if self.segment is None or self.target_pose is None or not self.pressed:
            return
        measured = self._pose()
        if (
            self.segment.first_motion_time is None
            and self.segment.command_times
            and np.linalg.norm(measured[:3] - self.segment.start_pose[:3]) >= self.args.motion_detect_m
        ):
            self.segment.first_motion_time = time.perf_counter()
        candidate = self.target_pose.copy()
        candidate[:3] += self._direction() * self.args.speed_m_s * self.period
        lead = candidate[:3] - measured[:3]
        lead_norm = float(np.linalg.norm(lead))
        if lead_norm > self.args.max_command_lead_m:
            candidate[:3] = measured[:3] + lead / lead_norm * self.args.max_command_lead_m
        if np.linalg.norm(candidate[:3] - self.session_start_pose[:3]) > self.args.max_distance_from_start_m:
            raise RuntimeError("TCP target exceeded --max-distance-from-start-m; servo stopped")
        if self.lower is not None and (
            np.any(candidate[:3] < self.lower) or np.any(candidate[:3] > self.upper)
        ):
            raise RuntimeError(f"TCP target outside workspace: {candidate[:3]}")
        sent_at = time.perf_counter()
        previous_target = self.target_pose.copy()
        self.robot.servo_tool(candidate.tolist())
        returned_at = time.perf_counter()
        self.target_pose = candidate
        self.segment.commanded_path_m += float(np.linalg.norm(candidate[:3] - previous_target[:3]))
        self.segment.record_command(
            sent_at,
            dispatch_ms=(returned_at - sent_at) * 1000.0,
            lateness_ms=(sent_at - deadline) * 1000.0,
        )

    def finish_segment(self, released_at: float) -> None:
        segment, self.segment = self.segment, None
        self.next_tick = None
        self.target_pose = None
        if segment is None:
            return
        if not self.robot.stop_servo():
            raise RuntimeError("control server did not acknowledge stop_servo")
        summary = segment.finish(
            released_at=released_at,
            stopped_at=time.perf_counter(),
            end_pose=self._pose(),
            target_hz=self.args.fps,
        )
        print_summary(summary)
        if self.args.log:
            self.args.log.parent.mkdir(parents=True, exist_ok=True)
            with self.args.log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(summary, ensure_ascii=False) + "\n")

    def command_gripper(self, command: str) -> None:
        """Start one asynchronous Hand command only while arm keys are released."""
        if self.pressed:
            print(f"Ignored gripper {command}: release all XYZ motion keys first.", flush=True)
            return
        # stop_servo is idempotent. It is the explicit ownership handoff the
        # driver requires before a Hand operation; the server verifies the
        # arm is idle again before accepting the asynchronous gripper command.
        try:
            if not self.robot.stop_servo():
                print(f"Gripper {command} not sent: control server rejected stop_servo.", flush=True)
                return
            if command == "open":
                motion = self.robot.start_open_gripper()
            elif command == "close":
                motion = self.robot.start_close_gripper(
                    width=self.args.gripper_width_m,
                    force=self.args.gripper_force_n,
                )
            else:
                raise ValueError(f"unsupported gripper command: {command}")
        except Exception as exc:  # command rejection should not kill arm teleop
            print(f"Gripper {command} rejected: {exc}", flush=True)
            return
        print(
            f"Gripper {command} started (id={motion.get('command_id')}, "
            f"width={self.args.gripper_width_m:.3f}m, force={self.args.gripper_force_n:.1f}N).",
            flush=True,
        )

    def arm_waypoint_recording(self) -> None:
        if self.pressed:
            print("Ignored waypoint recording: release all XYZ motion keys first.", flush=True)
            return
        self.record_waypoint_armed = True
        print("Waypoint recording armed: press 1-9 to save/overwrite the current TCP pose.", flush=True)

    def save_waypoint(self, slot: str, name: str) -> None:
        if self.pressed:
            print("Ignored waypoint save: release all XYZ motion keys first.", flush=True)
            return
        pose = self._pose()
        self.waypoints.rename(slot, name)
        self.waypoints.save_pose(slot, pose)
        self.record_waypoint_armed = False
        print(
            f"Saved waypoint {slot} ({self.waypoints.name(slot)}): "
            f"{np.array2string(pose, precision=5)}",
            flush=True,
        )

    def recall_waypoint(self, slot: str) -> None:
        if self.pressed:
            print("Ignored waypoint move: release all XYZ motion keys first.", flush=True)
            return
        pose = self.waypoints.pose(slot)
        if pose is None:
            print(f"Waypoint {slot} ({self.waypoints.name(slot)}) has not been recorded.", flush=True)
            return
        if self.lower is not None and (
            np.any(pose[:3] < self.lower) or np.any(pose[:3] > self.upper)
        ):
            print(
                f"Refusing waypoint {slot}: target {pose[:3]} lies outside configured workspace.",
                flush=True,
            )
            return
        try:
            if not self.robot.stop_servo():
                print("Waypoint move not sent: control server rejected stop_servo.", flush=True)
                return
            print(
                f"Moving to waypoint {slot} ({self.waypoints.name(slot)}) at "
                f"{self.args.waypoint_speed_m_s:.3f} m/s ...",
                flush=True,
            )
            measured = self.robot.move_tool(pose.tolist(), speed=self.args.waypoint_speed_m_s)
            self.target_pose = None
            print(
                f"Reached waypoint {slot}: "
                f"{np.array2string(np.asarray(measured), precision=5)}",
                flush=True,
            )
        except Exception as exc:  # a refused recalled move must not end the teleop process
            print(f"Waypoint {slot} move rejected: {exc}", flush=True)

    def stop(self) -> None:
        self.pressed.clear()
        self.segment = None
        self.next_tick = None
        self.target_pose = None
        self.robot.stop_servo()


def prompt_waypoint_name(teleop: Teleop, slot: str) -> None:
    """Prompt after keyboard capture has been released, then persist the pose."""
    current_name = teleop.waypoints.name(slot)
    try:
        name = input(f"\nName for waypoint {slot} [{current_name}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("Waypoint recording cancelled.", flush=True)
        teleop.record_waypoint_armed = False
        return
    teleop.save_waypoint(slot, name or current_name)


def advance_schedule(teleop: Teleop, deadline: float) -> None:
    teleop.tick(deadline)
    elapsed = max(1, math.floor((time.perf_counter() - deadline) / teleop.period) + 1)
    teleop.next_tick = deadline + elapsed * teleop.period


def require_motion_ready(robot: Any, timeout_s: float = 3.0) -> str:
    """Wait for the server PUSH stream after connect, then require Idle mode.

    ``connect()`` completes before the client's state-receiving thread is
    guaranteed to have received its first snapshot.  A blank mode during that
    short interval is unknown, not a robot-mode refusal.
    """
    deadline = time.monotonic() + timeout_s
    last_mode = ""
    while time.monotonic() < deadline:
        last_mode = str(robot.robot_mode_nowait())
        if "Idle" in last_mode:
            return last_mode
        time.sleep(0.05)
    if not last_mode:
        raise RuntimeError(
            "control-state cache remained empty for 3.0 s after connect; "
            "the robot mode is unknown (check control-server state stream)"
        )
    raise RuntimeError(
        f"robot is not motion-ready (reported mode={last_mode!r}; expected RobotMode.Idle)"
    )


def run_evdev(device: Any, teleop: Teleop) -> None:
    from evdev import ecodes

    code_to_name = {
        **{getattr(ecodes, name): name for name in KEY_DIRECTIONS},
        **{getattr(ecodes, name): name for name in GRIPPER_KEYS},
        **{getattr(ecodes, name): name for name in WAYPOINT_KEYS},
        getattr(ecodes, RECORD_WAYPOINT_KEY): RECORD_WAYPOINT_KEY,
    }
    print(
        "Ready: W/S=+X/-X, A/D=+Y/-Y, P/L=+Z/-Z; C=close, O=open (only when "
        "motion keys are released); R then 1-9=save waypoint, 1-9=recall; ESC exits.",
        flush=True,
    )

    def record_with_name(slot: str) -> None:
        # An evdev grab prevents the terminal from receiving the name. Release
        # it only while the arm is already stopped and the naming prompt is up.
        grabbed_here = not teleop.args.no_grab
        if grabbed_here:
            device.ungrab()
        try:
            time.sleep(0.08)  # let the just-pressed digit physically release
            prompt_waypoint_name(teleop, slot)
        finally:
            if grabbed_here:
                device.grab()
    while True:
        now = time.perf_counter()
        timeout = None if teleop.next_tick is None else max(0.0, teleop.next_tick - now)
        readable, _, _ = select.select([device.fd], [], [], timeout)
        if readable:
            for event in device.read():
                if event.type != ecodes.EV_KEY or event.value == 2:
                    continue
                if event.code == ecodes.KEY_ESC and event.value == 1:
                    return
                key = code_to_name.get(event.code)
                if key is not None:
                    received = time.perf_counter()
                    event_time = float(event.timestamp())
                    if abs(received - event_time) >= 5:
                        event_time = received
                    if key in GRIPPER_KEYS:
                        if event.value == 1:
                            teleop.command_gripper(GRIPPER_KEYS[key])
                    elif key == RECORD_WAYPOINT_KEY:
                        if event.value == 1:
                            teleop.arm_waypoint_recording()
                    elif key in WAYPOINT_KEYS:
                        if event.value == 1:
                            slot = WAYPOINT_KEYS[key]
                            if teleop.record_waypoint_armed:
                                record_with_name(slot)
                            else:
                                teleop.recall_waypoint(slot)
                    else:
                        teleop.key_event(key, event.value == 1, event_time)
        if teleop.next_tick is not None and time.perf_counter() >= teleop.next_tick:
            advance_schedule(teleop, teleop.next_tick)


def run_pynput(teleop: Teleop, *, suppress: bool) -> None:
    try:
        from pynput import keyboard
    except ImportError as exc:
        raise RuntimeError("pynput backend requires `pip install pynput`") from exc
    events: queue.Queue[tuple[str, bool, float]] = queue.Queue()
    # X11 auto-repeat (and ToDesk's X11 injection) can represent a held key as
    # KEY_RELEASE followed immediately by KEY_PRESS. Delay a release briefly;
    # a matching press cancels it, while a real physical release is retained.
    release_debounce_s = teleop.args.pynput_release_debounce_ms / 1000.0
    pending_releases: dict[str, float] = {}
    gripper_keys_down: set[str] = set()
    waypoint_keys_down: set[str] = set()

    def name_for(key: Any) -> str | None:
        if key == keyboard.Key.esc:
            return "KEY_ESC"
        char = getattr(key, "char", None)
        name = f"KEY_{char.upper()}" if isinstance(char, str) else ""
        if name in KEY_DIRECTIONS or name in GRIPPER_KEYS or name in WAYPOINT_KEYS:
            return name
        return RECORD_WAYPOINT_KEY if name == RECORD_WAYPOINT_KEY else None

    def receive(key: Any, down: bool) -> None:
        name = name_for(key)
        if name is not None:
            events.put((name, down, time.perf_counter()))

    def start_listener():
        listener = keyboard.Listener(
            on_press=lambda key: receive(key, True),
            on_release=lambda key: receive(key, False),
            suppress=suppress,
        )
        listener.start()
        listener.wait()
        if not listener.is_alive():
            raise RuntimeError("pynput keyboard listener failed to start")
        return listener

    listener = start_listener()
    print(
        "Ready via ToDesk/X11: W/S=+X/-X, A/D=+Y/-Y, P/L=+Z/-Z; "
        "C=close, O=open when motion keys are released; "
        "R then 1-9=save waypoint, 1-9=recall; "
        f"release debounce={teleop.args.pynput_release_debounce_ms:g}ms; ESC exits.",
        flush=True,
    )

    def flush_due_releases(now: float) -> None:
        for key, released_at in list(pending_releases.items()):
            if now >= released_at + release_debounce_s:
                del pending_releases[key]
                # Preserve the actual release timestamp: the statistics then
                # include debounce plus stop RPC time in release->stop ACK.
                if key in GRIPPER_KEYS:
                    gripper_keys_down.discard(key)
                elif key in WAYPOINT_KEYS or key == RECORD_WAYPOINT_KEY:
                    waypoint_keys_down.discard(key)
                else:
                    teleop.key_event(key, False, released_at)

    def record_with_name(slot: str) -> None:
        nonlocal listener
        # ToDesk/X11 text reaches the terminal only after the global listener
        # stops suppressing keys. No arm key is pressed here by construction.
        listener.stop()
        listener.join(timeout=1.0)
        pending_releases.clear()
        gripper_keys_down.clear()
        waypoint_keys_down.clear()
        try:
            time.sleep(0.08)  # let the just-pressed digit release before input()
            prompt_waypoint_name(teleop, slot)
        finally:
            listener = start_listener()

    def handle_event(key: str, down: bool, event_time: float) -> bool:
        if key == "KEY_ESC" and down:
            return True
        if down:
            pending_releases.pop(key, None)
            if key in GRIPPER_KEYS:
                if key not in gripper_keys_down:
                    gripper_keys_down.add(key)
                    teleop.command_gripper(GRIPPER_KEYS[key])
            elif key == RECORD_WAYPOINT_KEY:
                if key not in waypoint_keys_down:
                    waypoint_keys_down.add(key)
                    teleop.arm_waypoint_recording()
            elif key in WAYPOINT_KEYS:
                if key not in waypoint_keys_down:
                    waypoint_keys_down.add(key)
                    slot = WAYPOINT_KEYS[key]
                    if teleop.record_waypoint_armed:
                        record_with_name(slot)
                    else:
                        teleop.recall_waypoint(slot)
            else:
                teleop.key_event(key, True, event_time)
        else:
            pending_releases[key] = event_time
        return False

    try:
        while listener.is_alive():
            now = time.perf_counter()
            deadlines = []
            if teleop.next_tick is not None:
                deadlines.append(teleop.next_tick)
            if pending_releases:
                deadlines.append(min(pending_releases.values()) + release_debounce_s)
            timeout = None if not deadlines else max(0.0, min(deadlines) - now)
            try:
                key, down, event_time = events.get(timeout=timeout)
                if handle_event(key, down, event_time):
                    return
                while True:
                    key, down, event_time = events.get_nowait()
                    if handle_event(key, down, event_time):
                        return
            except queue.Empty:
                pass
            flush_due_releases(time.perf_counter())
            if teleop.next_tick is not None and time.perf_counter() >= teleop.next_tick:
                advance_schedule(teleop, teleop.next_tick)
    finally:
        listener.stop()
        listener.join(timeout=1.0)


def main() -> int:
    args = build_parser().parse_args()
    device = None
    waypoints = None
    try:
        validate_args(args)
        if not args.allow_motion:
            raise RuntimeError("refusing to move without --allow-motion")
        if args.input_backend == "pynput" and args.keyboard:
            raise ValueError("--keyboard only applies to --input-backend evdev")
        if args.input_backend == "pynput":
            try:
                from pynput import keyboard as _pynput_keyboard  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("pynput/X11 unavailable; run `pip install pynput`") from exc
        else:
            device = discover_keyboard(args.keyboard)
        waypoints = WaypointStore(args.waypoints_file)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    from evo_franka.control_client import FrankaArmControllerClient

    robot = None
    teleop = None
    grabbed = False
    connected = False
    try:
        if device is not None:
            try:
                device.set_clockid(time.CLOCK_MONOTONIC)
            except (AttributeError, OSError):
                pass
            if not args.no_grab:
                device.grab()
                grabbed = True
        robot = FrankaArmControllerClient(args.robot_ip)
        robot.connect()
        connected = True
        mode = require_motion_ready(robot)
        print(f"Robot motion-ready: {mode}", flush=True)
        teleop = Teleop(robot, args, waypoints)
        print(f"Start TCP={teleop.session_start_pose.tolist()}", flush=True)
        if device is None:
            run_pynput(teleop, suppress=not args.no_grab)
        else:
            print(f"Keyboard: {device.path} ({device.name})", flush=True)
            run_evdev(device, teleop)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"fatal: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if teleop is not None:
            try:
                teleop.stop()
            except Exception as exc:  # noqa: BLE001
                print(f"warning: final stop failed: {exc}", file=sys.stderr)
        if connected and robot is not None:
            try:
                robot.disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"warning: disconnect failed: {exc}", file=sys.stderr)
        if grabbed and device is not None:
            try:
                device.ungrab()
            except OSError:
                pass
        if device is not None:
            device.close()


if __name__ == "__main__":
    raise SystemExit(main())
