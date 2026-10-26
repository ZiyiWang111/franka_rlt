"""Record point1 -> point2 coarse approaches; pause and edit point2 XY between episodes."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from act_rlt.data_collection import record_sample_space as base
from act_rlt.data_collection.coarse_input import CoarseKeyboard, XY_KEYS
from act_rlt.data_collection.sampling import SAMPLE_Z_M, SAMPLE_ROTATION_VECTOR, SAMPLE_EULER_XYZ_DEG
from evo_franka.geometry import euler_xyz_deg_to_rotvec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POINT1_POSITION_M = (0.5534, 0.0548, 0.5594)
POINT1_EULER_XYZ_DEG = (174.0, 3.9, 92.0)
POINT1_POSE = (*POINT1_POSITION_M, *euler_xyz_deg_to_rotvec(POINT1_EULER_XYZ_DEG))
MODE = "coarse_point1_to_point2_v1_no_force"
RANGE_STEP_M = 0.005
RANGE_MAX_M = 0.05
POST_EPISODE_HOLD_S = 0.5
AUTO_PAUSE_EVERY = 5
POINT1_YAW_HALF_RANGE_DEG = 25.0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="act_coarse")
    parser.add_argument("--root", type=Path, help="Default: datasets/ethernet_coarse/DATASET")
    parser.add_argument("--episodes", type=int, default=10, help="Number of new episodes")
    parser.add_argument("--fps", type=int, choices=base.SUPPORTED_FPS, default=15)
    parser.add_argument("--episode-time", type=float, default=60.0)
    parser.add_argument("--move-speed", type=float, default=0.06, help="Recorded approach speed (m/s)")
    parser.add_argument("--return-speed", type=float, default=0.05, help="Unrecorded return speed (m/s)")
    parser.add_argument("--xy-range-cm", type=float, default=0.0, help="Point1 XY half range, 0..5 cm in 0.5 cm steps")
    parser.add_argument("--randomize-point1-yaw", action="store_true",
                        help="Sample point1 XYZ-Euler Rz within nominal +/-25 deg; disable every-5 automatic pauses")
    parser.add_argument("--teleop-speed", type=float, default=0.02, help="Paused WASD speed (m/s)")
    parser.add_argument("--teleop-step-mm", type=float, default=1.0, help="Per-keypress step for terminal backend only")
    parser.add_argument("--teleop-max-shift-cm", type=float, default=15.0, help="Maximum XY distance from captured point2")
    parser.add_argument("--input-backend", choices=("auto", "terminal", "pynput", "evdev"), default="auto",
                        help="Auto: X11 pynput if DISPLAY exists; otherwise terminal steps")
    parser.add_argument("--keyboard", help="evdev device path; use --input-backend evdev")
    parser.add_argument("--position-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--robot-ip", default=os.environ.get("ROBOT_IP", "172.16.0.2"))
    parser.add_argument("--wrist-camera", default="349622072679")
    parser.add_argument("--front-camera", default="233522075778")
    cameras = parser.add_mutually_exclusive_group()
    cameras.add_argument("--wrist_only", "--wrist-only", action="store_true")
    cameras.add_argument("--front_only", "--front-only", action="store_true")
    parser.add_argument("--task", default="Move from the sampled elevated start pose to the coarse approach target.")
    parser.add_argument("--vcodec", default="h264", choices=("h264", "hevc", "libsvtav1", "auto"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print geometry/settings without opening hardware or writing data")
    parser.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    return parser


def validate_args(args):
    if not args.dataset.strip() or Path(args.dataset).is_absolute() or ".." in Path(args.dataset).parts:
        raise ValueError("dataset must be a nonempty relative name without '..'")
    args.root = (args.root or PROJECT_ROOT / "datasets/ethernet_coarse" / args.dataset).expanduser().resolve()
    for name in ("episodes", "episode_time", "move_speed", "return_speed", "teleop_speed",
                 "teleop_step_mm", "teleop_max_shift_cm", "position_tolerance_mm", "orientation_tolerance_deg"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be finite and positive")
    if (not math.isfinite(args.xy_range_cm) or not 0 <= args.xy_range_cm <= 5
            or not math.isclose(args.xy_range_cm * 2, round(args.xy_range_cm * 2), abs_tol=1e-9)):
        raise ValueError("xy-range-cm must be 0..5 in 0.5 cm steps")
    if args.seed is not None and args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if args.keyboard and args.input_backend != "evdev":
        raise ValueError("--keyboard requires --input-backend evdev")
    if not (args.wrist_only or args.front_only) and args.wrist_camera == args.front_camera:
        raise ValueError("wrist and front camera serials must differ")
    if args.resume:
        marker = args.root / "meta/coarse_collection.json"
        if not marker.is_file() or json.loads(marker.read_text()).get("mode") != MODE:
            raise ValueError("--resume requires a dataset from this no-force coarse collector")
        info = json.loads((args.root / "meta/info.json").read_text())
        if any("force" in key or "torque" in key or "wrench" in key for key in info["features"]):
            raise ValueError("coarse dataset must not contain force/torque feedback")
        for p in (args.root / "meta/coarse_episodes").glob("*.json"):
            if not json.loads(p.read_text()).get("dataset_saved"):
                raise ValueError(f"Incomplete episode save: inspect {p} before resuming")
    elif args.root.exists():
        raise ValueError(f"Dataset root already exists: {args.root}; use --resume or a new name")


def point2_from_xy(pose):
    values = np.asarray(pose, dtype=float)
    if values.shape not in ((2,), (6,)) or not np.isfinite(values).all():
        raise ValueError("point2 requires a finite XY pair or 6D TCP pose")
    return [float(values[0]), float(values[1]), SAMPLE_Z_M, *SAMPLE_ROTATION_VECTOR]


def sample_point1(rng, half_range, *, randomize_yaw=False):
    if not math.isfinite(half_range) or not 0 <= half_range <= RANGE_MAX_M:
        raise ValueError("Point1 half range must be 0..0.05 m")
    pose = np.asarray(POINT1_POSE).copy()
    pose[:2] += rng.uniform(-half_range, half_range, size=2)
    if randomize_yaw:
        euler = list(POINT1_EULER_XYZ_DEG)
        euler[2] += float(rng.uniform(-POINT1_YAW_HALF_RANGE_DEG, POINT1_YAW_HALF_RANGE_DEG))
        pose[3:] = euler_xyz_deg_to_rotvec(euler)
    return pose.tolist()


def pose_error(actual, target):
    actual, target = np.asarray(actual), np.asarray(target)
    return (float(np.linalg.norm(actual[:3] - target[:3])),
            float(np.degrees((Rotation.from_rotvec(target[3:]) * Rotation.from_rotvec(actual[3:]).inv()).magnitude())))


def require_arrival(robot, target, args):
    error_m, error_deg = pose_error(base.measured_pose(robot), target)
    print(f"TCP error: {error_m * 1000:.2f} mm / {error_deg:.2f} deg", flush=True)
    if error_m > args.position_tolerance_mm / 1000 or error_deg > args.orientation_tolerance_deg:
        raise RuntimeError("Trajectory ended outside the measured TCP arrival tolerances")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class Controls:
    def __init__(self, keyboard, half_range, robot=None):
        self.keyboard = keyboard
        self.half_range = half_range
        self.pause_requested = False
        self.robot = robot
        self._last_probe = 0.0

    def poll(self):
        events = self.keyboard.drain()
        for key in events:
            if key == "q":
                raise KeyboardInterrupt
            if key == "x":
                raise base.EpisodeAborted("X: stop and discard; automatic motion paused")
            if key == "p":
                if not self.pause_requested:
                    print("P received: this episode will finish and save, then pause at point2.", flush=True)
                self.pause_requested = True
            if key in {"=", "-"}:
                steps = round(self.half_range / RANGE_STEP_M) + (1 if key == "=" else -1)
                self.half_range = max(0, min(10, steps)) * RANGE_STEP_M
                print(f"Next point1: XY +/-{self.half_range * 100:g} cm", flush=True)
        if self.robot is not None and time.monotonic() - self._last_probe >= 1:
            self._last_probe = time.monotonic()
            probe = self.robot.probe()
            if not probe.get("ok"):
                raise RuntimeError(f"Control server disconnected: {probe.get('error')}")
        return events

    def pressed(self):
        # EpisodeRecorder calls this before/after each frame. P never aborts a frame.
        self.poll()
        return False


def move_outside_episode(robot, target, controls, args):
    """Tracked asynchronous return: keyboard/timeout remain responsive; no frames."""
    print(f"Unrecorded move to point1: {base.format_pose(target)}", flush=True)
    robot._stop_keepalive()
    started = time.monotonic()
    try:
        robot.robot.move_tool_traj([[*target, args.return_speed, 0.0, 0.0]], is_async=True)
        while True:
            controls.poll()
            if time.monotonic() - started > args.episode_time:
                raise TimeoutError("Point1 reposition timed out")
            if not robot.robot.is_running():
                result = robot.robot._wait_idle(0.05)
                if result.get("success"):
                    break
                if not str(result.get("error", "")).startswith("timeout after"):
                    raise RuntimeError(f"Point1 reposition failed: {result}")
            time.sleep(0.02)
        require_arrival(robot, target, args)
    except BaseException:
        robot.robot.stop_move()
        raise
    finally:
        robot.resync_command_pose()
        if robot.is_connected and robot.config.enable_fci_keepalive:
            robot._start_keepalive()


def collect_episode(recorder, target, controls, args):
    recorder.start()
    recorder.abort_monitor = controls
    try:
        recorder.capture()
        base.run_recorded_move(recorder, target, speed=args.move_speed, label="point1 -> point2")
        require_arrival(recorder.robot, target, args)
        controls.poll()
        recorder.clear_command()
        return recorder.finish()
    except BaseException:
        recorder.dataset.clear_episode_buffer()
        raise
    finally:
        recorder.abort_monitor = None


class XYTeleop:
    """Base-frame XY servo with fixed point2 Z/rotation and bounded command lead."""

    def __init__(self, robot, point2, origin_xy, args):
        self.robot, self.args = robot, args
        self.target = np.asarray(point2).copy()
        self.origin_xy = np.asarray(origin_xy).copy()
        self.active = False

    def step(self, delta_xy):
        measured = np.asarray(base.measured_pose(self.robot))
        if abs(measured[2] - self.target[2]) > self.args.position_tolerance_mm / 1000:
            raise RuntimeError("Paused TCP Z changed; refuse XY servo until the target pose is restored")
        if pose_error(measured, self.target)[1] > self.args.orientation_tolerance_deg:
            raise RuntimeError("Paused TCP orientation changed; refuse XY servo")
        candidate = self.target.copy()
        candidate[:2] += delta_xy
        lead = candidate[:2] - measured[:2]
        if np.linalg.norm(lead) > 0.003:
            candidate[:2] = measured[:2] + lead / np.linalg.norm(lead) * 0.003
        if np.linalg.norm(candidate[:2] - self.origin_xy) > self.args.teleop_max_shift_cm / 100:
            self.stop()
            print("Point2 XY shift limit reached; move back toward the captured XY.", flush=True)
            return
        if not self.active:
            self.robot._stop_keepalive()
            self.active = True
        self.robot.robot.servo_tool(candidate.tolist())
        self.target = candidate

    def stop(self):
        if self.active:
            if not self.robot.robot.stop_servo():
                raise RuntimeError("Failed to stop paused XY servo")
            result = self.robot.robot._wait_idle(2.0)
            if not result.get("success"):
                raise RuntimeError(f"XY servo did not stop: {result}")
            self.active = False
            # Keep Z/rotation fixed, reset only XY to the measured stopped pose.
            self.target[:2] = base.measured_pose(self.robot)[:2]
            self.robot.resync_command_pose()
            if self.robot.is_connected and self.robot.config.enable_fci_keepalive:
                self.robot._start_keepalive()


def edit_point2(robot, point2, origin_xy, controls, args):
    print("\nPAUSED at point2. WASD: W/S=-X/+X, A/D=-Y/+Y; Z/orientation fixed.\n"
          "Release motion keys, Enter: save actual XY and resume; =/-: next point1 range; Q: quit.", flush=True)
    if controls.keyboard.backend == "terminal":
        print(f"Terminal mode: each WASD press/repeat steps {args.teleop_step_mm:g} mm.", flush=True)
    teleop = XYTeleop(robot, point2, origin_xy, args)
    previous = time.monotonic()
    last_step = previous
    try:
        while True:
            now = time.monotonic()
            dt = min(max(now - previous, 0), 1 / 30)
            previous = now
            events = controls.poll()
            if "enter" in events:
                if controls.keyboard.direction_keys_down():
                    print("Release WASD and press Enter again.", flush=True)
                else:
                    teleop.stop()
                    actual = base.measured_pose(robot)
                    updated = point2_from_xy(actual)
                    require_arrival(robot, updated, args)
                    controls.pause_requested = False
                    print(f"Updated point2: {base.format_pose(updated)}", flush=True)
                    return updated
            if controls.keyboard.backend == "terminal":
                for key in events:
                    if key in XY_KEYS:
                        teleop.step(XY_KEYS[key] * args.teleop_step_mm / 1000)
                # Keep discrete-step commands briefly alive; stop when no key repeats.
                if any(key in XY_KEYS for key in events):
                    last_step = now
                elif teleop.active and now - last_step > 0.12:
                    teleop.stop()
            else:
                direction = controls.keyboard.direction()
                if controls.keyboard.direction_keys_down() and np.any(direction):
                    teleop.step(direction * args.teleop_speed * dt)
                else:
                    teleop.stop()
            time.sleep(1 / 30)
    finally:
        try:
            teleop.stop()
        finally:
            robot.resync_command_pose()
            if robot.is_connected and robot.config.enable_fci_keepalive:
                robot._start_keepalive()


def wait_retry(robot, controls):
    print("Motion stopped; episode discarded. Correct the robot state. Enter: retry | Q: quit.", flush=True)
    while True:
        try:
            if "enter" in controls.poll() and base.robot_is_executable(robot):
                controls.pause_requested = False
                return
        except base.EpisodeAborted:
            pass
        time.sleep(0.05)


def capture_point2(robot, controls):
    print("Place TCP at the desired point2 XY. Enter: capture XY and begin | =/-: point1 range | Q: quit.", flush=True)
    while True:
        if "enter" in controls.poll():
            return point2_from_xy(base.measured_pose(robot))
        time.sleep(0.05)


def prepare_return(robot, point2, origin_xy, controls, args, recording_finished_at):
    """Hold without recording; a P request during this gap still pauses at point2."""
    deadline = recording_finished_at + POST_EPISODE_HOLD_S
    while True:
        controls.poll()
        if controls.pause_requested:
            point2 = edit_point2(robot, point2, origin_xy, controls, args)
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return point2
        time.sleep(min(0.02, remaining))


def run_collection(dataset, robot, recorder, controls, args, point2):
    rng = np.random.default_rng(args.seed)
    origin_xy = point2[:2].copy()
    saved = 0
    while saved < args.episodes:
        controls.poll()
        half_range = controls.half_range
        point1 = sample_point1(rng, half_range, randomize_yaw=args.randomize_point1_yaw)
        try:
            move_outside_episode(robot, point1, controls, args)
            frames = collect_episode(recorder, point2, controls, args)
            recording_finished_at = time.monotonic()
        except base.EpisodeAborted:
            wait_retry(robot, controls)
            continue
        index = int(dataset.meta.total_episodes)
        metadata = dict(episode_index=index, point1_pose=point1, point2_pose=point2.copy(),
                        point1_xy_half_range_m=half_range, speed_m_s=args.move_speed,
                        point1_randomize_yaw=args.randomize_point1_yaw,
                        frames=frames, dataset_saved=False)
        meta_path = args.root / "meta/coarse_episodes" / f"episode_{index:06d}.json"
        write_json(meta_path, metadata)
        base.normalize_lerobot_scalar_buffer(dataset)
        dataset.save_episode()
        metadata["dataset_saved"] = True
        write_json(meta_path, metadata)
        saved += 1
        print(f"Saved episode {index} ({saved}/{args.episodes} new, {frames} frames).", flush=True)
        if not args.randomize_point1_yaw and saved % AUTO_PAUSE_EVERY == 0:
            controls.pause_requested = True
            print(f"{saved} new episodes saved: automatically pausing at point2.", flush=True)
        point2 = prepare_return(robot, point2, origin_xy, controls, args, recording_finished_at)
        if saved == args.episodes:
            # Match the repeated workflow even on the final episode: unrecorded return.
            move_outside_episode(robot, point1, controls, args)
    return saved


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    if args.validate_only:
        return 0
    if args.dry_run:
        print(json.dumps(dict(dataset=str(args.root), point1_center=POINT1_POSE,
                              point2_z_m=SAMPLE_Z_M, point2_euler_xyz_deg=SAMPLE_EULER_XYZ_DEG,
                              point1_xy_half_range_cm=args.xy_range_cm, speed_m_s=args.move_speed,
                              return_speed_m_s=args.return_speed, post_episode_hold_s=POST_EPISODE_HOLD_S,
                              point1_randomize_yaw=args.randomize_point1_yaw,
                              point1_yaw_half_range_deg=POINT1_YAW_HALF_RANGE_DEG if args.randomize_point1_yaw else 0,
                              auto_pause_every=0 if args.randomize_point1_yaw else AUTO_PAUSE_EVERY,
                              force_feedback=False, fps=args.fps,
                              cameras=[name for name, enabled in (("wrist", not args.front_only),
                                                                 ("front", not args.wrist_only)) if enabled]), indent=2))
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    keyboard = CoarseKeyboard(args.input_backend, args.keyboard)
    dataset = robot = None
    encoding_manager_used = False
    try:
        dataset, robot, processor, action_names = base.build_dataset_and_robot(args, record_force_feedback=False)
        write_json(args.root / "meta/coarse_collection.json", dict(
            mode=MODE, force_feedback=False, frame="base", position_unit="m", orientation="rotation_vector_rad",
            point1_center=list(POINT1_POSE), point1_euler_xyz_deg=POINT1_EULER_XYZ_DEG,
            point1_yaw_sampling="intrinsic_XYZ_Rz",
            point1_yaw_half_range_deg=POINT1_YAW_HALF_RANGE_DEG if args.randomize_point1_yaw else 0,
            auto_pause_every=0 if args.randomize_point1_yaw else AUTO_PAUSE_EVERY,
            point2_z_m=SAMPLE_Z_M, point2_euler_xyz_deg=SAMPLE_EULER_XYZ_DEG,
        ))
        robot.connect()
        robot.get_observation()
        controls = Controls(keyboard, args.xy_range_cm / 100, robot)
        point2 = capture_point2(robot, controls)
        print(f"POINT2: {base.format_pose(point2)}; approach={args.move_speed:g} m/s; no force/torque recording.", flush=True)
        if args.randomize_point1_yaw:
            print("Point1 yaw: 92 +/-25 deg (XYZ Euler Rz); every-5 automatic pause disabled; P still pauses.", flush=True)
        recorder = base.EpisodeRecorder(robot=robot, dataset=dataset, observation_processor=processor,
                                        action_names=action_names, task=args.task, fps=args.fps,
                                        episode_time=args.episode_time, record_force_feedback=False)
        encoding_manager_used = True
        with base.VideoEncodingManager(dataset):
            run_collection(dataset, robot, recorder, controls, args, point2)
        return 0
    except KeyboardInterrupt:
        print("\nCoarse collection stopped.", flush=True)
        return 130
    finally:
        try:
            if robot is not None and robot.is_connected:
                try:
                    robot._stop_keepalive()
                    robot.robot.stop_move()
                finally:
                    robot.disconnect()
        finally:
            try:
                if dataset is not None and not encoding_manager_used:
                    dataset.finalize()
            finally:
                keyboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
