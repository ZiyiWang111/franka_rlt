"""Record ACT demonstrations: one soft approach to the taught pose offset by base -Z 1 cm.

Reuses the original sample-space collector's cameras, dataset schema, action
labels, sampling, hotkeys and reset motions. Only recorded task motion uses
the online reference integrator from act_rlt.soft_approach.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import threading
import time

import numpy as np

from act_rlt import soft_approach as soft
from act_rlt.data_collection import record_sample_space as base
from act_rlt.data_collection.record_sample_space import (
    COMMAND_ACTION_NAMES,
    EpisodeAborted,
    EpisodeAbortMonitor,
    SAMPLE_EULER_XYZ_DEG,
    VideoEncodingManager,
    format_pose,
    handle_delay_hotkey,
    insertion_pose_minus_z,
    nonrecorded_arm_move,
    normalize_lerobot_scalar_buffer,
    read_key_during_delay,
    robot_is_executable,
    sample_z_insertion_pose,
    teach_reference,
    wait_for_resume,
)
from act_rlt.data_collection.rgbd_capture import DepthEpisodeRecorder as EpisodeRecorder, RGBDFrankaRobot


MODE = "online_soft_approach_to_offset_goal_v2"
SOFT_PARAMETERS = (
    "acceleration", "jerk", "slowdown_distance", "lateral_gate_distance",
    "lateral_tolerance", "vertical_speed_floor", "lag_limit", "max_angular_velocity",
    "position_tolerance", "orientation_tolerance_deg", "settle_time",
)


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser(include_insertion_speed=False)
    parser.description = __doc__
    defaults = soft.build_parser().parse_args([])
    parser.add_argument("--approach-speed", type=float, default=defaults.speed,
                        help="Soft approach reference speed; move-speed remains the unrecorded reset speed")
    parser.add_argument("--control-fps", type=float, default=defaults.fps,
                        help="Online target updates (Hz), independent of camera/dataset --fps")
    parser.add_argument("--stage-timeout", type=float, default=defaults.timeout,
                        help="Maximum seconds per stage, also bounded by the episode timeout")
    for name in SOFT_PARAMETERS:
        parser.add_argument("--" + name.replace("_", "-"), type=float,
                            default=getattr(defaults, name),
                            help="Terminal braking distance scale (m); braking limits take precedence"
                            if name == "slowdown_distance" else f"As in soft_approach: {name}")
    parser.set_defaults(approach_speed=0.04, move_speed=0.016,
                        acceleration=0.05, jerk=0.2, slowdown_distance=0.008)
    return parser


def motion_args(args, *, remaining: float | None = None,
                start_pose=None, target_pose=None):
    result = soft.build_parser().parse_args([])
    for name in SOFT_PARAMETERS:
        setattr(result, name, getattr(args, name))
    result.fps = args.control_fps
    result.speed = args.approach_speed
    result.timeout = min(args.stage_timeout, remaining) if remaining is not None else args.stage_timeout
    result.finite_approach = True
    if start_pose is not None and target_pose is not None:
        lateral_distance = float(np.linalg.norm(
            np.asarray(target_pose[:2]) - np.asarray(start_pose[:2])))
        if lateral_distance > args.position_tolerance:
            # Sampled starts are only centimetres from p0. A fixed 60 mm gate
            # would already allow almost full descent at these starts. Fit the
            # transition ONCE to this stage, then use command-space errors only.
            result.lateral_gate_distance = min(args.lateral_gate_distance, lateral_distance)
            result.lateral_tolerance = min(args.lateral_tolerance, 0.15 * result.lateral_gate_distance)
    return result


def validate_args(args) -> None:
    base.validate_args(args)
    for name in ("approach_speed", "control_fps", "stage_timeout"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    soft.validate_args(motion_args(args))


def verify_dataset_mode(args) -> None:
    marker = args.root / "meta" / "soft_approach_collection.json"
    if args.resume:
        if not marker.exists() or json.loads(marker.read_text()).get("trajectory_mode") != MODE:
            raise ValueError("--resume requires a dataset created by record_soft_approach in single-stage mode; use a new dataset")
        metadata = json.loads(marker.read_text())
        if not metadata.get("depth_enabled", False):
            raise ValueError("RGB-only datasets cannot be resumed by this RGB-D collector; use a new dataset")
        for episode in (args.root / "depth").glob("episode_*/episode.json"):
            if not json.loads(episode.read_text()).get("dataset_saved", False):
                raise ValueError(f"unfinished RGB/depth save; inspect this episode before resuming: {episode}")


def write_mode_marker(args) -> None:
    marker = args.root / "meta" / "soft_approach_collection.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "trajectory_mode": MODE,
        "recorded_stages": 1,
        "goal_offset_base_z_m": -args.insert_minus_z,
        "depth_enabled": True,
        "depth": {"format": "uint16 PNG", "alignment": "color", "manifest": "depth/episode_*/frames.jsonl"},
        "action": "measured TCP delta and measured gripper width (original ACT schema)",
        "command_action": "latest reference pose and reference speed at observation acquisition start",
        "control_trace": "exact online target updates and measured states in soft_approach_traces/",
    }, indent=2) + "\n")


def build_dataset_and_robot(args):
    return base.build_dataset_and_robot(args, robot_factory=RGBDFrankaRobot)


def record_run_metadata(args, reference) -> Path:
    write_mode_marker(args)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = args.root / "soft_approach_traces" / run_name
    directory.mkdir(parents=True, exist_ok=False)
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    (directory / "run.json").write_text(json.dumps({
        "trajectory_mode": MODE, "reference_pose": list(reference),
        "final_pose": insertion_pose_minus_z(reference, args.insert_minus_z), "arguments": config,
    }, indent=2) + "\n")
    return directory


class _CancelableClient:
    def __init__(self, client, cancelled):
        self.client, self.cancelled = client, cancelled

    def _check(self):
        if self.cancelled.is_set():
            raise EpisodeAborted("recording stopped; no further reference targets are allowed")

    def _send(self, *args, **kwargs):
        self._check()
        return self.client._send(*args, **kwargs)

    def servo_tool(self, pose):
        self._check()
        return self.client.servo_tool(pose)

    def stop_servo(self):
        return self.client.stop_servo()


def run_recorded_episode(recorder, reference, final_pose, args, trace_path: Path) -> None:
    """Record on the calling thread; update targets on a separate control thread.

    The recording thread alone owns EpisodeRecorder and the dataset buffer.
    The control thread owns the planner and publishes immutable command copies.
    Always join it before the caller can save/clear the episode or move/reset.
    """
    robot, client = recorder.robot, recorder.robot.robot
    cancelled = threading.Event()
    command_lock = threading.Lock()
    latest_command = dict(zip(COMMAND_ACTION_NAMES, [*base.measured_pose(robot), 0.0], strict=True))
    failures = []
    succeeded = threading.Event()
    worker = None
    robot._stop_keepalive()
    try:
        with trace_path.open("w", encoding="utf-8", buffering=1) as trace:
            def log(event, **fields):
                trace.write(json.dumps({"event": event, "wall_time": time.time(),
                                        "monotonic": time.monotonic(), **fields}) + "\n")

            def emit(phase, event, **fields):
                if cancelled.is_set():
                    raise EpisodeAborted("recording cancelled")
                if event == "sample" and "command_pose" in fields:
                    speed = float(np.linalg.norm(fields.get("reference_velocity_m_s", [0, 0, 0])))
                    with command_lock:
                        latest_command.update(zip(COMMAND_ACTION_NAMES,
                                                  [*fields["command_pose"], speed], strict=True))
                log(event, phase=phase, **fields)

            def capture():
                # Pace BEFORE selecting the reference snapshot. The base recorder
                # also paces, but its remaining sleep is then effectively zero.
                # Otherwise this snapshot could precede camera acquisition by a
                # whole recording interval while the controller keeps advancing.
                if recorder.last_sample_started_at is not None:
                    base.precise_sleep(max(
                        recorder.last_sample_started_at + 1.0 / recorder.fps - time.perf_counter(), 0.0))
                # Set only the next observation's command snapshot. In particular,
                # don't retroactively overwrite the previous pending frame.
                with command_lock:
                    recorder.command_action = latest_command.copy()
                recorder.capture()

            def control():
                proxy = _CancelableClient(client, cancelled)
                try:
                    proxy._check()
                    remaining = args.episode_time - (time.perf_counter() - recorder.started_at)
                    if remaining <= 0:
                        raise TimeoutError("episode deadline reached before approach")
                    start = soft.checked_snapshot(soft.request(proxy, "get_state"))
                    parameters = motion_args(
                        args, remaining=remaining, start_pose=start["pose"], target_pose=final_pose)
                    log("stage_start", phase="approach", target=final_pose, state=start,
                        motion_parameters={name: getattr(parameters, name) for name in
                                           (*SOFT_PARAMETERS, "speed", "fps", "timeout", "finite_approach")})
                    soft.run_online_move(
                        proxy, start, final_pose, parameters,
                        lambda event, **fields: emit("approach", event, **fields))
                    log("stage_finished", phase="approach")
                    succeeded.set()
                except BaseException as exc:
                    failures.append(exc)
                    cancelled.set()
                    try:
                        if not client.stop_move():
                            logging.error("server did not confirm stopping failed online collection")
                    except Exception:
                        logging.exception("could not stop failed online collection")

            log("episode_start", reference=reference, final_pose=final_pose)
            capture()  # Obtain the first observation before issuing any task motion.
            worker = threading.Thread(target=control, name="act-soft-approach-control")
            worker.start()
            try:
                while worker.is_alive():
                    capture()
                worker.join()
                if failures:
                    raise failures[0]
                if not succeeded.is_set():
                    raise RuntimeError("control thread ended without reaching the offset goal")
                capture()  # Terminal observation after actual goal completion.
                log("episode_motion_finished", captured_frames=recorder.frames)
            except BaseException as exc:
                cancelled.set()
                try:
                    if not client.stop_move():
                        logging.error("server did not confirm stopping aborted collection")
                finally:
                    worker.join()
                log("episode_motion_aborted", error=f"{type(exc).__name__}: {exc}")
                raise
    finally:
        cancelled.set()
        if worker is not None and worker.is_alive():
            try:
                client.stop_move()
            finally:
                worker.join()
        robot.resync_command_pose()
        if robot.is_connected and robot.config.enable_fci_keepalive:
            robot._start_keepalive()


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    verify_dataset_mode(args)
    if args.task is None:
        args.task = f"One continuous soft approach to the taught pose offset {args.insert_minus_z:g} m along base -Z."
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(args.seed)
    dataset, robot, observation_processor, action_names = build_dataset_and_robot(args)
    encoding_manager_used = False
    trace_directory = None
    recorder = None
    try:
        write_mode_marker(args)
        soft.require_idle(robot.robot)
        robot.connect()
        robot.get_observation()  # Warm the gripper cache before recording.

        reference = teach_reference(robot)
        trace_directory = record_run_metadata(args, reference)
        calibration_path = trace_directory / "cameras.json"
        robot.save_depth_calibration(calibration_path)
        insertion_distance = args.insert_minus_z
        final_pose = insertion_pose_minus_z(reference, insertion_distance)
        print(f"REFERENCE p0: {format_pose(reference)}")
        print(f"FINAL (-Z {insertion_distance * 1000:.1f} mm): {format_pose(final_pose)}")
        print(
            "Sample bounds (robot base frame): "
            "x=x0+/-0.0250 m, y=[y0+0.0100, y0+0.0300] m, z=0.4020 m. "
            f"Sample orientation is fixed XYZ Euler {SAMPLE_EULER_XYZ_DEG} deg. "
            "The final goal retains the taught orientation."
        )

        recorder = EpisodeRecorder(
            depth_enabled=True,
            depth_calibration_path=calibration_path,
            robot=robot,
            dataset=dataset,
            observation_processor=observation_processor,
            action_names=action_names,
            task=args.task,
            fps=args.fps,
            episode_time=args.episode_time,
        )
        saved = 0
        attempt = 0
        print(f"Recording {args.fps} Hz; soft approach {args.control_fps:g} Hz, "
              f"<= {args.approach_speed:g} m/s directly to p0 -Z {args.insert_minus_z * 1000:g} mm. "
              "One continuous motion; no intermediate p0 stop or separate descent.")
        print(f"Reference acceleration <= {args.acceleration:g} m/s^2; jerk <= {args.jerk:g} m/s^3. "
              "Approach Z allowance scales smoothly with each stage's initial XY distance.")
        print("Approach profile: smooth acceleration, near-constant cruise, finite braking; "
              "short moves use a lower cruise speed so acceleration/braking leave room for cruise.")
        stop_requested = False
        encoding_manager_used = True
        with VideoEncodingManager(dataset):
            while saved < args.episodes:
                attempt += 1
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
                    trace_directory = record_run_metadata(args, reference)
                    recorder.depth_calibration_path = trace_directory / "cameras.json"
                    robot.save_depth_calibration(recorder.depth_calibration_path)
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
                        trace_path = trace_directory / f"attempt_{attempt:06d}.jsonl"
                        run_recorded_episode(recorder, reference, final_pose, args, trace_path)
                        recorder.clear_command()
                        frame_count = recorder.finish()
                except EpisodeAborted:
                    recorder.discard_depth()
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
                    recorder.discard_depth()
                    dataset.clear_episode_buffer()
                    raise
                finally:
                    recorder.abort_monitor = None

                normalize_lerobot_scalar_buffer(dataset)
                recorder.prepare_depth_save()
                dataset.save_episode()
                recorder.mark_depth_saved()
                with trace_path.open("a", encoding="utf-8") as trace:
                    trace.write(json.dumps({"event": "episode_saved", "wall_time": time.time(),
                                            "dataset_episode_count": dataset.num_episodes,
                                            "frames": frame_count}) + "\n")
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
        print("\nStopping soft-approach ACT collection.")
        return 130
    finally:
        try:
            try:
                if recorder is not None:
                    recorder.discard_depth()
            finally:
                if robot.is_connected:
                    robot.disconnect()
        finally:
            if not encoding_manager_used:
                dataset.finalize()


if __name__ == "__main__":
    raise SystemExit(main())
