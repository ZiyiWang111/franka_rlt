"""Wrist-only coarse ACT inference with vanilla actions and a measured Z stop gate."""
from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import select
import sys
import termios
import threading
import time
import tty
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation

from act_rlt.infer import (
    MAX_INFERENCE_S,
    _predict_full_action_chunk,
    bounded_action,
    checkpoint_camera_shapes,
    checkpoint_state_mode,
    policy_observation,
    prompt_choice,
    resolve_checkpoint,
    robot_camera_kwargs,
)
from evo_franka.kinematics import Q_MAX, Q_MIN
from act_rlt.infer_gc import EpisodeGarbageCollection
from act_rlt.coarse_z_gate import (
    ZGateConfig, add_z_gate_arguments, read_gate_pose, validate_z_gate_arguments,
)

CHUNK_SIZE = 10
JOINT_MARGIN_RAD = 0.01
# Actual first-frame joints from episode 8, the densest start neighbourhood
# (10/266 episodes within 0.005 rad in every joint). See coarse_start_joints.json.
COARSE_START_JOINTS = (
    -0.20299243927001953, -0.007241427432745695, 0.34285637736320496,
    -1.4956223964691162, -0.11038023233413696, 1.5422743558883667,
    2.5215561389923096,
)


def return_to_dataset_start(robot, args):
    """Return all model modes to the same measured dataset joint configuration."""
    if args.dry_run:
        print("Dry run: reset motion disabled.", flush=True)
        return False
    target = np.asarray(COARSE_START_JOINTS, dtype=float)
    if not np.all((target >= Q_MIN + JOINT_MARGIN_RAD) & (target <= Q_MAX - JOINT_MARGIN_RAD)):
        raise ValueError("Coarse start joints exceed the configured joint limits")
    print(f"Coarse start joints [rad]: {target.tolist()}", flush=True)
    robot.robot.move_joint(target.tolist(), speed=float(args.reset_joint_speed))
    robot.resync_command_pose()
    # IPC move completion and the state stream are separate channels. Wait for
    # post-completion snapshots before validating, rather than reading the
    # pre-move last-value cache. In-process controllers read the driver directly.
    read_after_move = getattr(robot.robot, "get_joint_angles_after_move", None)
    if callable(read_after_move):
        print("Waiting for fresh post-reset joint state...", flush=True)
        measured = np.asarray(read_after_move(timeout_s=2.0), dtype=float)
    else:
        measured = np.asarray(robot.robot.get_joint_angles(), dtype=float)
    if measured.shape != (7,) or not np.isfinite(measured).all():
        raise RuntimeError(f"Invalid measured joints after coarse reset: {measured}")
    if np.max(np.abs(measured - target)) > JOINT_MARGIN_RAD:
        raise RuntimeError(f"Coarse start joints not reached: {measured.tolist()}")
    print(f"Reached coarse start joints [rad]: {measured.tolist()}", flush=True)
    return True


def report_final_tcp(robot, *, dry_run, z_gate_config=None):
    """Print measured endpoint Z and intrinsic XYZ Euler angles, not action deltas."""
    pose = (read_gate_pose(robot, z_gate_config) if z_gate_config is not None else
            np.asarray(robot.robot.get_tool_pose(), dtype=float))
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("Invalid measured TCP pose at episode end")
    euler = Rotation.from_rotvec(pose[3:]).as_euler("XYZ", degrees=True)
    final = dict(z_m=float(pose[2]), euler_xyz_deg=euler.tolist(), rotvec_rad=pose[3:].tolist(),
                 euler_convention="intrinsic_XYZ", source="latest_streamed_state")
    label = "Dry-run end observation" if dry_run else "Stopped TCP"
    print(f"{label}: Z={final['z_m']:.6f} m ({final['z_m'] * 1000:.3f} mm) | "
          f"Euler intrinsic XYZ [deg]=[{euler[0]:.3f}, {euler[1]:.3f}, {euler[2]:.3f}]",
          flush=True)
    return final


def checkpoint_action_mode(config, checkpoint, requested="auto"):
    """Seven outputs alone cannot distinguish joint targets from TCP + gripper."""
    if config.type != "act" or config.chunk_size != CHUNK_SIZE:
        raise ValueError("Coarse inference requires an ACT checkpoint with chunk_size=10")
    cameras = checkpoint_camera_shapes(config)
    if cameras != {"wrist": (480, 640, 3)}:
        raise ValueError("Coarse inference requires only 640x480 wrist RGB")
    state = checkpoint_state_mode(config)
    output = tuple(config.output_features["action"].shape)
    if requested == "auto":
        training_file = Path(checkpoint) / "train_config.json"
        training = json.loads(training_file.read_text()) if training_file.is_file() else {}
        identity = " ".join(str(value) for value in (
            training.get("job_name", ""), training.get("dataset", {}).get("repo_id", ""),
        ))
        if "act_coarse_1020_rgb_joint_abs_joint" in identity:
            requested = "joint_absolute"
        elif any(name in identity for name in (
            "act_coarse_1020_rgb_z_delta_tcp", "coarse_pose_joint_hold1s", "coarse_z_hold1s",
        )):
            requested = "tcp_delta"
        else:
            raise ValueError("Unknown action semantics; explicitly set --action-mode joint_absolute or tcp_delta")
    expected = {"joint_absolute": {("joints", (7,))},
                "tcp_delta": {("tcp_z", (6,)), ("joints_pose", (6,))}}
    if requested not in expected or (state, output) not in expected[requested]:
        raise ValueError(f"{requested} does not match checkpoint state={state}, action={output}")
    # Training saved n_action_steps=4; use the complete model prediction instead.
    config.n_action_steps = CHUNK_SIZE
    config.temporal_ensemble_coeff = None
    return requested, cameras


def bounded_joint_target(predicted, measured, previous, max_step):
    """Keep absolute targets inside FR3 limits and near both measured/last commands."""
    predicted, measured = (np.asarray(v, dtype=float) for v in (predicted, measured))
    if any(v.shape != (7,) or not np.isfinite(v).all() for v in (predicted, measured)):
        raise ValueError("Expected finite seven-dimensional joint positions")
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("Joint step limit must be finite and positive")
    lower, upper = Q_MIN + JOINT_MARGIN_RAD, Q_MAX - JOINT_MARGIN_RAD
    if np.any(predicted < lower) or np.any(predicted > upper):
        raise ValueError("Predicted absolute joint target exceeds FR3 joint limits")
    anchor = measured if previous is None else np.asarray(previous, dtype=float)
    if anchor.shape != (7,) or not np.isfinite(anchor).all():
        raise ValueError("Invalid previous joint command")
    lower = np.maximum(lower, np.maximum(measured - max_step, anchor - max_step))
    upper = np.minimum(upper, np.minimum(measured + max_step, anchor + max_step))
    if np.any(lower > upper):
        raise RuntimeError("Measured joints diverged from the previous command")
    return np.clip(predicted, lower, upper)


class CoarseTemporalEnsembler:
    """Fuse predictions for one control step, with older forecasts weighted first."""

    def __init__(self, action_dim, coefficient):
        if action_dim not in (6, 7) or not np.isfinite(coefficient) or coefficient < 0:
            raise ValueError("Invalid temporal ensemble dimension or coefficient")
        self.action_dim = action_dim
        self.coefficient = coefficient
        self.chunks = {}

    def add(self, start_step, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (CHUNK_SIZE, self.action_dim) or not np.isfinite(values).all():
            raise ValueError("Expected finite full action chunk for temporal ensemble")
        self.chunks[start_step] = values.copy()

    def action_for(self, step):
        self.chunks = {start: values for start, values in self.chunks.items()
                       if start + CHUNK_SIZE > step}
        starts = sorted(start for start in self.chunks if start <= step)
        if not starts:
            raise RuntimeError(f"No temporal-ensemble prediction covers control step {step}; stopping")
        log_weights = -self.coefficient * np.arange(len(starts), dtype=float)
        weights = np.exp(log_weights - np.logaddexp.reduce(log_weights))
        values = np.stack([self.chunks[start][step - start] for start in starts])
        contributors = [dict(start_step=start, offset=step - start, weight=float(weight))
                        for start, weight in zip(starts, weights)]
        return np.sum(values * weights[:, None], axis=0), contributors


class EpisodeControls(AbstractContextManager):
    """Poll operator keys between commands and during fixed-rate waits."""
    def __enter__(self):
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self.settings = termios.tcgetattr(self.fd) if self.fd is not None else None
        if self.fd is not None:
            tty.setcbreak(self.fd)
        return self

    def poll(self):
        while self.fd is not None and select.select([self.fd], [], [], 0)[0]:
            key = os.read(self.fd, 1).lower()
            if not key or key in (b"x", b"q", b"\x1b"):
                return "operator_stop"
            if key == b"s":
                return "operator_success"
            if key == b"\x03":
                raise KeyboardInterrupt
        return None

    def wait_until(self, deadline):
        while True:
            reason = self.poll()
            if reason:
                return reason
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(remaining, 0.02))

    def __exit__(self, *exc):
        if self.settings is not None:
            termios.tcsetattr(self.fd, termios.TCSANOW, self.settings)
            termios.tcflush(self.fd, termios.TCIFLUSH)


class ChunkPipeline:
    """Produce full chunks using continuous or request-time observations."""

    def __init__(self, robot, policy, pre, post, args, mode, cameras, torch, emit):
        self.robot, self.policy, self.pre, self.post = robot, policy, pre, post
        self.args, self.mode, self.cameras, self.torch = args, mode, cameras, torch
        self.rtc = getattr(args, "rtc", True)
        self.temporal_ensemble = getattr(args, "temporal_ensemble", False)
        self.emit = emit
        self.stopping = threading.Event()
        self.observation_ready = threading.Condition()
        self.observation = None
        self.sequence = 0
        self.captured_at = None
        self.requests = queue.Queue(maxsize=1)
        self.ready = queue.Queue(maxsize=CHUNK_SIZE if self.temporal_ensemble else 1)
        self.errors = queue.SimpleQueue()
        self.threads = []

    def fail(self, error):
        self.errors.put(error)
        self.stopping.set()
        with self.observation_ready:
            self.observation_ready.notify_all()

    def check(self):
        try:
            error = self.errors.get_nowait()
        except queue.Empty:
            return
        raise error

    def start(self):
        workers = [(self.predict, "coarse-inference")]
        if self.rtc:
            workers.insert(0, (self.capture, "coarse-camera"))
        for target, name in workers:
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self.threads.append(thread)

    def request(self, index, step):
        self.check()
        if self.temporal_ensemble:
            # Keep the newest pending request when the model is slower than
            # control; never interrupt the prediction already in progress.
            try:
                skipped_index, skipped_step = self.requests.get_nowait()
            except queue.Empty:
                pass
            else:
                self.emit("refill_superseded", chunk=skipped_index, step=skipped_step)
        self.requests.put_nowait((index, step))
        self.emit("refill_requested", chunk=index, step=step)

    def capture(self):
        try:
            # RealSense is a 30 Hz stream. Rate limiting also avoids spinning
            # on a cached/synthetic observation source.
            period = 1 / max(30, self.args.fps)
            while not self.stopping.is_set():
                started = time.monotonic()
                observation = policy_observation(self.robot, self.policy.config)
                with self.observation_ready:
                    self.observation = observation
                    self.sequence += 1
                    self.captured_at = time.monotonic()
                    self.observation_ready.notify_all()
                self.stopping.wait(max(0, period - (time.monotonic() - started)))
        except BaseException as error:
            self.fail(error)

    def predict(self):
        last_sequence = 0
        try:
            # inference_mode is thread-local; enter it in the model worker.
            with self.torch.inference_mode():
                while not self.stopping.is_set():
                    try:
                        index, requested_at_step = self.requests.get(timeout=0.02)
                    except queue.Empty:
                        continue
                    if self.rtc:
                        with self.observation_ready:
                            available = self.observation_ready.wait_for(
                                lambda: self.stopping.is_set() or self.sequence > last_sequence,
                                timeout=MAX_INFERENCE_S,
                            )
                            if self.stopping.is_set():
                                return
                            if not available:
                                raise TimeoutError("No fresh wrist observation for coarse inference")
                            observation = self.observation
                            sequence = self.sequence
                            captured_at = self.captured_at
                    else:
                        # The request is issued after all ten control periods.
                        # Start a new observation read here; do not consume a
                        # cached frame captured while the previous chunk ran.
                        capture_started = time.monotonic()
                        observation = policy_observation(self.robot, self.policy.config)
                        captured_at = time.monotonic()
                        if captured_at - capture_started > MAX_INFERENCE_S:
                            raise TimeoutError("Post-chunk wrist observation took too long")
                        self.sequence += 1
                        sequence = self.sequence
                        self.emit("observation_captured", chunk=index,
                                  requested_at_step=requested_at_step, observation_seq=sequence,
                                  capture_s=captured_at - capture_started)
                    if self.stopping.is_set():
                        return
                    last_sequence = sequence
                    started = time.monotonic()
                    age = started - captured_at
                    if age > MAX_INFERENCE_S:
                        raise RuntimeError(f"Wrist observation is {age:.3f}s old; refusing stale chunk")
                    self.emit("inference_started", chunk=index, requested_at_step=requested_at_step,
                              observation_seq=sequence, observation_age_s=age)
                    values = _predict_full_action_chunk(
                        self.policy, self.pre, self.post, observation, self.cameras,
                    )
                    expected = (CHUNK_SIZE, 7 if self.mode == "joint_absolute" else 6)
                    if values.shape != expected or not np.isfinite(values).all():
                        raise ValueError(f"Expected finite full action chunk {expected}, got {values.shape}")
                    elapsed = time.monotonic() - started
                    if elapsed > MAX_INFERENCE_S:
                        raise RuntimeError(f"Inference took {elapsed:.3f}s; refusing stale chunk")
                    if self.stopping.is_set():
                        return
                    chunk = dict(index=index, values=values.copy(), inference_s=elapsed,
                                 observation_seq=sequence, captured_at=captured_at,
                                 start_step=requested_at_step)
                    self.ready.put_nowait(chunk)
                    self.emit("chunk_ready", chunk=index, observation_seq=sequence, inference_s=elapsed,
                              start_step=requested_at_step, predicted=values.tolist())
        except BaseException as error:
            self.fail(error)

    def take(self, index):
        self.check()
        try:
            chunk = self.ready.get_nowait()
        except queue.Empty:
            raise RuntimeError("Coarse action queue underrun at chunk boundary; stopping") from None
        if chunk["index"] != index:
            raise RuntimeError("Coarse action chunk arrived out of order")
        return chunk

    def close(self):
        self.stopping.set()
        with self.observation_ready:
            self.observation_ready.notify_all()
        for thread in self.threads:
            thread.join()


def run_episode(robot, policy, pre, post, args, mode, cameras, torch, controls, trace=None):
    """Run prefetched/sequential chunks or overlapping temporal predictions."""
    with EpisodeGarbageCollection():
        return _run_episode(robot, policy, pre, post, args, mode, cameras, torch, controls, trace)


def _run_episode(robot, policy, pre, post, args, mode, cameras, torch, controls, trace):
    policy.reset()
    robot.resync_command_pose()
    started = time.monotonic()
    chunks = steps = 0
    stop_acknowledged = False
    previous_joint = None
    reason = "duration"
    z_gate_config = ZGateConfig.from_args(args)
    rtc = getattr(args, "rtc", True)
    temporal = getattr(args, "temporal_ensemble", False)
    if temporal and not rtc:
        raise ValueError("--temporal-ensemble requires RTC; remove --no-rtc")
    coefficient = getattr(args, "temporal_ensemble_coeff", .01)
    ensemble = CoarseTemporalEnsembler(7 if mode == "joint_absolute" else 6, coefficient) if temporal else None
    prefetch_steps = args.prefetch_steps if rtc and not temporal else 0
    last_inference_s = 0.0
    events = queue.SimpleQueue()
    trace_errors = queue.SimpleQueue()

    def write_trace():
        try:
            while True:
                event = events.get()
                if event is None:
                    return
                trace.write(json.dumps(event, allow_nan=False) + "\n")
                trace.flush()
        except BaseException as error:
            trace_errors.put(error)

    def emit(event, **fields):
        if trace is not None:
            events.put(dict(event=event, elapsed_s=time.monotonic() - started, **fields))

    logger = threading.Thread(target=write_trace, name="coarse-trace", daemon=True) if trace is not None else None
    if logger is not None:
        logger.start()
    pipeline = ChunkPipeline(robot, policy, pre, post, args, mode, cameras, torch, emit)

    def check_workers():
        pipeline.check()
        try:
            error = trace_errors.get_nowait()
        except queue.Empty:
            return
        raise error

    def stop_policy_if_z_reached():
        if not z_gate_config.enabled or args.dry_run:
            return None
        pose = read_gate_pose(robot, z_gate_config)
        # A one-sided threshold catches crossings and starts below the gate.
        if pose[2] > z_gate_config.z_m:
            return None
        pipeline.stopping.set()
        emit("z_gate_reached", step=steps, measured_z_m=float(pose[2]),
             z_gate_m=z_gate_config.z_m)
        print(f"Z gate reached: Z={pose[2]:.6f} m <= {z_gate_config.z_m:.6f} m; "
              "stopping inference and all servo motion.", flush=True)
        return "z_gate"

    def wait_until(deadline):
        # Keep operator/error handling responsive even at low control rates.
        while True:
            check_workers()
            result = controls.wait_until(min(deadline, time.monotonic() + 0.02))
            if result:
                return result
            check_workers()
            result = stop_policy_if_z_reached()
            if result:
                return result
            if time.monotonic() >= deadline:
                return None

    def finished():
        if args.chunks is not None and chunks >= args.chunks:
            return "chunks"
        if chunks and time.monotonic() - started >= args.duration:
            return "duration"
        return None

    def wait_for_prediction():
        timeout = time.monotonic() + 2 * MAX_INFERENCE_S
        while pipeline.ready.empty():
            check_workers()
            result = controls.poll()
            if result:
                return result
            result = stop_policy_if_z_reached()
            if result:
                return result
            if time.monotonic() > timeout:
                raise TimeoutError("Timed out waiting for a coarse action chunk")
            pipeline.stopping.wait(0.005)
        check_workers()
        return None

    emit("episode_start", action_mode=mode, chunk_size=CHUNK_SIZE, fps=args.fps,
         dry_run=args.dry_run, checkpoint=str(args.checkpoint), prefetch_steps=prefetch_steps,
         rtc_enabled=rtc, camera_mode="continuous_latest" if rtc else "post_chunk",
         temporal_ensemble=temporal, temporal_ensemble_coeff=coefficient if temporal else None,
         z_gate=dict(enabled=z_gate_config.enabled, z_m=z_gate_config.z_m,
                     state_max_age_s=z_gate_config.state_max_age_s))
    try:
        reason = controls.poll() or stop_policy_if_z_reached()
        if reason is None:
            pipeline.start()
            pipeline.request(0, 0)
            reason = wait_for_prediction()
        if reason and reason != "z_gate":
            return dict(reason=reason, chunks=chunks, actions=steps)
        deadline = time.monotonic()
        while reason is None:
            check_workers()
            reason = controls.poll()
            if reason:
                break
            reason = stop_policy_if_z_reached() or finished()
            if reason:
                break
            if not rtc and chunks:
                pipeline.request(chunks, steps)
                reason = wait_for_prediction() or finished()
                if reason:
                    break
                # Observation and inference add a deliberate boundary gap.
                # Restart the schedule without catch-up commands.
                deadline = time.monotonic()
            if temporal:
                values = [None] * CHUNK_SIZE
            else:
                chunk = pipeline.take(chunks)
                values = chunk["values"]
                last_inference_s = chunk["inference_s"]
                age = time.monotonic() - chunk["captured_at"]
                if age > prefetch_steps / args.fps + MAX_INFERENCE_S:
                    raise RuntimeError(f"Prefetched coarse chunk observation is {age:.3f}s old")
                emit("chunk", chunk=chunks, inference_s=chunk["inference_s"],
                     observation_seq=chunk["observation_seq"], observation_age_s=age)
            for index, raw in enumerate(values):
                reason = wait_until(deadline)
                if reason:
                    break
                contributors = None
                if temporal:
                    while True:
                        try:
                            prediction = pipeline.ready.get_nowait()
                        except queue.Empty:
                            break
                        age = time.monotonic() - prediction["captured_at"]
                        if age > MAX_INFERENCE_S:
                            raise RuntimeError(f"Temporal prediction observation is {age:.3f}s old")
                        predicted = prediction["values"]
                        if mode == "joint_absolute" and (
                            np.any(predicted < Q_MIN + JOINT_MARGIN_RAD)
                            or np.any(predicted > Q_MAX - JOINT_MARGIN_RAD)
                        ):
                            raise ValueError("Predicted absolute joint target exceeds FR3 joint limits")
                        ensemble.add(prediction["start_step"], predicted)
                        last_inference_s = prediction["inference_s"]
                        emit("prediction_accepted", prediction=prediction["index"],
                             start_step=prediction["start_step"], accepted_at_step=steps,
                             observation_age_s=age)
                    raw, contributors = ensemble.action_for(steps)
                if mode == "joint_absolute":
                    measured = robot.robot.get_joint_angles()
                    target = bounded_joint_target(raw, measured, previous_joint, args.max_joint_step_rad)
                    reason = controls.poll() or stop_policy_if_z_reached()
                    if reason:
                        break
                    check_workers()
                    if not args.dry_run:
                        robot.robot.servo_joint(target.tolist())
                    previous_joint = target
                    command = target.tolist()
                else:
                    command = bounded_action(raw, args.max_step_m, args.max_step_rad)
                    check_workers()
                    reason = controls.poll() or stop_policy_if_z_reached()
                    if reason:
                        break
                    if not args.dry_run:
                        robot.send_action(command)
                emit("action", chunk=chunks, index=index, step=steps, command=command,
                     sent=not args.dry_run,
                     **(dict(ensemble_action=raw.tolist(), ensemble_votes=len(contributors),
                             ensemble_contributors=contributors) if temporal else {}))
                steps += 1
                if temporal:
                    more_steps = args.chunks is None or steps < args.chunks * CHUNK_SIZE
                    if more_steps and time.monotonic() - started < args.duration:
                        pipeline.request(steps, steps)
                elif rtc and CHUNK_SIZE - index - 1 == prefetch_steps:
                    more_chunks = args.chunks is None or chunks + 1 < args.chunks
                    if more_chunks and time.monotonic() - started < args.duration:
                        pipeline.request(chunks + 1, steps)
                deadline = max(deadline + 1 / args.fps, time.monotonic() + 1 / args.fps)
            if reason:
                break
            chunks += 1
            print(f"Coarse {mode}: chunks={chunks}, actions={steps}, "
                  f"inference={last_inference_s * 1000:.1f}ms",
                  flush=True)
            reason = wait_until(deadline)
            if reason:
                break
    except KeyboardInterrupt:
        reason = "operator_stop"
        raise
    except BaseException as exc:
        reason = "error"
        emit("error", message=str(exc))
        raise
    finally:
        # Cancel production before stopping hardware; join workers only after
        # stop_servo, so slow camera/model shutdown cannot postpone the stop.
        pipeline.stopping.set()
        try:
            if not args.dry_run and robot.is_connected:
                if not robot.robot.stop_servo():
                    reason = "error"
                    raise RuntimeError("Control server did not acknowledge stop_servo")
                stop_acknowledged = True
        finally:
            try:
                pipeline.close()
                robot.resync_command_pose()
                policy.reset()
            finally:
                final_tcp = None
                final_tcp_error = None
                if robot.is_connected and (args.dry_run or stop_acknowledged):
                    try:
                        final_tcp = report_final_tcp(robot, dry_run=args.dry_run,
                                                     z_gate_config=z_gate_config if z_gate_config.enabled else None)
                    except Exception as exc:
                        final_tcp_error = str(exc)
                else:
                    final_tcp_error = "robot disconnected" if not robot.is_connected else "stop_servo not acknowledged"
                if final_tcp_error:
                    print(f"End TCP unavailable: {final_tcp_error}", flush=True)
                emit("episode_end", reason=reason, chunks=chunks, actions=steps,
                     final_tcp=final_tcp, final_tcp_error=final_tcp_error,
                     stop_acknowledged=stop_acknowledged)
                if logger is not None:
                    events.put(None)
                    logger.join()
                    try:
                        error = trace_errors.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        raise error
    result = dict(reason=reason, chunks=chunks, actions=steps)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--action-mode", choices=("auto", "joint_absolute", "tcp_delta"), default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--duration", type=float, default=10,
                        help="Model execution budget at chunk boundaries; the Z gate can stop earlier")
    parser.add_argument("--chunks", type=int,
                        help="Model execution limit in full chunks; the Z gate can stop mid-chunk")
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=True,
                        help="Prefetch the next chunk (default); --no-rtc observes and predicts only after all ten steps")
    parser.add_argument("--temporal-ensemble", action=argparse.BooleanOptionalAction, default=False,
                        help="Fuse overlapping ten-step forecasts; requests a new forecast each control step (requires RTC)")
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=.01,
                        help="Exponential decay by prediction rank, oldest first; 0 gives equal weights (default: 0.01)")
    parser.add_argument("--prefetch-steps", type=int, default=2,
                        help="With RTC, predict when this many current actions remain (1..9); ignored with --no-rtc or temporal ensemble")
    parser.add_argument("--robot-ip", default=os.environ.get("ROBOT_IP", "172.16.0.2"))
    parser.add_argument("--wrist-camera", default="349622072679")
    parser.add_argument("--front-camera", default="233522075778", help=argparse.SUPPRESS)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.04)
    parser.add_argument("--max-step-m", type=float, default=0.01)
    parser.add_argument("--max-step-rad", type=float, default=0.05)
    parser.add_argument("--reset-joint-speed", type=float, default=0.5,
                        help="Joint velocity limit for returning to the fixed start, rad/s before dynamics scaling (default: 0.5)")
    parser.add_argument("--dry-run", action="store_true", help="Live camera inference; send no motion commands")
    parser.add_argument("--dry-run-episodes", type=int, help="Automatically run N dry-run episodes")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/act_coarse_infer"))
    add_z_gate_arguments(parser)
    return parser


def validate_args(parser, args):
    validate_z_gate_arguments(parser, args)
    for name in ("fps", "duration", "max_joint_step_rad", "max_step_m", "max_step_rad", "reset_joint_speed"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be finite and positive")
    if args.fps > 30:
        parser.error("fps must be <= 30 (training used 15 Hz)")
    if args.chunks is not None and args.chunks < 1:
        parser.error("chunks must be positive")
    if not 1 <= args.prefetch_steps < CHUNK_SIZE:
        parser.error("prefetch-steps must be between 1 and 9")
    if not np.isfinite(args.temporal_ensemble_coeff) or args.temporal_ensemble_coeff < 0:
        parser.error("temporal-ensemble-coeff must be finite and non-negative")
    if args.temporal_ensemble and not args.rtc:
        parser.error("--temporal-ensemble requires RTC; remove --no-rtc")
    if args.dry_run_episodes is not None and (not args.dry_run or args.dry_run_episodes < 1):
        parser.error("dry-run-episodes requires --dry-run and a positive count")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from evo_rlt.adapters.lerobot.franka_robot import FrankaRobot, FrankaRobotConfig

    checkpoint = resolve_checkpoint(args.checkpoint)
    config = PreTrainedConfig.from_pretrained(str(checkpoint), local_files_only=True)
    mode, cameras = checkpoint_action_mode(config, checkpoint, args.action_mode)
    config.device = args.device
    policy = ACTPolicy.from_pretrained(str(checkpoint), config=config, local_files_only=True).to(args.device).eval()
    pre, post = make_pre_post_processors(
        config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    robot = FrankaRobot(FrankaRobotConfig(
        robot_ip=args.robot_ip, **robot_camera_kwargs(cameras, args), enable_fci_keepalive=False,
    ))
    print(f"Checkpoint: {checkpoint}\nAction mode: {mode}; state={checkpoint_state_mode(config)}; wrist 640x480; "
          f"prediction horizon=10; {args.fps:g} Hz; rtc={args.rtc}; "
          f"prefetch_steps={args.prefetch_steps if args.rtc and not args.temporal_ensemble else 0}; "
          f"temporal ensemble={args.temporal_ensemble}; "
          f"ensemble coefficient={args.temporal_ensemble_coeff if args.temporal_ensemble else None}; "
          f"dry_run={args.dry_run}", flush=True)
    z_gate_config = ZGateConfig.from_args(args)
    print(f"Z gate: enabled={z_gate_config.enabled}; stop at measured "
          f"Z <= {z_gate_config.z_m:.6f} m; vanilla model actions until stop", flush=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    try:
        robot.connect()
        # Warm up without issuing any commands, before operator confirmation.
        with torch.inference_mode():
            _predict_full_action_chunk(policy, pre, post, policy_observation(robot, config), cameras)
        policy.reset()
        at_start = False
        episode = 0
        while args.dry_run_episodes is None or episode < args.dry_run_episodes:
            if args.dry_run_episodes is None:
                decision = prompt_choice(
                    ("At dataset start: Enter=start inference, r=return directly to dataset start, q=quit: "
                     if at_start or args.dry_run else
                     "Enter/r=move directly to dataset start, q=quit: "),
                    {"": "start", "s": "start", "r": "return", "q": "quit"},
                )
                if decision == "quit":
                    break
                if decision == "return":
                    at_start = return_to_dataset_start(robot, args)
                    continue
                if not args.dry_run and not at_start:
                    at_start = return_to_dataset_start(robot, args)
                    continue
            episode += 1
            log_path = args.log_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}_{episode}.jsonl"
            print(f"s=success/stop; x, q or Esc=stop; Ctrl+C=quit. Log: {log_path}", flush=True)
            with log_path.open("x") as trace, EpisodeControls() as controls:
                result = run_episode(robot, policy, pre, post, args, mode, cameras, torch, controls, trace)
            print(f"Episode {episode}: {result}", flush=True)
            at_start = False
    except KeyboardInterrupt:
        print("Stopped by operator.", flush=True)
    finally:
        policy.reset()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
