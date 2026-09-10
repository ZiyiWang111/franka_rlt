#!/usr/bin/env python3
"""Robot-host client for split FR3 hardware and remote π0.5 inference."""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evo_rlt.adapters.lerobot.franka_remote.execution import (
    GripperStateMachine,
    RelativeForceGuard,
    TcpActionFilter,
)
from evo_rlt.adapters.lerobot.franka_remote.protocol import (
    ACTION_CHUNK_SHAPE,
    ACTION_NAMES,
    InferenceResponse,
    decode_response,
    encode_inference_request,
    encode_ping,
)
from evo_rlt.adapters.lerobot.franka_remote.state import observation_image, observation_to_state15


DEFAULT_TASK = "pick up the vga and insert into the motherboard"
DEFAULT_EXPECTED_CHECKPOINT = (
    "/workspace/wangziyi/projects/franka_rlt/outputs/"
    "fr3_pi05_sft_60ep_state15_action7_bs4_30k/checkpoints/030000/pretrained_model"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, help="For example tcp://10.0.0.20:5559")
    parser.add_argument(
        "--network-only",
        action="store_true",
        help="Ping the model server; do not import hardware",
    )
    parser.add_argument(
        "--mode",
        choices=("shadow", "gripper-only", "arm-only", "integrated"),
        default="shadow",
    )
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--robotlab-path", default="/home/embint/robotLab")
    parser.add_argument("--wrist-camera-serial", default="349622072679")
    parser.add_argument("--front-camera-serial", default="233522075778")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=5,
        help="Synchronous steps per chunk, or async usable-prefix/fallback horizon",
    )
    parser.add_argument("--max-cycles", type=int, default=1, help="0 means run until interrupted")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--max-action-age-s", type=float, default=1.0)
    parser.add_argument("--expected-checkpoint", default=DEFAULT_EXPECTED_CHECKPOINT)
    parser.add_argument(
        "--max-translation-m",
        type=float,
        default=0.005,
        help="Maximum TCP translation per control step (speed limit = value * fps)",
    )
    parser.add_argument(
        "--max-rotation-rad",
        type=float,
        default=0.03,
        help="Maximum TCP rotation per control step (speed limit = value * fps)",
    )
    parser.add_argument("--action-low-pass-alpha", type=float, default=0.2)
    parser.add_argument("--max-translation-accel-m-s2", type=float, default=0.30)
    parser.add_argument("--max-rotation-accel-rad-s2", type=float, default=0.75)
    parser.add_argument(
        "--force-stop-n",
        type=float,
        default=None,
        help="Stop policy servo when base-frame translational force changes by this magnitude",
    )
    parser.add_argument("--force-baseline-samples", type=int, default=15)
    parser.add_argument("--force-retreat-m", type=float, default=0.05)
    parser.add_argument("--force-retreat-speed-m-s", type=float, default=0.03)
    parser.add_argument(
        "--require-initial-grasp",
        action="store_true",
        help="Refuse all motion unless the first observation reports a grasped object",
    )
    parser.add_argument("--workspace-min", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--workspace-max", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--close-threshold-m", type=float, default=0.040)
    parser.add_argument("--open-threshold-m", type=float, default=0.055)
    parser.add_argument("--gripper-confirm-steps", type=int, default=2)
    parser.add_argument(
        "--async-inference",
        action="store_true",
        help="Overlap observation/inference with action streaming and time-align replacement chunks",
    )
    parser.add_argument(
        "--chunk-blend-steps",
        type=int,
        default=6,
        help="Cross-fade this many time-aligned TCP steps at each async chunk replacement",
    )
    parser.add_argument(
        "--stop-after-gripper-close",
        action="store_true",
        help="Stop the run immediately after the first close command (integrated/gripper-only)",
    )
    parser.add_argument("--log", type=Path, default=None)
    return parser.parse_args()


def import_zmq():
    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError(
            "pyzmq is required; install the project with `pip install -e '.[lerobot]'`"
        ) from exc
    return zmq


class RemoteInferenceClient:
    def __init__(self, endpoint: str, timeout_s: float) -> None:
        self.zmq = import_zmq()
        self.context = self.zmq.Context.instance()
        self.endpoint = endpoint
        self.timeout_ms = int(timeout_s * 1000)
        self.socket = self._new_socket()

    def _new_socket(self):
        socket = self.context.socket(self.zmq.REQ)
        socket.setsockopt(self.zmq.LINGER, 0)
        socket.setsockopt(self.zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(self.zmq.RCVTIMEO, self.timeout_ms)
        socket.connect(self.endpoint)
        return socket

    def _reset_socket(self) -> None:
        self.socket.close(linger=0)
        self.socket = self._new_socket()

    def request(self, frames: list[bytes], request_id: int):
        try:
            self.socket.send_multipart(frames)
            response_frames = self.socket.recv_multipart()
        except self.zmq.Again as exc:
            self._reset_socket()
            raise TimeoutError(f"inference request {request_id} timed out; no action was executed") from exc
        return decode_response(response_frames, request_id)

    def ping(self) -> dict[str, Any]:
        response = self.request(encode_ping(0), 0)
        if not isinstance(response, dict):
            raise RuntimeError("expected ready response")
        return response

    def infer(
        self,
        *,
        request_id: int,
        state: np.ndarray,
        wrist: np.ndarray,
        front: np.ndarray,
        task: str,
    ) -> InferenceResponse:
        frames = encode_inference_request(
            request_id=request_id,
            timestamp_ns=time.time_ns(),
            task=task,
            state=state,
            wrist=wrist,
            front=front,
        )
        response = self.request(frames, request_id)
        if not isinstance(response, InferenceResponse):
            raise RuntimeError("expected inference result response")
        return response

    def close(self) -> None:
        self.socket.close(linger=0)


@dataclass(frozen=True)
class AsyncInferenceResult:
    request_id: int
    submitted_tick: int
    observation_started: float
    state: np.ndarray | None = None
    response: InferenceResponse | None = None
    error: BaseException | None = None


class AsyncInferenceWorker:
    """Single-owner inference socket plus background observation capture.

    The robotLab command client already serializes its DEALER socket, while arm
    state comes from its thread-safe PUSH cache.  Keeping camera reads and the
    inference REQ socket in this one worker lets the main thread maintain the
    30 Hz servo stream without sharing either ZMQ inference sockets or cameras.
    """

    def __init__(self, *, robot: Any, endpoint: str, timeout_s: float, task: str) -> None:
        self.robot = robot
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.task = task
        self._jobs: queue.Queue[tuple[int, int] | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[AsyncInferenceResult] = queue.Queue(maxsize=1)
        self._in_flight = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="fr3-inference")
        self._thread.start()

    @property
    def in_flight(self) -> bool:
        return self._in_flight

    def submit(self, *, request_id: int, submitted_tick: int) -> None:
        if self._in_flight:
            raise RuntimeError("an asynchronous inference request is already in flight")
        self._in_flight = True
        self._jobs.put_nowait((request_id, submitted_tick))

    def _run(self) -> None:
        client = RemoteInferenceClient(self.endpoint, self.timeout_s)
        try:
            while True:
                job = self._jobs.get()
                if job is None:
                    return
                request_id, submitted_tick = job
                observation_started = time.perf_counter()
                try:
                    observation = self.robot.get_observation()
                    state = observation_to_state15(observation)
                    response = client.infer(
                        request_id=request_id,
                        state=state,
                        wrist=observation_image(observation, "wrist"),
                        front=observation_image(observation, "front"),
                        task=self.task,
                    )
                    result = AsyncInferenceResult(
                        request_id=request_id,
                        submitted_tick=submitted_tick,
                        observation_started=observation_started,
                        state=state,
                        response=response,
                    )
                except BaseException as exc:  # surfaced on the motion thread before plan activation
                    result = AsyncInferenceResult(
                        request_id=request_id,
                        submitted_tick=submitted_tick,
                        observation_started=observation_started,
                        error=exc,
                    )
                self._results.put(result)
        finally:
            client.close()

    def _unwrap(self, result: AsyncInferenceResult) -> AsyncInferenceResult:
        self._in_flight = False
        if result.error is not None:
            raise result.error
        return result

    def poll(self) -> AsyncInferenceResult | None:
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            return None
        return self._unwrap(result)

    def wait(self) -> AsyncInferenceResult:
        # Two 1 s camera frame bounds plus the configured network timeout.
        try:
            result = self._results.get(timeout=self.timeout_s + 3.0)
        except queue.Empty as exc:
            raise TimeoutError("asynchronous observation/inference worker timed out") from exc
        return self._unwrap(result)

    def close(self) -> None:
        # A queued sentinel is consumed after any bounded in-flight request.
        self._jobs.put(None)
        self._thread.join(timeout=self.timeout_s + 4.0)
        if self._thread.is_alive():
            raise RuntimeError("asynchronous inference worker did not stop cleanly")


def sleep_control_period(started: float, fps: float) -> None:
    remaining = 1.0 / fps - (time.perf_counter() - started)
    if remaining > 0:
        time.sleep(remaining)


def require_expected_checkpoint(actual: str, expected: str) -> None:
    if actual != expected:
        raise RuntimeError(f"checkpoint mismatch: server={actual!r}, expected={expected!r}")


def execute_action_step(
    *,
    robot: Any,
    action: np.ndarray,
    step_index: int,
    mode: str,
    gripper: GripperStateMachine,
    action_filter: TcpActionFilter,
) -> tuple[dict[str, Any], bool]:
    arm_enabled = mode in {"arm-only", "integrated"}
    gripper_enabled = mode in {"gripper-only", "integrated"}
    record: dict[str, Any] = {"step": step_index, "predicted": action.tolist()}
    decision = gripper.update(float(action[6])) if gripper_enabled else None
    replan = False
    if decision is not None and decision.hold_arm:
        action_filter.reset()
        record["arm"] = "held_for_gripper_transition"
        if decision.command is not None:
            transition_prepared = False
            if arm_enabled:
                robot.prepare_gripper_transition()
                transition_prepared = True
                record["arm_servo_stopped"] = True
            try:
                if decision.command == "close":
                    target_width_m = float(action[6])
                    record["gripper_target_width_m"] = target_width_m
                    record["gripper_result"] = bool(robot.close_gripper(width=target_width_m))
                else:
                    open_result = robot.open_gripper()
                    record["gripper_result"] = None if open_result is None else bool(open_result)
            finally:
                if transition_prepared:
                    robot.finish_gripper_transition()
            gripper.mark_completed(decision.command)
            if hasattr(robot, "resync_command_pose"):
                robot.resync_command_pose()
            record["gripper_command"] = decision.command
            replan = True
    elif arm_enabled:
        tcp_action = action_filter.update(action)
        robot.send_action(tcp_action)
        record["arm"] = tcp_action
    else:
        record["arm"] = "disabled"
    return record, replan


def execute_action_chunk(
    *,
    robot: Any,
    chunk: np.ndarray,
    mode: str,
    execute_steps: int,
    fps: float,
    gripper: GripperStateMachine,
    action_filter: TcpActionFilter,
    action_deadline: float | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    replan = False
    for step_index, action in enumerate(chunk[:execute_steps]):
        if action_deadline is not None and time.perf_counter() > action_deadline:
            raise TimeoutError(
                f"refusing stale action chunk before step {step_index}; no further action was executed"
            )
        started = time.perf_counter()
        record, replan = execute_action_step(
            robot=robot,
            action=action,
            step_index=step_index,
            mode=mode,
            gripper=gripper,
            action_filter=action_filter,
        )
        records.append(record)
        if replan:
            break
        sleep_control_period(started, fps)
    return records, replan


def write_log(stream: Any, payload: dict[str, Any]) -> None:
    if stream is not None:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()


def run_async_control(
    *,
    args: argparse.Namespace,
    robot: Any,
    gripper: GripperStateMachine,
    action_filter: TcpActionFilter,
    log_stream: Any,
    force_guard: RelativeForceGuard | None = None,
) -> None:
    """Continuously servo while the next observation and inference run in parallel."""
    worker = AsyncInferenceWorker(
        robot=robot,
        endpoint=args.server,
        timeout_s=args.timeout_s,
        task=args.task,
    )
    total_ticks = 0
    requests_submitted = 0
    active: dict[str, Any] | None = None

    def may_submit() -> bool:
        return args.max_cycles == 0 or requests_submitted < args.max_cycles

    def submit() -> None:
        nonlocal requests_submitted
        requests_submitted += 1
        worker.submit(request_id=requests_submitted, submitted_tick=total_ticks)

    def flush_active(reason: str) -> None:
        nonlocal active
        if active is None:
            return
        sample: AsyncInferenceResult = active["sample"]
        response = sample.response
        assert response is not None and sample.state is not None
        write_log(
            log_stream,
            {
                "cycle": sample.request_id,
                "time_ns": time.time_ns(),
                "mode": args.mode,
                "async_inference": True,
                "state15": sample.state.tolist(),
                "task": args.task,
                "inference_ms": response.inference_ms,
                "activation_age_ms": active["activation_age_s"] * 1000.0,
                "checkpoint": response.checkpoint,
                "command_filter": action_filter.settings,
                "force_guard": None
                if force_guard is None
                else {
                    "threshold_n": force_guard.threshold_n,
                    "baseline_wrench": force_guard.baseline_wrench.tolist(),
                },
                "submitted_tick": sample.submitted_tick,
                "activated_tick": active["activated_tick"],
                "skipped_steps": active["skipped_steps"],
                "action_chunk": response.action_chunk.tolist(),
                "executed": active["records"],
                "retire_reason": reason,
            },
        )
        active = None

    def activate(sample: AsyncInferenceResult, previous: dict[str, Any] | None = None) -> None:
        nonlocal active
        response = sample.response
        if response is None or sample.state is None:
            raise RuntimeError("asynchronous worker returned an incomplete result")
        require_expected_checkpoint(response.checkpoint, args.expected_checkpoint)
        age_s = time.perf_counter() - sample.observation_started
        if age_s > args.max_action_age_s:
            raise TimeoutError(
                f"refusing stale asynchronous result: age={age_s:.3f}s "
                f"> max={args.max_action_age_s:.3f}s"
            )
        # Actions are a 30 Hz trajectory relative to the sampled observation.
        # If N control ticks ran while inference was pending, action[N] is the
        # time-aligned replacement; replaying action[0] would duplicate motion.
        skipped_steps = max(0, total_ticks - sample.submitted_tick)
        usable_steps = min(args.execute_steps, len(response.action_chunk))
        if skipped_steps >= usable_steps:
            raise TimeoutError(
                f"asynchronous result arrived after {skipped_steps} control ticks, "
                f"outside execute-steps={usable_steps}"
            )
        if sample.request_id == 1 and args.require_initial_grasp:
            width_m = float(sample.state[13])
            grasped = bool(sample.state[14])
            if not grasped:
                raise RuntimeError(
                    "second-stage run requires an initially grasped object; "
                    f"observed width={width_m:.5f}m grasped={grasped}; no action was executed"
                )
        if sample.request_id == 1 and args.stop_after_gripper_close:
            width_m = float(sample.state[13])
            grasped = bool(sample.state[14])
            if grasped or width_m < args.open_threshold_m:
                raise RuntimeError(
                    "stop-after-close run requires an initially open, empty gripper; "
                    f"observed width={width_m:.5f}m grasped={grasped}"
                )
        gripper.sync_from_observation(float(sample.state[13]), bool(sample.state[14]))
        blend = None
        if previous is not None and args.chunk_blend_steps > 0:
            previous_sample: AsyncInferenceResult = previous["sample"]
            previous_response = previous_sample.response
            assert previous_response is not None
            previous_step = max(0, total_ticks - previous_sample.submitted_tick)
            previous_usable_steps = min(args.execute_steps, len(previous_response.action_chunk))
            blend_steps = min(
                args.chunk_blend_steps,
                max(0, previous_usable_steps - previous_step),
            )
            if blend_steps > 0:
                blend = {
                    "sample": previous_sample,
                    "start_step": previous_step,
                    "steps": blend_steps,
                    "progress": 0,
                }
        active = {
            "sample": sample,
            "index": skipped_steps,
            "skipped_steps": skipped_steps,
            "activated_tick": total_ticks,
            "activation_age_s": age_s,
            "records": [],
            "blend": blend,
        }
        first = response.action_chunk[skipped_steps]
        print(
            f"cycle={sample.request_id} inference_ms={response.inference_ms:.1f} "
            f"activation_age_ms={age_s * 1000.0:.1f} skipped_steps={skipped_steps} "
            f"blend_steps={0 if blend is None else blend['steps']} "
            f"tcp_next={np.array2string(first[:6], precision=5)} "
            f"gripper={response.action_chunk[:, 6].min():.5f}.."
            f"{response.action_chunk[:, 6].max():.5f}",
            flush=True,
        )

    submit()
    stop_reason = "completed"
    try:
        initial = worker.wait()
        initial_latency_s = time.perf_counter() - initial.observation_started
        # With one inference worker, the first plan is already L seconds old at
        # activation and its replacement needs roughly another L seconds.  Fail
        # before moving if the configured age budget cannot bridge that seam.
        required_pipeline_budget_s = 2.0 * initial_latency_s + 2.0 / args.fps
        if args.max_action_age_s < required_pipeline_budget_s:
            raise RuntimeError(
                "max-action-age-s is too short for continuous async execution: "
                f"configured={args.max_action_age_s:.3f}s, first_pipeline_latency="
                f"{initial_latency_s:.3f}s, required>={required_pipeline_budget_s:.3f}s; "
                "no action was executed"
            )
        activate(initial)
        if may_submit():
            submit()

        while active is not None:
            started = time.perf_counter()
            if force_guard is not None:
                force_event = force_guard.check(robot.get_external_wrench_base())
                if force_event["triggered"]:
                    record = {
                        "global_tick": total_ticks,
                        "force_guard": force_event,
                        "arm": "force_stop_before_next_action",
                    }
                    active["records"].append(record)
                    print(
                        f"Force guard triggered: delta={force_event['force_delta_n']:.2f}N "
                        f">= {force_event['threshold_n']:.2f}N; stopping and retreating",
                        flush=True,
                    )
                    record["retreat"] = robot.stop_and_retreat_up(
                        distance_m=args.force_retreat_m,
                        speed_m_s=args.force_retreat_speed_m_s,
                    )
                    stop_reason = "force_guard_retreat"
                    break
            replacement = worker.poll()
            if replacement is not None:
                previous = active
                flush_active("superseded_by_blended_chunk")
                activate(replacement, previous=previous)
                if may_submit():
                    submit()

            assert active is not None
            sample = active["sample"]
            if time.perf_counter() - sample.observation_started > args.max_action_age_s:
                raise TimeoutError(
                    "refusing stale active action chunk; no further action was executed"
                )

            response = sample.response
            assert response is not None
            step_index = int(active["index"])
            usable_steps = min(args.execute_steps, len(response.action_chunk))
            if step_index >= usable_steps:
                if worker.in_flight:
                    # The safety choice is to hold rather than replay or extend
                    # a trajectory beyond the explicitly allowed prefix.
                    sleep_control_period(started, args.fps)
                    continue
                stop_reason = "execute_steps_exhausted"
                break

            policy_action = response.action_chunk[step_index]
            execution_action = policy_action
            blend_record = None
            blend = active.get("blend")
            if blend is not None:
                previous_sample = blend["sample"]
                previous_response = previous_sample.response
                assert previous_response is not None
                previous_step = int(blend["start_step"] + blend["progress"])
                progress = int(blend["progress"] + 1)
                blend_steps = int(blend["steps"])
                # Smoothstep gives zero slope at both ends of the cross-fade.
                phase = progress / blend_steps
                alpha = phase * phase * (3.0 - 2.0 * phase)
                execution_action = policy_action.copy()
                execution_action[:6] = (
                    (1.0 - alpha) * previous_response.action_chunk[previous_step, :6]
                    + alpha * policy_action[:6]
                )
                blend_record = {
                    "from_request_id": previous_sample.request_id,
                    "from_step": previous_step,
                    "to_request_id": sample.request_id,
                    "to_step": step_index,
                    "alpha": alpha,
                }
                blend["progress"] = progress
                if progress >= blend_steps:
                    active["blend"] = None

            record, replan = execute_action_step(
                robot=robot,
                action=execution_action,
                step_index=step_index,
                mode=args.mode,
                gripper=gripper,
                action_filter=action_filter,
            )
            if blend_record is not None:
                record["policy_predicted"] = policy_action.tolist()
                record["chunk_blend"] = blend_record
            record["global_tick"] = total_ticks
            active["records"].append(record)
            active["index"] = step_index + 1
            total_ticks += 1

            if record.get("gripper_command") == "close" and args.stop_after_gripper_close:
                result = record.get("gripper_result")
                print(
                    f"Stopping after gripper close: target={record.get('gripper_target_width_m')} "
                    f"grasped={result}",
                    flush=True,
                )
                stop_reason = "stop_after_gripper_close"
                break

            if replan:
                # A blocking Hand command changes the scene and TCP integration
                # anchor. Discard a pre-command result and resume from a fresh
                # post-command observation; a brief hold here is intentional.
                flush_active(f"replan_after_gripper_{record.get('gripper_command')}")
                if worker.in_flight:
                    worker.wait()
                if not may_submit():
                    stop_reason = "max_cycles_after_gripper"
                    break
                submit()
                activate(worker.wait())
                if may_submit():
                    submit()

            sleep_control_period(started, args.fps)
    except BaseException:
        stop_reason = "aborted"
        raise
    finally:
        flush_active(stop_reason)
        worker.close()


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or not 1 <= args.execute_steps <= 50:
        raise ValueError("fps must be positive and execute-steps must be within [1, 50]")
    if args.max_cycles < 0 or args.timeout_s <= 0 or args.max_action_age_s <= 0:
        raise ValueError(
            "max-cycles must be non-negative; timeout-s and max-action-age-s must be positive"
        )
    if args.max_translation_m <= 0 or args.max_rotation_rad <= 0:
        raise ValueError("TCP action limits must be positive")
    if not 0 < args.action_low_pass_alpha <= 1:
        raise ValueError("action-low-pass-alpha must be within (0, 1]")
    if args.max_translation_accel_m_s2 <= 0 or args.max_rotation_accel_rad_s2 <= 0:
        raise ValueError("TCP acceleration limits must be positive")
    if args.force_stop_n is not None and args.force_stop_n <= 0:
        raise ValueError("force-stop-n must be positive")
    if args.force_baseline_samples < 3:
        raise ValueError("force-baseline-samples must be at least 3")
    if args.force_retreat_m <= 0 or args.force_retreat_speed_m_s <= 0:
        raise ValueError("force retreat distance and speed must be positive")
    if args.chunk_blend_steps < 0 or args.chunk_blend_steps > ACTION_CHUNK_SHAPE[0]:
        raise ValueError(
            f"chunk-blend-steps must be within [0, {ACTION_CHUNK_SHAPE[0]}]"
        )
    if args.async_inference and args.mode not in {"arm-only", "integrated"}:
        raise RuntimeError("--async-inference requires arm-only or integrated mode")
    if args.force_stop_n is not None and not args.async_inference:
        raise RuntimeError("--force-stop-n currently requires --async-inference")
    if args.force_stop_n is not None and args.mode not in {"arm-only", "integrated"}:
        raise RuntimeError("--force-stop-n requires arm-only or integrated mode")
    if args.require_initial_grasp and args.mode not in {"arm-only", "integrated"}:
        raise RuntimeError("--require-initial-grasp requires arm-only or integrated mode")
    if args.async_inference:
        policy_horizon_s = ACTION_CHUNK_SHAPE[0] / args.fps
        if args.max_action_age_s > policy_horizon_s:
            raise RuntimeError(
                f"async max-action-age-s={args.max_action_age_s:.3f}s exceeds the "
                f"{policy_horizon_s:.3f}s model action horizon"
            )
    if args.stop_after_gripper_close and args.mode not in {"gripper-only", "integrated"}:
        raise RuntimeError("--stop-after-gripper-close requires gripper-only or integrated mode")
    if args.mode != "shadow" and not args.allow_motion:
        raise RuntimeError(f"mode {args.mode!r} requires the explicit --allow-motion flag")
    if args.mode in {"arm-only", "integrated"}:
        if args.workspace_min is None or args.workspace_max is None:
            raise RuntimeError("arm motion requires explicit --workspace-min X Y Z and --workspace-max X Y Z")

    client = RemoteInferenceClient(args.server, args.timeout_s)
    ready = client.ping()
    require_expected_checkpoint(str(ready["checkpoint"]), args.expected_checkpoint)
    print(
        f"Inference server ready: checkpoint={ready['checkpoint']} "
        f"action_chunk_shape={ready['action_chunk_shape']}",
        flush=True,
    )
    if args.network_only:
        client.close()
        print("Network-only check passed; no hardware module was imported", flush=True)
        return

    os.environ["ROBOTLAB_PATH"] = args.robotlab_path
    from evo_rlt.adapters.lerobot.franka_robot import FrankaRobot, FrankaRobotConfig

    robot = FrankaRobot(
        FrankaRobotConfig(
            robot_ip=args.robot_ip,
            robotlab_path=args.robotlab_path,
            camera_serial=args.wrist_camera_serial,
            front_camera_serial=args.front_camera_serial,
            include_gripper_action=True,
            # A gripper-only run must never issue an arm command.  The FCI
            # keepalive uses servo_joint, so reserve it for arm-enabled modes.
            enable_fci_keepalive=args.mode in {"arm-only", "integrated"},
            workspace_min_xyz=None if args.workspace_min is None else tuple(args.workspace_min),
            workspace_max_xyz=None if args.workspace_max is None else tuple(args.workspace_max),
        )
    )
    gripper = GripperStateMachine(
        close_threshold_m=args.close_threshold_m,
        open_threshold_m=args.open_threshold_m,
        confirm_steps=args.gripper_confirm_steps,
    )
    action_filter = TcpActionFilter(
        fps=args.fps,
        low_pass_alpha=args.action_low_pass_alpha,
        max_translation_step_m=args.max_translation_m,
        max_rotation_step_rad=args.max_rotation_rad,
        max_translation_accel_m_s2=args.max_translation_accel_m_s2,
        max_rotation_accel_rad_s2=args.max_rotation_accel_rad_s2,
    )
    log_stream = None
    if args.log is not None:
        args.log.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        log_stream = args.log.expanduser().resolve().open("a", buffering=1)

    cycle = 0
    try:
        robot.connect()
        print(
            f"Robot connected; mode={args.mode} command_filter={action_filter.settings}",
            flush=True,
        )
        force_guard = None
        if args.force_stop_n is not None:
            samples = []
            for _ in range(args.force_baseline_samples):
                samples.append(robot.get_external_wrench_base())
                time.sleep(1.0 / args.fps)
            sample_array = np.asarray(samples, dtype=np.float64)
            if np.all(sample_array == 0.0):
                raise RuntimeError(
                    "external-wrench stream returned only zeros during calibration; "
                    "force guard is unavailable and no action was executed"
                )
            baseline_wrench = np.median(sample_array, axis=0)
            baseline_noise_n = float(
                np.max(np.linalg.norm(sample_array[:, :3] - baseline_wrench[:3], axis=1))
            )
            if baseline_noise_n >= args.force_stop_n / 2.0:
                raise RuntimeError(
                    "external-wrench baseline is too unstable for the requested guard: "
                    f"max_delta={baseline_noise_n:.2f}N, "
                    f"threshold={args.force_stop_n:.2f}N; no action was executed"
                )
            force_guard = RelativeForceGuard(
                baseline_wrench=baseline_wrench,
                threshold_n=args.force_stop_n,
            )
            print(
                "Force guard calibrated while stationary: "
                f"baseline={np.array2string(baseline_wrench, precision=3)} "
                f"max_baseline_delta={baseline_noise_n:.2f}N "
                f"threshold={args.force_stop_n:.2f}N retreat=+Z{args.force_retreat_m:.3f}m",
                flush=True,
            )
        if args.async_inference:
            client.close()
            run_async_control(
                args=args,
                robot=robot,
                gripper=gripper,
                action_filter=action_filter,
                log_stream=log_stream,
                force_guard=force_guard,
            )
            return
        while args.max_cycles == 0 or cycle < args.max_cycles:
            cycle += 1
            # Age starts before observation capture, so the bound covers robot
            # state, both camera frames, transport, and model inference.
            observation_started = time.perf_counter()
            observation = robot.get_observation()
            state = observation_to_state15(observation)
            wrist = observation_image(observation, "wrist")
            front = observation_image(observation, "front")
            gripper.sync_from_observation(float(state[13]), bool(state[14]))
            response = client.infer(
                request_id=cycle,
                state=state,
                wrist=wrist,
                front=front,
                task=args.task,
            )
            require_expected_checkpoint(response.checkpoint, args.expected_checkpoint)
            action_age_s = time.perf_counter() - observation_started
            if action_age_s > args.max_action_age_s:
                raise TimeoutError(
                    f"refusing stale inference result: age={action_age_s:.3f}s "
                    f"> max={args.max_action_age_s:.3f}s; no action was executed"
                )
            chunk = response.action_chunk
            print(
                f"cycle={cycle} inference_ms={response.inference_ms:.1f} "
                f"action_age_ms={action_age_s * 1000.0:.1f} "
                f"tcp_first={np.array2string(chunk[0, :6], precision=5)} "
                f"gripper={chunk[:, 6].min():.5f}..{chunk[:, 6].max():.5f}",
                flush=True,
            )
            records: list[dict[str, Any]] = []
            replan = False
            if args.mode != "shadow":
                records, replan = execute_action_chunk(
                    robot=robot,
                    chunk=chunk,
                    mode=args.mode,
                    execute_steps=args.execute_steps,
                    fps=args.fps,
                    gripper=gripper,
                    action_filter=action_filter,
                    action_deadline=observation_started + args.max_action_age_s,
                )
            write_log(
                log_stream,
                {
                    "cycle": cycle,
                    "time_ns": time.time_ns(),
                    "mode": args.mode,
                    "state15": state.tolist(),
                    "task": args.task,
                    "inference_ms": response.inference_ms,
                    "action_age_ms": action_age_s * 1000.0,
                    "checkpoint": response.checkpoint,
                    "command_filter": action_filter.settings,
                    "action_chunk": chunk.tolist(),
                    "executed": records,
                    "replan_after_gripper": replan,
                },
            )
    except KeyboardInterrupt:
        print("Stopping robot client", flush=True)
    finally:
        if log_stream is not None:
            log_stream.close()
        robot.disconnect()
        client.close()


if __name__ == "__main__":
    main()
