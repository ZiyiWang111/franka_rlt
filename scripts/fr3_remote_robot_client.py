#!/usr/bin/env python3
"""Minimal robot-host client for split FR3 hardware and remote π0.5 inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from evo_rlt.adapters.lerobot.franka_remote.execution import (
    GripperStateMachine,
    RelativeForceGuard,
    clip_tcp_action,
)
from evo_rlt.adapters.lerobot.franka_remote.protocol import (
    ACTION_CHUNK_SHAPE,
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
    parser.add_argument("--wrist-camera-serial", default="349622072679")
    parser.add_argument("--front-camera-serial", default="233522075778")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Robot action playback rate; defaults to the requested 15 Hz",
    )
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=8,
        help="Consecutive model actions to execute before observing again",
    )
    parser.add_argument("--max-cycles", type=int, default=1, help="0 means run until interrupted")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--max-action-age-s",
        type=float,
        default=1.0,
        help="Reject a result if observation plus inference already exceeds this age",
    )
    parser.add_argument("--expected-checkpoint", default=DEFAULT_EXPECTED_CHECKPOINT)
    parser.add_argument(
        "--max-translation-m",
        type=float,
        default=0.005,
        help="Emergency norm bound for one predicted translation delta",
    )
    parser.add_argument(
        "--max-rotation-rad",
        type=float,
        default=0.03,
        help="Emergency norm bound for one predicted rotation delta",
    )
    parser.add_argument(
        "--force-stop-n",
        type=float,
        default=None,
        help="Stop and retreat when translational force changes by this magnitude",
    )
    parser.add_argument("--force-baseline-samples", type=int, default=15)
    parser.add_argument("--force-retreat-m", type=float, default=0.05)
    parser.add_argument("--force-retreat-speed-m-s", type=float, default=0.03)
    parser.add_argument(
        "--require-initial-grasp",
        action="store_true",
        help="Refuse arm motion unless the first observation reports a grasped object",
    )
    parser.add_argument("--workspace-min", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--workspace-max", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--close-threshold-m", type=float, default=0.040)
    parser.add_argument("--open-threshold-m", type=float, default=0.055)
    parser.add_argument("--gripper-confirm-steps", type=int, default=2)
    parser.add_argument(
        "--stop-after-gripper-close",
        action="store_true",
        help="Stop the run after the first close command",
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
    max_translation_m: float,
    max_rotation_rad: float,
) -> tuple[dict[str, Any], bool]:
    """Execute one model row directly, apart from hard per-step safety clipping."""
    arm_enabled = mode in {"arm-only", "integrated"}
    gripper_enabled = mode in {"gripper-only", "integrated"}
    record: dict[str, Any] = {"step": step_index, "predicted": action.tolist()}
    decision = gripper.update(float(action[6])) if gripper_enabled else None

    if decision is not None and decision.hold_arm:
        record["arm"] = "held_for_gripper_transition"
        if decision.command is None:
            return record, False

        target_width_m = float(action[6]) if decision.command == "close" else None
        if target_width_m is not None:
            record["gripper_target_width_m"] = target_width_m
        motion = robot.start_gripper_transition(decision.command, width=target_width_m)
        record["arm_servo_stopped"] = arm_enabled
        record["gripper_command_started"] = decision.command
        record["gripper_motion"] = motion
        return record, True

    if arm_enabled:
        tcp_action = clip_tcp_action(
            action,
            max_translation_m=max_translation_m,
            max_rotation_rad=max_rotation_rad,
        )
        robot.send_action(tcp_action)
        record["arm"] = tcp_action
    else:
        record["arm"] = "disabled"
    return record, False


def execute_action_chunk(
    *,
    robot: Any,
    chunk: np.ndarray,
    mode: str,
    execute_steps: int,
    fps: float,
    gripper: GripperStateMachine,
    max_translation_m: float,
    max_rotation_rad: float,
    force_guard: RelativeForceGuard | None = None,
    force_retreat_m: float = 0.05,
    force_retreat_speed_m_s: float = 0.03,
) -> tuple[list[dict[str, Any]], bool, bool]:
    """Play a model chunk in order at a fixed rate, without filtering or blending."""
    records: list[dict[str, Any]] = []
    for step_index, action in enumerate(chunk[:execute_steps]):
        started = time.perf_counter()
        if force_guard is not None:
            force_event = force_guard.check(robot.get_external_wrench_base())
            if force_event["triggered"]:
                record = {
                    "step": step_index,
                    "force_guard": force_event,
                    "arm": "force_stop_before_next_action",
                }
                print(
                    f"Force guard triggered: delta={force_event['force_delta_n']:.2f}N "
                    f">= {force_event['threshold_n']:.2f}N; stopping and retreating",
                    flush=True,
                )
                record["retreat"] = robot.stop_and_retreat_up(
                    distance_m=force_retreat_m,
                    speed_m_s=force_retreat_speed_m_s,
                )
                records.append(record)
                return records, False, True

        record, replan = execute_action_step(
            robot=robot,
            action=action,
            step_index=step_index,
            mode=mode,
            gripper=gripper,
            max_translation_m=max_translation_m,
            max_rotation_rad=max_rotation_rad,
        )
        records.append(record)
        if replan:
            return records, True, False
        sleep_control_period(started, fps)
    return records, False, False


def write_log(stream: Any, payload: dict[str, Any]) -> None:
    if stream is not None:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()


def calibrate_force_guard(robot: Any, args: argparse.Namespace) -> RelativeForceGuard | None:
    if args.force_stop_n is None:
        return None
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
            f"max_delta={baseline_noise_n:.2f}N, threshold={args.force_stop_n:.2f}N; "
            "no action was executed"
        )
    print(
        "Force guard calibrated while stationary: "
        f"baseline={np.array2string(baseline_wrench, precision=3)} "
        f"max_baseline_delta={baseline_noise_n:.2f}N "
        f"threshold={args.force_stop_n:.2f}N retreat=+Z{args.force_retreat_m:.3f}m",
        flush=True,
    )
    return RelativeForceGuard(baseline_wrench=baseline_wrench, threshold_n=args.force_stop_n)


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0 or not 1 <= args.execute_steps <= ACTION_CHUNK_SHAPE[0]:
        raise ValueError(
            f"fps must be positive and execute-steps must be within [1, {ACTION_CHUNK_SHAPE[0]}]"
        )
    if args.max_cycles < 0 or args.timeout_s <= 0 or args.max_action_age_s <= 0:
        raise ValueError(
            "max-cycles must be non-negative; timeout-s and max-action-age-s must be positive"
        )
    if args.max_translation_m <= 0 or args.max_rotation_rad <= 0:
        raise ValueError("TCP action limits must be positive")
    if args.force_stop_n is not None and args.force_stop_n <= 0:
        raise ValueError("force-stop-n must be positive")
    if args.force_baseline_samples < 3:
        raise ValueError("force-baseline-samples must be at least 3")
    if args.force_retreat_m <= 0 or args.force_retreat_speed_m_s <= 0:
        raise ValueError("force retreat distance and speed must be positive")
    if args.force_stop_n is not None and args.mode not in {"arm-only", "integrated"}:
        raise RuntimeError("--force-stop-n requires arm-only or integrated mode")
    if args.require_initial_grasp and args.mode not in {"arm-only", "integrated"}:
        raise RuntimeError("--require-initial-grasp requires arm-only or integrated mode")
    if args.stop_after_gripper_close and args.mode not in {"gripper-only", "integrated"}:
        raise RuntimeError("--stop-after-gripper-close requires gripper-only or integrated mode")
    if args.mode != "shadow" and not args.allow_motion:
        raise RuntimeError(f"mode {args.mode!r} requires the explicit --allow-motion flag")
    if args.mode in {"arm-only", "integrated"}:
        if args.workspace_min is None or args.workspace_max is None:
            raise RuntimeError(
                "arm motion requires explicit --workspace-min X Y Z and --workspace-max X Y Z"
            )


def main() -> None:
    args = parse_args()
    validate_args(args)

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

    from evo_rlt.adapters.lerobot.franka_robot import FrankaRobot, FrankaRobotConfig

    robot = FrankaRobot(
        FrankaRobotConfig(
            robot_ip=args.robot_ip,
            camera_serial=args.wrist_camera_serial,
            front_camera_serial=args.front_camera_serial,
            include_gripper_action=True,
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
    log_stream = None
    if args.log is not None:
        log_path = args.log.expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("a", buffering=1)

    cycle = 0
    stop_run = False
    pending_gripper_command: str | None = None
    try:
        robot.connect()
        print(
            f"Robot connected; mode={args.mode} control_rate={args.fps:g}Hz "
            f"execute_steps={args.execute_steps} pipeline=synchronous_vanilla",
            flush=True,
        )
        force_guard = calibrate_force_guard(robot, args)

        # A cycle budget may stop new arm work, but it must never tear down a
        # Hand Future that was accepted on the final cycle. Keep observing until
        # that transition reaches a terminal state, then exit cleanly.
        while not stop_run and (
            pending_gripper_command is not None
            or args.max_cycles == 0
            or cycle < args.max_cycles
        ):
            cycle += 1
            observation_started = time.perf_counter()
            observation = robot.get_observation()
            state = observation_to_state15(observation)
            gripper_motion = robot.get_gripper_transition_state()
            gripper_transition_running = False
            completed_gripper_motion = None
            if pending_gripper_command is not None:
                status = gripper_motion.get("status")
                if status == "running":
                    gripper_transition_running = True
                elif status == "finished":
                    completed_gripper_motion = dict(gripper_motion)
                    robot.finish_gripper_transition()
                    gripper.mark_completed(pending_gripper_command)
                    robot.resync_command_pose()
                    pending_gripper_command = None
                else:
                    robot.finish_gripper_transition()
                    failed_command = pending_gripper_command
                    pending_gripper_command = None
                    raise RuntimeError(
                        f"asynchronous gripper {failed_command} ended with status={status}: "
                        f"{gripper_motion.get('error') or gripper_motion}")
            if not gripper_transition_running:
                gripper.sync_from_observation(float(state[13]), bool(state[14]))

            if cycle == 1 and args.require_initial_grasp and not bool(state[14]):
                raise RuntimeError(
                    "run requires an initially grasped object; "
                    f"observed width={float(state[13]):.5f}m grasped=False; no action was executed"
                )
            if cycle == 1 and args.stop_after_gripper_close:
                if bool(state[14]) or float(state[13]) < args.open_threshold_m:
                    raise RuntimeError(
                        "stop-after-close run requires an initially open, empty gripper; "
                        f"observed width={float(state[13]):.5f}m grasped={bool(state[14])}"
                    )

            response = client.infer(
                request_id=cycle,
                state=state,
                wrist=observation_image(observation, "wrist"),
                front=observation_image(observation, "front"),
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
            force_stopped = False
            if completed_gripper_motion is not None and args.stop_after_gripper_close \
                    and completed_gripper_motion.get("command") == "close":
                records = [{
                    "arm": "held_after_gripper_transition",
                    "gripper_command": "close",
                    "gripper_motion": completed_gripper_motion,
                }]
                print(
                    "Stopping after gripper close completed: "
                    f"target={completed_gripper_motion.get('width')} "
                    f"grasped={completed_gripper_motion.get('result')}",
                    flush=True,
                )
                stop_run = True
            elif (completed_gripper_motion is not None
                  and args.max_cycles != 0 and cycle > args.max_cycles):
                records = [{
                    "arm": "held_after_gripper_transition",
                    "gripper_motion": completed_gripper_motion,
                    "cycle_budget_exhausted": True,
                }]
                stop_run = True
            elif gripper_transition_running:
                # Keep observing/inferencing for fresh perception, but never run
                # actions produced while the Hand is still moving. The first
                # post-completion observation is replanned before arm servo resumes.
                records = [{
                    "arm": "held_for_gripper_transition",
                    "gripper_motion": gripper_motion,
                    "discarded_action_chunk": True,
                }]
                replan = True
            elif args.mode != "shadow":
                records, replan, force_stopped = execute_action_chunk(
                    robot=robot,
                    chunk=chunk,
                    mode=args.mode,
                    execute_steps=args.execute_steps,
                    fps=args.fps,
                    gripper=gripper,
                    max_translation_m=args.max_translation_m,
                    max_rotation_rad=args.max_rotation_rad,
                    force_guard=force_guard,
                    force_retreat_m=args.force_retreat_m,
                    force_retreat_speed_m_s=args.force_retreat_speed_m_s,
                )
                started = next(
                    (record.get("gripper_command_started") for record in records
                     if record.get("gripper_command_started") is not None),
                    None,
                )
                if started is not None:
                    pending_gripper_command = str(started)
            if force_stopped:
                stop_run = True

            write_log(
                log_stream,
                {
                    "cycle": cycle,
                    "time_ns": time.time_ns(),
                    "mode": args.mode,
                    "pipeline": "synchronous_vanilla",
                    "fps": args.fps,
                    "execute_steps": args.execute_steps,
                    "state15": state.tolist(),
                    "task": args.task,
                    "inference_ms": response.inference_ms,
                    "action_age_ms": action_age_s * 1000.0,
                    "checkpoint": response.checkpoint,
                    "safety_clip": {
                        "max_translation_m": args.max_translation_m,
                        "max_rotation_rad": args.max_rotation_rad,
                    },
                    "action_chunk": chunk.tolist(),
                    "executed": records,
                    "replan_after_gripper": replan,
                    "force_stopped": force_stopped,
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
