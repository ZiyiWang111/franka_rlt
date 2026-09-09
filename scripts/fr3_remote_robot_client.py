#!/usr/bin/env python3
"""Robot-host client for split FR3 hardware and remote π0.5 inference."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from evo_rlt.adapters.lerobot.franka_remote.execution import GripperStateMachine, clip_tcp_action
from evo_rlt.adapters.lerobot.franka_remote.protocol import (
    ACTION_NAMES,
    InferenceResponse,
    decode_response,
    encode_inference_request,
    encode_ping,
)
from evo_rlt.adapters.lerobot.franka_remote.state import observation_image, observation_to_state15


DEFAULT_TASK = "pick up the vga and insert into the motherboard"


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
    parser.add_argument("--execute-steps", type=int, default=5)
    parser.add_argument("--max-cycles", type=int, default=1, help="0 means run until interrupted")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--max-translation-m", type=float, default=0.005)
    parser.add_argument("--max-rotation-rad", type=float, default=0.03)
    parser.add_argument("--workspace-min", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--workspace-max", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--close-threshold-m", type=float, default=0.040)
    parser.add_argument("--open-threshold-m", type=float, default=0.055)
    parser.add_argument("--gripper-confirm-steps", type=int, default=2)
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
) -> tuple[list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    replan = False
    arm_enabled = mode in {"arm-only", "integrated"}
    gripper_enabled = mode in {"gripper-only", "integrated"}
    for step_index, action in enumerate(chunk[:execute_steps]):
        started = time.perf_counter()
        record: dict[str, Any] = {"step": step_index, "predicted": action.tolist()}
        decision = gripper.update(float(action[6])) if gripper_enabled else None
        if decision is not None and decision.hold_arm:
            record["arm"] = "held_for_gripper_transition"
            if decision.command is not None:
                if decision.command == "close":
                    record["gripper_result"] = bool(robot.close_gripper())
                else:
                    open_result = robot.open_gripper()
                    record["gripper_result"] = None if open_result is None else bool(open_result)
                gripper.mark_completed(decision.command)
                if hasattr(robot, "resync_command_pose"):
                    robot.resync_command_pose()
                record["gripper_command"] = decision.command
                replan = True
                records.append(record)
                break
        elif arm_enabled:
            tcp_action = clip_tcp_action(
                action,
                max_translation_m=max_translation_m,
                max_rotation_rad=max_rotation_rad,
            )
            robot.send_action(tcp_action)
            record["arm"] = tcp_action
        else:
            record["arm"] = "disabled"
        records.append(record)
        sleep_control_period(started, fps)
    return records, replan


def write_log(stream: Any, payload: dict[str, Any]) -> None:
    if stream is not None:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or not 1 <= args.execute_steps <= 50:
        raise ValueError("fps must be positive and execute-steps must be within [1, 50]")
    if args.max_cycles < 0 or args.timeout_s <= 0:
        raise ValueError("max-cycles must be non-negative and timeout-s must be positive")
    if args.max_translation_m <= 0 or args.max_rotation_rad <= 0:
        raise ValueError("TCP action limits must be positive")
    if args.mode != "shadow" and not args.allow_motion:
        raise RuntimeError(f"mode {args.mode!r} requires the explicit --allow-motion flag")
    if args.mode in {"arm-only", "integrated"}:
        if args.workspace_min is None or args.workspace_max is None:
            raise RuntimeError("arm motion requires explicit --workspace-min X Y Z and --workspace-max X Y Z")

    client = RemoteInferenceClient(args.server, args.timeout_s)
    ready = client.ping()
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
            enable_fci_keepalive=args.mode != "shadow",
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
        args.log.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        log_stream = args.log.expanduser().resolve().open("a", buffering=1)

    cycle = 0
    try:
        robot.connect()
        print(f"Robot connected; mode={args.mode}", flush=True)
        while args.max_cycles == 0 or cycle < args.max_cycles:
            cycle += 1
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
            chunk = response.action_chunk
            print(
                f"cycle={cycle} inference_ms={response.inference_ms:.1f} "
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
                    max_translation_m=args.max_translation_m,
                    max_rotation_rad=args.max_rotation_rad,
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
