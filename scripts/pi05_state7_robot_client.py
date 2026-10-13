"""Robot client for joint7, wrist RGB and TCP6 PI05; keeps the existing grasp."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from evo_rlt.adapters.lerobot.franka_remote import protocol_state7 as wire
from scripts.fr3_remote_robot_client import (
    RemoteInferenceClient, calibrate_force_guard, execute_action_chunk,
    require_expected_checkpoint, write_log,
)

TASK = "insert the ethernet cable into the hole"
DEFAULT_CHECKPOINT = "/workspace/wangziyi/projects/franka_rlt/outputs/pi05_dataset_928_state7_action6/checkpoints/030000/pretrained_model"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", required=True)
    p.add_argument("--expected-checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--network-only", action="store_true")
    p.add_argument("--mode", choices=("shadow", "arm-only"), default="shadow")
    p.add_argument("--allow-motion", action="store_true")
    p.add_argument("--robot-ip", default="172.16.0.2")
    p.add_argument("--wrist-camera-serial", default="349622072679")
    p.add_argument("--task", default=TASK)
    p.add_argument("--fps", type=float, choices=(30.0,), default=30.0)
    p.add_argument("--execute-steps", type=int, default=4)
    p.add_argument("--max-cycles", type=int, default=1, help="0 means run until interrupted")
    p.add_argument("--timeout-s", type=float, default=60)
    p.add_argument("--max-action-age-s", type=float, default=1)
    p.add_argument("--max-translation-m", type=float, default=.005)
    p.add_argument("--max-rotation-rad", type=float, default=.03)
    p.add_argument("--workspace-min", type=float, nargs=3)
    p.add_argument("--workspace-max", type=float, nargs=3)
    p.add_argument("--force-stop-n", type=float)
    p.add_argument("--force-baseline-samples", type=int, default=15)
    p.add_argument("--force-retreat-m", type=float, default=.05)
    p.add_argument("--force-retreat-speed-m-s", type=float, default=.03)
    p.add_argument("--log", type=Path)
    return p.parse_args()


def validate_args(args):
    if not 1 <= args.execute_steps <= 10 or args.max_cycles < 0:
        raise ValueError("execute-steps must be 1..10 and max-cycles nonnegative")
    numbers = [args.fps, args.timeout_s, args.max_action_age_s,
               args.max_translation_m, args.max_rotation_rad,
               args.force_retreat_m, args.force_retreat_speed_m_s]
    if any(not np.isfinite(v) or v <= 0 for v in numbers) or args.force_baseline_samples < 3:
        raise ValueError("Timing, clipping and retreat limits must be finite and positive")
    if args.force_stop_n is not None and (not np.isfinite(args.force_stop_n) or args.force_stop_n <= 0):
        raise ValueError("force-stop-n must be finite and positive")
    if not args.task.strip():
        raise ValueError("task must be nonempty")
    if args.mode == "arm-only":
        if not args.allow_motion or args.workspace_min is None or args.workspace_max is None:
            raise ValueError("arm-only requires --allow-motion and explicit workspace bounds")
    if (args.workspace_min is None) != (args.workspace_max is None):
        raise ValueError("Provide both workspace bounds")
    if args.workspace_min is not None:
        low, high = np.array(args.workspace_min), np.array(args.workspace_max)
        if not np.isfinite([low, high]).all() or not (low < high).all():
            raise ValueError("Workspace bounds must be finite with min < max")


def model_observation(observation):
    state = np.asarray([observation[name] for name in wire.STATE_NAMES], dtype=np.float32)
    wrist = np.asarray(observation["wrist"])
    # Use the wire validator to enforce dimensions, RGB dtype and finiteness.
    wire.encode_inference_request(request_id=0, timestamp_ns=0, task=TASK, state=state, wrist=wrist)
    return state, wrist


def require_grasp(observation):
    if not bool(observation.get("gripper_grasped", False)):
        raise RuntimeError("Existing object grasp is required for arm motion")


def shadow_observation(robot):
    # Consume the already-running control server's state; no FCI connect/disconnect RPC.
    snapshot = robot.robot._snap()
    stamp = snapshot.get("ts")
    if (snapshot.get("state_stale", True) or snapshot.get("session_fault")
            or stamp is None or abs(time.monotonic() - stamp) > 1):
        raise RuntimeError("Shadow requires a fresh existing robot state stream")
    joints = snapshot.get("joints")
    if joints is None or len(joints) != 7:
        raise RuntimeError("State stream does not contain seven joint angles")
    frame = robot._read_camera_frame("wrist")
    return {**dict(zip(wire.STATE_NAMES, joints)),
            "wrist": np.asanyarray(frame.get_color_frame().get_data())}


def close_shadow(robot):
    if robot.camera is not None:
        try:
            robot.camera.stop()
        except RuntimeError:
            pass
    controller = robot.robot
    controller._stream_on = False
    controller._rx.join(timeout=2)
    controller._sock.close(linger=0)
    if not controller._rx.is_alive():
        controller._pull.close(linger=0)


def main():
    args = parse_args()
    validate_args(args)
    client = RemoteInferenceClient(args.server, args.timeout_s, protocol=wire)
    robot = None
    stream = None
    try:
        ready = client.ping()
        require_expected_checkpoint(ready["checkpoint"], args.expected_checkpoint)
        print(f"READY {ready}", flush=True)
        if args.network_only:
            return
        from evo_rlt.adapters.lerobot.franka_robot import FrankaRobot, FrankaRobotConfig
        robot = FrankaRobot(FrankaRobotConfig(
            robot_ip=args.robot_ip, camera_serial=args.wrist_camera_serial,
            camera_width=1280, camera_height=720, enable_front_camera=False,
            include_gripper_action=False, enable_fci_keepalive=args.mode == "arm-only",
            workspace_min_xyz=None if args.workspace_min is None else tuple(args.workspace_min),
            workspace_max_xyz=None if args.workspace_max is None else tuple(args.workspace_max),
        ))
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            stream = args.log.open("a", buffering=1)
        if args.mode == "shadow":
            if not robot.robot.ping():
                raise RuntimeError("Shadow requires an already-running control server")
            robot.camera.start(robot.camera_cfg)
            for _ in range(15):
                robot._read_camera_frame("wrist")
        else:
            robot.connect()
        guard = calibrate_force_guard(robot, args) if args.mode == "arm-only" else None
        cycle = 0
        while args.max_cycles == 0 or cycle < args.max_cycles:
            cycle += 1
            started = time.perf_counter()
            observation = shadow_observation(robot) if args.mode == "shadow" else robot.get_observation()
            if args.mode == "arm-only":
                require_grasp(observation)
            state, wrist = model_observation(observation)
            result = client.request(wire.encode_inference_request(
                request_id=cycle, timestamp_ns=time.time_ns(), task=args.task,
                state=state, wrist=wrist,
            ), cycle)
            if not isinstance(result, wire.InferenceResponse):
                raise RuntimeError("Expected an inference result")
            require_expected_checkpoint(result.checkpoint, args.expected_checkpoint)
            age = time.perf_counter() - started
            records, stopped = [], False
            if args.mode == "arm-only":
                require_grasp(robot.get_observation())
                age = time.perf_counter() - started
                if age > args.max_action_age_s:
                    raise TimeoutError(f"Stale action chunk: age={age:.3f}s")
                records, _, stopped = execute_action_chunk(
                    robot=robot, chunk=result.action_chunk, mode="arm-only", gripper=None,
                    execute_steps=args.execute_steps, fps=args.fps,
                    max_translation_m=args.max_translation_m, max_rotation_rad=args.max_rotation_rad,
                    force_guard=guard, force_retreat_m=args.force_retreat_m,
                    force_retreat_speed_m_s=args.force_retreat_speed_m_s,
                )
            print(f"cycle={cycle} inference_ms={result.inference_ms:.1f} age_s={age:.3f} tcp_first={result.action_chunk[0]}", flush=True)
            write_log(stream, dict(cycle=cycle, state7=state.tolist(), task=args.task,
                                  checkpoint=result.checkpoint, fps=args.fps, execute_steps=args.execute_steps,
                                  inference_ms=result.inference_ms, action_age_s=age,
                                  action_chunk=result.action_chunk.tolist(), executed=records, force_stopped=stopped))
            if stopped:
                break
    except KeyboardInterrupt:
        print("Stopping robot client", flush=True)
    finally:
        if robot is not None:
            if args.mode == "shadow":
                close_shadow(robot)
            else:
                robot.disconnect()
        if stream:
            stream.close()
        client.close()


if __name__ == "__main__":
    main()
