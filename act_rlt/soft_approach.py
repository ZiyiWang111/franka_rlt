"""Teach a TCP target, then repeatedly return with online lateral/descent speed allocation.

Run from Evo-RLT: PYTHONPATH=src python -m act_rlt.soft_approach
Uses the existing servo_tool API; no controller configuration is changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Control server host")
    parser.add_argument("--cmd-port", type=int, default=5555)
    parser.add_argument("--state-port", type=int, default=5556)
    parser.add_argument("--speed", type=float, default=0.05,
                        help="Maximum reference TCP translation speed in m/s")
    parser.add_argument("--max-angular-velocity", type=float, default=0.4, help="Reference rotation speed in rad/s")
    parser.add_argument("--fps", type=float, default=30.0, help="Target update frequency (Hz)")
    parser.add_argument("--acceleration", type=float, default=0.1, help="Reference acceleration limit (m/s^2)")
    parser.add_argument("--jerk", type=float, default=0.5, help="Reference jerk limit (m/s^3)")
    parser.add_argument("--slowdown-distance", type=float, default=0.03,
                        help="Reference speed decreases inside this remaining distance (m)")
    parser.add_argument("--lateral-gate-distance", type=float, default=0.06,
                        help="Increase vertical allowance below this commanded XY error (m)")
    parser.add_argument("--lateral-tolerance", type=float, default=0.01,
                        help="Fully allow vertical motion below this commanded XY error (m)")
    parser.add_argument("--vertical-speed-floor", type=float, default=0.15,
                        help="Minimum vertical allowance while XY error is large (fraction in [0, 1])")
    parser.add_argument("--lag-limit", type=float, default=0.008,
                        help="Pause reference translation if measured tracking lag reaches this (m)")
    parser.add_argument("--position-tolerance", type=float, default=0.001, help="Measured final position tolerance (m)")
    parser.add_argument("--orientation-tolerance-deg", type=float, default=0.3,
                        help="Measured final orientation tolerance (degrees)")
    parser.add_argument("--settle-time", type=float, default=0.4,
                        help="Require measured goal and low velocity continuously for this many seconds")
    parser.add_argument("--timeout", type=float, default=40.0,
                        help="Total online motion and settling timeout (s)")
    parser.add_argument("--log", type=Path, help="JSONL log; default: logs/act_rlt_soft_approach/<time>.jsonl")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("speed", "max_angular_velocity", "timeout", "fps", "acceleration", "jerk",
                 "slowdown_distance", "lateral_gate_distance", "lateral_tolerance", "lag_limit",
                 "position_tolerance", "orientation_tolerance_deg", "settle_time"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 10 <= args.fps <= 100:
        raise ValueError("fps must be within [10, 100]")
    if args.lateral_tolerance >= args.lateral_gate_distance:
        raise ValueError("lateral_tolerance must be smaller than lateral_gate_distance")
    if not math.isfinite(args.vertical_speed_floor) or not 0 <= args.vertical_speed_floor <= 1:
        raise ValueError("vertical_speed_floor must be within [0, 1]")


def smooth_fraction(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def limit_norm(vector: np.ndarray, ceiling: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector * min(1.0, ceiling / max(norm, 1e-12))


class OnlineApproach:
    """Integrate a persistent reference; measurements ONLY govern tracking lag.

    These are script-level reference limits, not changes to Ruckig. Hard lag
    pauses and the tiny final snap take precedence over reference smoothing.
    """

    def __init__(self, start, target, args):
        self.args = args
        self.command = np.asarray(start, dtype=float).copy()
        self.target = np.asarray(target, dtype=float).copy()
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.rotation_start = Rotation.from_rotvec(self.command[3:])
        rotation_delta = (self.rotation_start.inv() * Rotation.from_rotvec(self.target[3:])).as_rotvec()
        self.rotation_total = float(np.linalg.norm(rotation_delta))
        self.rotation_axis = rotation_delta / max(self.rotation_total, 1e-12)
        self.rotation_progress = 0.0
        self.angular_velocity = 0.0
        self.angular_acceleration = 0.0

    def step(self, measured, dt: float) -> dict:
        if not math.isfinite(dt) or not 0 < dt <= 0.15:
            raise RuntimeError("目标更新间隔过长，停止运动，避免恢复后跳跃推进目标。")
        measured = np.asarray(measured, dtype=float)
        error = self.target[:3] - self.command[:3]
        lateral_error = float(np.linalg.norm(error[:2]))
        vertical_gate = self.args.vertical_speed_floor + (1 - self.args.vertical_speed_floor) * smooth_fraction(
            (self.args.lateral_gate_distance - lateral_error)
            / (self.args.lateral_gate_distance - self.args.lateral_tolerance))
        lag = float(np.linalg.norm(self.command[:3] - measured[:3]))
        angular_lag = float((Rotation.from_rotvec(measured[3:]).inv()
                             * Rotation.from_rotvec(self.command[3:])).magnitude())
        # Slow down in the second half of the permitted lead; never rebase the
        # integrator onto measured position. Orientation lead is limited to 0.1 rad.
        tracking_scale = min(smooth_fraction(2 * (1 - lag / self.args.lag_limit)),
                             smooth_fraction(2 * (1 - angular_lag / 0.1)))
        desired = np.zeros(3)
        if lateral_error > 1e-12:
            desired[:2] = error[:2] / lateral_error * self.args.speed \
                * min(1.0, lateral_error / self.args.slowdown_distance)
        desired[2] = self.args.speed * vertical_gate \
            * float(np.clip(error[2] / self.args.slowdown_distance, -1.0, 1.0))
        desired = limit_norm(desired, self.args.speed) * tracking_scale
        desired_acceleration = limit_norm((desired - self.velocity) / 0.15, self.args.acceleration)
        self.acceleration += limit_norm(desired_acceleration - self.acceleration, self.args.jerk * dt)
        new_velocity = limit_norm(self.velocity + self.acceleration * dt, self.args.speed)
        delta = (self.velocity + new_velocity) * (0.5 * dt)
        candidate = self.command[:3] + delta
        # Keep each reference coordinate inside the remaining interval, preventing
        # inertia in the smoother from overshooting the goal or moving backward.
        candidate = np.minimum(np.maximum(candidate, np.minimum(self.command[:3], self.target[:3])),
                               np.maximum(self.command[:3], self.target[:3]))
        position_paused = lag >= self.args.lag_limit \
            or np.linalg.norm(candidate - measured[:3]) > self.args.lag_limit
        if position_paused:
            self.velocity[:] = 0.0
            self.acceleration[:] = 0.0
        else:
            self.command[:3] = candidate
            self.velocity = new_velocity
            blocked = np.abs(candidate - self.target[:3]) < 1e-12
            self.velocity[blocked] = 0.0
            self.acceleration[blocked] = 0.0
            if np.linalg.norm(self.target[:3] - self.command[:3]) < 1e-4 \
                    and np.linalg.norm(self.velocity) < 5e-4:
                self.command[:3] = self.target[:3]
                self.velocity[:] = 0.0
                self.acceleration[:] = 0.0

        remaining_angle = max(0.0, self.rotation_total - self.rotation_progress)
        desired_angular_velocity = self.args.max_angular_velocity \
            * min(1.0, remaining_angle / 0.35) * tracking_scale
        desired_angular_acceleration = float(np.clip(
            (desired_angular_velocity - self.angular_velocity) / 0.15, -0.8, 0.8))
        self.angular_acceleration += float(np.clip(
            desired_angular_acceleration - self.angular_acceleration, -4.0 * dt, 4.0 * dt))
        angular_velocity = float(np.clip(
            self.angular_velocity + self.angular_acceleration * dt, 0.0, self.args.max_angular_velocity))
        progress = min(self.rotation_total, self.rotation_progress
                       + (self.angular_velocity + angular_velocity) * 0.5 * dt)
        candidate_rotation = (self.rotation_start * Rotation.from_rotvec(self.rotation_axis * progress)).as_rotvec()
        candidate_angular_lag = float((Rotation.from_rotvec(measured[3:]).inv()
                                       * Rotation.from_rotvec(candidate_rotation)).magnitude())
        rotation_paused = angular_lag >= 0.1 or candidate_angular_lag > 0.1
        if rotation_paused:
            self.angular_velocity = self.angular_acceleration = 0.0
        else:
            self.rotation_progress = progress
            self.angular_velocity = angular_velocity
            if self.rotation_total - progress < 5e-4 and angular_velocity < 0.002:
                self.rotation_progress = self.rotation_total
                self.angular_velocity = self.angular_acceleration = 0.0
            if self.rotation_progress == self.rotation_total:
                self.command[3:] = self.target[3:]
            else:
                self.command[3:] = candidate_rotation
        return {"lateral_command_error_m": lateral_error,
                "vertical_command_error_m": float(error[2]), "vertical_gate": vertical_gate,
                "tracking_lag_m": lag, "angular_tracking_lag_rad": angular_lag,
                "tracking_scale": tracking_scale, "position_paused": bool(position_paused),
                "rotation_paused": bool(rotation_paused),
                "reference_velocity_m_s": self.velocity.tolist()}

    def at_goal(self) -> bool:
        return np.array_equal(self.command[:3], self.target[:3]) \
            and self.rotation_progress == self.rotation_total


def request(client, command: str) -> dict:
    response = client._send(command)
    if not response or not response.get("success"):
        raise RuntimeError(f"{command}: {(response or {}).get('error', 'no server reply')}")
    return response


def require_idle(client) -> dict:
    status = request(client, "ping")
    if status.get("busy"):
        raise RuntimeError("control server 正在运动，请先结束 inference / 其他运动脚本。")
    if status.get("connected"):
        snap = request(client, "get_state").get("state", {})
        if snap.get("is_running") or "move" in str(snap.get("robot_mode", "")).lower():
            raise RuntimeError("机械臂仍在控制运动，请先结束 inference / 其他运动脚本。")
    return status


def checked_snapshot(response: dict, *, after: float = 0.0) -> dict:
    snap = response.get("state", {})
    timestamp = float(snap.get("ts", 0.0))
    age = time.monotonic() - timestamp
    if snap.get("state_stale") or snap.get("session_fault"):
        raise RuntimeError(f"机器人状态失效: {snap.get('session_fault')}")
    if timestamp < after or not 0 <= age < 1.0:
        raise RuntimeError("机器人状态尚未刷新或已过期。")
    for name, length in (("pose", 6), ("joints", 7)):
        values = np.asarray(snap.get(name, []), dtype=float)
        if values.shape != (length,) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"无效的 {name} 状态")
    return snap


def wait_fresh_snapshot(client, *, after: float, timeout: float = 3.0) -> dict:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        response = request(client, "get_state")
        try:
            return checked_snapshot(response, after=after)
        except RuntimeError as exc:
            if (response.get("state") or {}).get("session_fault"):
                raise
            last_error = exc
        time.sleep(0.05)
    raise RuntimeError(f"无法获得新鲜 TCP 状态: {last_error}")


def pose_error(pose, target) -> tuple[float, float]:
    position = float(np.linalg.norm(np.asarray(pose[:3]) - np.asarray(target[:3])))
    orientation = float((Rotation.from_rotvec(target[3:]).inv()
                         * Rotation.from_rotvec(pose[3:])).magnitude())
    return position * 1000.0, math.degrees(orientation)


def format_pose(pose) -> str:
    return "XYZ [m] " + ", ".join(f"{v:+.4f}" for v in pose[:3]) + " | rotvec [rad] " \
        + ", ".join(f"{v:+.4f}" for v in pose[3:])


def run_online_move(client, start: dict, target, args, emit) -> dict:
    planner = OnlineApproach(start["pose"], target, args)
    period = 1.0 / args.fps
    started = time.monotonic()
    deadline = started + args.timeout
    previous = start
    previous_tick = started - period
    next_tick = started
    settled_since = None
    previous_print = -float("inf")
    fault_baseline = int(start.get("servo_fault_seq", 0))
    servo_started = False
    while time.monotonic() < deadline:
        delay = next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        response = request(client, "get_state")
        snap = checked_snapshot(response)
        now = time.monotonic()
        if now >= deadline:
            break
        if int(snap.get("servo_fault_seq", 0)) > fault_baseline:
            raise RuntimeError(f"servo 控制失败: {snap.get('servo_fault')}")
        mode = str(snap.get("robot_mode", "")).lower()
        if any(word in mode for word in ("reflex", "userstopped", "guiding")):
            raise RuntimeError(f"机械臂不能继续运动: {snap.get('robot_mode')}")
        error_mm, error_deg = pose_error(snap["pose"], target)
        speed = None
        angular_speed = None
        if snap["seq"] != previous["seq"]:
            dt = snap["ts"] - previous["ts"]
            if dt > 0:
                speed = float(np.linalg.norm(np.asarray(snap["pose"][:3])
                                             - np.asarray(previous["pose"][:3])) / dt)
                angular_speed = float((Rotation.from_rotvec(previous["pose"][3:]).inv()
                                       * Rotation.from_rotvec(snap["pose"][3:])).magnitude()) / dt
        joint_speeds = np.asarray(snap.get("joint_speeds", []), dtype=float)
        low_joint_speed = joint_speeds.shape == (7,) and np.all(np.isfinite(joint_speeds)) \
            and float(np.max(np.abs(joint_speeds))) < 0.01
        inside_tolerance = error_mm <= args.position_tolerance * 1000 \
            and error_deg <= args.orientation_tolerance_deg
        arrived = (planner.at_goal() or not servo_started) and inside_tolerance and low_joint_speed \
            and speed is not None and speed < 0.002 \
            and angular_speed is not None and angular_speed < 0.02
        if arrived:
            if settled_since is None:
                settled_since = now
            if now - settled_since >= args.settle_time:
                if servo_started and not client.stop_servo():
                    raise RuntimeError("已到位，但 server 未确认结束 servo。")
                emit("settled", state=snap, position_error_mm=error_mm, orientation_error_deg=error_deg)
                return snap
        else:
            settled_since = None
        if not servo_started and inside_tolerance and low_joint_speed:
            # No degenerate zero-distance JointMotion when the arm is already at
            # the goal: verify stability without opening a servo stream.
            emit("sample", state=snap, command_pose=planner.command.tolist(),
                 initial_goal_wait=True, position_error_mm=error_mm, orientation_error_deg=error_deg)
            if snap["seq"] != previous["seq"]:
                previous = snap
            previous_tick = now
            next_tick = now + period
            continue
        diagnostics = planner.step(snap["pose"], now - previous_tick)
        client.servo_tool(planner.command.tolist())
        servo_started = True
        emit("sample", state=snap, command_pose=planner.command.tolist(),
             tcp_speed_m_s=speed, position_error_mm=error_mm, orientation_error_deg=error_deg,
             command_at_goal=planner.at_goal(), **diagnostics)
        if now - previous_print >= 0.1:
            speed_text = "--" if speed is None else f"{speed:.4f}"
            phase = "目标保持" if planner.at_goal() else "跟踪等待" \
                if diagnostics["position_paused"] or diagnostics["rotation_paused"] else "接近"
            print(f"{phase} | TCP v≈{speed_text} m/s | 误差 {error_mm:.2f} mm / {error_deg:.2f}°"
                  f" | 滞后 {diagnostics['tracking_lag_m'] * 1000:.2f} mm"
                  f" | Z放行 {diagnostics['vertical_gate']:.0%}", flush=True)
            previous_print = now
        if snap["seq"] != previous["seq"]:
            previous = snap
        previous_tick = now
        # Never integrate a catch-up burst after a delayed iteration.
        next_tick = now + period
    raise RuntimeError(f"超过 {args.timeout:g} s 尚未实际到位并稳定，停止运动。")


def release_connection(client) -> None:
    client.disconnect()
    if request(client, "ping").get("connected"):
        raise RuntimeError("FCI 连接未释放，不能开始手动引导。")


def run(args: argparse.Namespace) -> None:
    from evo_franka.control_client import FrankaArmControllerClient

    log_path = args.log or Path("logs/act_rlt_soft_approach") / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log, FrankaArmControllerClient(
            host=args.host, cmd_port=args.cmd_port, state_port=args.state_port) as client:
        def emit(event: str, **fields) -> None:
            log.write(json.dumps({"event": event, "wall_time": time.time(), **fields},
                                 ensure_ascii=False) + "\n")

        connected = False
        moving = False
        try:
            require_idle(client)
            emit("configuration", speed=args.speed, fps=args.fps,
                 reference_acceleration_m_s2=args.acceleration, reference_jerk_m_s3=args.jerk,
                 slowdown_distance_m=args.slowdown_distance, lag_limit_m=args.lag_limit,
                 lateral_gate_distance_m=args.lateral_gate_distance,
                 lateral_tolerance_m=args.lateral_tolerance, vertical_speed_floor=args.vertical_speed_floor,
                 path_mode="online_command_integrator")
            print(f"日志: {log_path.resolve()}")
            print(f"目标更新 {args.fps:g} Hz，参考速度上限 {args.speed:g} m/s；不修改底层参数。")
            print("目标是完整 TCP 位姿，包含位置和朝向。")
            print("按指令误差分配 XY/Z 速度；实际 TCP 只用于监测滞后和到位。")
            print(f"Z 初始放行 {args.vertical_speed_floor:.0%}，XY 误差从"
                  f" {args.lateral_gate_distance * 1000:g} 到 {args.lateral_tolerance * 1000:g} mm 时逐渐完全放行。")
            input("\n把机械臂放在目标点，松开引导按钮，确认 Execution + FCI active。按 Enter 记录目标：")
            require_idle(client)
            connected = True  # also clean up a partially failed connect
            connected_at = time.monotonic()
            client.connect()
            target_state = wait_fresh_snapshot(client, after=connected_at)
            target = list(target_state["pose"])
            emit("target", state=target_state)
            print("已记录目标：", format_pose(target), flush=True)
            release_connection(client)
            connected = False

            cycle = 1
            while True:
                answer = input(
                    f"\n第 {cycle} 次：已释放 FCI。用手动引导移到起点，再松开按钮并确认 Execution + FCI active。"
                    "按 Enter 返回已记录的目标，或输入 q 后 Enter 退出（Ctrl+C 取消）：").strip().lower()
                if answer == "q":
                    emit("session_end", completed_cycles=cycle - 1)
                    break
                if answer:
                    print("按 Enter 开始运动，或输入 q 退出。")
                    continue
                require_idle(client)
                connected = True
                connected_at = time.monotonic()
                client.connect()
                start = wait_fresh_snapshot(client, after=connected_at)
                error_mm, error_deg = pose_error(start["pose"], target)
                emit("start", cycle=cycle, state=start,
                     position_error_mm=error_mm, orientation_error_deg=error_deg)
                print("起点：", format_pose(start["pose"]), flush=True)
                print(f"目标距离 {error_mm:.2f} mm / {error_deg:.2f}°。开始运动；Ctrl+C 立即停止。", flush=True)
                moving = True  # stop on cancellation, including a just-dispatched servo target
                final = run_online_move(client, start, target, args, emit)
                moving = False
                error_mm, error_deg = pose_error(final["pose"], target)
                emit("finished", cycle=cycle, state=final,
                     position_error_mm=error_mm, orientation_error_deg=error_deg)
                print(f"本轮完成。实测最终误差：{error_mm:.2f} mm / {error_deg:.2f}°。", flush=True)
                release_connection(client)
                connected = False
                cycle += 1
        except BaseException as exc:
            emit("aborted", reason=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if moving:
                try:
                    if not client.stop_move():
                        print("server 未确认停止请求，请检查机械臂状态。", file=sys.stderr)
                except Exception as exc:
                    print(f"停止请求失败：{exc}", file=sys.stderr)
            if connected:
                try:
                    release_connection(client)
                except Exception as exc:
                    print(f"释放连接失败：{exc}", file=sys.stderr)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        run(args)
    except (KeyboardInterrupt, EOFError):
        print("\n已取消。", flush=True)
        return 130
    except (ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
