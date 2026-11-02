#!/usr/bin/env python3
"""Record Stage-2-style physical TCP deltas, return to the start, and replay."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty

import numpy as np
from scipy.spatial.transform import Rotation


class Terminal:
    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("Run in an interactive terminal")
        self.saved = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def key(self):
        return os.read(sys.stdin.fileno(), 1).decode().lower() if select.select([sys.stdin], [], [], 0)[0] else ""

    def __exit__(self, *args):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.saved)


class DeltaServo:
    """Same target integration/lead limits as FrankaRobot.send_action; no cameras."""
    def __init__(self, client):
        self.client = client
        self.target = None
        self.active = False

    def stop(self):
        if not self.client.stop_servo():
            raise RuntimeError("stop_servo was not acknowledged")
        self.target = None
        self.active = False

    def send(self, delta):
        measured = np.asarray(self.client.get_fresh_tool_pose(), dtype=float)
        if self.target is None:
            self.target = measured.copy()
        self.target[:3] += delta[:3]
        self.target[3:] = (Rotation.from_rotvec(self.target[3:])
                           * Rotation.from_rotvec(delta[3:])).as_rotvec()
        lead = self.target[:3] - measured[:3]
        distance = np.linalg.norm(lead)
        if distance > 0.010:
            self.target[:3] = measured[:3] + lead * (0.010 / distance)
        measured_rotation = Rotation.from_rotvec(measured[3:])
        lead_rotation = (measured_rotation.inv()
                         * Rotation.from_rotvec(self.target[3:])).as_rotvec()
        angle = np.linalg.norm(lead_rotation)
        if angle > 0.1:
            self.target[3:] = (measured_rotation
                              * Rotation.from_rotvec(lead_rotation * (0.1 / angle))).as_rotvec()
        self.client.servo_tool(self.target.tolist())
        self.active = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--linear-speed", type=float, default=0.0135)
    parser.add_argument("--angular-speed", type=float, default=0.09)
    parser.add_argument("--deadzone", type=float, default=0.05)
    parser.add_argument("--return-speed", type=float, default=0.3,
                        help="Return joint speed limit in rad/s, further scaled by server dynamics")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    for name in ("fps", "linear_speed", "angular_speed", "return_speed"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.return_speed > 1 or not np.isfinite(args.deadzone) or not 0 <= args.deadzone < 1:
        parser.error("return-speed must be <= 1; deadzone must be in [0, 1)")

    from act_rlt.spacemouse_input import SpaceMouseInput
    from evo_franka.control_client import FrankaArmControllerClient

    output = args.output or Path("outputs/spacemouse") / time.strftime("%Y%m%d_%H%M%S.json")
    period = 1 / args.fps
    rows = []
    client = FrankaArmControllerClient(host=args.host)
    servo = DeltaServo(client)
    mouse = None
    recording = None

    def save():
        if recording is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(recording, indent=2) + "\n")

    try:
        with Terminal() as terminal:
            mouse = SpaceMouseInput(args.linear_speed, args.angular_speed, args.deadzone)
            client.connect()
            servo.stop()
            start_joints = client.get_joint_angles_after_move()
            start_pose = client.get_tool_pose_after_move()
            recording = {
                "format": "stage2_physical_delta_v1", "fps": args.fps,
                "action_names": ["dx", "dy", "dz", "drx", "dry", "drz"],
                "units": ["m", "m", "m", "rad", "rad", "rad"],
                "rotation_frame": "tool_local", "normalized": False,
                "linear_speed": args.linear_speed, "angular_speed": args.angular_speed,
                "deadzone": args.deadzone, "start_joints": start_joints,
                "start_pose": start_pose, "steps": rows,
            }
            save()
            print(f"Start TCP: {start_pose}\nMove SpaceMouse to record. "
                  "s = finish, return and replay; q / Esc / Ctrl+C = abort.", flush=True)
            started = False
            while True:
                tick = time.monotonic()
                key = terminal.key()
                if key in ("q", "\x1b"):
                    return
                if key == "s":
                    break
                delta = mouse.velocity() * period
                moving = bool(np.any(delta))
                if moving:
                    if not servo.active:
                        servo.target = np.asarray(client.get_fresh_tool_pose(), dtype=float)
                    # Stage-2 converts spatial angular increments into tool-local ones.
                    delta[3:] = Rotation.from_rotvec(servo.target[3:]).inv().apply(delta[3:])
                    servo.send(delta)
                    started = True
                elif servo.active:
                    servo.stop()
                if started:
                    rows.append({"action": delta.tolist(), "send": moving})
                time.sleep(max(0, period - (time.monotonic() - tick)))
            servo.stop()
            save()
            if not rows:
                print(f"No motion recorded. Start pose saved to {output}", flush=True)
                return
            print(f"Saved {len(rows)} steps ({len(rows) * period:.1f}s) to {output}. "
                  "Returning to start joints, then replaying...", flush=True)
            # No automatic retry/replay after a failed return.
            client.move_joint(start_joints, speed=args.return_speed)
            returned = np.asarray(client.get_joint_angles_after_move())
            if np.max(np.abs(returned - start_joints)) > 0.02:
                raise RuntimeError("Return did not reach start joints; replay cancelled")
            client.get_tool_pose_after_move()
            servo.target = None
            for row in rows:
                tick = time.monotonic()
                if terminal.key() in ("q", "\x1b"):
                    return
                if row["send"]:
                    servo.send(np.asarray(row["action"]))
                elif servo.active:
                    servo.stop()
                time.sleep(max(0, period - (time.monotonic() - tick)))
            servo.stop()
            print("Replay complete.", flush=True)
    finally:
        try:
            servo.stop()
        finally:
            try:
                save()
            finally:
                if mouse is not None:
                    mouse.close()
                client.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
