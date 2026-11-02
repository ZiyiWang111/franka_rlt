"""Bridge the official franka_spacemouse Twist topic to Evo-RLT servo_tool.

Run with ROS Humble's Python 3.10 environment. Preview is the default and never
constructs a robot client. For --execute the control server must be running;
this node asks that existing server to establish/heal its FCI connection before
waiting for fresh state. It never starts a second control server.
"""
from __future__ import annotations

import argparse
import math
import os
import select
import sys
import termios
import time
import tty

import numpy as np
from scipy.spatial.transform import Rotation

YAW_SPEED_MULTIPLIER = 1.2


class EscapeKey:
    """Read a foreground terminal without requiring Enter; restore it on exit."""

    def __init__(self, stream=None):
        self.stream = sys.stdin if stream is None else stream
        self.fd = None
        self.saved = None

    def open(self):
        if self.stream.isatty():
            self.fd = self.stream.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

    def pressed(self):
        if self.fd is None or not select.select([self.fd], [], [], 0)[0]:
            return False
        data = os.read(self.fd, 128)
        # Arrow/function keys start with ESC too; a standalone ESC requests exit.
        if data == b'\x1b' and select.select([self.fd], [], [], 0.01)[0]:
            data += os.read(self.fd, 128)
        return data == b'\x1b'

    def close(self):
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            self.saved = None


class PoseIntegrator:
    """Integrate normalized velocities in the client's working coordinate frame."""

    def __init__(self, linear_speed=0.009, angular_speed=0.06, deadzone=0.05):
        if not all(math.isfinite(v) for v in (linear_speed, angular_speed, deadzone)):
            raise ValueError('parameters must be finite')
        if linear_speed <= 0 or angular_speed <= 0 or not 0 <= deadzone < 1:
            raise ValueError('speeds must be positive and deadzone in [0, 1)')
        self.scale = np.array([linear_speed] * 3 +
                              [angular_speed, angular_speed, angular_speed * YAW_SPEED_MULTIPLIER])
        self.deadzone = deadzone

    def velocity(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError('Twist must contain six finite values')
        values = np.clip(values, -1, 1)
        values = np.sign(values) * np.maximum(0, np.abs(values) - self.deadzone) / (1 - self.deadzone)
        return values * self.scale

    def step(self, pose, velocity, dt):
        pose = np.asarray(pose, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        if pose.shape != (6,) or velocity.shape != (6,) or not np.isfinite([pose, velocity]).all():
            raise ValueError('pose and velocity must contain six finite values')
        if not math.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError('dt must be in (0, 0.1]')
        result = pose.copy()
        result[:3] += velocity[:3] * dt
        # Spatial angular velocity: compose rotations, never add rotation vectors.
        result[3:] = (Rotation.from_rotvec(velocity[3:] * dt)
                      * Rotation.from_rotvec(pose[3:])).as_rotvec()
        return result


def check_excursion(pose, origin, max_distance, max_rotation):
    distance = np.linalg.norm(np.asarray(pose[:3]) - origin[:3])
    angle = (Rotation.from_rotvec(pose[3:]) * Rotation.from_rotvec(origin[3:]).inv()).magnitude()
    if ((max_distance is not None and distance > max_distance)
            or (max_rotation is not None and angle > max_rotation)):
        raise RuntimeError('Target exceeded the starting-pose translation or rotation limit')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Connect through the existing Evo-RLT server and send targets')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--topic', default='/franka_controller/target_cartesian_velocity_percent')
    parser.add_argument('--seconds', type=float, default=None, help='Optional duration; default: run until Esc or Ctrl+C')
    parser.add_argument('--linear-speed', type=float, default=0.009, help='m/s at full deflection')
    parser.add_argument('--angular-speed', type=float, default=0.06,
                        help='roll/pitch rad/s at full deflection; yaw uses 1.2 times this speed')
    parser.add_argument('--deadzone', type=float, default=0.05)
    parser.add_argument('--max-distance', type=float, default=None, help='Optional starting TCP distance limit in metres; default: disabled')
    parser.add_argument('--max-rotation', type=float, default=None, help='Optional starting TCP rotation limit in radians; default: disabled')
    args = parser.parse_args(argv)
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error('--seconds must be finite and positive')
    if any(v is not None and (not math.isfinite(v) or v <= 0)
           for v in (args.max_distance, args.max_rotation)):
        parser.error('excursion limits must be finite and positive')
    integrator = PoseIntegrator(args.linear_speed, args.angular_speed, args.deadzone)

    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.qos import QoSProfile, HistoryPolicy, DurabilityPolicy, ReliabilityPolicy

    rclpy.init(args=[])
    node = rclpy.create_node('evo_franka_spacemouse_bridge')
    client = None
    active = False
    received = 0
    nonzero = 0
    velocity = np.zeros(6)
    received_at = float('-inf')
    target = np.zeros(6)
    origin = target.copy()
    sent = 0
    started = time.monotonic()
    previous = started
    last_log = float('-inf')
    escape = EscapeKey()

    def receive(msg):
        nonlocal velocity, received_at, received, nonzero
        velocity = integrator.velocity([msg.linear.x, msg.linear.y, msg.linear.z,
                                        msg.angular.x, msg.angular.y, msg.angular.z])
        received_at = time.monotonic()
        received += 1
        nonzero += int(np.any(velocity))

    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                     reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE)
    node.create_subscription(Twist, args.topic, receive, qos)
    try:
        escape.open()
        if args.execute:
            from evo_franka.control_client import FrankaArmControllerClient
            client = FrankaArmControllerClient(host=args.host)
            # Match teleop.py: initialize the session through the shared server.
            # Client construction alone only opens ZMQ, not the robot connection.
            print('Connecting robot through the existing Evo-RLT control server...', flush=True)
            try:
                client.connect()
            except RuntimeError as exc:
                raise RuntimeError(f'Robot connection through control server failed: {exc}') from exc
            deadline = time.monotonic() + 3
            while True:
                if escape.pressed():
                    print('Esc pressed: stopping and exiting.', flush=True)
                    return
                try:
                    target = np.asarray(client.get_fresh_tool_pose(), dtype=float)
                    origin = target.copy()
                    break
                except RuntimeError as exc:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            'Control server connection completed, but no fresh TCP state '
                            'arrived within 3 seconds; refusing to send motion targets'
                        ) from exc
                    time.sleep(0.02)
            print(f'Robot connected; fresh TCP={target.tolist()}', flush=True)
        # Count optional run duration from readiness, excluding connection setup.
        started = previous = time.monotonic()
        print(f'{"EXECUTE" if client else "PREVIEW"}: listening on {args.topic}', flush=True)
        print('Press Esc or Ctrl+C in this terminal to stop and exit.', flush=True)
        while rclpy.ok() and (args.seconds is None or time.monotonic() - started < args.seconds):
            if escape.pressed():
                print('Esc pressed: stopping and exiting.', flush=True)
                break
            rclpy.spin_once(node, timeout_sec=0.005)
            now = time.monotonic()
            if now - previous < 1 / 30:
                continue
            dt = min(now - previous, 0.05)
            previous = now
            moving = now - received_at <= 0.2 and bool(np.any(velocity))
            if moving:
                if client:
                    measured = np.asarray(client.get_fresh_tool_pose())
                    if not active:
                        target = measured.copy()
                    candidate = integrator.step(target, velocity, dt)
                    pos_lead = np.linalg.norm(candidate[:3] - measured[:3])
                    rot_lead = (Rotation.from_rotvec(candidate[3:])
                                * Rotation.from_rotvec(measured[3:]).inv()).magnitude()
                    if pos_lead > 0.01 or rot_lead > 0.1:
                        raise RuntimeError('TCP target lead exceeded 1 cm or 0.1 rad')
                    check_excursion(candidate, origin, args.max_distance, args.max_rotation)
                    target = candidate
                    # Set first so a failed send still attempts a stop in finally.
                    active = True
                    client.servo_tool(target.tolist())
                    sent += 1
                else:
                    target = integrator.step(target, velocity, dt)
            elif active:
                if not client.stop_servo():
                    raise RuntimeError('control server did not acknowledge stop')
                active = False
            if now - last_log >= 0.5:
                print(f'messages={received} nonzero={nonzero} fresh={now-received_at <= 0.2} '
                      f'velocity={velocity.round(4).tolist()} target={target.round(4).tolist()}', flush=True)
                last_log = now
    except KeyboardInterrupt:
        pass
    finally:
        escape.close()
        if client:
            try:
                if active and not client.stop_servo():
                    print('ERROR: control server did not acknowledge final stop', flush=True)
            finally:
                client.close()
        node.destroy_node()
        rclpy.shutdown()
        print(f'SUMMARY messages={received} nonzero={nonzero} commands_sent={sent}', flush=True)


if __name__ == '__main__':
    main()
