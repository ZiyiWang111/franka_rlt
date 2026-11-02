"""Forward the official publisher's Twist as JSON; no robot connection."""
from __future__ import annotations

import argparse
import json
import os
import time

from evo_franka.spacemouse_bridge import PoseIntegrator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--linear-speed', type=float, default=0.009)
    parser.add_argument('--angular-speed', type=float, default=0.06,
                        help='roll/pitch rad/s at full deflection; yaw uses 1.2 times this speed')
    parser.add_argument('--deadzone', type=float, default=0.05)
    args = parser.parse_args()
    integrator = PoseIntegrator(args.linear_speed, args.angular_speed, args.deadzone)
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from spacemouse_publisher.pyspacemouse_publisher import SpaceMousePublisher

    rclpy.init(args=['--ros-args', '-p', f'device_path:={os.environ.get("SPACEMOUSE_DEVICE", "/dev/hidraw4")}'])
    node = None
    try:
        node = SpaceMousePublisher()
        if not node._device_open_success:
            raise RuntimeError('SpaceMouse HID open failed; check SPACEMOUSE_DEVICE and permissions')

        def receive(msg):
            velocity = integrator.velocity([msg.linear.x, msg.linear.y, msg.linear.z,
                                           msg.angular.x, msg.angular.y, msg.angular.z])
            print('SPACEMOUSE ' + json.dumps([time.monotonic(), velocity.tolist()]), flush=True)

        node.create_subscription(
            Twist, '/franka_controller/target_cartesian_velocity_percent', receive,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
