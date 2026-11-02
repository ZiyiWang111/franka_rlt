import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from evo_franka.spacemouse_bridge import PoseIntegrator


def test_excursion_limits():
    from evo_franka.spacemouse_bridge import check_excursion
    origin = np.array([0.4, 0, 0.3, 0, 0, 0])
    check_excursion(origin, origin, 0.02, 0.1)
    with pytest.raises(RuntimeError):
        check_excursion([0.421, 0, 0.3, 0, 0, 0], origin, 0.02, 0.1)
    with pytest.raises(RuntimeError):
        check_excursion([0.4, 0, 0.3, 0.11, 0, 0], origin, 0.02, 0.1)
    check_excursion([0.8, 0, 0.5, 1, 0, 0], origin, None, None)
    check_excursion([0.8, 0, 0.5, 0, 0, 0], origin, None, 0.1)
    check_excursion([0.4, 0, 0.3, 1, 0, 0], origin, 0.02, None)


def test_deadzone_and_full_scale():
    bridge = PoseIntegrator(linear_speed=0.03, angular_speed=0.2)
    np.testing.assert_allclose(bridge.velocity([0.04, 1, -1, 1, -1, 0]),
                               [0, 0.03, -0.03, 0.2, -0.2, 0])
    with pytest.raises(ValueError):
        bridge.velocity([float('nan')] * 6)


def test_rotation_is_composed_in_working_frame():
    bridge = PoseIntegrator()
    pose = np.array([0.4, 0.1, 0.3, 0, 0, np.pi / 2])
    result = bridge.step(pose, [0.03, 0, 0, 0.2, 0, 0], 0.05)
    expected = Rotation.from_rotvec([0.01, 0, 0]) * Rotation.from_rotvec(pose[3:])
    np.testing.assert_allclose(result[:3], [0.4015, 0.1, 0.3])
    np.testing.assert_allclose(Rotation.from_rotvec(result[3:]).as_matrix(), expected.as_matrix(), atol=1e-14)


def test_zero_input_and_delayed_tick():
    bridge = PoseIntegrator()
    pose = [0.4, 0, 0.3, 0.2, 0.1, 0]
    np.testing.assert_allclose(bridge.step(pose, np.zeros(6), 1 / 30), pose, atol=1e-15)
    with pytest.raises(ValueError):
        bridge.step(pose, np.zeros(6), 1)


def test_escape_key_and_terminal_restore():
    import os
    import pty
    import termios
    from evo_franka.spacemouse_bridge import EscapeKey
    master, slave = pty.openpty()
    stream = os.fdopen(slave, 'r')
    keyboard = EscapeKey(stream)
    saved = termios.tcgetattr(slave)
    try:
        keyboard.open()
        assert not keyboard.pressed()
        os.write(master, b'\x1b[A')
        assert not keyboard.pressed()
        os.write(master, b'\x1b')
        assert keyboard.pressed()
    finally:
        keyboard.close()
        assert termios.tcgetattr(slave) == saved
        stream.close()
        os.close(master)


@pytest.mark.parametrize('exit_with_escape', [False, True])
def test_ros_input_dispatch_and_timeout_stop(monkeypatch, exit_with_escape):
    """Real ROS transport, fake robot client: no FCI or control server needed."""
    import threading
    import time
    rclpy = pytest.importorskip('rclpy')
    from geometry_msgs.msg import Twist
    from evo_franka import control_client, spacemouse_bridge

    class Client:
        def __init__(self, **kwargs):
            self.commands = []
            self.stops = 0
            self.closed = False
            self.connected = False
            self.events = []

        def connect(self):
            self.events.append('connect')
            self.connected = True

        def get_fresh_tool_pose(self):
            assert self.connected
            self.events.append('state')
            return [0.4, 0, 0.3, 0, 0, 0]

        def servo_tool(self, pose):
            assert self.connected
            self.events.append('servo')
            self.commands.append(pose)

        def stop_servo(self):
            self.stops += 1
            return True

        def close(self):
            self.closed = True

    client = Client()
    monkeypatch.setattr(control_client, 'FrankaArmControllerClient', lambda **kw: client)
    if exit_with_escape:
        monkeypatch.setattr(spacemouse_bridge.EscapeKey, 'pressed', lambda self: len(client.commands) >= 2)
    topic = '/evo_spacemouse_test/twist'
    errors = []

    def publish():
        try:
            while not rclpy.ok():
                time.sleep(0.01)
            node = rclpy.create_node('evo_spacemouse_test_source')
            try:
                pub = node.create_publisher(Twist, topic, 1)
                deadline = time.monotonic() + 1.0
                while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                msg = Twist()
                msg.linear.x = 0.5
                msg.angular.y = -0.5
                for _ in range(10):
                    if not rclpy.ok():
                        break
                    pub.publish(msg)
                    time.sleep(0.02)
                # Stop publishing entirely: exercise the 0.2 s freshness timeout.
            finally:
                node.destroy_node()
        except rclpy._rclpy_pybind11.RCLError as exc:
            # Esc shuts down the context while this test source is publishing.
            if rclpy.ok():
                errors.append(exc)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=publish, daemon=True)
    thread.start()
    spacemouse_bridge.main(['--execute', '--seconds', '2', '--topic', topic])
    thread.join(timeout=2)
    assert not thread.is_alive() and not errors
    assert client.closed and client.stops == 1
    assert client.events[:2] == ['connect', 'state']
    assert len(client.commands) >= 2
    assert all(p[0] > 0.4 and p[4] < 0 for p in client.commands)


def test_connection_failure_closes_client_without_motion(monkeypatch):
    pytest.importorskip('rclpy')
    from evo_franka import control_client, spacemouse_bridge

    class FailedClient:
        closed = False

        def connect(self):
            raise RuntimeError('FCI unavailable')

        def get_fresh_tool_pose(self):
            pytest.fail('must not read state after failed connect')

        def servo_tool(self, *args):
            pytest.fail('must not send motion after failed connect')

        def disconnect(self):
            pytest.fail('must not disconnect the shared server')

        def close(self):
            self.closed = True

    client = FailedClient()
    monkeypatch.setattr(control_client, 'FrankaArmControllerClient', lambda **kw: client)
    with pytest.raises(RuntimeError, match='connection through control server failed.*FCI unavailable'):
        spacemouse_bridge.main(['--execute', '--seconds', '0.05'])
    assert client.closed


def test_preview_does_not_construct_robot_client(monkeypatch):
    pytest.importorskip('rclpy')
    from evo_franka import control_client, spacemouse_bridge

    def forbidden(**kwargs):
        pytest.fail('preview must not construct a robot client')

    monkeypatch.setattr(control_client, 'FrankaArmControllerClient', forbidden)
    spacemouse_bridge.main(['--seconds', '0.05'])
