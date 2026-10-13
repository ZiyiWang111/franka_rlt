"""Robot-free checks of motion-local limits and the two-Enter experiment.

Run with the Franka-capable Python:
PYTHONPATH=src python -m unittest discover -s tests -p test_franka_soft_approach.py
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from act_rlt import soft_approach as experiment

try:
    import evo_franka.motion as motion_module
    from evo_franka.kinematics import FR3Kinematics
    from evo_franka.constants import LEFT_HOME_JOINTS
except ImportError:
    motion_module = None


@unittest.skipIf(motion_module is None, "franky unavailable")
class MotionLocalDynamicsTests(unittest.TestCase):
    def make_driver(self):
        class ForbiddenRobotMutation:
            relative_dynamics_factor = (0.15, 0.15, 0.08)

            def __setattr__(self, name, value):
                raise AssertionError(f"robot-wide mutation: {name}")

            @property
            def joint_velocity_limit(self):
                return SimpleNamespace(get=lambda: np.full(7, 2.0))

        class Driver(motion_module.MotionMixin):
            def __init__(self):
                self.robot = ForbiddenRobotMutation()
                self.relative_dynamics_factor = 0.15
                self.kin = FR3Kinematics()
                self.q = np.asarray(LEFT_HOME_JOINTS)
                self.pose = motion_module.matrix_to_pose(self.kin.fk(self.q))
                self.issued = []
                self.fail = False

            def _require(self): pass
            def _require_kin(self): pass
            def _require_hand_idle_for_arm_motion(self): pass
            def _warn_and_recover(self): pass
            def _log_traj_excursion(self, *args): pass
            def _new_to_base(self, pose): return pose
            def _tool_pose_base(self): return self.pose
            def get_tool_pose(self): return self.pose
            def get_joint_angles(self): return self.q.tolist()

            def _safe_move(self, motion, is_async):
                self.issued.append((motion, is_async))
                if self.fail:
                    raise RuntimeError("simulated control failure")

        return Driver()

    def test_local_native_factor_does_not_leak_into_next_motion(self):
        driver = self.make_driver()
        target = list(driver.pose)
        target[2] -= 0.04
        constructor = motion_module.JointWaypointMotion
        with patch.object(motion_module, "JointWaypointMotion", wraps=constructor) as record:
            driver.move_tool(target, speed=0.05, is_async=True, motion_dynamics=[1, 0.1, 0.002])
            factor = record.call_args.kwargs["relative_dynamics_factor"]
            self.assertAlmostEqual(factor.velocity, 1.0)
            self.assertAlmostEqual(factor.acceleration, 0.1)
            self.assertAlmostEqual(factor.jerk, 0.002)
            driver.move_tool(target, speed=0.05)
            self.assertEqual(record.call_args.kwargs, {})
        self.assertEqual(driver.robot.relative_dynamics_factor, (0.15, 0.15, 0.08))
        self.assertEqual(driver.relative_dynamics_factor, 0.15)
        self.assertTrue(driver.issued[0][1])

    def test_control_failure_does_not_change_global_dynamics(self):
        driver = self.make_driver()
        target = list(driver.pose)
        target[2] -= 0.04
        driver.fail = True
        with self.assertRaisesRegex(RuntimeError, "simulated control failure"):
            driver.move_tool(target, motion_dynamics=[1, 0.1, 0.002])
        driver.fail = False
        constructor = motion_module.JointWaypointMotion
        with patch.object(motion_module, "JointWaypointMotion", wraps=constructor) as record:
            driver.move_tool(target)
            self.assertEqual(record.call_args.kwargs, {})
        self.assertEqual(driver.robot.relative_dynamics_factor, (0.15, 0.15, 0.08))

    def test_invalid_dynamics_never_issue_a_motion(self):
        driver = self.make_driver()
        for factors in ([1, 0, 0.01], [1, 1.1, 0.01], [1, float("nan"), 0.1], [1, 0.1]):
            with self.subTest(factors=factors), self.assertRaises(ValueError):
                driver.move_tool_traj([driver.pose], motion_dynamics=factors)
        self.assertEqual(driver.issued, [])

    def test_blended_waypoint_respects_reduced_velocity_and_braking_limits(self):
        driver = self.make_driver()
        pose1, pose2 = list(driver.pose), list(driver.pose)
        pose1[2] -= 0.02
        pose2[2] -= 0.05
        parsed = driver._parse_tcp_waypoints(
            [pose1 + [0.05, 0, 1], pose2 + [0.05, 0, 0]], "test")
        waypoints = driver._solve_branch_continuous_ik(
            driver.pose, parsed, 0.4, motion_dynamics=[0.1, 0.01, 0.002])
        q1, q2 = waypoints[0].target.position, waypoints[1].target.position
        velocity = np.abs(waypoints[0].target.velocity)
        self.assertTrue(np.all(velocity <= driver._joint_velocity_ceiling() * 0.1))
        brake_cap = np.sqrt(2 * motion_module.FR3_MIN_RATED_JOINT_ACCEL_RAD_S2
                            * motion_module.ACCELERATION_DYNAMICS_FACTOR * 0.01
                            * np.abs(q2 - q1) * motion_module.PASSTHROUGH_BRAKE_MARGIN)
        self.assertTrue(np.all(velocity <= brake_cap + 1e-12))
        np.testing.assert_array_equal(waypoints[-1].target.velocity, np.zeros(7))



class OnlinePlannerTests(unittest.TestCase):
    def test_xy_priority_with_early_z_and_exact_reference_goal(self):
        args = experiment.build_parser().parse_args([])
        start = np.array([0.4, 0.1, 0.4, 0, 0, 0])
        goal = np.array([0.45, 0, 0.32, 0, 0, 0.2])
        planner = experiment.OnlineApproach(start, goal, args)
        descended = False
        for _ in range(1200):
            previous = planner.command.copy()
            diagnostics = planner.step(previous, 1 / 30)
            if abs(planner.command[2] - start[2]) > 1e-5 and not descended:
                self.assertGreater(diagnostics["lateral_command_error_m"], args.lateral_gate_distance)
                self.assertAlmostEqual(diagnostics["vertical_gate"], args.vertical_speed_floor)
                descended = True
            self.assertLessEqual(np.linalg.norm(planner.command[:3] - previous[:3]) * 30, args.speed + 1e-9)
            if planner.at_goal():
                break
        self.assertTrue(descended)
        self.assertTrue(planner.at_goal())
        np.testing.assert_array_equal(planner.command, goal)

    def test_vertical_allowance_opens_gradually_over_a_wider_xy_range(self):
        args = experiment.build_parser().parse_args([])
        allowances = []
        for error in (0.08, 0.06, 0.04, 0.02, 0.01, 0.0):
            planner = experiment.OnlineApproach([0]*6, [error, 0, -0.1, 0, 0, 0], args)
            allowances.append(planner.step([0]*6, 1 / 30)["vertical_gate"])
        np.testing.assert_allclose(allowances, [0.15, 0.15, 0.4492, 0.9116, 1.0, 1.0])
        self.assertTrue(np.all(np.diff(allowances) >= 0))
        for value in ("-0.1", "1.1", "nan"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                experiment.validate_args(experiment.build_parser().parse_args(["--vertical-speed-floor", value]))

    def test_small_measured_offset_does_not_rebase_or_change_reference_velocity(self):
        args = experiment.build_parser().parse_args([])
        start, target = [0, 0, 0, 0, 0, 0], [0.1, 0, -0.1, 0, 0, 0]
        first, second = experiment.OnlineApproach(start, target, args), experiment.OnlineApproach(start, target, args)
        first.step(start, 1 / 30)
        second.step([-0.001, 0, 0, 0, 0, 0], 1 / 30)
        np.testing.assert_array_equal(first.command, second.command)
        np.testing.assert_array_equal(first.velocity, second.velocity)

    def test_large_tracking_lag_pauses_without_snapping_to_actual_pose(self):
        args = experiment.build_parser().parse_args([])
        planner = experiment.OnlineApproach([0, 0, 0, 0, 0, 0], [0.1, 0, -0.1, 0, 0, 0], args)
        for _ in range(20):
            planner.step(planner.command.copy(), 1 / 30)
        command = planner.command.copy()
        measured = command.copy()
        measured[0] -= 0.02
        diagnostics = planner.step(measured, 1 / 30)
        self.assertTrue(diagnostics["position_paused"])
        np.testing.assert_array_equal(planner.command[:3], command[:3])
        self.assertNotEqual(planner.command[0], measured[0])

    def test_reference_acceleration_slew_is_limited_and_delayed_tick_is_rejected(self):
        args = experiment.build_parser().parse_args([])
        planner = experiment.OnlineApproach([0]*6, [0.1, 0, 0, 0, 0, 0], args)
        for _ in range(20):
            acceleration = planner.acceleration.copy()
            planner.step(planner.command.copy(), 1 / 30)
            self.assertLessEqual(np.linalg.norm(planner.acceleration - acceleration), args.jerk / 30 + 1e-12)
        with self.assertRaisesRegex(RuntimeError, "间隔过长"):
            planner.step(planner.command.copy(), 0.3)


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.connected = False
        self.connect_count = 0
        self.events = []
        self.motion_kwargs = None
        self.targets = []
        self.pose = [0.4, 0.0, 0.4, 0, 0, 0]
        self.seq = 0
        self.move_complete = False
        self.servo_active = False
        self.servo_commands = []

    def __enter__(self): return self
    def __exit__(self, *args): self.events.append("close")

    def connect(self):
        self.events.append("connect")
        self.connected = True
        self.connect_count += 1
        self.pose[0] = 0.4 if self.connect_count == 1 else 0.5

    def disconnect(self):
        self.events.append("disconnect")
        self.connected = False

    def _send(self, command):
        if command == "ping":
            return {"success": True, "connected": self.connected, "busy": False,
                    "capabilities": []}
        assert command == "get_state"
        self.seq += 1
        return {"success": True, "state": {
            "pose": list(self.pose), "joints": [0.0] * 7, "seq": self.seq,
            "joint_speeds": [0.0] * 7, "servo_fault_seq": 0,
            "ts": time.monotonic(), "robot_mode": "Move" if self.servo_active else "Idle",
            "is_running": self.servo_active,
            "worker_status": "idle", "state_stale": False,
        }, "result": {"success": True} if self.move_complete else None}

    def move_tool_traj(self, path, **kwargs):
        raise AssertionError("online script must use servo_tool, not move_tool_traj")

    def servo_tool(self, pose):
        if not self.servo_active:
            self.events.append("servo_start")
        self.servo_active = True
        self.servo_commands.append(list(pose))
        self.pose = list(pose)

    def stop_servo(self):
        self.events.append("stop_servo")
        self.targets.append(list(self.pose))
        self.servo_active = False
        return True

    def stop_move(self):
        self.events.append("stop")
        return True


class VirtualClock:
    def __init__(self): self.now = 100.0
    def monotonic(self): return self.now
    def sleep(self, seconds): self.now += seconds


class ExperimentTests(unittest.TestCase):
    def stream(self, client, target, args):
        clock = VirtualClock()
        with patch.object(experiment.time, "monotonic", clock.monotonic), \
                patch.object(experiment.time, "sleep", clock.sleep), contextlib.redirect_stdout(io.StringIO()):
            start = client._send("get_state")["state"]
            return experiment.run_online_move(client, start, target, args, lambda *a, **k: None)

    def exercise(self, client, directory, interrupt=False, cycles=1):
        args = experiment.build_parser().parse_args(["--log", str(Path(directory) / "test.jsonl")])
        clock = VirtualClock()
        with patch("evo_franka.control_client.FrankaArmControllerClient", return_value=client), \
                patch("builtins.input", side_effect=[""] * (cycles + 1) + ["q"]), \
                patch.object(experiment.time, "monotonic", clock.monotonic), \
                patch.object(experiment.time, "sleep", clock.sleep), \
                contextlib.redirect_stdout(io.StringIO()):
            if interrupt:
                with patch.object(experiment, "run_online_move", side_effect=KeyboardInterrupt), \
                        self.assertRaises(KeyboardInterrupt):
                    experiment.run(args)
            else:
                experiment.run(args)
        return [json.loads(line) for line in args.log.read_text().splitlines()]

    def test_two_enter_workflow_releases_hand_guiding_and_reports_final_error(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            rows = self.exercise(client, directory)
        self.assertEqual(client.events, ["connect", "disconnect", "connect", "servo_start",
                                         "stop_servo", "disconnect", "close"])
        self.assertGreater(len(client.servo_commands), 10)
        self.assertEqual(rows[0]["path_mode"], "online_command_integrator")
        self.assertTrue(any(row["event"] == "settled" for row in rows))
        finished = [row for row in rows if row["event"] == "finished"]
        self.assertEqual(finished[-1]["position_error_mm"], 0)
        self.assertEqual(rows[-1]["event"], "session_end")

    def test_repeated_moves_keep_original_target_and_release_each_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            rows = self.exercise(client, directory, cycles=2)
        self.assertEqual(client.events, ["connect", "disconnect", "connect", "servo_start", "stop_servo",
                                         "disconnect", "connect", "servo_start", "stop_servo", "disconnect", "close"])
        self.assertEqual(len(client.targets), 2)
        self.assertEqual(client.targets[0], client.targets[1])
        self.assertEqual(client.targets[0], [0.4, 0.0, 0.4, 0, 0, 0])
        self.assertEqual([row["cycle"] for row in rows if row["event"] == "finished"], [1, 2])
        self.assertEqual(rows[-1]["completed_cycles"], 2)

    def test_interrupt_stops_before_disconnect(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            rows = self.exercise(client, directory, interrupt=True)
        self.assertEqual(client.events[-3:], ["stop", "disconnect", "close"])
        self.assertEqual(rows[-1]["event"], "aborted")

    def test_stale_state_is_rejected_and_rotation_error_handles_rotvec_wrap(self):
        with self.assertRaisesRegex(RuntimeError, "过期"):
            experiment.checked_snapshot({"state": {"ts": time.monotonic() - 2}})
        error_mm, error_deg = experiment.pose_error(
            [0, 0, 0, 0, 0, np.pi - 0.01], [0, 0, 0, 0, 0, -np.pi + 0.01])
        self.assertEqual(error_mm, 0)
        self.assertAlmostEqual(error_deg, np.degrees(0.02))

    def test_actual_goal_must_be_reached_even_after_reference_finishes(self):
        class LaggingClient(FakeClient):
            def servo_tool(self, pose):
                super().servo_tool(pose)
                self.pose[0] += 0.003
        client = LaggingClient()
        client.pose[0] = 0.5
        args = experiment.build_parser().parse_args(["--timeout", "12"])
        with self.assertRaisesRegex(RuntimeError, "尚未实际到位"):
            self.stream(client, [0.4, 0, 0.4, 0, 0, 0], args)
        self.assertEqual(client.servo_commands[-1], [0.4, 0, 0.4, 0, 0, 0])
        self.assertNotIn("stop_servo", client.events)

    def test_already_at_goal_checks_stability_without_starting_zero_distance_motion(self):
        client = FakeClient()
        target = list(client.pose)
        self.stream(client, target, experiment.build_parser().parse_args([]))
        self.assertEqual(client.servo_commands, [])

    def test_new_servo_fault_is_not_reported_as_success(self):
        class FaultClient(FakeClient):
            def _send(self, command):
                response = super()._send(command)
                if command == "get_state" and len(self.servo_commands) >= 2:
                    response["state"].update(servo_fault_seq=1, servo_fault={"error": "simulated reflex"})
                return response
        client = FaultClient()
        client.pose[0] = 0.5
        with self.assertRaisesRegex(RuntimeError, "servo 控制失败"):
            self.stream(client, [0.4, 0, 0.4, 0, 0, 0], experiment.build_parser().parse_args([]))


if __name__ == "__main__":
    unittest.main()
