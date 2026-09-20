"""Diagnostics must not touch hardware, especially after a connection failure."""
import time
from types import SimpleNamespace

from evo_franka.control_server import ControlServer, _Cmd
from evo_franka.errors import FrankaSessionLost
from evo_franka.motion import MotionMixin
from evo_franka.state import StateMixin
from evo_franka._franky import NetworkException


class DeadRobot:
    @property
    def state(self):
        raise AssertionError("diagnostics attempted hardware I/O")


def cached():
    return dict(ts=time.monotonic(), wall_time=time.time(), mode="RobotMode.Move",
                ccsr=0.985, current_errors="[]", last_motion_errors="[]", has_errors=False)


def test_fault_classification_uses_cache_only():
    ctrl = SimpleNamespace(robot=DeadRobot(), _diagnostic_state=cached())
    fault = MotionMixin._classify_move_fault(ctrl, NetworkException("server closed connection"), "async")
    assert isinstance(fault, FrankaSessionLost)
    assert "last_cached_robot_mode=RobotMode.Move" in str(fault)
    assert "Programming" not in str(fault)


def test_server_reports_cached_evidence_and_stops_polling(caplog):
    server = ControlServer("192.0.2.220", 0.15)
    server._ctrl = SimpleNamespace(robot=DeadRobot(), _diagnostic_state=cached())
    server._cycle_gap_max_ms = 37
    server._record_evidence()
    assert server._evidence[-1]["cycle_gap_max_ms"] == 37
    assert server._evidence[-1]["ccsr"] == 0.985
    server._snap = {"pose": [1] * 6, "ts": 123}
    result = server._err(FrankaSessionLost("link lost"))
    assert "FCI pre-fault evidence" in caplog.text
    assert "age_ms" in caplog.text
    snap = server._read_snapshot()
    assert snap["state_stale"] and snap["ts"] == 123
    assert snap["session_fault"] == result
    server._busy = True
    server._move_active = _Cmd("move_tool_traj", {})
    server._poll_active_move()
    assert server._worker_result == result and not server._busy
    cmd = _Cmd("move_joint", {})
    server._process(cmd)
    assert cmd.event.is_set() and cmd.result == result


def test_evidence_is_bounded_and_keeps_sample_age():
    server = ControlServer("192.0.2.220", 0.15)
    server._ctrl = SimpleNamespace(_diagnostic_state=cached())
    server._ctrl._diagnostic_state["ts"] -= 2
    for _ in range(230):
        server._last_evidence_ts = 0
        server._record_evidence()
    assert len(server._evidence) == 200
    assert server._evidence[-1]["age_ms"] >= 2000


def test_mode_sampling_reuses_single_existing_state_read():
    class Robot:
        reads = 0
        @property
        def state(self):
            self.reads += 1
            return SimpleNamespace(robot_mode="Move", control_command_success_rate=0.998,
                                   current_errors=[], last_motion_errors=[])
    robot = Robot()
    ctrl = SimpleNamespace(robot=robot, _read_with_fallback=lambda key, fn: fn())
    assert StateMixin.robot_mode(ctrl) == "Move"
    assert robot.reads == 1
    assert ctrl._diagnostic_state["ccsr"] == 0.998


def test_explicit_recovery_clears_latch_only_on_success():
    server = ControlServer("192.0.2.220", 0.15)
    def fail():
        raise FrankaSessionLost("still disconnected")
    server._ctrl = SimpleNamespace(recover=fail)
    server._session_fault = {"success": False, "error": "lost"}
    failed = _Cmd("recover", {})
    server._process(failed)
    assert not failed.result["success"] and server._session_fault is not None
    server._ctrl.recover = lambda: None
    recovered = _Cmd("recover", {})
    server._process(recovered)
    assert recovered.result["success"] and server._session_fault is None
