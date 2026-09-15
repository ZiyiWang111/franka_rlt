#!/usr/bin/env python3
"""Validate the decoupled control IPC WITHOUT a robot.

A FakeController stands in for ``evo_franka.driver.FrankaArmController``
(monkeypatched into control_server), so the full client<->server ZMQ path is exercised
deterministically: transport + msgpack, the blocking accepted+poll handshake,
state streaming DURING a move, stop_move preemption, servo fire-and-forget +
coalescing, the servo-gap watchdog, error propagation, and busy rejection.

Needs pyzmq + msgpack + franky importable (control_server imports the real
controller module at import time). Run on host-009:

    cd ~/demo && ~/miniconda3/envs/bm2/bin/python tests/test_control_ipc.py

Exits non-zero on any failure. Also discoverable by pytest (test_* functions).
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# control_server imports the real franka driver (franky); both modules pull in
# pyzmq + msgpack. None are present offline (this Mac), so the IPC stack is
# import-guarded -- the tests skip here and run for real on host-009 (bm2 env),
# mirroring tests/test_eval_loop.py / tests/test_robot_interface.py.
try:
    import evo_franka.control_server as cs            # noqa: E402  (imports franky)
    from evo_franka.control_client import FrankaArmControllerClient  # noqa: E402
    IPC_AVAILABLE, _IPC_ERR = True, None
except Exception as _exc:  # noqa: BLE001
    cs = FrankaArmControllerClient = None
    IPC_AVAILABLE, _IPC_ERR = False, _exc

# Under pytest, skip the whole module when the IPC stack (msgpack/zmq) is absent
# -- as it is on the Mac dev box. The plain-script __main__ path below already
# skips via _Skip, but the pytest entry points did not, so they errored on
# cs=None. This just wires the module's documented intent ("the tests skip here
# and run for real on host-009") into the pytest path. Guarded so the plain
# `python tests/test_control_ipc.py` run still works if pytest is not installed.
try:
    import pytest as _pytest
    pytestmark = _pytest.mark.skipif(
        not IPC_AVAILABLE,
        reason=f"IPC stack (msgpack/zmq) unavailable offline: "
               f"{type(_IPC_ERR).__name__ if _IPC_ERR else 'n/a'}")
except Exception:  # noqa: BLE001 - pytest absent on the plain-script path
    pass


class _Skip(Exception):
    pass


class FakeController:
    """Pure-Python stand-in for FrankaArmController (no franky, no robot)."""

    def __init__(self, ip, rdf=0.15, *a, **k):
        self.ip = ip
        self.q = [0.0] * 7
        self.pose = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
        self.connected = False
        self.state_fallbacks_total = 0
        self.servo_count = 0
        self.last_servo = None
        self.stop_move_called = 0
        self.stop_servo_called = 0
        self._stop = False
        self._fail_next = False
        self.gripper_started_at = None
        self.gripper_motion = {
            "command_id": 0, "command": None, "status": "idle", "result": None,
            "error": None, "width": 0.076, "grasped": False,
            "measurement_stale": False,
        }
        self.move_steps = 20          # tests override for long/short moves
        self.move_dt = 0.02
        self._lock = threading.Lock()

    # lifecycle
    def connect(self, *a, **k): self.connected = True
    def disconnect(self, *a, **k): self.connected = False
    def reset_session(self, *a, **k): return None
    def heal_session(self, *a, **k): self.connected = True
    def recover(self): return None
    def wait_until_ready(self, *a, **k): return True

    # state reads (the hot snapshot)
    def get_joint_angles(self):
        with self._lock:
            return list(self.q)

    def get_joint_speeds(self): return [0.0] * 7
    def get_tool_pose(self):
        with self._lock:
            return list(self.pose)

    def get_tool_force_raw(self): return [0.0] * 6
    def is_running(self): return False
    def robot_mode(self): return "Idle"
    def is_user_stopped(self): return False
    def reset_fallback_counters(self):
        self.state_fallbacks_total = 0
        return 0

    def has_gripper(self): return True
    def get_gripper_width(self): return self.gripper_motion["width"]
    def is_grasped(self): return self.gripper_motion["grasped"]
    def start_close_gripper(self, width=0.0, **kwargs):
        self.gripper_started_at = time.monotonic()
        self.gripper_motion.update(
            command_id=self.gripper_motion["command_id"] + 1,
            command="close", status="running", result=None, error=None,
            measurement_stale=True,
        )
        self.gripper_target = width
        return dict(self.gripper_motion)

    def get_gripper_motion_state(self):
        if (self.gripper_motion["status"] == "running"
                and time.monotonic() - self.gripper_started_at >= 0.25):
            self.gripper_motion.update(
                status="finished", result=True, width=self.gripper_target,
                grasped=True, measurement_stale=False,
            )
        return dict(self.gripper_motion)

    def stop_gripper(self):
        self.gripper_motion.update(status="stopped", result=True, measurement_stale=False)
        return True

    # a slow, preemptible blocking move (mimics sliced-join + abort-on-stop)
    def _slow_move(self, target):
        self._stop = False
        if self._fail_next:
            self._fail_next = False
            raise RuntimeError("Franka move rejected [control reflex]: simulated")
        for _ in range(self.move_steps):
            if self._stop:
                raise RuntimeError("move aborted [stopped]")
            with self._lock:
                self.q[0] += 0.01
            time.sleep(self.move_dt)
        with self._lock:
            self.pose = list(target)
        return list(self.pose)

    def move_tool(self, pose, *a, **k): return self._slow_move(list(pose))
    def move_joint(self, q, *a, **k): return self._slow_move(self.pose)
    def move_tool_traj(self, path, *a, **k):
        return self._slow_move(list(path[-1][:6]) if path else self.pose)
    def move_until_force(self, pose, *a, **k): return (self._slow_move(list(pose)), False)
    def check_move_hang(self): pass   # driver owns the async hang cap (F4); no-op stand-in
    def finish_move(self): return True

    # servo (fast / non-blocking)
    def servo_tool(self, pose, *a, **k):
        self.servo_count += 1
        self.last_servo = list(pose)
        with self._lock:
            self.pose = list(pose)

    def servo_joint(self, q, *a, **k): self.servo_count += 1

    # stops
    def stop_move(self):
        self._stop = True
        self.stop_move_called += 1

    def stop_servo(self, *a, **k): self.stop_servo_called += 1
    def stop_joints(self, *a, **k): pass
    def stop_tool(self, *a, **k): pass


# ── harness ─────────────────────────────────────────────────────────────────
_PORT = [16100]


def _setup():
    cs.FrankaArmController = FakeController          # inject the fake backend
    _PORT[0] += 2
    cmd, state = _PORT[0], _PORT[0] + 1
    srv = cs.ControlServer("192.0.2.1", 0.15, cmd_port=cmd, state_port=state, state_poll_hz=100.0)
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    time.sleep(0.4)                                  # let it bind
    cli = FrankaArmControllerClient(host="127.0.0.1", cmd_port=cmd, state_port=state)
    return srv, cli, th


def _teardown(srv, cli, th):
    try:
        cli.close()
    except Exception:
        pass
    srv._running = False
    srv._event.set()
    th.join(timeout=3.0)
    time.sleep(0.15)


def _bg_move(cli, target, sink=None):
    try:
        r = cli.move_tool(target)
        if sink is not None:
            sink.append(("ok", r))
    except Exception as e:
        if sink is not None:
            sink.append(("err", str(e)))


def _close(a, b, tol=1e-9):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


# ── tests ───────────────────────────────────────────────────────────────────
def test_connect_and_state_stream():
    """connect succeeds, .robot becomes truthy, and state reads come off the stream."""
    srv, cli, th = _setup()
    try:
        cli.connect()
        assert cli.robot, "client.robot truthy after connect"
        assert srv._ctrl.connected, "server controller connected"
        time.sleep(0.2)                              # let the PUSH stream prime the cache
        q = cli.get_joint_angles()
        assert isinstance(q, list) and len(q) == 7, q
        assert cli.robot_mode() == "Idle", cli.robot_mode()
    finally:
        _teardown(srv, cli, th)


def test_explicit_async_move_tool_uses_managed_completion_path():
    """Async move_tool must be polled/joined, while blocking move_tool keeps its helper."""
    srv = cs.ControlServer.__new__(cs.ControlServer)
    routed = []
    srv._start_move = lambda cmd: routed.append(("managed", cmd.name))
    srv._spawn_move = lambda cmd: routed.append(("helper", cmd.name))

    srv._process(cs._Cmd("move_tool", {"kwargs": {"is_async": True}}))
    srv._process(cs._Cmd("move_tool", {"kwargs": {}}))
    srv._process(cs._Cmd("move_tool_traj", {"kwargs": {}}))

    assert routed == [
        ("managed", "move_tool"),
        ("helper", "move_tool"),
        ("managed", "move_tool_traj"),
    ]


def test_managed_async_move_stays_busy_until_finish_move_joins():
    class TrackedController:
        def __init__(self):
            self.running = True
            self.issued = []
            self.finish_calls = 0

        def move_tool_traj(self, path, *, is_async):
            self.issued.append((path, is_async))

        def check_move_hang(self):
            return None

        def is_running(self):
            return self.running

        def finish_move(self):
            self.finish_calls += 1
            return True

    srv = cs.ControlServer.__new__(cs.ControlServer)
    srv._ctrl = TrackedController()
    srv._servo_active = False
    srv._move_active = None
    srv._move_t0 = 0.0
    srv._worker_result = None
    srv._busy = True
    srv._fault_detail = lambda: "test"
    cmd = cs._Cmd("move_tool_traj", {"args": [[[0.1] * 9]], "kwargs": {}})

    srv._start_move(cmd)
    assert srv._busy and srv._move_active is cmd
    assert srv._ctrl.issued == [([[0.1] * 9], True)]

    srv._move_t0 = time.monotonic() - cs._MOVE_ENGAGE_GRACE_S - 0.01
    srv._poll_active_move()
    assert srv._busy and srv._ctrl.finish_calls == 0

    srv._ctrl.running = False
    srv._poll_active_move()
    assert not srv._busy
    assert srv._move_active is None
    assert srv._ctrl.finish_calls == 1
    assert srv._worker_result["success"] is True


def test_blocking_move_returns_value():
    """A blocking move returns the controller's reached pose across the IPC."""
    srv, cli, th = _setup()
    try:
        cli.connect()
        srv._ctrl.move_steps = 3
        target = [0.5, 0.1, 0.2, 0.0, 0.0, 0.0]
        out = cli.move_tool(target)
        assert _close(out, target), f"move returned reached pose: {out}"
    finally:
        _teardown(srv, cli, th)


def test_state_readable_during_move():
    """The core property: state keeps streaming (and changing) WHILE a move runs."""
    srv, cli, th = _setup()
    try:
        cli.connect()
        srv._ctrl.move_steps, srv._ctrl.move_dt = 40, 0.02      # ~0.8 s
        sink = []
        mover = threading.Thread(target=_bg_move, args=(cli, [0.9, 0, 0, 0, 0, 0], sink))
        mover.start()
        time.sleep(0.15)
        first = cli.get_joint_angles()[0]
        busy = (cli._send("get_state").get("state") or {}).get("worker_status")
        time.sleep(0.25)
        second = cli.get_joint_angles()[0]
        mover.join(timeout=3.0)
        assert busy == "busy", f"worker busy during move, got {busy}"
        assert second > first, f"joints live during move: {first} -> {second}"
        assert not mover.is_alive() and sink and sink[0][0] == "ok", f"move finished ok: {sink}"
    finally:
        _teardown(srv, cli, th)


def test_state_and_cached_gripper_observation_continue_during_async_grasp():
    """A Hand Future never occupies the controller loop or the client RPC socket."""
    srv, cli, th = _setup()
    try:
        cli.connect()
        time.sleep(0.1)
        before_seq = cli._snap()["seq"]
        t0 = time.monotonic()
        started = cli.start_close_gripper(width=0.019)
        assert time.monotonic() - t0 < 0.1, "async grasp start must return promptly"
        assert started["status"] == "running"
        time.sleep(0.08)
        during = cli.get_gripper_motion_state()
        after_seq = cli._snap()["seq"]
        assert during["status"] == "running"
        assert during["measurement_stale"] is True
        assert cli.get_gripper_width() == 0.076
        assert cli.is_grasped() is False
        assert after_seq > before_seq, "state snapshots must continue during Hand motion"
        time.sleep(0.25)
        done = cli.get_gripper_motion_state()
        assert done["status"] == "finished" and done["result"] is True
        assert cli.get_gripper_width() == 0.019
        assert cli.is_grasped() is True
    finally:
        _teardown(srv, cli, th)


def test_stop_move_preempts():
    """stop_move from another thread aborts a wedged move promptly (the headline)."""
    srv, cli, th = _setup()
    try:
        cli.connect()
        srv._ctrl.move_steps, srv._ctrl.move_dt = 200, 0.02     # ~4 s if uninterrupted
        sink = []
        mover = threading.Thread(target=_bg_move, args=(cli, [0.9, 0, 0, 0, 0, 0], sink))
        mover.start()
        time.sleep(0.2)
        t0 = time.monotonic()
        cli.stop_move()
        mover.join(timeout=3.0)
        elapsed = time.monotonic() - t0
        assert not mover.is_alive(), "move did not stop"
        assert elapsed < 1.5, f"stop too slow: {elapsed:.2f}s (move was ~4 s)"
        assert srv._ctrl.stop_move_called >= 1, "controller.stop_move was called"
        assert sink and sink[0][0] == "err" and "abort" in sink[0][1].lower(), \
            f"aborted move raised across IPC: {sink}"
    finally:
        _teardown(srv, cli, th)


def test_servo_fire_and_forget_and_coalesce():
    """servo_tool is non-blocking, reaches the controller, and is coalesced to the latest."""
    srv, cli, th = _setup()
    try:
        cli.connect()
        last = None
        for i in range(50):
            last = [0.30 + i * 0.001, 0.0, 0.0, 0.0, 0.0, 0.0]
            cli.servo_tool(last)                     # returns None, never blocks
        time.sleep(0.2)                              # < servo-gap; let the loop apply
        assert srv._ctrl.servo_count >= 1, "servo reached the controller"
        assert srv._ctrl.last_servo is not None and _close(srv._ctrl.last_servo, last), \
            f"latest-wins: applied {srv._ctrl.last_servo}, sent {last}"
        assert srv._ctrl.servo_count < 50, \
            f"coalesced: {srv._ctrl.servo_count} applies for 50 sends"
    finally:
        _teardown(srv, cli, th)


def test_servo_gap_watchdog_stops():
    """After the servo gap with no new target, the server gracefully stops the servo."""
    srv, cli, th = _setup()
    try:
        cli.connect()
        cli.servo_tool([0.3, 0, 0, 0, 0, 0])
        time.sleep(0.1)
        assert srv._ctrl.stop_servo_called == 0, "no premature stop within the gap"
        time.sleep(0.4)                              # exceed CONTROL_SERVER_SERVO_GAP_S (0.3 s)
        assert srv._ctrl.stop_servo_called >= 1, "watchdog stopped the servo after the gap"
    finally:
        _teardown(srv, cli, th)


def test_move_error_propagates_with_tag():
    """A controller-side move failure surfaces as a client RuntimeError, tag preserved."""
    srv, cli, th = _setup()
    try:
        cli.connect()
        srv._ctrl.move_steps = 3
        srv._ctrl._fail_next = True
        raised = None
        try:
            cli.move_tool([0.5, 0, 0, 0, 0, 0])
        except Exception as e:
            raised = str(e)
        assert raised is not None, "move should raise"
        assert "control reflex" in raised, f"failure tag preserved across IPC: {raised}"
    finally:
        _teardown(srv, cli, th)


def test_client_reraises_server_error_types():
    """The server's _err() envelope carries error_type; the client re-raises it
    TYPED, so typed branches (the collector's start-resample loop, teleop's
    session healing) work identically in-process and behind the server.
    Unknown or missing types stay RuntimeError."""
    from evo_franka.base_errors import JointSwingExceeded
    from evo_franka.control_client import _raise_failure
    from evo_franka.errors import FrankaMotionRefused

    def raised_by(result):
        try:
            _raise_failure(result, "fallback")
        except Exception as e:  # noqa: BLE001
            return e
        return None

    e = raised_by({"error": "IK did not converge", "error_type": "FrankaMotionRefused"})
    assert isinstance(e, FrankaMotionRefused) and "IK did not converge" in str(e)
    e = raised_by({"error": "77deg over budget", "error_type": "JointSwingExceeded"})
    assert isinstance(e, JointSwingExceeded)
    e = raised_by({"error": "worker busy"})                       # no error_type
    assert type(e) is RuntimeError
    e = raised_by({"error": "??", "error_type": "SomethingNew"})  # unknown name
    assert type(e) is RuntimeError


def test_busy_rejects_second_move():
    """A second blocking move while one is in flight is rejected, not queued."""
    srv, cli, th = _setup()
    try:
        cli.connect()
        srv._ctrl.move_steps, srv._ctrl.move_dt = 60, 0.02      # ~1.2 s
        mover = threading.Thread(target=_bg_move, args=(cli, [0.9, 0, 0, 0, 0, 0], None))
        mover.start()
        time.sleep(0.2)
        raised = None
        try:
            cli.move_tool([0.1, 0, 0, 0, 0, 0])
        except Exception as e:
            raised = str(e)
        mover.join(timeout=3.0)
        assert raised and "busy" in raised.lower(), f"second move rejected while busy: {raised}"
    finally:
        _teardown(srv, cli, th)


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = skipped = 0
    for fn in tests:
        try:
            if not IPC_AVAILABLE:
                raise _Skip(f"IPC stack absent: {type(_IPC_ERR).__name__}")
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except _Skip as s:
            print(f"SKIP  {fn.__name__}: {s}")
            skipped += 1
        except Exception as e:
            import traceback
            print(f"FAIL  {fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
