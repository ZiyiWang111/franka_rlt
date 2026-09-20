"""Ownership regressions; all addresses are documentation-only, no hardware."""
import os
import subprocess
import sys
import threading
import selectors

import pytest

from evo_franka.runtime_owner import OwnershipError, RuntimeOwner, require_owner

IP = "192.0.2.219"


def child(code):
    env = dict(os.environ, PYTHONPATH=os.path.abspath("src"))
    return subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=10)


def test_exclusion_and_release():
    owner = RuntimeOwner(IP)
    owner.acquire()
    try:
        require_owner(IP)
        with pytest.raises(OwnershipError):
            RuntimeOwner(IP).acquire()
        result = child(f"from evo_franka.runtime_owner import RuntimeOwner; RuntimeOwner('{IP}').acquire()")
        assert result.returncode != 0 and "already owned" in result.stderr
    finally:
        owner.release()
    with pytest.raises(OwnershipError):
        require_owner(IP)
    result = child(f"from evo_franka.runtime_owner import RuntimeOwner; RuntimeOwner('{IP}').acquire()")
    assert result.returncode == 0, result.stderr
    owner.acquire()  # child exit releases its lock even without explicit release
    owner.release()


def test_fork_does_not_inherit_authorization_or_unlock_parent():
    owner = RuntimeOwner(IP)
    owner.acquire()
    try:
        pid = os.fork()
        if pid == 0:
            try:
                require_owner(IP)
            except OwnershipError:
                os._exit(0)
            os._exit(1)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        require_owner(IP)
        result = child(f"from evo_franka.runtime_owner import RuntimeOwner; RuntimeOwner('{IP}').acquire()")
        assert result.returncode != 0
    finally:
        owner.release()


def test_native_factories_and_all_connection_paths_refuse_without_owner(monkeypatch):
    from evo_franka import _franky as native
    from evo_franka.driver import FrankaArmController
    from evo_franka.gripper import FrankaGripperSession
    calls = []
    monkeypatch.setattr(native, "Robot", lambda *a, **k: calls.append("arm"))
    monkeypatch.setattr(native, "Gripper", lambda *a, **k: calls.append("hand"))
    ctrl = FrankaArmController(IP)
    for call in (lambda: native.create_robot(IP), lambda: native.create_gripper(IP),
                 ctrl.connect, ctrl._connect_hand, ctrl._rebuild_hand,
                 FrankaGripperSession(IP).connect):
        with pytest.raises(OwnershipError):
            call()
    assert calls == []
    owner = RuntimeOwner(IP)
    owner.acquire()
    try:
        native.create_robot(IP)
        native.create_gripper(IP)
        assert calls == ["arm", "hand"]
    finally:
        owner.release()


def test_server_start_failure_releases_owner(monkeypatch):
    from evo_franka.control_server import ControlServer
    server = ControlServer(IP, 0.15)
    def fail():
        require_owner(IP)
        raise ValueError("startup failed")
    monkeypatch.setattr(server, "_serve_owned_loop", fail)
    with pytest.raises(ValueError, match="startup failed"):
        server.run()
    owner = RuntimeOwner(IP)
    owner.acquire()
    owner.release()


def test_session_disconnect_and_rebuild_keep_ownership(monkeypatch):
    from evo_franka import _franky as native
    from evo_franka.driver import FrankaArmController
    from types import SimpleNamespace
    calls = []
    def robot(ip):
        calls.append("arm")
        return SimpleNamespace(recover_from_errors=lambda: None,
                               set_cartesian_impedance=lambda value: None)
    def hand(ip):
        calls.append("hand")
        return SimpleNamespace(width=0.08)
    monkeypatch.setattr(native, "Robot", robot)
    monkeypatch.setattr(native, "Gripper", hand)
    ctrl = FrankaArmController(IP)
    monkeypatch.setattr(ctrl, "_post_connect", lambda r: None)
    monkeypatch.setattr(ctrl, "_apply_dynamics_factor", lambda r: None)
    monkeypatch.setattr(ctrl, "stop_move", lambda: None)
    monkeypatch.setattr(ctrl, "get_gripper_motion_state", lambda: {"status": "idle"})
    owner = RuntimeOwner(IP)
    owner.acquire()
    try:
        ctrl.connect()
        ctrl.reset_session()
        assert ctrl._rebuild_hand()
        ctrl.disconnect()
        require_owner(IP)
        ctrl.connect()
        assert calls == ["arm", "hand", "arm", "hand", "hand", "arm", "hand"]
        with pytest.raises(OwnershipError):
            RuntimeOwner(IP).acquire()
    finally:
        ctrl.disconnect()
        owner.release()


def test_shutdown_keeps_ownership_until_helper_finishes(monkeypatch):
    from evo_franka.control_server import ControlServer
    from types import SimpleNamespace
    server = ControlServer(IP, 0.15)
    stopped = threading.Event()
    finish = threading.Event()
    errors = []
    def serve():
        server._ctrl = SimpleNamespace(stop_move=stopped.set, disconnect=lambda: None)
        server._helper = threading.Thread(target=lambda: finish.wait(5))
        server._helper.start()
    monkeypatch.setattr(server, "_serve_owned_loop", serve)
    def run():
        try:
            server.run()
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert stopped.wait(3)
        with pytest.raises(OwnershipError):
            RuntimeOwner(IP).acquire()
    finally:
        finish.set()
        thread.join(3)
    assert not thread.is_alive()
    assert not errors
    owner = RuntimeOwner(IP)
    owner.acquire()
    owner.release()


def test_state_port_bind_failure_cleans_up(monkeypatch):
    import zmq
    from types import SimpleNamespace
    from evo_franka.control_server import ControlServer
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUSH)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    server = ControlServer(IP, 0.15, cmd_port=0, state_port=port)
    monkeypatch.setattr(server, "_build_controller",
                        lambda: SimpleNamespace(disconnect=lambda: None))
    try:
        with pytest.raises(zmq.ZMQError):
            server.run()
        owner = RuntimeOwner(IP)
        owner.acquire()
        owner.release()
    finally:
        socket.close()
        ctx.term()


def test_sigterm_finishes_cleanup_before_releasing_lock():
    code = f"""
import time
from types import SimpleNamespace
from evo_franka.control_server import ControlServer
from evo_franka.runtime_owner import require_owner
server = ControlServer('{IP}', 0.15)
def cleanup():
    require_owner('{IP}')
    print('CLEANED', flush=True)
def serve():
    server._ctrl = SimpleNamespace(disconnect=cleanup)
    print('READY', flush=True)
    while server._running:
        time.sleep(0.01)
server._serve_owned_loop = serve
server.run()
"""
    proc = subprocess.Popen([sys.executable, "-c", code],
                            env=dict(os.environ, PYTHONPATH=os.path.abspath("src")),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=10), "child did not start"
        assert proc.stdout.readline().strip() == "READY"
        with pytest.raises(OwnershipError):
            RuntimeOwner(IP).acquire()
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=10)
        assert proc.returncode == 0, stderr
        assert "CLEANED" in stdout
        owner = RuntimeOwner(IP)
        owner.acquire()
        owner.release()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=5)
