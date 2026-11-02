from collections import namedtuple
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from act_rlt.idle_xy import read_idle_xy_choice
from act_rlt.human_input import HumanInputMonitor
from act_rlt.train_stage2 import FrankaInsertionStage2Env


def run_idle(monkeypatch, key="KEY_D", *, fail=False, outside=False, held_on_enter=False):
    pose = np.array([.65, -.10, .402, .1, .2, .3])
    initial = pose.copy()
    events = []
    clock = [0.]
    monitor = SimpleNamespace(frame=0)

    def poll():
        monitor.frame += 1
        if fail and monitor.frame == 5:
            raise RuntimeError("keyboard disconnected")
        return "" if monitor.frame == 14 else None

    def servo(target):
        events.append(("servo", np.array(target)))
        pose[:] = target

    def stop():
        events.append(("stop", None))
        return True

    monitor.poll = poll
    monitor.pressed_keys = lambda: {key} if monitor.frame <= 10 or held_on_enter else set()
    monitor.close = lambda: events.append(("close", None))
    robot = SimpleNamespace(get_tool_pose=lambda: pose.copy(), servo_tool=servo, stop_servo=stop)
    env = SimpleNamespace(
        robot=SimpleNamespace(robot=robot, resync_command_pose=lambda: events.append(("resync", None))),
        workspace_min=initial[:3] - .05, workspace_max=initial[:3] + .05,
        teleop_input_backend="pynput", teleop_keyboard=None,
    )
    if outside:
        env.workspace_max[1] = pose[1]
    monkeypatch.setattr("act_rlt.idle_xy.HumanInputMonitor", lambda *args, **kwargs: monitor)
    monkeypatch.setattr("act_rlt.idle_xy.time.perf_counter", lambda: clock[0])
    monkeypatch.setattr("act_rlt.idle_xy.time.sleep", lambda duration: clock.__setitem__(0, clock[0] + .02))
    return env, initial, pose, events


@pytest.mark.parametrize("key,axis,sign", [
    ("KEY_W", 0, -1), ("KEY_S", 0, 1), ("KEY_A", 1, -1), ("KEY_D", 1, 1),
])
def test_held_wasd_uses_continuous_teleop_servo(monkeypatch, key, axis, sign):
    env, initial, pose, events = run_idle(monkeypatch, key)
    assert read_idle_xy_choice(env, "Ready?") == ""
    commands = [target for kind, target in events if kind == "servo"]
    assert len(commands) >= 3
    assert sign * (pose[axis] - initial[axis]) >= .0059
    for target in commands:
        np.testing.assert_array_equal(target[2:], initial[2:])
    # Releasing WASD stops before the later Enter closes the menu listener.
    assert [kind for kind, _ in events][-4:] == ["stop", "stop", "close", "resync"]


def test_enter_stops_motion_even_if_wasd_still_held(monkeypatch):
    env, _, _, events = run_idle(monkeypatch, held_on_enter=True)
    assert read_idle_xy_choice(env, "Ready?") == ""
    assert [kind for kind, _ in events][-3:] == ["stop", "close", "resync"]


def test_wasd_allows_motion_when_workspace_is_disabled(monkeypatch):
    env, initial, pose, events = run_idle(monkeypatch, outside=True)
    env.workspace_min = env.workspace_max = None
    assert read_idle_xy_choice(env, "Ready?") == ""
    assert pose[1] - initial[1] >= .0059
    assert any(kind == "servo" for kind, _ in events)


@pytest.mark.parametrize("failure", ["keyboard", "workspace"])
def test_failure_stops_servo_closes_keyboard_and_resyncs(monkeypatch, failure):
    env, _, _, events = run_idle(monkeypatch, fail=failure == "keyboard", outside=failure == "workspace")
    with pytest.raises(RuntimeError, match="keyboard disconnected|outside workspace"):
        read_idle_xy_choice(env, "Ready?")
    assert [kind for kind, _ in events][-3:] == ["stop", "close", "resync"]
    if failure == "workspace":
        assert not any(kind == "servo" for kind, _ in events)


def test_reset_menu_uses_continuous_controller(monkeypatch):
    env = object.__new__(FrankaInsertionStage2Env)
    calls = []
    monkeypatch.setattr("act_rlt.idle_xy.read_idle_xy_choice", lambda owner, prompt: calls.append((owner, prompt)) or "r")
    assert env._read_idle_reset_choice("Ready?") == "r"
    assert calls == [(env, "Ready?")]


@pytest.mark.parametrize("idle", [False, True])
def test_pynput_s_mapping_release_debounce_and_menu_keys(monkeypatch, idle):
    class Listener:
        def __init__(self, **callbacks):
            self.__dict__.update(callbacks)
            self.alive = True

        def start(self):
            pass

        def wait(self):
            pass

        def is_alive(self):
            return self.alive

        def stop(self):
            self.alive = False

        def join(self, **kwargs):
            pass

    special = SimpleNamespace(**{name: object() for name in ("space", "f8", "f9", "esc", "enter", "up", "down", "left", "right")})
    keyboard = SimpleNamespace(Key=special, Listener=Listener)
    monkeypatch.setitem(sys.modules, "pynput", SimpleNamespace(keyboard=keyboard))
    clock = [0.]
    monkeypatch.setattr("act_rlt.human_input.time.monotonic", lambda: clock[0])
    char = namedtuple("Char", "char")
    monitor = HumanInputMonitor("pynput", idle_xy=idle)
    listener = monitor._listener
    try:
        for backend in ("pynput", "evdev"):
            monitor.backend = backend
            listener.on_press(char("b"))
            listener.on_press(char("b"))
            assert monitor.consume_downward_bias() is (not idle)
            assert monitor.consume_downward_bias() is False
            assert monitor.poll() is None
            assert not monitor.pressed_keys()
        monitor.backend = "pynput"
        listener.on_press(char("s"))
        if not idle:
            assert monitor.poll() == "s"
            assert monitor.pressed_keys() == set()
            return
        assert monitor.poll() is None
        assert monitor.direction().tolist() == [1., 0., 0.]
        listener.on_release(char("s"))
        clock[0] += .01
        listener.on_press(char("s"))  # X11 auto-repeat cancels the pending release.
        clock[0] += .1
        assert monitor.pressed_keys() == {"KEY_S"}
        listener.on_release(char("s"))
        clock[0] += .07
        assert monitor.pressed_keys() == set()
        listener.on_press(special.enter)
        assert monitor.poll() == ""
        listener.on_press(char("p"))
        assert monitor.poll() == "p"
        for name in ("up", "down", "left", "right"):
            key = getattr(special, name)
            listener.on_press(key)
            assert monitor.poll() == f"KEY_{name.upper()}"
            listener.on_press(key)
            assert monitor.poll() is None
            listener.on_release(key)
            clock[0] += .01
            listener.on_press(key)  # ToDesk/X11 synthetic repeat release.
            assert monitor.poll() is None
            listener.on_release(key)
            clock[0] += .07
            listener.on_press(key)
            assert monitor.poll() == f"KEY_{name.upper()}"
        listener.on_press(char("0"))
        assert monitor.poll() == "KEY_0"
        for name in ("n", "m"):
            key = char(name)
            listener.on_press(key)
            assert monitor.poll() == f"KEY_{name.upper()}"
            assert monitor.pressed_keys() == set()
            listener.on_press(key)
            assert monitor.poll() is None
            listener.on_release(key)
            clock[0] += .01
            listener.on_press(key)
            assert monitor.poll() is None
            listener.on_release(key)
            clock[0] += .07
            listener.on_press(key)
            assert monitor.poll() == f"KEY_{name.upper()}"
        # evdev has real releases and filters native repeat events upstream.
        monitor.backend = "evdev"
        for key, name in ((special.up, "KEY_UP"), (char("n"), "KEY_N"), (char("m"), "KEY_M")):
            listener.on_release(key)
            listener.on_press(key)
            assert monitor.poll() == name
            listener.on_press(key)
            assert monitor.poll() is None
            listener.on_release(key)
            listener.on_press(key)
            assert monitor.poll() == name
    finally:
        monitor.close()
    assert not listener.is_alive()


def test_spacemouse_monitor_d_deletes_but_idle_d_still_moves(monkeypatch):
    from act_rlt.spacemouse_input import SpaceMouseHumanInputMonitor
    class Listener:
        def __init__(self, **callbacks):
            self.__dict__.update(callbacks)
        def start(self): pass
        def wait(self): pass
        def is_alive(self): return True
        def stop(self): pass
        def join(self, **kwargs): pass
    special = SimpleNamespace(**{name: object() for name in
                                ('space', 'f8', 'f9', 'esc', 'enter', 'up', 'down', 'left', 'right')})
    monkeypatch.setitem(sys.modules, 'pynput', SimpleNamespace(
        keyboard=SimpleNamespace(Key=special, Listener=Listener)))
    motion = SimpleNamespace(velocity=lambda: np.zeros(6))
    monitor = SpaceMouseHumanInputMonitor('pynput', None, motion)
    idle = HumanInputMonitor('pynput', idle_xy=True)
    try:
        key = namedtuple('Character', 'char')('d')
        monitor._listener.on_press(key)
        assert monitor.poll() == 'd'
        assert not monitor.pressed_keys()
        idle._listener.on_press(key)
        assert idle.poll() is None
        np.testing.assert_array_equal(idle.direction(), [0, 1, 0])
    finally:
        monitor.close()
        idle.close()
