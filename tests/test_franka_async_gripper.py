from __future__ import annotations

import pytest

from evo_franka.errors import FrankaGripperError, FrankaMotionRefused
from evo_franka.gripper import GripperMixin
from evo_franka.motion import MotionMixin


class _Future:
    def __init__(self, finish):
        self.ready = False
        self._finish = finish

    def wait(self, timeout=None):
        return self.ready

    def get(self):
        return self._finish()


class _Hand:
    def __init__(self):
        self.width = 0.076
        self.is_grasped = False
        self.future = None

    def grasp_async(self, width, speed, force, **kwargs):
        def finish():
            self.width = width
            self.is_grasped = True
            return True

        self.future = _Future(finish)
        return self.future

    def open_async(self, speed):
        def finish():
            self.width = 0.08
            self.is_grasped = False
            return True

        self.future = _Future(finish)
        return self.future

    def stop(self):
        return True


class _Controller(MotionMixin, GripperMixin):
    def __init__(self):
        self.gripper = _Hand()
        self.robot = object()

    def _require(self):
        return None

    def is_running(self):
        return False


def test_native_async_grasp_keeps_last_measurement_until_future_finishes():
    ctrl = _Controller()
    started = ctrl.start_close_gripper(width=0.019)
    assert started["status"] == "running"
    assert started["width"] == pytest.approx(0.076)
    assert ctrl.get_gripper_width() == pytest.approx(0.076)
    assert ctrl.is_grasped() is False
    with pytest.raises(FrankaGripperError, match="is running"):
        ctrl.start_open_gripper()

    ctrl.gripper.future.ready = True
    done = ctrl.get_gripper_motion_state()
    assert done["status"] == "finished"
    assert done["result"] is True
    assert done["measurement_stale"] is False
    assert done["width"] == pytest.approx(0.019)
    assert done["grasped"] is True


def test_arm_motion_gate_can_observe_hand_future_without_blocking():
    ctrl = _Controller()
    ctrl.start_open_gripper()
    with pytest.raises(FrankaMotionRefused, match="arm motion refused"):
        ctrl._require_hand_idle_for_arm_motion()
