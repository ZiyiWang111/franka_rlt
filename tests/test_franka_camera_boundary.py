"""Boundary freshness is exposure/state time, never image delivery time."""
from collections import deque
from types import SimpleNamespace
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from evo_franka.control_client import FrankaArmControllerClient
from evo_rlt.adapters.lerobot.franka_robot.camera_timing import ExposureClock, TimedObservation
from evo_rlt.adapters.lerobot.franka_robot.franka_robot import FrankaRobot
from act_rlt.stage2_runtime import FixedChunkRuntime
from act_rlt.stage2_timing import Stage2Timing


def convert(clock, number, **overrides):
    # Readout 10 ms ago, exposure midpoint 5 ms before readout, 4 ms exposure.
    mono = 100 + number / 30
    args = dict(domain="global_time", timestamp_ms=(1_700_000_000 + mono - .010) * 1000,
                frame_us=1_000_000 + number * 33333,
                sensor_us=995_000 + number * 33333, exposure_us=4000,
                number=number, wall_s=1_700_000_000 + mono, monotonic_s=mono)
    args.update(overrides)
    return clock.convert(**args)


def test_exposure_clock_maps_exposure_start_with_margin():
    clock = ExposureClock()
    assert convert(clock, 1)[1] == "clock_warming_up"
    assert convert(clock, 2)[1] == "clock_warming_up"
    (start, midpoint), reason = convert(clock, 3)
    assert reason is None
    assert midpoint == pytest.approx(100.1 - .015, abs=1e-6)
    assert start == pytest.approx(100.1 - .019, abs=1e-6)


@pytest.mark.parametrize("overrides,reason", [
    ({"domain": "hardware_clock"}, "timestamp_domain_hardware_clock"),
    ({"domain": "system_time"}, "timestamp_domain_system_time"),
    ({"timestamp_ms": float("nan")}, "nonfinite_timestamp"),
    ({"exposure_us": 0}, "invalid_exposure_metadata"),
    ({"sensor_us": 2_000_000}, "invalid_exposure_metadata"),
    ({"timestamp_ms": 1_800_000_000_000}, "implausible_frame_age"),
    ({"timestamp_ms": 1_600_000_000_000}, "implausible_frame_age"),
])
def test_invalid_camera_times_cannot_enable_early_boundary(overrides, reason):
    assert convert(ExposureClock(), 1, **overrides) == (None, reason)


def test_clock_jump_and_duplicate_require_revalidation():
    clock = ExposureClock()
    for i in range(1, 4):
        convert(clock, i)
    assert convert(clock, 3)[1] == "nonmonotonic_frame"
    assert convert(clock, 4)[1] == "clock_warming_up"
    assert convert(clock, 5, wall_s=1_700_000_101)[1] == "host_clock_jump"


def test_metadata_clock_rollover_is_supported():
    clock = ExposureClock()
    for i in range(1, 4):
        value, reason = convert(clock, i, frame_us=2000 + (i - 3) * 33333, sensor_us=(2000 + (i - 3) * 33333 - 5000) % 2**32)
    assert reason is None
    assert value[1] == pytest.approx(100.1 - .015, abs=1e-6)


def state_client(host="127.0.0.1"):
    client = object.__new__(FrankaArmControllerClient)
    client._host = host
    client._cache_lock = threading.Lock()
    client._state_history = deque([
        {"ts": 10.00, "joints": [0] * 7},
        {"ts": 10.01, "joints": [1] * 7},
        {"ts": 10.02, "joints": [2] * 7, "state_stale": True},
    ])
    return client


def test_state_history_uses_nearest_valid_snapshot_without_rpc():
    client = state_client()
    assert client.get_state_near(10.008)["joints"] == [1] * 7
    assert client.get_state_near(10.019)["ts"] == 10.01
    assert client.get_state_near(11) is None
    assert state_client("192.168.1.2").get_state_near(10.008) is None


def adapter(exposure=(10.02, 10.025), reason=None, state_ts=10.026):
    robot = object.__new__(FrankaRobot)
    robot._timestamped_observations = True
    robot._timing_status = None
    robot._has_gripper = False
    robot.robot = SimpleNamespace(get_state_near=Mock(return_value={
        "ts": state_ts, "joints": [1] * 7, "joint_speeds": [0] * 7,
        "pose": [0] * 6,
    }))
    robot._proprio_observation = lambda snapshot=None: {"joint_0": 0 if snapshot is None else 1}
    def images(timing):
        timing["wrist"] = (exposure, reason)
        return {"wrist": np.zeros((2, 2, 3), dtype=np.uint8)}
    robot._image_observation = images
    return robot


def test_timed_observation_replaces_only_state_with_aligned_snapshot():
    robot = adapter()
    timed = robot.get_timed_observation()
    assert timed.fresh_after == 10.02
    assert timed.observation["joint_0"] == 1
    assert timed.observation["wrist"].shape == (2, 2, 3)
    assert timed.diagnostics["camera_boundary_mode"] == "exposure_timestamp"
    assert timed.diagnostics["camera_state_skew_ms"] == pytest.approx(1)


def test_state_before_exposure_limits_observation_freshness():
    timed = adapter(state_ts=10.015).get_timed_observation()
    assert timed.fresh_after == 10.015


def test_unverifiable_timing_keeps_original_state_and_start_rule():
    robot = adapter(exposure=None, reason="missing_exposure_metadata")
    timed = robot.get_timed_observation()
    assert timed.observation["joint_0"] == 0
    assert timed.diagnostics["camera_boundary_mode"] == "legacy_fallback"
    assert "missing_exposure_metadata" in timed.diagnostics["camera_boundary_fallback_reason"]
    robot.robot.get_state_near.assert_not_called()


def test_missing_state_falls_back():
    robot = adapter()
    robot.robot.get_state_near.return_value = None
    timed = robot.get_timed_observation()
    assert timed.observation["joint_0"] == 0
    assert "no_local_state" in timed.diagnostics["camera_boundary_fallback_reason"]


def boundary_runtime(history):
    runtime = object.__new__(FixedChunkRuntime)
    runtime.env = SimpleNamespace(pre=lambda x: x)
    runtime.observation_ready = threading.Condition()
    runtime.observation_history = deque(history, maxlen=8)
    runtime.stop = threading.Event()
    return runtime


def test_boundary_accepts_inflight_capture_only_when_exposure_is_new(monkeypatch):
    monkeypatch.setattr("act_rlt.stage2_runtime.observation_frame", lambda x: x)
    runtime = boundary_runtime([
        # Both reads started before boundary=10, but only second exposure is new.
        (9.97, 10.01, 9.99, {"frame": 1}, {"camera_boundary_mode": "exposure_timestamp"}),
        (9.99, 10.04, 10.01, {"frame": 2}, {"camera_boundary_mode": "exposure_timestamp"}),
        (10.03, 10.07, 10.04, {"frame": 3}, {"camera_boundary_mode": "exposure_timestamp"}),
    ])
    timing = Stage2Timing()
    assert runtime._boundary_batch(10, timing) == {"frame": 2}
    assert timing.values["observation_start_after_request_ms"] < 0
    assert timing.values["observation_fresh_after_request_ms"] > 0


def test_legacy_fallback_rejects_capture_started_before_boundary(monkeypatch):
    monkeypatch.setattr("act_rlt.stage2_runtime.observation_frame", lambda x: x)
    runtime = boundary_runtime([
        (9.99, 10.04, 9.99, {"frame": 1}, {}),
        (10.04, 10.07, 10.04, {"frame": 2}, {}),
    ])
    assert runtime._boundary_batch(10) == {"frame": 2}


def test_camera_to_global_mapping_jump_is_rejected():
    clock = ExposureClock()
    for i in range(1, 4):
        convert(clock, i)
    # Device time advances 33 ms, global mapping suddenly advances 5 ms less.
    number = 4
    shifted = (1_700_000_000 + 100 + number / 30 - .015) * 1000
    assert convert(clock, number, timestamp_ms=shifted)[1] == "camera_clock_mapping_jump"


def test_missing_metadata_invalidates_clock_even_after_warmup():
    robot = object.__new__(FrankaRobot)
    clock = ExposureClock()
    for i in range(1, 4):
        convert(clock, i)
    robot._exposure_clocks = {"wrist": clock}
    frame = SimpleNamespace(get_frame_timestamp_domain=lambda: "global_time",
                            supports_frame_metadata=lambda _: False)
    assert robot._exposure_time("wrist", frame)[1] == "missing_exposure_metadata"
    assert clock.valid_frames == 0
