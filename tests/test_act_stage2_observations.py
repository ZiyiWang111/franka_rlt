"""Cached Stage-2 observations: sequence freshness is not boundary freshness."""
from collections import deque
from types import SimpleNamespace
import threading
import time
from unittest.mock import Mock

import pytest

from act_rlt.stage2_runtime import FixedChunkRuntime
from act_rlt.stage2_timing import Stage2Timing
from act_rlt.train_stage2 import build_parser


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr("act_rlt.stage2_runtime.observation_frame", lambda observation: observation)
    runtime = object.__new__(FixedChunkRuntime)
    runtime.env = SimpleNamespace(pre=lambda observation: observation)
    runtime.observation_selection = "latest"
    runtime.observation_ready = threading.Condition()
    runtime.observation_history = deque(maxlen=8)
    runtime.observation_seq = 0
    runtime.last_observation_seq = 0
    runtime.stop = threading.Event()
    return runtime


def publish(runtime, *, fresh_after=None):
    completed = time.monotonic()
    if fresh_after is None:
        fresh_after = completed - .02
    with runtime.observation_ready:
        runtime.observation_seq += 1
        sequence = runtime.observation_seq
        runtime.observation_history.append((
            completed - .03, completed, fresh_after, {"frame": sequence},
            {"observation_seq": sequence, "camera_boundary_mode": "legacy_fallback"},
        ))
        runtime.observation_ready.notify_all()
    return sequence


def test_latest_uses_newest_cached_frame_even_before_last_command(runtime):
    for _ in range(3):
        publish(runtime)
    after = time.monotonic()
    timing = Stage2Timing()
    runtime.env.robot = SimpleNamespace(get_observation=Mock(side_effect=AssertionError("cache must be used")))

    assert runtime._boundary_batch(after, timing) == {"frame": 3}
    assert runtime.last_observation_seq == 3
    assert timing.values["camera_cache_hit"] is True
    assert timing.values["observation_selection"] == "latest"
    assert timing.values["observation_seq_gap"] == 3
    assert timing.values["observation_fresh_after_request_ms"] < 0
    assert timing.values["observation_start_after_request_ms"] < 0
    assert timing.values["observation_cache_age_ms"] >= 0
    assert timing.values["observation_age_ms"] >= timing.values["observation_cache_age_ms"]
    runtime.env.robot.get_observation.assert_not_called()


@pytest.mark.parametrize("stop_waiting", [False, True])
def test_latest_waits_without_reusing_previous_frame(runtime, stop_waiting):
    publish(runtime)
    assert runtime._boundary_batch(time.monotonic()) == {"frame": 1}
    waiting = threading.Event()
    finished = threading.Event()
    results = []
    original_wait = runtime.observation_ready.wait_for

    def wait_for(predicate, timeout):
        waiting.set()
        return original_wait(predicate, timeout)

    runtime.observation_ready.wait_for = wait_for
    timing = Stage2Timing()
    after = time.monotonic()

    def consume():
        try:
            results.append(runtime._boundary_batch(after, timing))
        finally:
            finished.set()

    worker = threading.Thread(target=consume)
    worker.start()
    try:
        assert waiting.wait(1)
        with runtime.observation_ready:
            assert not finished.is_set()
            assert runtime.last_observation_seq == 1
            if stop_waiting:
                runtime.stop.set()
                runtime.observation_ready.notify_all()
            else:
                # A later delivered frame may still have pre-boundary exposure.
                publish(runtime, fresh_after=after - .01)
        assert finished.wait(1)
    finally:
        runtime.stop.set()
        with runtime.observation_ready:
            runtime.observation_ready.notify_all()
        worker.join(timeout=1)
    assert not worker.is_alive()
    if stop_waiting:
        assert results == [None]
        assert runtime.last_observation_seq == 1
    else:
        assert results == [{"frame": 2}]
        assert runtime.last_observation_seq == 2
        assert timing.values["camera_cache_hit"] is False
        assert timing.values["observation_seq_gap"] == 1
        assert timing.values["observation_fresh_after_request_ms"] < 0


def test_latest_timeout_does_not_reuse_stale_frame(runtime):
    publish(runtime)
    runtime._boundary_batch(time.monotonic())
    runtime.observation_ready.wait_for = Mock(return_value=False)
    with pytest.raises(RuntimeError, match="newer observation.*latest"):
        runtime._boundary_batch(time.monotonic())
    assert runtime.last_observation_seq == 1
    assert runtime.observation_ready.wait_for.call_args.kwargs["timeout"] == 2.0


def test_boundary_option_still_requires_post_command_exposure(runtime):
    runtime.observation_selection = "boundary"
    after = time.monotonic()
    publish(runtime, fresh_after=after - .01)
    publish(runtime, fresh_after=after + .01)
    publish(runtime, fresh_after=after + .02)
    timing = Stage2Timing()

    assert runtime._boundary_batch(after, timing) == {"frame": 2}
    assert runtime.last_observation_seq == 2
    assert timing.values["observation_fresh_after_request_ms"] > 0
    # The history contains frame 2, but it cannot be consumed twice.
    assert runtime._boundary_batch(after) == {"frame": 3}


def test_train_cli_defaults_to_latest_and_allows_boundary():
    parser = build_parser()
    base = ["--stage1-checkpoint", "/tmp/stage1", "--act-checkpoint", "/tmp/act"]
    assert parser.parse_args(base).observation_selection == "latest"
    assert parser.parse_args(base + ["--observation-selection", "boundary"]).observation_selection == "boundary"

