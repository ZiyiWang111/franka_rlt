import io
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from evo_rlt.adapters.lerobot.franka_remote.execution import (
    GripperStateMachine,
    RelativeForceGuard,
    TcpActionFilter,
    clip_tcp_action,
)
from evo_rlt.adapters.lerobot.franka_remote.protocol import (
    ACTION_CHUNK_SHAPE,
    ACTION_NAMES,
    STATE_NAMES,
    InferenceResponse,
    decode_request,
    decode_response,
    encode_inference_request,
    encode_inference_response,
    encode_ready_response,
)
from evo_rlt.adapters.lerobot.franka_remote.state import observation_to_state15
import scripts.fr3_remote_robot_client as remote_client
from scripts.fr3_remote_robot_client import (
    AsyncInferenceResult,
    execute_action_chunk,
    require_expected_checkpoint,
    run_async_control,
)


def test_protocol_round_trip_preserves_arrays_and_schema():
    state = np.arange(15, dtype=np.float32)
    wrist = np.zeros((480, 640, 3), dtype=np.uint8)
    front = np.full((480, 640, 3), 127, dtype=np.uint8)
    frames = encode_inference_request(
        request_id=7,
        timestamp_ns=123,
        task="insert the VGA",
        state=state,
        wrist=wrist,
        front=front,
    )
    kind, request_id, request = decode_request(frames)
    assert kind == "infer"
    assert request_id == 7
    assert request is not None
    np.testing.assert_array_equal(request.state, state)
    np.testing.assert_array_equal(request.wrist, wrist)
    np.testing.assert_array_equal(request.front, front)

    action = np.arange(350, dtype=np.float32).reshape(ACTION_CHUNK_SHAPE)
    response = decode_response(
        encode_inference_response(
            request_id=7,
            action_chunk=action,
            inference_ms=42.5,
            checkpoint="030000",
        ),
        expected_request_id=7,
    )
    assert isinstance(response, InferenceResponse)
    np.testing.assert_array_equal(response.action_chunk, action)


def test_ready_response_checks_the_full_schema():
    ready = decode_response(encode_ready_response(0, "030000"), expected_request_id=0)
    assert ready["state_names"] == list(STATE_NAMES)
    assert ready["action_names"] == list(ACTION_NAMES)
    assert ready["action_chunk_shape"] == list(ACTION_CHUNK_SHAPE)


def test_checkpoint_binding_is_exact():
    require_expected_checkpoint("/models/030000/pretrained_model", "/models/030000/pretrained_model")
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        require_expected_checkpoint("/models/029000/pretrained_model", "/models/030000/pretrained_model")


def test_observation_to_state15_drops_joint_velocities_in_the_expected_order():
    observation = {name: float(index) for index, name in enumerate(STATE_NAMES)}
    observation.update({f"joint_vel_{index}": 100.0 + index for index in range(7)})
    state = observation_to_state15(observation)
    np.testing.assert_array_equal(state, np.arange(15, dtype=np.float32))


def test_gripper_transition_holds_arm_and_only_fires_once():
    machine = GripperStateMachine(confirm_steps=2)
    machine.sync_from_observation(0.076, grasped=False)
    first = machine.update(0.019)
    second = machine.update(0.018)
    assert first.hold_arm and first.command is None
    assert second.hold_arm and second.command == "close"
    machine.mark_completed("close")
    assert machine.update(0.018).command is None
    assert not machine.update(0.018).hold_arm


def test_tcp_clip_preserves_direction_and_limits_vector_norms():
    action = np.asarray([0.03, 0.04, 0.0, 0.0, 0.06, 0.08, 0.019], dtype=np.float32)
    clipped = clip_tcp_action(action, max_translation_m=0.005, max_rotation_rad=0.02)
    assert np.linalg.norm([clipped[name] for name in ACTION_NAMES[:3]]) <= 0.00500001
    assert np.linalg.norm([clipped[name] for name in ACTION_NAMES[3:6]]) <= 0.02000001


def test_tcp_action_filter_low_passes_and_limits_acceleration_and_speed():
    action_filter = TcpActionFilter(
        fps=30.0,
        low_pass_alpha=1.0,
        max_translation_step_m=0.005,
        max_rotation_step_rad=0.02,
        max_translation_accel_m_s2=0.30,
        max_rotation_accel_rad_s2=0.90,
    )
    action = np.asarray([0.02, 0, 0, 0.08, 0, 0, 0.076], dtype=np.float32)
    first = action_filter.update(action)
    second = action_filter.update(action)
    assert first["dx"] == pytest.approx(0.30 / 30.0**2)
    assert second["dx"] == pytest.approx(2.0 * 0.30 / 30.0**2)
    assert first["drx"] == pytest.approx(0.90 / 30.0**2)
    assert second["drx"] == pytest.approx(2.0 * 0.90 / 30.0**2)
    for _ in range(100):
        output = action_filter.update(action)
    assert output["dx"] == pytest.approx(0.005)
    assert output["drx"] == pytest.approx(0.02)


def test_tcp_action_filter_reset_restarts_from_zero_velocity():
    action_filter = TcpActionFilter(
        fps=30.0,
        low_pass_alpha=0.2,
        max_translation_step_m=0.005,
        max_rotation_step_rad=0.02,
        max_translation_accel_m_s2=0.30,
        max_rotation_accel_rad_s2=0.90,
    )
    action = np.asarray([0.005, 0, 0, 0, 0, 0, 0.076], dtype=np.float32)
    first = action_filter.update(action)
    action_filter.update(action)
    action_filter.reset()
    restarted = action_filter.update(action)
    assert restarted == first


def test_relative_force_guard_uses_translational_change_from_baseline():
    guard = RelativeForceGuard(
        baseline_wrench=np.asarray([1.0, -2.0, 3.0, 0.1, 0.2, 0.3]),
        threshold_n=7.0,
    )
    below = guard.check(np.asarray([4.0, 2.0, 7.0, 9.0, 9.0, 9.0]))
    assert below["force_delta_n"] == pytest.approx(np.sqrt(41.0))
    assert below["triggered"] is False
    at_limit = guard.check(np.asarray([8.0, -2.0, 3.0, 0.1, 0.2, 0.3]))
    assert at_limit["force_delta_n"] == pytest.approx(7.0)
    assert at_limit["triggered"] is True


class _FakeRobot:
    def __init__(self):
        self.arm_actions = []
        self.close_count = 0
        self.close_widths = []
        self.resync_count = 0
        self.transition_prepare_count = 0
        self.transition_finish_count = 0
        self.transition_prepared = False
        self.external_wrench = np.zeros(6, dtype=np.float64)
        self.force_retreats = []

    def send_action(self, action):
        self.arm_actions.append(action)

    def close_gripper(self, width=None):
        assert self.transition_prepared
        self.close_count += 1
        self.close_widths.append(width)
        return True

    def open_gripper(self):
        raise AssertionError("unexpected open")

    def resync_command_pose(self):
        self.resync_count += 1

    def prepare_gripper_transition(self):
        assert not self.transition_prepared
        self.transition_prepared = True
        self.transition_prepare_count += 1

    def finish_gripper_transition(self):
        assert self.transition_prepared
        self.transition_prepared = False
        self.transition_finish_count += 1

    def get_external_wrench_base(self):
        return self.external_wrench.copy()

    def stop_and_retreat_up(self, *, distance_m, speed_m_s):
        self.force_retreats.append((distance_m, speed_m_s))
        return {
            "start_pose": [0.0] * 6,
            "target_pose": [0.0, 0.0, distance_m, 0.0, 0.0, 0.0],
            "measured_pose": [0.0, 0.0, distance_m, 0.0, 0.0, 0.0],
            "distance_m": distance_m,
            "speed_m_s": speed_m_s,
        }


def _test_action_filter(*, fps=1e9):
    return TcpActionFilter(
        fps=fps,
        low_pass_alpha=1.0,
        max_translation_step_m=0.005,
        max_rotation_step_rad=0.03,
        max_translation_accel_m_s2=1e20,
        max_rotation_accel_rad_s2=1e20,
    )


def test_integrated_executor_freezes_arm_during_gripper_confirmation_and_command():
    robot = _FakeRobot()
    machine = GripperStateMachine(confirm_steps=2)
    machine.sync_from_observation(0.076, grasped=False)
    chunk = np.zeros(ACTION_CHUNK_SHAPE, dtype=np.float32)
    chunk[:, :6] = 0.003
    chunk[:, 6] = 0.019
    records, replan = execute_action_chunk(
        robot=robot,
        chunk=chunk,
        mode="integrated",
        execute_steps=5,
        fps=1e9,
        gripper=machine,
        action_filter=_test_action_filter(),
    )
    assert replan
    assert robot.arm_actions == []
    assert robot.close_count == 1
    assert robot.close_widths == [pytest.approx(0.019)]
    assert robot.resync_count == 1
    assert robot.transition_prepare_count == 1
    assert robot.transition_finish_count == 1
    assert not robot.transition_prepared
    assert records[-1]["gripper_command"] == "close"
    assert records[-1]["arm_servo_stopped"] is True


def test_executor_refuses_expired_chunk_before_arm_action():
    robot = _FakeRobot()
    machine = GripperStateMachine(confirm_steps=2)
    chunk = np.zeros(ACTION_CHUNK_SHAPE, dtype=np.float32)
    with pytest.raises(TimeoutError, match="stale action chunk"):
        execute_action_chunk(
            robot=robot,
            chunk=chunk,
            mode="arm-only",
            execute_steps=1,
            fps=1e9,
            gripper=machine,
            action_filter=_test_action_filter(),
            action_deadline=0.0,
        )
    assert robot.arm_actions == []


def _async_args(**overrides):
    values = {
        "server": "tcp://unused:5559",
        "timeout_s": 1.0,
        "task": "pick",
        "max_cycles": 2,
        "execute_steps": 30,
        "fps": 1e9,
        "expected_checkpoint": "checkpoint",
        "max_action_age_s": 1.0,
        "mode": "arm-only",
        "max_translation_m": 0.001,
        "max_rotation_rad": 0.005,
        "action_low_pass_alpha": 1.0,
        "max_translation_accel_m_s2": 1e20,
        "max_rotation_accel_rad_s2": 1e20,
        "chunk_blend_steps": 6,
        "open_threshold_m": 0.055,
        "stop_after_gripper_close": False,
        "require_initial_grasp": False,
        "force_retreat_m": 0.05,
        "force_retreat_speed_m_s": 0.03,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _async_sample(request_id, submitted_tick, chunk, *, age_s=0.0):
    state = np.zeros(15, dtype=np.float32)
    state[13] = 0.076
    return AsyncInferenceResult(
        request_id=request_id,
        submitted_tick=submitted_tick,
        observation_started=time.perf_counter() - age_s,
        state=state,
        response=InferenceResponse(
            request_id=request_id,
            action_chunk=chunk,
            inference_ms=10.0,
            checkpoint="checkpoint",
        ),
    )


def _async_action_filter(args):
    return TcpActionFilter(
        fps=args.fps,
        low_pass_alpha=args.action_low_pass_alpha,
        max_translation_step_m=args.max_translation_m,
        max_rotation_step_rad=args.max_rotation_rad,
        max_translation_accel_m_s2=args.max_translation_accel_m_s2,
        max_rotation_accel_rad_s2=args.max_rotation_accel_rad_s2,
    )


def test_async_replacement_skips_ticks_executed_while_inference_was_pending(monkeypatch):
    chunk = np.zeros(ACTION_CHUNK_SHAPE, dtype=np.float32)
    chunk[:, 0] = np.arange(ACTION_CHUNK_SHAPE[0], dtype=np.float32) / 10000.0

    class FakeWorker:
        def __init__(self, **kwargs):
            self.in_flight = False
            self.pending = None
            self.poll_count = 0

        def submit(self, *, request_id, submitted_tick):
            self.in_flight = True
            self.pending = _async_sample(request_id, submitted_tick, chunk)

        def wait(self):
            self.in_flight = False
            result, self.pending = self.pending, None
            return result

        def poll(self):
            self.poll_count += 1
            if self.pending is None or self.poll_count < 2:
                return None
            return self.wait()

        def close(self):
            pass

    monkeypatch.setattr(remote_client, "AsyncInferenceWorker", FakeWorker)
    robot = _FakeRobot()
    log = io.StringIO()
    args = _async_args(max_cycles=2, execute_steps=3)
    run_async_control(
        args=args,
        robot=robot,
        gripper=GripperStateMachine(confirm_steps=2),
        action_filter=_async_action_filter(args),
        log_stream=log,
    )
    entries = [json.loads(line) for line in log.getvalue().splitlines()]
    assert entries[0]["executed"][0]["global_tick"] == 0
    assert entries[1]["skipped_steps"] == 1
    assert entries[1]["executed"][0]["step"] == 1


def test_async_replacement_cross_fades_time_aligned_tcp_actions(monkeypatch):
    first_chunk = np.zeros(ACTION_CHUNK_SHAPE, dtype=np.float32)
    first_chunk[:, 0] = 0.0008
    second_chunk = np.zeros(ACTION_CHUNK_SHAPE, dtype=np.float32)
    second_chunk[:, 0] = -0.0008

    class FakeWorker:
        def __init__(self, **kwargs):
            self.in_flight = False
            self.pending = None
            self.poll_count = 0

        def submit(self, *, request_id, submitted_tick):
            self.in_flight = True
            chunk = first_chunk if request_id == 1 else second_chunk
            self.pending = _async_sample(request_id, submitted_tick, chunk)

        def wait(self):
            self.in_flight = False
            result, self.pending = self.pending, None
            return result

        def poll(self):
            self.poll_count += 1
            if self.pending is None or self.poll_count < 2:
                return None
            return self.wait()

        def close(self):
            pass

    monkeypatch.setattr(remote_client, "AsyncInferenceWorker", FakeWorker)
    robot = _FakeRobot()
    log = io.StringIO()
    args = _async_args(max_cycles=2, execute_steps=4, chunk_blend_steps=2)
    run_async_control(
        args=args,
        robot=robot,
        gripper=GripperStateMachine(confirm_steps=2),
        action_filter=_async_action_filter(args),
        log_stream=log,
    )
    entries = [json.loads(line) for line in log.getvalue().splitlines()]
    first_new_record = entries[1]["executed"][0]
    assert first_new_record["step"] == 1
    assert first_new_record["policy_predicted"][0] == pytest.approx(-0.0008)
    assert first_new_record["predicted"][0] == pytest.approx(0.0, abs=1e-8)
    assert first_new_record["chunk_blend"] == {
        "from_request_id": 1,
        "from_step": 1,
        "to_request_id": 2,
        "to_step": 1,
        "alpha": 0.5,
    }
    assert robot.arm_actions[1]["dx"] == pytest.approx(0.0, abs=1e-8)


def test_async_half_task_stops_immediately_after_close_command(monkeypatch):
    chunk = np.zeros(ACTION_CHUNK_SHAPE, dtype=np.float32)
    chunk[:, 6] = 0.019

    class FakeWorker:
        def __init__(self, **kwargs):
            self.in_flight = False
            self.pending = None

        def submit(self, *, request_id, submitted_tick):
            self.in_flight = True
            self.pending = _async_sample(request_id, submitted_tick, chunk)

        def wait(self):
            self.in_flight = False
            result, self.pending = self.pending, None
            return result

        def poll(self):
            return None

        def close(self):
            self.in_flight = False

    monkeypatch.setattr(remote_client, "AsyncInferenceWorker", FakeWorker)
    robot = _FakeRobot()
    args = _async_args(
        max_cycles=1,
        mode="integrated",
        stop_after_gripper_close=True,
    )
    run_async_control(
        args=args,
        robot=robot,
        gripper=GripperStateMachine(confirm_steps=2),
        action_filter=_async_action_filter(args),
        log_stream=io.StringIO(),
    )
    assert robot.arm_actions == []
    assert robot.close_count == 1
    assert robot.close_widths == [pytest.approx(0.019)]


def test_async_force_guard_stops_before_action_and_retreats(monkeypatch):
    chunk = np.zeros(ACTION_CHUNK_SHAPE, dtype=np.float32)

    class FakeWorker:
        def __init__(self, **kwargs):
            self.in_flight = False
            self.pending = None

        def submit(self, *, request_id, submitted_tick):
            self.in_flight = True
            self.pending = _async_sample(request_id, submitted_tick, chunk)

        def wait(self):
            self.in_flight = False
            result, self.pending = self.pending, None
            return result

        def poll(self):
            return None

        def close(self):
            self.in_flight = False

    monkeypatch.setattr(remote_client, "AsyncInferenceWorker", FakeWorker)
    robot = _FakeRobot()
    robot.external_wrench[2] = 8.0
    args = _async_args(max_cycles=1)
    log = io.StringIO()
    run_async_control(
        args=args,
        robot=robot,
        gripper=GripperStateMachine(confirm_steps=2),
        action_filter=_async_action_filter(args),
        log_stream=log,
        force_guard=RelativeForceGuard(
            baseline_wrench=np.zeros(6),
            threshold_n=7.0,
        ),
    )
    assert robot.arm_actions == []
    assert robot.force_retreats == [(0.05, 0.03)]
    entry = json.loads(log.getvalue())
    assert entry["retire_reason"] == "force_guard_retreat"
    assert entry["executed"][0]["force_guard"]["triggered"] is True


def test_async_rejects_insufficient_double_buffer_budget_before_motion(monkeypatch):
    chunk = np.zeros(ACTION_CHUNK_SHAPE, dtype=np.float32)

    class SlowFakeWorker:
        def __init__(self, **kwargs):
            self.in_flight = False
            self.pending = None

        def submit(self, *, request_id, submitted_tick):
            self.in_flight = True
            self.pending = _async_sample(request_id, submitted_tick, chunk, age_s=0.55)

        def wait(self):
            self.in_flight = False
            result, self.pending = self.pending, None
            return result

        def poll(self):
            return None

        def close(self):
            pass

    monkeypatch.setattr(remote_client, "AsyncInferenceWorker", SlowFakeWorker)
    robot = _FakeRobot()
    with pytest.raises(RuntimeError, match="too short for continuous async execution"):
        args = _async_args(max_cycles=1, max_action_age_s=1.0)
        run_async_control(
            args=args,
            robot=robot,
            gripper=GripperStateMachine(confirm_steps=2),
            action_filter=_async_action_filter(args),
            log_stream=io.StringIO(),
        )
    assert robot.arm_actions == []
