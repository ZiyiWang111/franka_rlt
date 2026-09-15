import sys

import numpy as np
import pytest

from evo_rlt.adapters.lerobot.franka_remote.execution import (
    GripperStateMachine,
    RelativeForceGuard,
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
from scripts.fr3_remote_robot_client import (
    execute_action_chunk,
    parse_args,
    require_expected_checkpoint,
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
    require_expected_checkpoint("/models/030000", "/models/030000")
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        require_expected_checkpoint("/models/base", "/models/030000")


def test_vanilla_defaults_match_openpi_droid_loop(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fr3_remote_robot_client.py", "--server", "tcp://test"])
    args = parse_args()
    assert args.fps == 15.0
    assert args.execute_steps == 8
    assert not hasattr(args, "async_inference")
    assert not hasattr(args, "chunk_blend_steps")
    assert not hasattr(args, "action_low_pass_alpha")


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
        self.gripper_motion = None
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

    def start_gripper_transition(self, command, width=None):
        assert not self.transition_prepared
        self.transition_prepared = True
        self.transition_prepare_count += 1
        self.close_count += command == "close"
        if command == "close":
            self.close_widths.append(width)
        self.gripper_motion = {
            "command_id": 1, "command": command, "status": "running",
            "result": None, "width": 0.076, "grasped": False,
            "measurement_stale": True,
        }
        return dict(self.gripper_motion)

    def get_external_wrench_base(self):
        return self.external_wrench.copy()

    def stop_and_retreat_up(self, *, distance_m, speed_m_s):
        self.force_retreats.append((distance_m, speed_m_s))
        return {"distance_m": distance_m, "speed_m_s": speed_m_s}


def test_executor_sends_unfiltered_model_rows_in_order():
    robot = _FakeRobot()
    machine = GripperStateMachine(confirm_steps=2)
    chunk = np.zeros(ACTION_CHUNK_SHAPE, dtype=np.float32)
    chunk[:3, 0] = [0.001, -0.002, 0.003]
    records, replan, force_stopped = execute_action_chunk(
        robot=robot,
        chunk=chunk,
        mode="arm-only",
        execute_steps=3,
        fps=1e9,
        gripper=machine,
        max_translation_m=0.005,
        max_rotation_rad=0.03,
    )
    assert not replan
    assert not force_stopped
    assert [action["dx"] for action in robot.arm_actions] == pytest.approx(
        [0.001, -0.002, 0.003]
    )
    assert [record["step"] for record in records] == [0, 1, 2]


def test_integrated_executor_starts_async_gripper_and_leaves_arm_held():
    robot = _FakeRobot()
    machine = GripperStateMachine(confirm_steps=2)
    machine.sync_from_observation(0.076, grasped=False)
    chunk = np.zeros(ACTION_CHUNK_SHAPE, dtype=np.float32)
    chunk[:, :6] = 0.003
    chunk[:, 6] = 0.019
    records, replan, force_stopped = execute_action_chunk(
        robot=robot,
        chunk=chunk,
        mode="integrated",
        execute_steps=5,
        fps=1e9,
        gripper=machine,
        max_translation_m=0.005,
        max_rotation_rad=0.03,
    )
    assert replan
    assert not force_stopped
    assert robot.arm_actions == []
    assert robot.close_count == 1
    assert robot.close_widths == [pytest.approx(0.019)]
    assert robot.transition_prepare_count == 1
    assert robot.transition_finish_count == 0
    assert robot.resync_count == 0
    assert records[-1]["gripper_command_started"] == "close"
    assert records[-1]["gripper_motion"]["status"] == "running"
    # Completion is deliberately not claimed until the Future reports it.
    assert machine.current == "open"


def test_force_guard_stops_before_next_model_action_and_retreats():
    robot = _FakeRobot()
    robot.external_wrench[2] = 8.0
    chunk = np.zeros(ACTION_CHUNK_SHAPE, dtype=np.float32)
    records, replan, force_stopped = execute_action_chunk(
        robot=robot,
        chunk=chunk,
        mode="arm-only",
        execute_steps=8,
        fps=1e9,
        gripper=GripperStateMachine(confirm_steps=2),
        max_translation_m=0.005,
        max_rotation_rad=0.03,
        force_guard=RelativeForceGuard(baseline_wrench=np.zeros(6), threshold_n=7.0),
        force_retreat_m=0.05,
        force_retreat_speed_m_s=0.03,
    )
    assert not replan
    assert force_stopped
    assert robot.arm_actions == []
    assert robot.force_retreats == [(0.05, 0.03)]
    assert records[0]["force_guard"]["triggered"] is True
