import numpy as np

from evo_rlt.adapters.lerobot.franka_remote.execution import GripperStateMachine, clip_tcp_action
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
from scripts.fr3_remote_robot_client import execute_action_chunk


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


class _FakeRobot:
    def __init__(self):
        self.arm_actions = []
        self.close_count = 0
        self.resync_count = 0

    def send_action(self, action):
        self.arm_actions.append(action)

    def close_gripper(self):
        self.close_count += 1
        return True

    def open_gripper(self):
        raise AssertionError("unexpected open")

    def resync_command_pose(self):
        self.resync_count += 1


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
        max_translation_m=0.005,
        max_rotation_rad=0.03,
    )
    assert replan
    assert robot.arm_actions == []
    assert robot.close_count == 1
    assert robot.resync_count == 1
    assert records[-1]["gripper_command"] == "close"
