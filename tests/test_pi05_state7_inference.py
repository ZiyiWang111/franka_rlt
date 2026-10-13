import json
import sys

import numpy as np
import pytest

from evo_rlt.adapters.lerobot.franka_remote import protocol_state7 as wire
from evo_rlt.adapters.lerobot.franka_remote.protocol import encode_ready_response as old_ready
from scripts.pi05_state7_robot_client import model_observation, parse_args, require_grasp, validate_args
from scripts.fr3_remote_robot_client import execute_action_chunk


def test_new_protocol_preserves_task_arrays_and_rejects_old_server():
    state = np.arange(7, dtype=np.float32)
    image = np.zeros(wire.IMAGE_SHAPE, dtype=np.uint8)
    frames = wire.encode_inference_request(request_id=1, timestamp_ns=123,
        task="insert the ethernet cable into the hole", state=state, wrist=image)
    _, _, request = wire.decode_request(frames)
    np.testing.assert_array_equal(request.state, state)
    np.testing.assert_array_equal(request.wrist, image)
    assert request.task == "insert the ethernet cable into the hole"
    chunk = np.arange(60, dtype=np.float32).reshape(10, 6)
    result = wire.decode_response(wire.encode_inference_response(
        request_id=1, action_chunk=chunk, inference_ms=1, checkpoint="new"), 1)
    np.testing.assert_array_equal(result.action_chunk, chunk)
    assert wire.decode_response(wire.encode_ready_response(0, "new"), 0)["camera_names"] == ["wrist"]
    with pytest.raises(wire.ProtocolError, match="version"):
        wire.decode_response(old_ready(0, "old"), 0)
    frames[2] = frames[2][:-1]
    with pytest.raises(wire.ProtocolError, match="byte lengths"):
        wire.decode_request(frames)


def test_state7_ignores_extra_hardware_state_and_grasp_is_required():
    observation = {name: float(i) for i, name in enumerate(wire.STATE_NAMES)}
    observation.update(wrist=np.zeros(wire.IMAGE_SHAPE, np.uint8), ee_x=99, gripper_width=.0128)
    state, _ = model_observation(observation)
    assert state.shape == (7,)
    with pytest.raises(RuntimeError, match="grasp"):
        require_grasp(observation)
    observation["gripper_grasped"] = True
    require_grasp(observation)


def test_tcp6_execution_never_commands_gripper(monkeypatch):
    import scripts.fr3_remote_robot_client as old
    monkeypatch.setattr(old, "sleep_control_period", lambda *a: None)
    class Robot:
        actions = []
        def send_action(self, action):
            self.actions.append(action)
        def start_gripper_transition(self, *a, **kw):
            pytest.fail("Unexpected gripper command")
    robot = Robot()
    records, replan, stopped = execute_action_chunk(
        robot=robot, chunk=np.ones((10, 6), np.float32), mode="arm-only", execute_steps=4,
        fps=30, gripper=None, max_translation_m=.005, max_rotation_rad=.03)
    assert len(records) == len(robot.actions) == 4 and not replan and not stopped
    for action in robot.actions:
        assert set(action) == set(wire.ACTION_NAMES)
        assert np.linalg.norm([action[k] for k in wire.ACTION_NAMES[:3]]) <= .005 + 1e-9


def test_shadow_defaults_and_motion_requires_bounds(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["client", "--server", "tcp://test"])
    args = parse_args()
    validate_args(args)
    assert args.mode == "shadow" and args.fps == 30 and args.execute_steps == 4
    args.mode = "arm-only"
    with pytest.raises(ValueError, match="allow-motion"):
        validate_args(args)
