from __future__ import annotations

import threading
from types import SimpleNamespace

import torch
import pytest

from act_rlt import infer_stage2
from act_rlt.stage2_runtime import FixedChunkRuntime


def test_frozen_runtime_uses_actor_mean_without_sampling() -> None:
    class Actor(torch.nn.Module):
        def forward(self, state, reference, training=False):
            assert not training
            return torch.full_like(reference, 0.0001), torch.full_like(reference, 0.1)

        def sample(self, *args, **kwargs):
            raise AssertionError("frozen inference must not sample exploration noise")

    runtime = object.__new__(FixedChunkRuntime)
    runtime.warmup = False
    runtime.deterministic = True
    runtime.actor = Actor()
    runtime.actor_lock = threading.Lock()
    runtime.actor_state = None
    runtime.env = SimpleNamespace(
        config=SimpleNamespace(chunk_length=4, action_dim=6),
        post=lambda action: action,
        max_step_m=0.002,
        max_step_rad=0.02,
        _normalize_executed_action=lambda action: action[:6],
    )
    state = torch.zeros(1, 775)
    reference = torch.zeros(1, 4, 6)
    full = torch.zeros(1, 16, 7)
    prediction = runtime._prediction(state, reference, full)

    assert len(prediction.commands) == 4
    assert prediction.commands[0]["dx"] == prediction.mean_commands[0]["dx"]
    torch.testing.assert_close(prediction.executed, torch.full((4, 6), 0.0001))


@pytest.mark.parametrize("options,timestamped,selection", [
    ([], True, "latest"),
    (["--camera-boundary-mode", "legacy"], False, "latest"),
    (["--observation-selection", "boundary"], True, "boundary"),
    (["--camera-boundary-mode", "legacy", "--observation-selection", "boundary"], False, "boundary"),
])
@pytest.mark.parametrize("act_action_dim", [6, 7])
def test_infer_main_wires_camera_modes_before_connect(monkeypatch, options, timestamped, selection, act_action_dim):
    from evo_rlt.adapters.lerobot import franka_robot
    from lerobot.policies import factory

    events = []
    policy = SimpleNamespace(stage1=SimpleNamespace(_act=SimpleNamespace(config=object())), act_action_dim=act_action_dim)
    config = SimpleNamespace(act_checkpoint="/tmp/act", stage1_checkpoint="/tmp/stage1",
                             device="cpu", max_step_m=0.002, max_step_rad=0.02)
    monkeypatch.setattr(infer_stage2, "resolve_stage2_checkpoint", lambda path: path)
    monkeypatch.setattr(infer_stage2, "load_frozen_policy", lambda *args, **kwargs: (policy, config))
    monkeypatch.setattr(infer_stage2, "resolve_pretrained_model", lambda path: path)
    monkeypatch.setattr(infer_stage2, "checkpoint_camera_shapes", lambda config: {"wrist": (480, 640, 3)})
    monkeypatch.setattr(infer_stage2, "robot_camera_kwargs", lambda shapes, args: {})
    monkeypatch.setattr(factory, "make_pre_post_processors", lambda *args, **kwargs: (object(), object()))

    class Robot:
        def __init__(self, config):
            self.config = config
            assert config.include_gripper_action == (act_action_dim == 7)
            self.is_connected = False
            self.robot = SimpleNamespace(stop_move=lambda: events.append("stop"))

        def enable_timestamped_observations(self):
            assert not self.is_connected
            events.append("timestamp")

        def connect(self):
            self.is_connected = True
            events.append("connect")

        def disconnect(self):
            self.is_connected = False
            events.append("disconnect")

    class Env:
        def __init__(self, **kwargs):
            assert kwargs["robot"].is_connected
            assert kwargs["observation_selection"] == selection
            events.append("env")

        def close(self):
            events.append("close")

    monkeypatch.setattr(franka_robot, "FrankaRobot", Robot)
    monkeypatch.setattr(infer_stage2, "FrankaInsertionStage2Env", Env)
    monkeypatch.setattr(infer_stage2, "run_episodes", lambda *args: events.append("run"))
    assert infer_stage2.main([
        "--checkpoint", "/tmp/stage2", "--allow-motion",
        "--workspace-min", "0.3", "-0.3", "0.3",
        "--workspace-max", "0.7", "0.1", "0.5",
        *options,
    ]) == 0
    assert events == (["timestamp"] if timestamped else []) + ["connect", "env", "run", "close", "stop", "disconnect"]


def _saved_reference():
    return dict(version=1, mode="z_insertion", frame="base", position_unit="m",
                orientation="rotation_vector_rad", reference_pose=[0.4, -0.2, 0.4, 1, 2, 3],
                workspace_min_xyz=[0.35, -0.25, 0.35], workspace_max_xyz=[0.45, -0.15, 0.45])


@pytest.mark.parametrize("choice", ["t"])
def test_run_reference_uses_saved_training_point(monkeypatch, choice):
    import io
    import numpy as np
    from act_rlt.infer_stage2_terminal import InferenceTerminal

    monkeypatch.setattr("builtins.input", lambda prompt: choice)
    monkeypatch.setattr(infer_stage2, "capture_centered_workspace",
                        lambda *a, **k: pytest.fail("training choice must not capture TCP"))
    reference, lower, upper, initial = infer_stage2.select_run_reference(
        object(), _saved_reference(), randomize_xy=True,
        terminal=InferenceTerminal(1, 10, stream=io.StringIO()),
    )
    np.testing.assert_allclose(reference, [0.4, -0.2, 0.4, 1, 2, 3])
    np.testing.assert_allclose(lower, [0.35, -0.30, 0.35])
    np.testing.assert_allclose(upper, [0.45, -0.10, 0.45])
    np.testing.assert_array_equal(initial, reference)
    assert initial is not reference


@pytest.mark.parametrize("saved,answers", [(_saved_reference(), ["", "train", "invalid", "c", ""]),
                                          (None, [""])])
def test_run_reference_captures_current_tcp_and_new_workspace(monkeypatch, saved, answers):
    import io
    import numpy as np
    from act_rlt.infer_stage2_terminal import InferenceTerminal

    answers = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    # Real capture reads the TCP only after its Enter confirmation.
    current = [0.6, 0.1, 0.41, 4, 5, 6]
    robot = SimpleNamespace(robot=SimpleNamespace(get_tool_pose=lambda: current))
    reference, lower, upper, initial = infer_stage2.select_run_reference(
        robot, saved, randomize_xy=True,
        terminal=InferenceTerminal(1, 10, stream=io.StringIO()),
    )
    np.testing.assert_array_equal(reference, current)
    np.testing.assert_allclose(lower, [0.55, 0.00, 0.36])
    np.testing.assert_allclose(upper, [0.65, 0.20, 0.46])
    assert initial is None  # No training-pose shortcut in the newly captured workspace.


def test_run_reference_quit_does_not_read_or_move_robot(monkeypatch):
    import io
    from act_rlt.infer_stage2_terminal import InferenceTerminal

    monkeypatch.setattr("builtins.input", lambda prompt: "q")
    with pytest.raises(KeyboardInterrupt):
        infer_stage2.select_run_reference(
            object(), _saved_reference(), randomize_xy=True,
            terminal=InferenceTerminal(1, 10, stream=io.StringIO()),
        )
