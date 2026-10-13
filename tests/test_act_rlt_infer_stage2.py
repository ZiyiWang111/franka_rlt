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
def test_infer_main_wires_camera_modes_before_connect(monkeypatch, options, timestamped, selection):
    from evo_rlt.adapters.lerobot import franka_robot
    from lerobot.policies import factory

    events = []
    policy = SimpleNamespace(stage1=SimpleNamespace(_act=SimpleNamespace(config=object())))
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
