from __future__ import annotations

import threading
from types import SimpleNamespace

import torch

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
