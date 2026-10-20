from types import SimpleNamespace
import threading

import numpy as np
import pytest
import torch
from torch import nn

from act_rlt.stage2 import ACTStage2Config, ACTStage2Policy, Stage2Learner
from act_rlt.stage2_runtime import FixedChunkRuntime
from act_rlt.train_stage2 import FrankaInsertionStage2Env
from lerobot.types import TransitionKey


class TokenEncoder(nn.Linear):
    def encode(self, hidden):
        return self(hidden.mean(dim=1))


class FrozenStage1(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.config = SimpleNamespace(rl_token_dim=768)
        self.rl_token = TokenEncoder(8, 768)
        self._act = nn.Module()
        self._act.config = SimpleNamespace(
            chunk_size=16, action_feature=SimpleNamespace(shape=(action_dim,)),
            robot_state_feature=SimpleNamespace(shape=(state_dim,)) if state_dim else None,
        )


@pytest.mark.parametrize('state_dim,action_dim', [(0, 6), (1, 6), (6, 6), (7, 7)])
def test_model_contract_encoding_learning_and_roundtrip(state_dim, action_dim, monkeypatch, tmp_path):
    config = ACTStage2Config(
        stage1_checkpoint='/tmp/stage1', act_checkpoint='/tmp/act', device='cpu',
        proprio_dim=state_dim, batch_size=2, actor_hidden_dim=16, critic_hidden_dim=16,
        fusion_dim=8, actor_update_interval=1,
    )
    frozen = FrozenStage1(state_dim, action_dim)
    monkeypatch.setattr('act_rlt.stage2.predict_act_chunk_with_encoder_hidden_and_pos',
        lambda act, batch: (torch.zeros(2, 16, action_dim), torch.ones(2, 5, 8), torch.zeros(1, 5, 8)))
    policy = ACTStage2Policy(config, stage1_policy=frozen)
    batch = {'action': None, 'observation.images.wrist': torch.zeros(2, 3, 8, 8)}
    if state_dim:
        batch['observation.state'] = torch.randn(2, state_dim)
    state, reference, full = policy.encode_and_reference(batch)
    assert state.shape == (2, 768 + state_dim)
    assert full.shape[-1] == action_dim
    if state_dim:
        torch.testing.assert_close(state[:, 768:], batch['observation.state'])
    assert all(not p.requires_grad for p in frozen.parameters())
    assert (policy.actor.fusion.proprio is None) == (state_dim == 0)
    learner = Stage2Learner(policy, config)
    learner.configure_action_projection(lambda x: x * .02)
    replay_batch = dict(
        state_vec=state, next_state_vec=state, exec_chunk_flat=reference.flatten(1),
        ref_chunk_flat=reference.flatten(1), next_ref_flat=reference.flatten(1),
        reward_seq=torch.zeros(2, 4), done=torch.zeros(2), actual_steps=torch.full((2,), 4),
    )
    before = [p.detach().clone() for p in policy.actor.parameters()]
    result = learner.update_once(SimpleNamespace(sample=lambda n: replay_batch))
    assert np.isfinite(result.critic_loss) and np.isfinite(result.actor_loss)
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.actor.parameters()))
    assert all(p.grad is None for p in frozen.parameters())
    path = tmp_path / 'learner.pt'
    torch.save(learner.state_dict(), path)
    restored = Stage2Learner(ACTStage2Policy(config, stage1_policy=FrozenStage1(state_dim, action_dim)), config)
    restored.load_state_dict(torch.load(path, weights_only=True))
    for key, value in policy.actor.state_dict().items():
        torch.testing.assert_close(value, restored.policy.actor.state_dict()[key])


@pytest.mark.parametrize('state_dim,action_dim', [(0, 6), (1, 6), (6, 6), (7, 7)])
def test_observation_and_runtime_action_roundtrip(state_dim, action_dim):
    env = object.__new__(FrankaInsertionStage2Env)
    env.config = SimpleNamespace(proprio_dim=state_dim, action_dim=6, chunk_length=4, device='cpu')
    env.camera_shapes = {'wrist': (8, 8, 3)}
    raw = {'wrist': np.zeros((8, 8, 3), dtype=np.uint8), 'ee_z': .402,
           **{f'joint_{i}': float(i) for i in range(7)}}
    calls = []
    def wrench():
        calls.append(True)
        return np.arange(6, dtype=np.float32)
    env.robot = SimpleNamespace(get_observation=lambda: raw, get_external_wrench_base=wrench)
    env.pre = lambda frame: frame
    frame = env._processed_observation()
    assert ('observation.state' in frame) == bool(state_dim)
    assert len(calls) == (1 if state_dim == 6 else 0)
    if state_dim:
        expected = torch.tensor([.402]) if state_dim == 1 else torch.arange(state_dim, dtype=torch.float32)
        torch.testing.assert_close(frame['observation.state'], expected)
    if state_dim == 1:
        # Every new observation uses measured absolute Z, not the reset height.
        raw['ee_z'] = .391
        torch.testing.assert_close(env._processed_observation()['observation.state'], torch.tensor([.391]))
    env.post = lambda x: x * .02
    env.max_step_m, env.max_step_rad = .002, .02
    env.action_normalizer = lambda t: {TransitionKey.ACTION: t[TransitionKey.ACTION] / .02}
    runtime = object.__new__(FixedChunkRuntime)
    runtime.env, runtime.warmup = env, True
    reference = torch.ones(1, 4, 6)
    full = torch.ones(1, 16, action_dim)
    prediction = runtime._prediction(torch.zeros(1, 768 + state_dim), reference, full)
    assert prediction.executed.shape == (4, 6)
    assert len(prediction.gripper_values) == (4 if action_dim == 7 else 0)
    assert all(prediction.clipped)
    torch.testing.assert_close(prediction.executed[:, :3].norm(dim=-1), torch.full((4,), .1))
    assert raw.keys() == {'wrist', 'ee_z', *[f'joint_{i}' for i in range(7)]}


@pytest.mark.parametrize('selection', ['latest', 'boundary'])
def test_tcp_z_cached_observation_uses_same_frame_state(selection):
    from collections import deque

    env = object.__new__(FrankaInsertionStage2Env)
    env.config = SimpleNamespace(proprio_dim=1)
    env.camera_shapes = {'wrist': (8, 8, 3)}
    env.pre = lambda frame: frame
    runtime = object.__new__(FixedChunkRuntime)
    runtime.env = env
    runtime.observation_selection = selection
    runtime.observation_seq, runtime.last_observation_seq = 1, 0
    runtime.observation_ready = threading.Condition()
    runtime.stop = threading.Event()
    raw = {'ee_z': .391, 'wrist': np.zeros((8, 8, 3), dtype=np.uint8)}
    runtime.observation_history = deque([(1., 2., 1., raw, {'observation_seq': 1})])
    frame = runtime._boundary_batch(0.)
    torch.testing.assert_close(frame['observation.state'], torch.tensor([.391]))


@pytest.mark.parametrize('z', [None, float('nan'), float('inf')])
def test_tcp_z_rejects_missing_or_nonfinite_measurement(z):
    env = object.__new__(FrankaInsertionStage2Env)
    env.config = SimpleNamespace(proprio_dim=1)
    env.camera_shapes = {}
    raw = {} if z is None else {'ee_z': z}
    with pytest.raises(ValueError, match='TCP Z|tcp_z'):
        env._observation_frame(raw)
