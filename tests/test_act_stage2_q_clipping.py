from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from act_rlt.stage2 import ACTStage2Config, OnlineStage2Metrics, Stage2Learner
from act_rlt.stage2_networks import Stage2ChunkActor
from act_rlt.train_stage2 import build_parser, load_checkpoint, save_checkpoint
from evo_rlt.core.replay_buffer import ReplayBuffer


def _config(**overrides):
    values = dict(stage1_checkpoint="/tmp/stage1", act_checkpoint="/tmp/act", device="cpu",
                  batch_size=1, warmup_steps=8, total_env_steps=16, replay_capacity=32,
                  fusion_dim=16, actor_hidden_dim=16, critic_hidden_dim=16,
                  actor_num_layers=1, critic_num_layers=1, bc_init_steps=0, critic_init_steps=0)
    values.update(overrides)
    return ACTStage2Config(**values)


class _FakeStage2Policy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.actor = Stage2ChunkActor(config.state_dim, config.proprio_dim, config.chunk_dim,
                                     fusion_dim=config.fusion_dim, hidden_dim=config.actor_hidden_dim,
                                     num_layers=config.actor_num_layers, fixed_std=config.actor_fixed_std)


@pytest.mark.parametrize("clipping", [True, False])
@pytest.mark.parametrize("raw_q", [90.0, -2.0])
def test_targets_optionally_clip_q_and_always_mask_terminal_bootstrap(clipping, raw_q):
    config = _config(batch_size=3, target_q_clipping=clipping)
    learner = Stage2Learner(_FakeStage2Policy(config), config)
    batch = {
        "state_vec": torch.zeros(3, 775), "next_state_vec": torch.zeros(3, 775),
        "exec_chunk_flat": torch.zeros(3, 24), "ref_chunk_flat": torch.zeros(3, 24),
        "next_ref_flat": torch.zeros(3, 24), "reward_seq": torch.tensor([[0.,0,0,0], [0,1,0,0], [0,0,0,0]]),
        "actual_steps": torch.tensor([4,2,1]), "done": torch.tensor([0.,1,1]),
    }
    for network in (learner.critic, learner.target_critic):
        for parameter in network.parameters():
            nn.init.zeros_(parameter)
    for q in (learner.target_critic.q1, learner.target_critic.q2):
        q.net[-1].bias.data.fill_(raw_q)
    metrics = learner.update_once(SimpleNamespace(sample=lambda n: batch))
    bootstrapped_q = min(1.0, max(0.0, raw_q)) if clipping else raw_q
    expected = torch.tensor([0.99**4 * bootstrapped_q, 0.99, 0.])
    assert metrics.critic_loss == pytest.approx(2 * expected.square().mean().item())
    assert metrics.diagnostics["target_q_clip_fraction"] == float(clipping)
    assert metrics.diagnostics["target_q_clipping_enabled"] is clipping
    assert metrics.diagnostics["td_target_mean"] == pytest.approx(expected.mean().item())
    assert metrics.diagnostics["q_reference_mean"] == pytest.approx(0.0)
    assert metrics.diagnostics["q_reference_max"] == pytest.approx(0.0)
    assert all(p.grad is None for p in learner.target_actor.parameters())
    assert all(p.grad is None for p in learner.target_critic.parameters())


def test_target_q_clipping_cli_defaults_on_and_can_be_disabled():
    parser = build_parser()
    base = ["--stage1-checkpoint", "/tmp/stage1", "--act-checkpoint", "/tmp/act"]
    assert parser.parse_args(base).target_q_clipping is True
    assert parser.parse_args(base + ["--no-target-q-clipping"]).target_q_clipping is False
    assert parser.parse_args(base + ["--target-q-clipping"]).target_q_clipping is True


@pytest.mark.parametrize("clipping", [True, False])
def test_target_q_clipping_still_rejects_nonfinite_td_targets(clipping):
    config = _config(target_q_clipping=clipping)
    learner = Stage2Learner(_FakeStage2Policy(config), config)
    batch = {
        "state_vec": torch.zeros(1, 775), "next_state_vec": torch.zeros(1, 775),
        "exec_chunk_flat": torch.zeros(1, 24), "ref_chunk_flat": torch.zeros(1, 24),
        "next_ref_flat": torch.zeros(1, 24), "reward_seq": torch.zeros(1, 4),
        "actual_steps": torch.tensor([4]), "done": torch.tensor([0.]),
    }
    for q in (learner.target_critic.q1, learner.target_critic.q2):
        q.net[-1].bias.data.fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite TD target"):
        learner.update_once(SimpleNamespace(sample=lambda n: batch))


@pytest.mark.parametrize("saved_flag,new_flag", [(None, False), (False, True), (False, False)])
def test_q_clipping_checkpoint_persistence_and_resume_override(tmp_path, saved_flag, new_flag):
    config = _config(target_q_clipping=True if saved_flag is None else saved_flag)
    learner = Stage2Learner(_FakeStage2Policy(config), config)
    save_checkpoint(tmp_path, config, learner, ReplayBuffer(config.replay_capacity), OnlineStage2Metrics())
    path = tmp_path / "checkpoints/latest.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["config"]["target_q_clipping"] is config.target_q_clipping
    if saved_flag is None:
        del saved["config"]["target_q_clipping"]  # Checkpoints predating the switch.
        torch.save(saved, path)
    resumed_config = _config(target_q_clipping=new_flag)
    resumed = Stage2Learner(_FakeStage2Policy(resumed_config), resumed_config)
    load_checkpoint(path, resumed_config, resumed, ReplayBuffer(resumed_config.replay_capacity))
    assert resumed.config.target_q_clipping is new_flag


