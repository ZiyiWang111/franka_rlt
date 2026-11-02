from types import SimpleNamespace

import pytest
import torch
from torch import nn

from act_rlt import infer_stage2
from act_rlt.stage2 import ACTStage2Config, OnlineStage2Metrics, Stage2Learner
from act_rlt.stage2_networks import ChunkInputFusion, Stage2ChunkActor, Stage2TwinCritic
from act_rlt.train_stage2 import import_replay, load_checkpoint, save_checkpoint
from evo_rlt.core.interfaces import ChunkTransition
from evo_rlt.core.replay_buffer import ReplayBuffer


@pytest.mark.parametrize("proprio_dim", [0, 1, 6, 7])
def test_token_is_unchanged_auxiliary_and_action_project_and_receive_gradients(proprio_dim):
    fusion = ChunkInputFusion(768 + proprio_dim, proprio_dim, 24, 128, input_fusion="token_identity")
    state = (torch.randn(3, 768 + proprio_dim) * 5).requires_grad_()
    action = torch.randn(3, 24, requires_grad=True)
    output = fusion(state, action)
    width = 928 if proprio_dim else 896
    assert output.shape == (3, width)
    assert isinstance(fusion.token, nn.Identity)
    torch.testing.assert_close(output[:, :768], state[:, :768], atol=0, rtol=0)
    assert output[:, 768:].abs().max() <= 1
    assert output[:, :768].abs().max() > 1  # Token values must not pass through tanh.
    if proprio_dim:
        assert fusion.proprio[0].weight.shape == (32, proprio_dim)
    assert fusion.action[0].weight.shape == (128, 24)
    output.square().mean().backward()
    assert state.grad[:, :768].abs().sum() > 0
    if proprio_dim:
        assert state.grad[:, 768:].abs().sum() > 0
    assert action.grad.abs().sum() > 0


@pytest.mark.parametrize("proprio_dim", [0, 1, 6, 7])
@pytest.mark.parametrize("width", [16, 128])
def test_balanced_projection_normalizes_each_input_and_preserves_gradients(proprio_dim, width):
    fusion = ChunkInputFusion(768 + proprio_dim, proprio_dim, 24, width)
    state = torch.randn(3, 768 + proprio_dim)
    state[:, :768] *= 100  # Deliberately different input scales.
    state[:, 768:] *= .001
    state.requires_grad_()
    action = torch.randn(3, 24, requires_grad=True)
    output = fusion(state, action)
    branches = [fusion.token, *([fusion.proprio] if proprio_dim else []), fusion.action]
    assert output.shape == (3, len(branches) * width)
    assert fusion.token[0].weight.shape == (width, 768)
    assert fusion.action[0].weight.shape == (width, 24)
    if proprio_dim:
        assert fusion.proprio[0].weight.shape == (width, proprio_dim)
    for branch, values in zip(branches, [state[:, :768], *([state[:, 768:]] if proprio_dim else []), action]):
        normalized = branch[1](branch[0](values))
        torch.testing.assert_close(normalized.mean(-1), torch.zeros(3), atol=1e-5, rtol=0)
        assert (normalized.square().mean(-1) > .9).all()
    assert torch.isfinite(output).all() and output.abs().max() <= 1
    output.square().mean().backward()
    assert state.grad[:, :768].abs().sum() > 0
    if proprio_dim:
        assert state.grad[:, 768:].abs().sum() > 0
    assert action.grad.abs().sum() > 0
    assert all(branch[0].weight.grad.abs().sum() > 0 for branch in branches)


def test_actor_and_each_critic_have_independent_384d_inputs_and_targets_match():
    config = ACTStage2Config(stage1_checkpoint="a", act_checkpoint="b", device="cpu", proprio_dim=1)
    actor = Stage2ChunkActor(769, 1, 24)
    policy = SimpleNamespace(actor=actor)
    learner = Stage2Learner(policy, config)
    state, chunk = torch.randn(2, 769), torch.randn(2, 24)
    branches = [actor, learner.critic.q1, learner.critic.q2,
                learner.target_actor, learner.target_critic.q1, learner.target_critic.q2]
    assert config.input_fusion == "balanced"
    assert all(branch.fusion.output_dim == 384 for branch in branches)
    assert all(branch.net[0].in_features == 384 for branch in branches)
    assert len({branch.fusion.token[0].weight.data_ptr() for branch in branches}) == 6
    assert len({branch.fusion.proprio[0].weight.data_ptr() for branch in branches}) == 6
    assert len({branch.fusion.action[0].weight.data_ptr() for branch in branches}) == 6
    mean, std = actor(state, chunk)
    assert mean.shape == std.shape == (2, 24)
    assert all(q.shape == (2, 1) for q in learner.critic(state, chunk))


def make_learner(config):
    actor = Stage2ChunkActor(config.state_dim, config.proprio_dim, config.chunk_dim,
                            fusion_dim=config.fusion_dim, input_fusion=config.input_fusion)
    return Stage2Learner(SimpleNamespace(actor=actor), config)


def test_checkpoint_roundtrip_and_legacy_resume_rejection_before_weight_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    values = dict(stage1_checkpoint="a", act_checkpoint="b", device="cpu", proprio_dim=1, batch_size=1)
    config = ACTStage2Config(**values)
    learner = make_learner(config)
    save_checkpoint(tmp_path / "new", config, learner, ReplayBuffer(), OnlineStage2Metrics())
    path = tmp_path / "new/checkpoints/latest.pt"
    restored = make_learner(config)
    load_checkpoint(path, config, restored, ReplayBuffer())
    for key, value in learner.policy.actor.state_dict().items():
        torch.testing.assert_close(restored.policy.actor.state_dict()[key], value)
    legacy_config = ACTStage2Config(**values, input_fusion="token_identity")
    legacy_learner = make_learner(legacy_config)
    saved = torch.load(path, weights_only=False)
    saved["config"] = legacy_config.to_dict()
    saved["learner"] = legacy_learner.state_dict()
    saved["replay"] = [ChunkTransition(
        state_vec=torch.zeros(769), next_state_vec=torch.ones(769),
        exec_chunk=torch.zeros(4, 6), ref_chunk=torch.ones(4, 6), next_ref_chunk=torch.ones(4, 6),
        reward_seq=torch.zeros(4), done=torch.tensor(0.), intervention=torch.tensor(0.),
        actual_steps=torch.tensor(4),
    )]
    old_path = tmp_path / "legacy.pt"
    torch.save(saved, old_path)
    weights = {key: value.clone() for key, value in restored.policy.actor.state_dict().items()}
    with pytest.raises(ValueError, match="--replay-from"):
        load_checkpoint(old_path, config, restored, ReplayBuffer())
    with pytest.raises(ValueError, match="--replay-from"):
        restored.load_state_dict(saved["learner"])
    for key, value in weights.items():
        torch.testing.assert_close(restored.policy.actor.state_dict()[key], value)
    monkeypatch.setattr("act_rlt.train_stage2.resolve_pretrained_model", lambda path: path)
    reused = ReplayBuffer()
    metrics = import_replay(old_path, config, reused)
    assert len(reused) == 1 and metrics.online_env_steps == 0
    torch.testing.assert_close(reused.buffer[0].state_vec, saved["replay"][0].state_vec)
    torch.testing.assert_close(reused.buffer[0].exec_chunk, saved["replay"][0].exec_chunk)


def test_earliest_balanced_checkpoint_without_layout_metadata_resumes(tmp_path):
    config = ACTStage2Config(stage1_checkpoint="a", act_checkpoint="b", device="cpu", proprio_dim=1)
    learner = make_learner(config)
    save_checkpoint(tmp_path, config, learner, ReplayBuffer(), OnlineStage2Metrics())
    path = tmp_path / "checkpoints/latest.pt"
    saved = torch.load(path, weights_only=False)
    del saved["config"]["input_fusion"]
    del saved["learner"]["input_fusion"]
    torch.save(saved, path)
    restored = make_learner(config)
    load_checkpoint(path, config, restored, ReplayBuffer())
    state, action = torch.randn(2, 769), torch.randn(2, 24)
    torch.testing.assert_close(restored.policy.actor(state, action)[0], learner.policy.actor(state, action)[0])


@pytest.mark.parametrize("layout", ["balanced", "token_identity"])
def test_frozen_inference_selects_original_or_new_architecture(tmp_path, monkeypatch, layout):
    config = ACTStage2Config(stage1_checkpoint="a", act_checkpoint="b", device="cpu", proprio_dim=1,
                            input_fusion=layout)
    learner = make_learner(config)
    state = learner.state_dict()
    state["initialized"] = True
    config_data = config.to_dict()
    if layout == "balanced":
        del config_data["input_fusion"]
        del state["input_fusion"]
    checkpoint = tmp_path / "model.pt"
    torch.save({"config": config_data, "learner": state}, checkpoint)

    class Policy(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.actor = make_learner(cfg).policy.actor

    monkeypatch.setattr(infer_stage2, "ACTStage2Policy", Policy)
    monkeypatch.setattr(infer_stage2, "resolve_pretrained_model", lambda path: path)
    policy, loaded_config = infer_stage2.load_frozen_policy(checkpoint, device="cpu")
    assert loaded_config.input_fusion == layout
    assert policy.actor.fusion.output_dim == (384 if layout == "balanced" else 928)
    state_vec, actions = torch.randn(2, 769), torch.randn(2, 24)
    torch.testing.assert_close(policy.actor(state_vec, actions)[0], learner.policy.actor(state_vec, actions)[0])
