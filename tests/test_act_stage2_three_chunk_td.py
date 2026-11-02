from types import SimpleNamespace

import pytest
import torch

from act_rlt.stage2 import ACTStage2Config, Stage2Learner
from act_rlt.stage2_networks import Stage2ChunkActor
from evo_rlt.core.interfaces import ChunkTransition
from evo_rlt.core.replay_buffer import ReplayBuffer
from evo_rlt.core.losses import discounted_chunk_return


def chunk(start, *, count=4, done=False, episode=0, legacy=False, dim=1):
    rewards = torch.zeros(4)
    if done:
        rewards[count - 1] = 1
    return ChunkTransition(
        state_vec=torch.full((dim,), float(start)),
        next_state_vec=torch.full((dim,), float(start + count)),
        exec_chunk=torch.full((4, 6), float(start + 100)),
        ref_chunk=torch.full((4, 6), float(start)),
        next_ref_chunk=torch.full((4, 6), float(start + count)),
        reward_seq=rewards, actual_steps=torch.tensor(count), done=torch.tensor(float(done)),
        intervention=torch.tensor(0.), episode_id=torch.tensor(episode),
        start_step=None if legacy else start,
        bc_target_chunk=torch.full((4, 6), float(start + 200)),
    )


def test_three_chunk_nonterminal_horizon_and_online_readiness():
    replay = ReplayBuffer()
    replay.add(chunk(0))
    with pytest.raises(ValueError, match="no complete TD"):
        replay.sample(256, td_chunks=3)
    replay.add(chunk(4))
    with pytest.raises(ValueError):
        replay.sample(256, td_chunks=3)
    replay.add(chunk(8))
    batch = replay.sample(256, td_chunks=3, include_bc_target=True)
    assert batch["state_vec"].tolist() == [[0.]]
    assert batch["td_steps"].tolist() == [12]
    assert batch["td_next_state_vec"].tolist() == [[12.]]
    assert batch["td_next_ref_flat"].eq(12).all()
    assert batch["exec_chunk_flat"].shape == (1, 24)
    assert batch["exec_chunk_flat"].eq(100).all()
    assert batch["bc_target_flat"].eq(200).all()
    assert batch["actual_steps"].tolist() == [4]  # BC mask stays one chunk.
    assert batch["td_done"].tolist() == [0.]


@pytest.mark.parametrize("terminal", [True, False])
def test_terminal_partial_tail_and_truncation_do_not_cross_episode(terminal):
    replay = ReplayBuffer()
    for start, count in [(0, 4), (4, 4), (8, 2)]:
        item = chunk(start, count=count, done=start == 8)
        item.terminated = torch.tensor(float(terminal and start == 8))
        item.truncated = torch.tensor(float(not terminal and start == 8))
        replay.add(item)
    replay.add(chunk(0, episode=1))
    batch = replay.sample(256, td_chunks=3)
    rows = {int(state): i for i, state in enumerate(batch["state_vec"][:, 0])}
    assert set(rows) == {0, 4, 8}
    for start, steps in [(0, 10), (4, 6), (8, 2)]:
        i = rows[start]
        assert batch["td_steps"][i] == steps
        assert batch["td_done"][i] == 1
        assert batch["td_next_state_vec"][i] == 10
        assert discounted_chunk_return(batch["td_reward_seq"], .99, batch["td_steps"])[i] == pytest.approx(.99 ** (steps - 1))


def test_overlap_uses_t_plus_12_not_adjacent_insertion_or_stride_2():
    replay = ReplayBuffer()
    for start in [0, 4, 8, 12, 2, 6, 10]:
        replay.add(chunk(start))
    batch = replay.sample(256, td_chunks=3)
    endpoints = {int(state): int(end) for state, end in zip(batch["state_vec"][:, 0], batch["td_next_state_vec"][:, 0])}
    assert endpoints == {0: 12, 4: 16, 2: 14}


def test_missing_middle_chunk_or_mismatched_boundary_is_not_eligible():
    replay = ReplayBuffer()
    replay.add(chunk(0))
    replay.add(chunk(8))
    with pytest.raises(ValueError):
        replay.sample(1, td_chunks=3)
    invalid = chunk(4)
    invalid.state_vec.fill_(999)
    replay.add(invalid)
    with pytest.raises(ValueError):
        replay.sample(1, td_chunks=3)


def test_eviction_invalidates_old_horizons_and_checkpoint_roundtrip(tmp_path):
    replay = ReplayBuffer(3)
    for start in (0, 4, 8):
        replay.add(chunk(start))
    replay.sample(1, td_chunks=3)
    replay.add(chunk(12))
    assert replay.sample(10, td_chunks=3)["state_vec"].tolist() == [[4.]]
    path = tmp_path / "replay.pt"
    torch.save(list(replay.buffer), path)
    restored = ReplayBuffer()
    for item in torch.load(path, weights_only=False):
        restored.add(item)
    assert restored.sample(10, td_chunks=3)["td_next_state_vec"].tolist() == [[16.]]


def test_legacy_completed_originals_reused_and_overlap_tail_excluded():
    replay = ReplayBuffer()
    for start in (0, 4, 8, 2, 6):
        replay.add(chunk(start, done=start == 8, legacy=True))
    replay.add(chunk(0, episode=1, legacy=True))
    batch = replay.sample(256, td_chunks=3)
    assert sorted(batch["state_vec"][:, 0].tolist()) == [0., 4., 8.]
    assert batch["td_next_state_vec"].eq(12).all()


@pytest.mark.parametrize("done,count", [(False, 4), (True, 2)])
def test_learner_uses_three_chunk_discount_endpoint_and_terminal_mask(monkeypatch, done, count):
    monkeypatch.setattr("random.sample", lambda population, n: sorted(population)[:n])
    config = ACTStage2Config(stage1_checkpoint="a", act_checkpoint="b", device="cpu",
                             proprio_dim=0, batch_size=1, actor_hidden_dim=8, critic_hidden_dim=8, td_chunks=3)
    assert config.td_chunks == 3
    actor = Stage2ChunkActor(config.state_dim, 0, config.chunk_dim, hidden_dim=8)
    learner = Stage2Learner(SimpleNamespace(actor=actor), config)
    for network in (learner.critic, learner.target_critic):
        for parameter in network.parameters():
            parameter.data.zero_()
    learner.target_critic.q1.net[-1].bias.data.fill_(.4)
    learner.target_critic.q2.net[-1].bias.data.fill_(.7)
    replay = ReplayBuffer()
    for start in (0, 4, 8):
        replay.add(chunk(start, count=count if start == 8 else 4,
                         done=done and start == 8, dim=config.state_dim))
    seen = []
    hook = learner.target_actor.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    result = learner.update_once(replay, update_actor=False)
    hook.remove()
    steps = 8 + count
    target = .99 ** (steps - 1) if done else .99 ** steps * .4
    assert result.critic_loss == pytest.approx(2 * target ** 2, rel=1e-5)
    assert result.diagnostics["td_steps_mean"] == steps
    assert seen[0].eq(steps).all()


@pytest.mark.parametrize("missing_field", [True, False])
@pytest.mark.parametrize("current_horizon", [1, 3])
def test_checkpoint_resumes_with_requested_td_horizon(tmp_path, missing_field, current_horizon):
    from dataclasses import replace
    from act_rlt.stage2 import OnlineStage2Metrics
    from act_rlt.train_stage2 import save_checkpoint, load_checkpoint

    config = ACTStage2Config(stage1_checkpoint="a", act_checkpoint="b", device="cpu", proprio_dim=0,
                             actor_hidden_dim=8, critic_hidden_dim=8, td_chunks=current_horizon)

    def learner(cfg):
        actor = Stage2ChunkActor(cfg.state_dim, 0, cfg.chunk_dim, hidden_dim=8)
        return Stage2Learner(SimpleNamespace(actor=actor), cfg)

    old = learner(replace(config, td_chunks=4 - current_horizon))
    replay = ReplayBuffer()
    for start in (0, 4, 8):
        replay.add(chunk(start, done=start == 8, legacy=True, dim=config.state_dim))
    save_checkpoint(tmp_path, old.config, old, replay, OnlineStage2Metrics())
    path = tmp_path / "checkpoints/latest.pt"
    if missing_field:
        saved = torch.load(path, weights_only=False)
        del saved["config"]["td_chunks"]
        torch.save(saved, path)
    restored, data = learner(config), ReplayBuffer()
    load_checkpoint(path, config, restored, data)
    assert restored.config.td_chunks == current_horizon
    for key, value in old.critic.state_dict().items():
        torch.testing.assert_close(restored.critic.state_dict()[key], value)
    assert sorted(data.sample(256, td_chunks=3)["td_steps"].tolist()) == [4, 8, 12]


def test_three_chunk_td_is_opt_in():
    from act_rlt.train_stage2 import build_parser

    parser = build_parser()
    base = ["--stage1-checkpoint", "a", "--act-checkpoint", "b", "--dry-run"]
    assert not parser.parse_args(base).three_chunk_td
    assert parser.parse_args(base + ["--three-chunk-td"]).three_chunk_td
    assert ACTStage2Config(stage1_checkpoint="a", act_checkpoint="b").td_chunks == 1
