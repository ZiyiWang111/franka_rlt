from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from act_rlt.act_encoder import predict_act_chunk_with_encoder_hidden
from act_rlt.stage2 import (
    ACTStage2Config,
    ChunkExecution,
    OnlineStage2Metrics,
    Stage2Learner,
    run_online_stage2,
)
from act_rlt.train_stage2 import (
    FrankaInsertionStage2Env,
    RESET_RETREAT_Y_M,
    RESET_XY_MARGIN_M,
    build_parser,
    load_checkpoint,
    retreat_pose_along_positive_y,
    sample_workspace_pose,
    save_checkpoint,
    validate_args,
)
from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.replay_buffer import ReplayBuffer
from evo_rlt.core.utils import unflatten_chunk
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor import NormalizerProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.types import TransitionKey


def _config(**overrides) -> ACTStage2Config:
    values = {
        "stage1_checkpoint": "/tmp/stage1",
        "act_checkpoint": "/tmp/act",
        "device": "cpu",
        "batch_size": 1,
        "warmup_steps": 8,
        "total_env_steps": 16,
        "replay_capacity": 32,
        "actor_hidden_dim": 16,
        "critic_hidden_dim": 16,
    }
    values.update(overrides)
    return ACTStage2Config(**values)


class _FakeEncoder(nn.Module):
    def forward(self, tokens):
        return tokens + 1


class _FakeACTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _FakeEncoder()


class _FakeACTPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeACTModel()
        self.calls = 0

    def predict_action_chunk(self, batch):
        self.calls += 1
        batch_size = batch["observation.state"].shape[0]
        tokens = torch.zeros(5, batch_size, 8)
        self.model.encoder(tokens)
        return torch.full((batch_size, 16, 7), 0.25)


def test_act_reference_and_hidden_come_from_one_forward() -> None:
    policy = _FakeACTPolicy().eval()
    action, hidden = predict_act_chunk_with_encoder_hidden(
        policy, {"observation.state": torch.zeros(2, 7)}
    )
    assert policy.calls == 1
    assert action.shape == (2, 16, 7)
    assert hidden.shape == (2, 5, 8)
    assert torch.equal(hidden, torch.ones_like(hidden))


def test_actor_outputs_final_action_not_reference_plus_residual() -> None:
    actor = ChunkActor(state_dim=775, chunk_dim=24, hidden_dim=16, num_layers=2)
    for parameter in actor.parameters():
        nn.init.zeros_(parameter)
    state = torch.zeros(2, 775)
    reference = torch.ones(2, 24)
    mean, _ = actor(state, reference)
    assert torch.equal(mean, torch.zeros_like(mean))


class _FakeStage2Policy(nn.Module):
    def __init__(self, config: ACTStage2Config):
        super().__init__()
        self.config = config
        self.actor = ChunkActor(
            config.state_dim,
            config.chunk_dim,
            hidden_dim=config.actor_hidden_dim,
            num_layers=config.actor_num_layers,
            fixed_std=config.actor_fixed_std,
            ref_dropout_p=config.actor_ref_dropout_p,
        )

    def encode_and_reference(self, batch):
        batch_size = batch["observation.state"].shape[0]
        state = torch.zeros(batch_size, self.config.state_dim)
        ref = torch.full(
            (batch_size, self.config.chunk_length, self.config.action_dim), 0.1
        )
        full = torch.full((batch_size, 16, 7), 0.1)
        return state, ref, full

    def actor_chunk(self, state_vec, ref_chunk, *, deterministic):
        sampled, mean = self.actor.sample(state_vec, ref_chunk.flatten(start_dim=-2))
        return unflatten_chunk(
            mean if deterministic else sampled, self.config.chunk_length
        )


class _FakeEnvironment:
    def __init__(self, config: ACTStage2Config):
        self.config = config
        self.executed: list[torch.Tensor] = []

    def reset(self, *, episode_id: int, warmup: bool):
        return {"observation.state": torch.zeros(1, 7)}

    def execute_chunk(self, action_chunk, full_reference):
        self.executed.append(action_chunk.detach().cpu())
        return ChunkExecution(
            next_batch={"observation.state": torch.zeros(1, 7)},
            exec_chunk=action_chunk.squeeze(0).detach().cpu(),
            reward_seq=torch.zeros(self.config.chunk_length),
            actual_steps=self.config.chunk_length,
            done=False,
            terminated=False,
            truncated=False,
        )


class _ZeroStepWorkspaceEnvironment(_FakeEnvironment):
    def __init__(self, config):
        super().__init__(config)
        self.calls = 0
        self.resets = 0

    def reset(self, *, episode_id: int, warmup: bool):
        self.resets += 1
        return super().reset(episode_id=episode_id, warmup=warmup)

    def execute_chunk(self, action_chunk, full_reference):
        self.calls += 1
        if self.calls == 1:
            return ChunkExecution(
                next_batch={"observation.state": torch.zeros(1, 7)},
                exec_chunk=torch.zeros(self.config.chunk_length, self.config.action_dim),
                reward_seq=torch.zeros(self.config.chunk_length),
                actual_steps=0,
                done=True,
                terminated=False,
                truncated=True,
                info={"workspace_violation": True},
            )
        return super().execute_chunk(action_chunk, full_reference)


def test_warmup_has_no_updates_and_utd_is_per_chunk() -> None:
    torch.manual_seed(0)
    config = _config()
    policy = _FakeStage2Policy(config)
    learner = Stage2Learner(policy, config)
    replay = ReplayBuffer(config.replay_capacity)
    env = _FakeEnvironment(config)

    metrics = run_online_stage2(policy, learner, env, replay, config)

    assert metrics.chunks == 6
    assert metrics.env_steps == 24
    assert metrics.warmup_env_steps == 8
    assert metrics.online_env_steps == 16
    assert metrics.critic_updates == 4 * config.utd_ratio == 20
    assert metrics.actor_updates == 10
    assert [int(t.source) for t in replay.buffer] == [1, 1, 2, 2, 2, 2]
    # The two warmup chunks are exactly the frozen reference.
    assert torch.equal(env.executed[0], torch.full((1, 4, 6), 0.1))
    assert torch.equal(env.executed[1], torch.full((1, 4, 6), 0.1))


def test_zero_step_workspace_refusal_resets_without_fake_replay_transition() -> None:
    config = _config(warmup_steps=4, total_env_steps=4)
    policy = _FakeStage2Policy(config)
    learner = Stage2Learner(policy, config)
    replay = ReplayBuffer(config.replay_capacity)
    env = _ZeroStepWorkspaceEnvironment(config)
    observed = []

    metrics = run_online_stage2(
        policy,
        learner,
        env,
        replay,
        config,
        on_chunk=lambda current, execution: (
            observed.append(execution.info.copy())
            if execution.info.get("workspace_violation")
            else None
        ),
    )

    assert len(replay) == 2
    assert metrics.env_steps == 8
    assert metrics.chunks == 2
    assert metrics.episodes == 1
    assert observed == [{"workspace_violation": True, "warmup": True}]
    assert env.resets == 2


def test_environment_converts_workspace_refusal_to_truncation() -> None:
    config = _config()

    class Control:
        def __init__(self):
            self.stopped = False

        def stop_move(self):
            self.stopped = True

    class Robot:
        def __init__(self):
            self.robot = Control()

        def resync_command_pose(self):
            pass

        def send_action(self, _command):
            raise RuntimeError(
                "refusing TCP target outside workspace: target=[0 0 0]"
            )

    env = object.__new__(FrankaInsertionStage2Env)
    env.config = config
    env.robot = Robot()
    env.post = lambda action: action
    env.max_step_m = 0.002
    env.max_step_rad = 0.02
    env.period_s = 0.0
    env._monitor = None
    env._normalize_executed_action = lambda action: action.reshape(-1)[:6]
    env._processed_observation = lambda: {"observation.state": torch.zeros(1, 7)}

    result = env.execute_chunk(
        torch.zeros(1, config.chunk_length, config.action_dim),
        torch.zeros(1, 16, 7),
    )

    assert result.actual_steps == 0
    assert result.done and result.truncated and not result.terminated
    assert result.info["workspace_violation"] is True
    assert env.robot.robot.stopped is True


def test_callback_observes_completed_episode_and_warmup_phase() -> None:
    config = _config(warmup_steps=4, total_env_steps=4)
    policy = _FakeStage2Policy(config)
    learner = Stage2Learner(policy, config)
    replay = ReplayBuffer(config.replay_capacity)
    env = _FakeEnvironment(config)

    def terminal_chunk(action_chunk, full_reference):
        result = _FakeEnvironment.execute_chunk(env, action_chunk, full_reference)
        result.done = True
        result.terminated = True
        result.stop_requested = True
        result.info["success"] = True
        return result

    env.execute_chunk = terminal_chunk
    observed = []
    metrics = run_online_stage2(
        policy,
        learner,
        env,
        replay,
        config,
        on_chunk=lambda current, execution: observed.append(
            (current.episodes, current.successes, execution.info["warmup"])
        ),
    )

    assert metrics.episodes == 1
    assert metrics.successes == 1
    assert observed == [(1, 1, True)]


def test_confirmed_stage2_default_shapes() -> None:
    config = _config()
    assert config.state_dim == 775
    assert config.chunk_dim == 24
    assert config.chunk_length == 4
    assert config.action_dim == 6
    assert config.utd_ratio == 5


def test_stage2_default_episode_time_is_five_seconds() -> None:
    args = build_parser().parse_args(
        ["--stage1-checkpoint", "/tmp/stage1", "--act-checkpoint", "/tmp/act"]
    )
    assert args.episode_time == 5.0
    assert args.warmup_steps == 4_000
    assert args.total_env_steps == 10_000


def test_automatic_reset_retreat_and_workspace_sample() -> None:
    lower = torch.tensor([0.3, -0.2, 0.1]).numpy()
    upper = torch.tensor([0.7, 0.2, 0.5]).numpy()
    measured = torch.tensor([0.5, 0.0, 0.3, 0.1, 0.2, 0.3]).numpy()
    retreat = retreat_pose_along_positive_y(measured, lower, upper)
    assert np.isclose(retreat[1], measured[1] + RESET_RETREAT_Y_M)
    assert np.allclose(retreat[[0, 2, 3, 4, 5]], measured[[0, 2, 3, 4, 5]])

    sampled = sample_workspace_pose(lower, upper, measured[3:])
    assert (sampled[:2] >= lower[:2] + RESET_XY_MARGIN_M).all()
    assert (sampled[:2] <= upper[:2] - RESET_XY_MARGIN_M).all()
    assert lower[2] <= sampled[2] <= upper[2]
    assert np.allclose(sampled[3:], measured[3:])


def test_automatic_reset_can_use_explicit_narrow_sample_box() -> None:
    workspace_min = np.array([0.570, -0.223, 0.128])
    workspace_max = np.array([0.710, -0.060, 0.180])
    sample_min = np.array([0.630, -0.120, 0.130])
    sample_max = np.array([0.670, -0.080, 0.175])
    sampled = sample_workspace_pose(
        workspace_min,
        workspace_max,
        np.array([0.1, 0.2, 0.3]),
        sample_min=sample_min,
        sample_max=sample_max,
    )
    assert (sampled[:3] >= sample_min).all()
    assert (sampled[:3] <= sample_max).all()


def test_stage2_parser_accepts_narrow_sample_box() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--stage1-checkpoint", "/tmp/stage1", "--act-checkpoint", "/tmp/act",
            "--allow-motion",
            "--workspace-min", "0.570", "-0.223", "0.128",
            "--workspace-max", "0.710", "-0.060", "0.180",
            "--sample-min", "0.630", "-0.120", "0.130",
            "--sample-max", "0.670", "-0.080", "0.175",
        ]
    )
    validate_args(parser, args)
    assert args.sample_min == [0.630, -0.120, 0.130]
    assert args.sample_max == [0.670, -0.080, 0.175]


def test_workspace_sample_rejects_xy_box_without_two_centimetre_margins() -> None:
    with pytest.raises(ValueError, match="exceed 4 cm"):
        sample_workspace_pose(
            np.array([0.0, 0.0, 0.1]),
            np.array([0.04, 0.10, 0.2]),
            np.zeros(3),
        )


def test_timeout_requires_manual_success_or_failure_confirmation(monkeypatch) -> None:
    answers = iter(["invalid", "s"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    assert FrankaInsertionStage2Env._prompt_outcome_after_timeout() == "s"


def test_saved_act_action_stats_round_trip_after_safety_edit() -> None:
    act_config = ACTConfig(
        device="cpu",
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,))
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    stats = {
        "observation.state": {"mean": torch.zeros(7), "std": torch.ones(7)},
        "action": {
            "mean": torch.arange(7, dtype=torch.float32),
            "std": torch.full((7,), 2.0),
        },
    }
    pre, post = make_act_pre_post_processors(act_config, stats)
    normalizer = next(step for step in pre.steps if isinstance(step, NormalizerProcessorStep))
    normalized = torch.tensor([[0.5, -0.5, 0.0, 1.0, -1.0, 0.25, -0.25]])
    physical = post(normalized)
    round_trip = normalizer(create_transition(action=physical))[TransitionKey.ACTION]
    assert torch.allclose(round_trip, normalized)


def test_stage2_checkpoint_restores_learner_replay_and_counters() -> None:
    config = _config(total_env_steps=12)
    policy = _FakeStage2Policy(config)
    learner = Stage2Learner(policy, config)
    replay = ReplayBuffer(config.replay_capacity)
    metrics = run_online_stage2(
        policy, learner, _FakeEnvironment(config), replay, config
    )

    with TemporaryDirectory() as directory:
        output = Path(directory)
        save_checkpoint(output, config, learner, replay, metrics)

        restored_policy = _FakeStage2Policy(config)
        restored_learner = Stage2Learner(restored_policy, config)
        restored_replay = ReplayBuffer(config.replay_capacity)
        restored_metrics = load_checkpoint(
            output / "checkpoints/latest.pt",
            config,
            restored_learner,
            restored_replay,
        )

    assert len(restored_replay) == len(replay)
    assert restored_learner.critic_updates == learner.critic_updates
    assert restored_learner.actor_updates == learner.actor_updates
    assert restored_metrics.env_steps == metrics.env_steps
    assert restored_metrics.chunks == metrics.chunks


@pytest.mark.parametrize(
    ("saved_warmup", "saved_total", "warmup_progress", "online_progress", "new_warmup", "new_total"),
    [
        (5_000, 100_000, 3_700, 0, 4_000, 10_000),
        (4_000, 10_000, 4_000, 10_000, 4_000, 20_000),
    ],
)
def test_resume_allows_safe_schedule_changes(
    saved_warmup,
    saved_total,
    warmup_progress,
    online_progress,
    new_warmup,
    new_total,
) -> None:
    saved_config = _config(warmup_steps=saved_warmup, total_env_steps=saved_total)
    saved_policy = _FakeStage2Policy(saved_config)
    saved_learner = Stage2Learner(saved_policy, saved_config)
    saved_replay = ReplayBuffer(saved_config.replay_capacity)
    saved_metrics = OnlineStage2Metrics(
        env_steps=warmup_progress + online_progress,
        warmup_env_steps=warmup_progress,
        online_env_steps=online_progress,
    )

    resumed_config = _config(warmup_steps=new_warmup, total_env_steps=new_total)
    resumed_policy = _FakeStage2Policy(resumed_config)
    resumed_learner = Stage2Learner(resumed_policy, resumed_config)
    resumed_replay = ReplayBuffer(resumed_config.replay_capacity)

    with TemporaryDirectory() as directory:
        checkpoint_root = Path(directory)
        save_checkpoint(
            checkpoint_root,
            saved_config,
            saved_learner,
            saved_replay,
            saved_metrics,
        )
        restored = load_checkpoint(
            checkpoint_root / "checkpoints/latest.pt",
            resumed_config,
            resumed_learner,
            resumed_replay,
        )

    assert restored.warmup_env_steps == warmup_progress
    assert restored.online_env_steps == online_progress
