from __future__ import annotations

import threading
import time
from collections import deque
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from torch import nn

from act_rlt.act_encoder import (
    predict_act_chunk_with_encoder_hidden,
    predict_act_chunk_with_encoder_hidden_and_pos,
)
from act_rlt.stage2_networks import ChunkInputFusion, Stage2ChunkActor, Stage2TwinCritic
from act_rlt.human_input import HumanInputMonitor
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
    RESET_RETREAT_Z_M,
    RESET_XY_MARGIN_M,
    STARTUP_WORKSPACE_HALF_RANGE_M,
    build_parser,
    capture_centered_workspace,
    _format_terminal_metrics,
    load_checkpoint,
    retreat_pose_along_positive_y,
    sample_z_insertion_workspace_pose,
    sample_workspace_pose,
    save_checkpoint,
    validate_args,
)
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
        "fusion_dim": 16,
        "bc_init_steps": 0,
        "critic_init_steps": 0,
    }
    values.update(overrides)
    return ACTStage2Config(**values)


def test_stage2_uses_only_checkpoint_camera_at_reset_and_chunk_boundary():
    from act_rlt.stage2_runtime import FixedChunkRuntime

    observation = {f"joint_{index}": 0.0 for index in range(7)}
    observation["wrist"] = np.zeros((720, 1280, 3), dtype=np.uint8)
    env = object.__new__(FrankaInsertionStage2Env)
    env.camera_shapes = {"wrist": (720, 1280, 3)}
    env.robot = SimpleNamespace(get_observation=lambda: observation)
    env.pre = lambda frame: frame

    reset_frame = env._processed_observation()
    assert set(reset_frame) == {"observation.state", "observation.images.wrist"}
    assert reset_frame["observation.images.wrist"].shape == (3, 720, 1280)

    runtime = object.__new__(FixedChunkRuntime)
    runtime.env = env
    runtime.observation_ready = threading.Condition()
    runtime.observation = observation
    runtime.observation_started = 1.0
    runtime.stop = threading.Event()
    boundary_frame = runtime._boundary_batch(0.0)
    assert set(boundary_frame) == set(reset_frame)
    torch.testing.assert_close(boundary_frame["observation.images.wrist"], reset_frame["observation.images.wrist"])


class _FakeEncoder(nn.Module):
    def forward(self, tokens):
        return tokens + 1


class _FakeACTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _FakeEncoder()
        self.decoder = _FakeDecoder()


class _FakeDecoder(nn.Module):
    def forward(self, hidden, *, encoder_pos_embed):
        return hidden


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
        self.model.decoder(tokens, encoder_pos_embed=torch.arange(5).view(5, 1, 1).expand(5, 1, 8))
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


def test_act_reference_and_position_come_from_same_forward() -> None:
    policy = _FakeACTPolicy().eval()
    batch = {"observation.state": torch.zeros(2, 7)}
    baseline = policy.predict_action_chunk(batch)
    action, hidden, pos = predict_act_chunk_with_encoder_hidden_and_pos(policy, batch)
    assert policy.calls == 2
    assert torch.equal(action, baseline)
    assert hidden.shape == (2, 5, 8)
    assert pos.shape == (1, 5, 8)
    assert torch.equal(pos[0, :, 0], torch.arange(5))


def test_actor_outputs_final_action_not_reference_plus_residual() -> None:
    actor = Stage2ChunkActor(
        state_dim=775, proprio_dim=7, chunk_dim=24,
        fusion_dim=16, hidden_dim=16, num_layers=2,
    )
    for parameter in actor.parameters():
        nn.init.zeros_(parameter)
    state = torch.zeros(2, 775)
    reference = torch.ones(2, 24)
    mean, _ = actor(state, reference)
    assert torch.equal(mean, torch.zeros_like(mean))


def test_stage2_fusion_gives_each_input_equal_width_and_bounded_scale() -> None:
    fusion = ChunkInputFusion(state_dim=775, proprio_dim=7, chunk_dim=24, fusion_dim=16)
    state = torch.randn(3, 775, requires_grad=True)
    reference = torch.randn(3, 24, requires_grad=True)
    features = fusion(state, reference)
    assert features.shape == (3, 48)
    assert torch.isfinite(features).all()
    assert features.abs().max() <= 1
    features.square().sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert reference.grad is not None and torch.isfinite(reference.grad).all()
    assert state.grad[:, :768].abs().sum() > 0
    assert state.grad[:, 768:].abs().sum() > 0
    assert reference.grad.abs().sum() > 0


def test_stage2_actor_and_twin_critic_use_balanced_production_shapes() -> None:
    state = torch.randn(2, 775)
    chunk = torch.randn(2, 24)
    actor = Stage2ChunkActor(state_dim=775, proprio_dim=7, chunk_dim=24)
    critic = Stage2TwinCritic(state_dim=775, proprio_dim=7, chunk_dim=24)
    assert actor.fusion(state, chunk).shape == (2, 384)
    mean, std = actor(state, chunk)
    q1, q2 = critic(state, chunk)
    assert mean.shape == std.shape == (2, 24)
    assert q1.shape == q2.shape == (2, 1)
    assert torch.isfinite(mean).all() and torch.isfinite(q1).all() and torch.isfinite(q2).all()


def test_terminal_metrics_are_compact_while_json_fields_remain_available() -> None:
    line = _format_terminal_metrics({
        "warmup": False, "env_steps": 12, "warmup_env_steps": 4,
        "online_env_steps": 8, "episodes": 2, "successes": 1,
        "actual_steps": 4, "done": True, "success": True,
        "workspace_violation": False, "safety_clip_steps": 0,
        "critic_loss": 0.25, "actor_loss": None, "q1_mean": 0.2,
        "q2_mean": 0.3, "q_reference_mean": 0.1,
        "episode_exploration_vs_actor_mean_tcp_target_translation_norm_mm": 1.2,
        "episode_sampled_vs_reference_tcp_target_translation_norm_mm": 2.3,
    })
    assert line == (
        "[ONLINE] env=12 warm=4 online=8 ep=2 succ=1 k=4 DONE=success "
        "loss(critic=0.250) Q(q1=0.200 q2=0.300 ref=0.100) "
        "Δtcp_mm(explore=1.20 sampled-ref=2.30)"
    )


def test_v2_targets_clip_high_q_and_mask_terminal_bootstrap():
    config = _config(batch_size=3)
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
        q.net[-1].bias.data.fill_(90.)
    metrics = learner.update_once(SimpleNamespace(sample=lambda n: batch))
    expected = torch.tensor([0.99**4, 0.99, 0.])
    assert metrics.critic_loss == pytest.approx(2 * expected.square().mean().item())
    assert metrics.diagnostics["target_q_clip_fraction"] == 1
    assert metrics.diagnostics["td_target_mean"] == pytest.approx(expected.mean().item())
    assert metrics.diagnostics["q_reference_mean"] == pytest.approx(0.0)
    assert metrics.diagnostics["q_reference_max"] == pytest.approx(0.0)
    assert all(p.grad is None for p in learner.target_actor.parameters())
    assert all(p.grad is None for p in learner.target_critic.parameters())


def test_v2_projection_matches_physical_norm_clip_and_has_gradients():
    from act_rlt.infer import bounded_action
    config = _config()
    learner = Stage2Learner(_FakeStage2Policy(config), config)
    scale = torch.tensor([.01,.02,.03,.04,.05,.06,1.])
    offset = torch.tensor([.001,0.,0.,0.,0.,0.,0.])
    learner.configure_action_projection(lambda x: x * scale + offset)
    action = torch.ones(1,24, requires_grad=True)
    projected = learner.action_projection(action)
    physical = projected.reshape(4,6) * scale[:6] + offset[:6]
    expected = list(bounded_action((scale+offset).numpy(), .002, .02).values())
    torch.testing.assert_close(physical[0], torch.tensor(expected, dtype=torch.float32))
    projected.sum().backward()
    assert torch.isfinite(action.grad).all()


def test_v2_bc_initialization_reduces_reference_error_and_roundtrips():
    torch.manual_seed(123)
    config = _config(bc_init_steps=1000)
    policy = _FakeStage2Policy(config)
    learner = Stage2Learner(policy, config)
    batch = {"state_vec": torch.zeros(1,775), "ref_chunk_flat": torch.full((1,24), .1)}
    class Replay:
        def __len__(self): return 1
        def sample(self, n): return batch
    before = (policy.actor(batch["state_vec"], batch["ref_chunk_flat"])[0] - .1).square().mean()
    learner.initialize(Replay())
    after = (policy.actor(batch["state_vec"], batch["ref_chunk_flat"])[0] - .1).square().mean()
    assert after < before * .1
    restored = Stage2Learner(_FakeStage2Policy(config), config)
    restored.load_state_dict(learner.state_dict())
    assert restored.initialized
    for p, q in zip(restored.target_actor.parameters(), policy.actor.parameters()):
        torch.testing.assert_close(p, q)


def test_discounted_return_masks_unexecuted_padding():
    from evo_rlt.core.losses import discounted_chunk_return
    value = discounted_chunk_return(torch.tensor([[0.,1.,99.,99.]]), .99, torch.tensor([2]))
    assert value.item() == pytest.approx(.99)


def test_v2_critic_learns_terminal_success_and_failure():
    torch.manual_seed(7)
    config = _config(batch_size=2, critic_lr=.003)
    learner = Stage2Learner(_FakeStage2Policy(config), config)
    states = torch.zeros(2,775)
    states[1,0] = 1
    batch = {"state_vec": states, "next_state_vec": states,
             "exec_chunk_flat": torch.zeros(2,24), "ref_chunk_flat": torch.zeros(2,24),
             "next_ref_flat": torch.zeros(2,24), "done": torch.ones(2),
             "actual_steps": torch.ones(2, dtype=torch.long),
             "reward_seq": torch.tensor([[0.,0,0,0],[1.,0,0,0]])}
    replay = SimpleNamespace(sample=lambda n: batch)
    for _ in range(600):
        learner.update_once(replay, update_actor=False)
    q1,q2 = learner.critic(states,batch["exec_chunk_flat"])
    for q in (q1,q2):
        torch.testing.assert_close(q.flatten(), torch.tensor([0.,1.]), atol=.05, rtol=0)
    assert learner.actor_updates == 0


def test_v2_resume_reproduces_next_update():
    import copy
    config = _config(actor_ref_dropout_p=.5)
    first = Stage2Learner(_FakeStage2Policy(config), config)
    batch = {"state_vec": torch.zeros(1,775), "next_state_vec": torch.zeros(1,775),
             "exec_chunk_flat": torch.zeros(1,24), "ref_chunk_flat": torch.ones(1,24),
             "next_ref_flat": torch.ones(1,24), "done": torch.ones(1),
             "actual_steps": torch.ones(1, dtype=torch.long),
             "reward_seq": torch.tensor([[1.,0,0,0]])}
    replay = SimpleNamespace(sample=lambda n: batch)
    first.update_once(replay)
    second = Stage2Learner(_FakeStage2Policy(config), config)
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    rng = torch.get_rng_state()
    a = first.update_once(replay)
    torch.set_rng_state(rng)
    b = second.update_once(replay)
    assert a == b
    for model_a,model_b in ((first.policy.actor, second.policy.actor),
                            (first.critic, second.critic),
                            (first.target_actor, second.target_actor),
                            (first.target_critic, second.target_critic)):
        for p,q in zip(model_a.parameters(), model_b.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)


class _FakeStage2Policy(nn.Module):
    def __init__(self, config: ACTStage2Config):
        super().__init__()
        self.config = config
        self.actor = Stage2ChunkActor(
            config.state_dim,
            config.proprio_dim,
            config.chunk_dim,
            fusion_dim=config.fusion_dim,
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


def test_human_chunk_uses_executed_bc_target_with_act_reference_preserved():
    config = _config(warmup_steps=4, total_env_steps=4)
    policy = _FakeStage2Policy(config)
    learner = Stage2Learner(policy, config)
    replay = ReplayBuffer(config.replay_capacity)
    env = _FakeEnvironment(config)
    ordinary_execute = env.execute_chunk

    def human_first_chunk(action_chunk, full_reference):
        result = ordinary_execute(action_chunk, full_reference)
        if len(env.executed) == 1:
            result.exec_chunk = torch.full((4, 6), .2)
            result.bc_target_chunk = result.exec_chunk.clone()
            result.intervention = True
        return result

    env.execute_chunk = human_first_chunk
    metrics = run_online_stage2(policy, learner, env, replay, config)
    assert metrics.human_chunks == 1
    assert metrics.human_env_steps == 4
    human, policy_transition = replay.buffer
    assert int(human.source) == 3
    assert float(human.intervention) == 1
    torch.testing.assert_close(human.ref_chunk, torch.full((4, 6), .1))
    torch.testing.assert_close(human.bc_target_chunk, torch.full((4, 6), .2))
    assert policy_transition.bc_target_chunk is None
    sampled = replay.sample(2, include_bc_target=True)
    assert sorted(sampled["bc_target_flat"].mean(dim=1).tolist()) == pytest.approx([.1, .2])


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


def _threaded_env(config, *, camera_delay=.003, period=.02, refusal_at=None):
    class Robot:
        def __init__(self):
            self.stops = 0
            self.robot = SimpleNamespace(stop_servo=self.stop_servo)
            self.sent_at = []

        def stop_servo(self):
            self.stops += 1
            return True

        def get_observation(self):
            time.sleep(camera_delay)
            return {"observation.state": torch.zeros(1, 7)}

        def resync_command_pose(self):
            pass

        def send_action(self, command):
            if refusal_at is not None and len(self.sent_at) == refusal_at:
                raise RuntimeError("refusing TCP target outside workspace: test")
            self.sent_at.append(time.monotonic())

    env = object.__new__(FrankaInsertionStage2Env)
    env.config = config
    env.robot = Robot()
    env.pre = lambda observation: observation
    env.post = lambda action: action
    env.period_s = period
    env.episode_time_s = 10
    env.max_step_m = .002
    env.max_step_rad = .02
    env._monitor = SimpleNamespace(poll=lambda: None, close=lambda: None)
    env._episode_start = time.monotonic()
    env._runtime = None
    env._normalize_executed_action = lambda action: action[:config.action_dim]
    return env


@pytest.mark.parametrize("timestamped", [False, True])
def test_persistent_workers_cross_chunk_boundaries_without_waiting_for_learner(timestamped):
    from act_rlt.stage2_runtime import FixedChunkRuntime
    config = _config()
    env = _threaded_env(config)
    policy = _FakeStage2Policy(config).eval()
    calls = []

    def encode(batch):
        time.sleep(.003)
        calls.append(len(calls))
        return (
            torch.full((1, config.state_dim), float(len(calls))),
            torch.full((1, 4, 6), .1), torch.full((1, 16, 7), .1),
        )
    policy.encode_and_reference = encode
    if timestamped:
        from evo_rlt.adapters.lerobot.franka_robot.camera_timing import TimedObservation
        def timed_observation():
            observation = env.robot.get_observation()
            return TimedObservation(observation, time.monotonic(),
                                    {"camera_boundary_mode": "exposure_timestamp"})
        env.robot.get_timed_observation = timed_observation
    with patch("act_rlt.stage2_runtime.observation_frame", side_effect=lambda obs: obs):
        runtime = FixedChunkRuntime(env, policy, {}, warmup=True, step_budget=12)
        try:
            first = runtime.next_result()
            workers = tuple(runtime.threads)
            # Simulate replay updates/checkpoint IO taking longer than a chunk.
            time.sleep(.13)
            assert len(env.robot.sent_at) > 4
            second = runtime.next_result()
            third = runtime.next_result()
            assert tuple(runtime.threads) == workers
        finally:
            runtime.close()
    import json
    for result in (first, second, third):
        json.dumps(result.info)  # Timing objects/events must not leak into JSONL.
        assert result.info["next_boundary_total_wall_ms"] >= result.info["next_camera_wait_wall_ms"]
        assert result.info["current_encode_wall_ms"] >= 0
        assert result.info["send_action_wall_ms_sum"] >= 0
    assert second.info["chunk_boundary_command_interval_ms"] > 0
    assert third.info["result_queue_wait_wall_ms"] > 0
    assert len(calls) == 4  # initial state + one next state per chunk, no duplicates
    assert len(env.robot.sent_at) == 12
    torch.testing.assert_close(first.next_state_vec, second.state_vec)
    torch.testing.assert_close(second.next_state_vec, third.state_vec)
    torch.testing.assert_close(first.next_ref_chunk, second.ref_chunk)
    assert all(x.actual_steps == 4 for x in (first, second, third))
    assert not first.info["collector_paused"]
    assert third.info["collector_paused"]
    assert env.robot.stops == 1
    assert max(np.diff(env.robot.sent_at)) < .08
    assert min(np.diff(env.robot.sent_at)) > .01
    assert all(not thread.is_alive() for thread in workers)


def test_human_takeover_waits_for_complete_policy_chunk_and_returns_at_boundary():
    from act_rlt.stage2_runtime import FixedChunkRuntime

    config = _config()
    env = _threaded_env(config, period=.01)
    env.enable_human_intervention = True
    env.teleop_speed_m_s = .01
    env._control_mode = "policy"
    commands = []
    original_send = env.robot.send_action

    def send(command):
        commands.append(command.copy())
        original_send(command)
        if len(commands) == 5:
            env._monitor.pending = True

    env.robot.send_action = send

    class Monitor:
        def __init__(self):
            self.switches = 0
            # A request already waiting before the first policy action must
            # still let that entire four-step chunk complete.
            self.pending = True

        def poll(self):
            return None

        def direction(self):
            return np.array([1., 0., 0.])

        def consume_toggle(self, *, human):
            if self.pending:
                self.pending = False
                self.switches += 1
                return True
            return False

    env._monitor = Monitor()
    with patch("act_rlt.stage2_runtime.observation_frame", side_effect=lambda obs: obs):
        runtime = FixedChunkRuntime(
            env, _FakeStage2Policy(config),
            {"observation.state": torch.zeros(1, 7)}, warmup=True, step_budget=12,
        )
        try:
            results = [runtime.next_result() for _ in range(3)]
        finally:
            runtime.close()

    assert [result.actual_steps for result in results] == [4, 4, 4]
    assert [result.intervention for result in results] == [False, True, False]
    assert [result.info.get("handoff_to") for result in results] == ["human", "policy", None]
    assert env.robot.stops == 3
    assert len(commands) == 12
    assert all(command["dx"] == pytest.approx(.0001) for command in commands[4:8])
    assert results[1].bc_target_chunk is not None
    torch.testing.assert_close(results[1].bc_target_chunk, results[1].exec_chunk)
    torch.testing.assert_close(results[0].next_state_vec, results[1].state_vec)
    torch.testing.assert_close(results[1].next_state_vec, results[2].state_vec)


def test_pynput_repeat_does_not_toggle_or_release_human_motion():
    monitor = object.__new__(HumanInputMonitor)
    monitor.backend = "pynput"
    monitor._lock = threading.Lock()
    monitor._stopping = threading.Event()
    monitor._pressed = set()
    monitor._pending_release = {}
    monitor._toggle_down = False
    monitor._toggle_pending = False
    monitor._toggle_release_deadline = None
    monitor._outcomes = deque()
    monitor._failure = None
    monitor._listener = None
    with patch("act_rlt.human_input.time.monotonic") as clock:
        clock.return_value = 0.0
        monitor._event("KEY_SPACE", True)
        assert monitor.consume_toggle(human=False)
        clock.return_value = .01
        monitor._event("KEY_SPACE", False)
        monitor._event("KEY_SPACE", True)
        assert not monitor.consume_toggle(human=True)
        clock.return_value = .02
        monitor._event("KEY_SPACE", False)
        clock.return_value = .20
        monitor._event("KEY_SPACE", True)
        assert monitor.consume_toggle(human=True)


def test_stage2_human_keys_reserve_s_f_for_outcomes_and_x_for_negative_x():
    monitor = object.__new__(HumanInputMonitor)
    monitor.backend = "evdev"
    monitor._lock = threading.Lock()
    monitor._stopping = threading.Event()
    monitor._pressed = set()
    monitor._pending_release = {}
    monitor._toggle_down = False
    monitor._toggle_pending = False
    monitor._toggle_release_deadline = None
    monitor._outcomes = deque()
    monitor._failure = None
    monitor._listener = None
    monitor._event("KEY_X", True)
    assert monitor.direction().tolist() == [-1.0, 0.0, 0.0]
    monitor._event("KEY_S", True)
    monitor._event("KEY_F", True)
    assert monitor.direction().tolist() == [-1.0, 0.0, 0.0]
    assert monitor.poll() == "s"
    assert monitor.poll() == "f"


def test_human_key_release_stops_servo_without_leaving_human_mode():
    from act_rlt.stage2_runtime import FixedChunkRuntime

    config = _config()
    env = _threaded_env(config, period=.01)
    env.enable_human_intervention = True
    env.teleop_speed_m_s = .01
    env._control_mode = "human"

    class Monitor:
        def __init__(self):
            self.calls = 0

        def poll(self):
            return None

        def direction(self):
            self.calls += 1
            return np.array([1., 0., 0.]) if self.calls <= 2 else np.zeros(3)

        def consume_toggle(self, *, human):
            return False

    env._monitor = Monitor()
    with patch("act_rlt.stage2_runtime.observation_frame", side_effect=lambda obs: obs):
        runtime = FixedChunkRuntime(
            env, _FakeStage2Policy(config),
            {"observation.state": torch.zeros(1, 7)}, warmup=True, step_budget=4,
        )
        try:
            result = runtime.next_result()
        finally:
            runtime.close()
    assert result.actual_steps == 4
    assert result.intervention
    assert result.info["human_idle_steps"] == 2
    assert len(env.robot.sent_at) == 2
    assert env.robot.stops == 2  # key release, then phase boundary
    assert env._control_mode == "human"


def test_default_online_loop_uses_persistent_collection_and_respects_phase_budget():
    config = _config(warmup_steps=4, total_env_steps=4)
    env = _threaded_env(config)
    env.reset = lambda **kwargs: {"observation.state": torch.zeros(1, 7)}
    policy = _FakeStage2Policy(config)
    learner = Stage2Learner(policy, config)
    replay = ReplayBuffer(config.replay_capacity)
    with patch("act_rlt.stage2_runtime.observation_frame", side_effect=lambda obs: obs):
        try:
            metrics = run_online_stage2(policy, learner, env, replay, config)
        finally:
            env.close()
    assert metrics.warmup_env_steps == 4
    assert metrics.online_env_steps == 4
    assert metrics.critic_updates == 5
    assert len(env.robot.sent_at) == 8
    assert [int(t.source) for t in replay.buffer] == [1, 2]


@pytest.mark.parametrize("refusal_at", [0, 2])
def test_threaded_workspace_refusal_preserves_only_executed_actions(refusal_at):
    config = _config()
    env = _threaded_env(config, refusal_at=refusal_at)
    with patch("act_rlt.stage2_runtime.observation_frame", side_effect=lambda obs: obs):
        try:
            result = env.execute_policy_chunk(
                _FakeStage2Policy(config), {"observation.state": torch.zeros(1, 7)},
                warmup=True, remaining_steps=8
            )
        finally:
            env.close()
    assert result.actual_steps == refusal_at
    assert result.done and result.truncated
    assert result.info["workspace_violation"]
    assert torch.count_nonzero(result.exec_chunk[refusal_at:]) == 0
    assert len(env.robot.sent_at) == refusal_at
    assert env.robot.stops == 1


def test_threaded_inference_failure_stops_servo_and_joins_workers():
    from act_rlt.stage2_runtime import FixedChunkRuntime
    config = _config()
    env = _threaded_env(config)
    policy = _FakeStage2Policy(config)
    policy.encode_and_reference = lambda batch: (_ for _ in ()).throw(RuntimeError("bad inference"))
    runtime = FixedChunkRuntime(env, policy, {}, warmup=True, step_budget=8)
    with pytest.raises(RuntimeError, match="bad inference"):
        runtime.next_result()
    with pytest.raises(RuntimeError, match="bad inference"):
        runtime.close()
    assert env.robot.stops == 1
    assert not env.robot.sent_at
    assert all(not thread.is_alive() for thread in runtime.threads)


@pytest.mark.parametrize("outcome", ["s", "f", "q", "timeout"])
def test_threaded_outcome_stops_before_prompt_or_retreat(outcome):
    config = _config()
    env = _threaded_env(config)
    env._monitor.poll = lambda: None if outcome == "timeout" else outcome
    calls = []

    def after_stop(name):
        assert env.robot.stops == 1
        calls.append(name)
        return "s"

    env._retreat_after_outcome = lambda: after_stop("retreat")
    env._prompt_outcome_after_timeout = lambda: after_stop("prompt")
    if outcome == "timeout":
        env._episode_start -= 100
    with patch("act_rlt.stage2_runtime.observation_frame", side_effect=lambda obs: obs):
        try:
            result = env.execute_policy_chunk(
                _FakeStage2Policy(config), {"observation.state": torch.zeros(1, 7)},
                warmup=True, remaining_steps=8,
            )
        finally:
            env.close()
    assert result.actual_steps == (4 if outcome in {"s", "f"} else 1)
    assert result.done
    assert result.reward_seq.sum().item() == int(outcome in {"s", "timeout"})
    if outcome == "s":
        assert result.reward_seq.tolist() == [0.0, 0.0, 0.0, 1.0]
    assert result.stop_requested == (outcome == "q")
    assert calls == ([] if outcome == "q" else ["prompt", "retreat"] if outcome == "timeout" else ["retreat"])
    # Warmup executes the ACT reference exactly, so all three endpoint
    # comparisons must be zero after one completed episode.
    assert result.info["episode_tcp_target_comparison_steps"] == result.actual_steps
    assert result.info["episode_exploration_vs_actor_mean_tcp_target_translation_norm_mm"] == pytest.approx(0.0)
    assert result.info["episode_actor_mean_vs_reference_tcp_target_translation_norm_mm"] == pytest.approx(0.0)
    assert result.info["episode_sampled_vs_reference_tcp_target_translation_norm_mm"] == pytest.approx(0.0)


def test_single_thread_outcome_waits_for_full_chunk():
    config = _config()
    env = _threaded_env(config, period=0.0)
    env._monitor.poll = lambda: "s"
    env._processed_observation = lambda: {"observation.state": torch.zeros(1, 7)}
    env._retreat_after_outcome = lambda: None
    result = env.execute_chunk(
        torch.zeros(1, config.chunk_length, config.action_dim),
        torch.zeros(1, 16, config.action_dim + 1),
    )
    assert result.actual_steps == config.chunk_length
    assert len(env.robot.sent_at) == config.chunk_length
    assert result.done and result.terminated
    assert result.reward_seq.tolist() == [0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize("outcome", ["s", "f"])
def test_human_mode_outcome_ends_episode_before_time_limit(outcome):
    config = _config()
    env = _threaded_env(config)
    env.enable_human_intervention = True
    env.teleop_speed_m_s = .01
    env._control_mode = "human"
    env.episode_time_s = 100.0
    env._monitor.poll = lambda: outcome
    env._monitor.direction = lambda: np.zeros(3)
    env._monitor.consume_toggle = lambda *, human: False
    env._retreat_after_outcome = lambda: None
    with patch("act_rlt.stage2_runtime.observation_frame", side_effect=lambda obs: obs):
        try:
            result = env.execute_policy_chunk(
                _FakeStage2Policy(config), {"observation.state": torch.zeros(1, 7)},
                warmup=True, remaining_steps=8,
            )
        finally:
            env.close()
    assert result.actual_steps == 4
    assert result.done and result.terminated and not result.truncated
    assert result.reward_seq.sum().item() == int(outcome == "s")
    assert not result.info.get("timed_out")


def test_outcome_during_final_control_period_ends_before_timeout():
    config = _config()
    env = _threaded_env(config, period=.01)
    env.enable_human_intervention = True
    env._control_mode = "policy"
    env.episode_time_s = 100.0
    polls = iter([None, None, None, None, "s"])
    env._monitor.poll = lambda: next(polls)
    env._monitor.consume_toggle = lambda *, human: False
    env._retreat_after_outcome = lambda: None
    with patch("act_rlt.stage2_runtime.observation_frame", side_effect=lambda obs: obs):
        try:
            result = env.execute_policy_chunk(
                _FakeStage2Policy(config), {"observation.state": torch.zeros(1, 7)},
                warmup=True, remaining_steps=8,
            )
        finally:
            env.close()
    assert result.actual_steps == 4
    assert result.done and result.terminated
    assert result.reward_seq.tolist() == [0.0, 0.0, 0.0, 1.0]
    assert not result.info.get("timed_out")


def test_late_boundary_prediction_is_logged_without_repeating_or_bursting_actions():
    from act_rlt.stage2_runtime import FixedChunkRuntime
    config = _config()
    env = _threaded_env(config, period=.02)
    policy = _FakeStage2Policy(config)
    encode = policy.encode_and_reference
    calls = 0

    def delayed_encode(batch):
        nonlocal calls
        calls += 1
        if calls == 2:
            time.sleep(.06)
        return encode(batch)

    policy.encode_and_reference = delayed_encode
    with patch("act_rlt.stage2_runtime.observation_frame", side_effect=lambda obs: obs):
        runtime = FixedChunkRuntime(
            env, policy, {"observation.state": torch.zeros(1, 7)},
            warmup=True, step_budget=8,
        )
        try:
            first, second = runtime.next_result(), runtime.next_result()
        finally:
            runtime.close()
    assert first.info["boundary_inference_ms"] >= 60
    assert second.info["max_command_interval_ms"] >= 60
    assert second.info["deadline_misses"] >= 1
    assert len(env.robot.sent_at) == 8
    assert min(np.diff(env.robot.sent_at)) > .01


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


@pytest.mark.parametrize("entrypoint", ["train", "infer"])
def test_z_reset_xy_randomization_cli_is_opt_in(entrypoint) -> None:
    if entrypoint == "train":
        parser = build_parser()
        validate = validate_args
        base = ["--stage1-checkpoint", "/tmp/stage1", "--act-checkpoint", "/tmp/act", "--dry-run"]
    else:
        from act_rlt.infer_stage2 import build_parser as infer_parser, validate_args as infer_validate

        parser = infer_parser()
        validate = infer_validate
        base = ["--checkpoint", "/tmp/stage2", "--dry-run"]
    default = parser.parse_args(base + ["--z-insertion-mode"])
    assert default.randomize_reset_xy is False
    validate(parser, default)
    enabled = parser.parse_args(base + ["--z-insertion-mode", "--randomize-reset-xy"])
    assert enabled.randomize_reset_xy is True
    validate(parser, enabled)
    with pytest.raises(SystemExit):
        validate(parser, parser.parse_args(base + ["--randomize-reset-xy"]))


@pytest.mark.parametrize("randomize_xy", [False, True])
def test_z_insertion_reset_uses_confirmed_p0_and_retreats_up(monkeypatch, randomize_xy) -> None:
    p0 = np.array([0.65, -0.10, 0.387, 0.1, 0.2, 0.3])
    lower = np.array([0.62, -0.13, 0.357])
    upper = np.array([0.68, -0.07, 0.417])
    env = object.__new__(FrankaInsertionStage2Env)
    env.z_insertion_mode = True
    env.randomize_reset_xy = randomize_xy
    env._reference_pose = None
    env._reset_orientation = None
    env._first_reset_move = True
    env.workspace_min = lower
    env.workspace_max = upper
    env.robot = SimpleNamespace(robot=SimpleNamespace(get_tool_pose=lambda: p0))
    moves = []
    env._move_reset_pose = lambda target, *, label: moves.append((target, label))
    answers = iter(["", "", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    env._move_to_accepted_workspace_sample()
    assert np.array_equal(env._reference_pose, p0)
    target, _ = moves[0]
    assert abs(target[0] - p0[0]) <= 0.01
    assert abs(target[1] - p0[1]) <= 0.01
    if not randomize_xy:
        np.testing.assert_array_equal(target[:2], p0[:2])
    assert target[2] == 0.402
    expected_rotation = Rotation.from_euler("XYZ", [179.7, 1.4, 90.6], degrees=True)
    np.testing.assert_allclose(Rotation.from_rotvec(target[3:]).as_matrix(), expected_rotation.as_matrix())
    for z in (0.38, 0.40):
        reference = p0.copy()
        reference[2] = z
        for _ in range(20):
            sampled = sample_z_insertion_workspace_pose(reference, lower, upper, randomize_xy=randomize_xy)
            assert abs(sampled[0] - reference[0]) <= 0.01
            assert abs(sampled[1] - reference[1]) <= 0.01
            assert sampled[2] == 0.402
            if randomize_xy:
                assert not np.array_equal(sampled[:2], reference[:2])
            else:
                np.testing.assert_array_equal(sampled[:2], reference[:2])
            np.testing.assert_allclose(
                Rotation.from_rotvec(sampled[3:]).as_matrix(), expected_rotation.as_matrix()
            )

    env._retreat_after_outcome()
    retreat, label = moves[-1]
    assert label == "+Z 1 cm retreat"
    assert np.isclose(retreat[2], p0[2] + RESET_RETREAT_Z_M)
    assert np.array_equal(retreat[[0, 1, 3, 4, 5]], p0[[0, 1, 3, 4, 5]])
    with pytest.raises(ValueError, match="outside the safe workspace"):
        sample_z_insertion_workspace_pose(p0, lower, np.array([0.66, -0.07, 0.401]))


def test_workspace_center_is_sampled_when_enter_is_pressed(monkeypatch) -> None:
    pose = np.array([0.65, -0.10, 0.15, 0.1, 0.2, 0.3])
    pressed = False

    def confirm(_prompt):
        nonlocal pressed
        pressed = True
        return ""

    def get_pose():
        assert pressed
        return pose

    monkeypatch.setattr("builtins.input", confirm)
    robot = SimpleNamespace(robot=SimpleNamespace(get_tool_pose=get_pose))
    captured, lower, upper = capture_centered_workspace(robot)
    assert np.array_equal(captured, pose)
    np.testing.assert_allclose(lower, pose[:3] - STARTUP_WORKSPACE_HALF_RANGE_M)
    np.testing.assert_allclose(upper, pose[:3] + STARTUP_WORKSPACE_HALF_RANGE_M)


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


def test_stage2_z_insertion_mode_captures_workspace_at_startup() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--stage1-checkpoint", "/tmp/stage1", "--act-checkpoint", "/tmp/act",
        "--allow-motion", "--z-insertion-mode",
    ])
    validate_args(parser, args)
    assert args.z_insertion_mode
    assert args.fps == 30.0
    assert args.workspace_min is None and args.workspace_max is None


@pytest.mark.parametrize(
    ("display", "extra_args", "expected"),
    [
        (":1", [], "pynput"),
        (None, [], "evdev"),
        (":1", ["--teleop-keyboard", "/dev/input/event3"], "evdev"),
        (":1", ["--teleop-input-backend", "evdev"], "evdev"),
    ],
)
def test_stage2_keyboard_backend_auto_selection(monkeypatch, display, extra_args, expected) -> None:
    if display is None:
        monkeypatch.delenv("DISPLAY", raising=False)
    else:
        monkeypatch.setenv("DISPLAY", display)
    parser = build_parser()
    args = parser.parse_args([
        "--stage1-checkpoint", "/tmp/stage1",
        "--act-checkpoint", "/tmp/act",
        "--dry-run",
        *extra_args,
    ])
    validate_args(parser, args)
    assert args.teleop_input_backend == expected


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


def test_old_stage2_weights_require_fresh_learner() -> None:
    config = _config()
    learner = Stage2Learner(_FakeStage2Policy(config), config)
    with TemporaryDirectory() as directory:
        output = Path(directory)
        save_checkpoint(output, config, learner, ReplayBuffer(config.replay_capacity), OnlineStage2Metrics())
        path = output / "checkpoints/latest.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        state["learner"]["learner_version"] = 2
        torch.save(state, path)
        with pytest.raises(ValueError, match="--replay-from"):
            load_checkpoint(path, config, learner, ReplayBuffer(config.replay_capacity))


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
