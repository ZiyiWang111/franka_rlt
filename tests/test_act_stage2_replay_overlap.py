from collections import deque
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest
import torch

from act_rlt.replay_overlap import EpisodeReplayOverlap, ReplayObservation
from act_rlt.stage2 import ACTStage2Config, LearnerStepMetrics, OnlineStage2Metrics, run_online_stage2
from act_rlt.stage2_runtime import FixedChunkRuntime
from act_rlt.train_stage2 import FrankaInsertionStage2Env, build_parser, load_checkpoint, save_checkpoint
from evo_rlt.core.interfaces import ChunkTransition
from evo_rlt.core.replay_buffer import ReplayBuffer


def snapshot(step):
    return ReplayObservation({"step": step}, step + 1, float(step), float(step))


def encode(observation):
    step = observation["step"]
    return torch.tensor([float(step)]), torch.full((4, 6), 1000. + step)


def transition(start, *, count=4, done=False, human=False, episode=0, success=True):
    actions = torch.arange(start, start + 4).float()[:, None].expand(4, 6).clone()
    rewards = torch.zeros(4)
    if done and success:
        rewards[count - 1] = 1
    return ChunkTransition(
        state_vec=torch.tensor([float(start)]), exec_chunk=actions,
        ref_chunk=torch.full((4, 6), 1000. + start), reward_seq=rewards,
        next_state_vec=torch.tensor([float(start + count)]),
        next_ref_chunk=torch.full((4, 6), 1000. + start + count),
        actual_steps=torch.tensor(count), done=torch.tensor(float(done)),
        intervention=torch.tensor(float(human)), source=torch.tensor(3 if human else 2),
        episode_id=torch.tensor(episode), terminated=torch.tensor(float(done)),
        bc_target_chunk=actions + 100 if human else None,
    )


def test_cross_chunk_windows_use_exact_actions_and_t_plus_c_states():
    builder = EpisodeReplayOverlap(4, 2)
    calls = []
    for start in (0, 4, 8):
        builder.add(transition(start, done=start == 8), {2: snapshot(start + 2)})
    extra, stats = builder.finish(lambda obs: (calls.append(obs["step"]) or encode(obs)),
                                  done=True, terminated=True, truncated=False)
    assert calls == [2, 6, 10]  # x6 is encoded once and reused as both state and next state.
    assert len(extra) == 2
    for item, start in zip(extra, (2, 6)):
        assert item.state_vec.item() == start
        assert item.next_state_vec.item() == start + 4
        torch.testing.assert_close(item.exec_chunk[:, 0], torch.arange(start, start + 4).float())
        assert item.ref_chunk[0, 0] == 1000 + start
        assert item.next_ref_chunk[0, 0] == 1004 + start
        assert int(item.actual_steps) == 4
        assert not item.done and not item.reward_seq.any()
    assert stats["replay_overlap_added"] == 2
    assert not builder.chunks and not builder.observations and not builder.anchors


@pytest.mark.parametrize("success,truncated", [(True, False), (False, False), (False, True)])
def test_short_original_tail_can_complete_one_full_terminal_overlap(success, truncated):
    builder = EpisodeReplayOverlap(4, 2)
    builder.add(transition(0), {2: snapshot(2)})
    tail = transition(4, count=2, done=True, success=success)
    tail.exec_chunk[2:] = 999  # Padding must never enter the overlapping action window.
    builder.add(tail, {})
    extra, _ = builder.finish(encode, done=True, terminated=not truncated, truncated=truncated)
    assert len(extra) == 1
    item = extra[0]
    torch.testing.assert_close(item.exec_chunk[:, 0], torch.tensor([2., 3., 4., 5.]))
    assert item.next_state_vec.item() == 6
    assert bool(item.done) and bool(item.truncated) == truncated
    assert bool(item.terminated) != truncated
    assert item.reward_seq.tolist() == [0., 0., 0., float(success)]


def test_missing_intermediate_state_skips_window_without_fabrication():
    builder = EpisodeReplayOverlap(4, 2)
    for start in (0, 4, 8):
        builder.add(transition(start), {} if start == 4 else {2: snapshot(start + 2)})
    extra, stats = builder.finish(lambda obs: pytest.fail("do not encode unusable windows"),
                                  done=False, terminated=False, truncated=False)
    assert extra == []
    assert stats["replay_overlap_skipped_missing"] == 2


def test_handoffs_are_not_crossed_and_human_bc_targets_are_preserved():
    builder = EpisodeReplayOverlap(4, 2)
    for start in (0, 4, 8):
        builder.add(transition(start, human=start > 0), {2: snapshot(start + 2)})
    extra, stats = builder.finish(encode, done=True, terminated=True, truncated=False)
    assert len(extra) == 1 and extra[0].state_vec.item() == 6
    assert stats["replay_overlap_skipped_source"] == 1
    assert bool(extra[0].intervention) and int(extra[0].source) == 3
    torch.testing.assert_close(extra[0].bc_target_chunk[:, 0], torch.arange(106., 110.))
    assert extra[0].ref_chunk[0, 0] == 1006  # Keep the frozen ACT reference separate.


def test_episode_isolation_and_cache_release_after_encoding_error():
    builder = EpisodeReplayOverlap(4, 2)
    builder.add(transition(0), {2: snapshot(2)})
    with pytest.raises(ValueError, match="mix episodes"):
        builder.add(transition(4, episode=1), {})
    builder.add(transition(4), {2: snapshot(6)})
    with pytest.raises(RuntimeError, match="encode failed"):
        builder.finish(lambda obs: (_ for _ in ()).throw(RuntimeError("encode failed")),
                       done=True, terminated=True, truncated=False)
    assert not builder.observations and not builder.chunks and builder.steps == 0


def test_replay_selection_is_nonblocking_and_does_not_consume_control_frames():
    runtime = object.__new__(FixedChunkRuntime)
    runtime.observation_ready = threading.Condition()
    runtime.last_observation_seq, runtime._last_replay_sequence = 3, 0
    runtime.observation_selection = "latest"
    observation = {"step": 2}
    runtime.observation_history = deque([(1., 2., 1., observation, {"observation_seq": 4})])
    item = runtime._replay_observation(after=3.)
    assert item.observation is observation and item.sequence == 4
    assert runtime.last_observation_seq == 3
    assert runtime._replay_observation(after=3.) is None
    runtime._last_replay_sequence = 0
    runtime.observation_selection = "boundary"
    assert runtime._replay_observation(after=3.) is None  # Never wait for a fresh replay frame.
    assert runtime._replay_observation(after=.5).sequence == 4


def test_camera_owns_replay_images_even_if_sensor_reuses_its_buffer():
    runtime = object.__new__(FixedChunkRuntime)
    runtime.replay_stride = 2
    runtime.stop = threading.Event()
    runtime.observation_ready = threading.Condition()
    runtime.observation_seq = 0
    runtime.observation_history = deque(maxlen=8)
    shared = np.zeros((2, 2, 3), dtype=np.uint8)
    reads = []

    def read():
        reads.append(True)
        shared.fill(len(reads))
        if len(reads) == 3:
            runtime.stop.set()
        return {"wrist": shared}

    runtime.env = SimpleNamespace(config=SimpleNamespace(chunk_length=4),
                                  robot=SimpleNamespace(get_observation=read))
    runtime._camera()
    assert runtime.observation_history[0][3]["wrist"].tolist() == np.ones_like(shared).tolist()
    assert not np.shares_memory(runtime.observation_history[0][3]["wrist"], shared)


class Actor(torch.nn.Module):
    def forward(self, state, reference, training=False):
        return torch.zeros_like(reference), torch.ones_like(reference)

    def sample(self, state, reference, training=False):
        mean, _ = self.forward(state, reference, training)
        return mean + .25, mean


class Learner:
    def __init__(self, initialized=True):
        self.initialized = initialized
        self.critic_updates, self.actor_updates = 0, 0
        self.initial_replay = None

    def initialize(self, replay):
        self.initial_replay = list(replay.buffer)
        self.initialized = True

    def update_once(self, replay):
        self.critic_updates += 1
        return LearnerStepMetrics(0., None, 0., None, self.critic_updates, 0)


def training_env(monkeypatch, config, *, episode_steps=None, refuse_at=None):
    env = object.__new__(FrankaInsertionStage2Env)
    env.config = config
    env._runtime, env._monitor = None, None
    env.enable_human_intervention = False
    env._control_mode = "policy"
    env._episode_sampled_tcp_command_sum = np.zeros(6)
    env._episode_reference_tcp_command_sum = np.zeros(6)
    env._episode_mean_tcp_command_sum = np.zeros(6)
    env.sent, env.encodes, env.stops = [], [], []
    env.pre = lambda observation: observation
    env.post = lambda action: action * 1e-6
    env._normalize_executed_action = lambda action: action[:6] / 1e-6
    env._observation_frame = lambda observation: observation
    env._processed_observation = lambda: {"step": len(env.sent)}
    env._move_to_accepted_workspace_sample = lambda: None
    env._retreat_after_outcome = lambda: None
    env.period_s, env.episode_time_s = .004, 100.
    env.max_step_m, env.max_step_rad = .002, .02
    env.observation_selection = "boundary"
    monitor_start = [0]

    def monitor():
        monitor_start[0] = len(env.sent)
        return SimpleNamespace(poll=lambda: "s" if episode_steps is not None
                               and len(env.sent) - monitor_start[0] >= episode_steps else None,
                               close=lambda: None)

    def send(command):
        if refuse_at is not None and len(env.sent) == refuse_at:
            raise RuntimeError("refusing TCP target outside workspace: test")
        env.sent.append(command.copy())

    env.robot = SimpleNamespace(
        get_observation=lambda: (time.sleep(.001) or {"step": len(env.sent)}),
        resync_command_pose=lambda: None, send_action=send,
        robot=SimpleNamespace(stop_servo=lambda: (env.stops.append(len(env.sent)) or True)),
    )
    monkeypatch.setattr("act_rlt.train_stage2.OutcomeMonitor", monitor)

    def mid_observation(runtime, after):
        return snapshot(len(env.sent))

    monkeypatch.setattr(FixedChunkRuntime, "_replay_observation", mid_observation)

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = Actor()

        def encode_and_reference(self, batch):
            step = batch["step"]
            env.encodes.append((threading.current_thread().name, step, len(env.sent)))
            return (torch.full((1, config.state_dim), float(step)),
                    torch.full((1, 4, 6), 1000. + step), torch.zeros(1, 16, 6))

    return env, Policy()


@pytest.mark.parametrize("stride,extra", [(4, 0), (2, 2)])
def test_actual_training_flushes_after_stop_without_extra_steps_or_updates(monkeypatch, stride, extra):
    config = ACTStage2Config(stage1_checkpoint="/tmp/stage1", act_checkpoint="/tmp/act",
                            device="cpu", batch_size=1, warmup_steps=1, total_env_steps=12,
                            replay_stride=stride, utd_ratio=3)
    env, policy = training_env(monkeypatch, config)
    learner, replay, records = Learner(), ReplayBuffer(), []
    try:
        metrics = run_online_stage2(policy, learner, env, replay, config,
                                   initial_metrics=OnlineStage2Metrics(env_steps=1, warmup_env_steps=1),
                                   on_chunk=lambda metrics, result: records.append(result.info.copy()))
    finally:
        env.close()
    assert metrics.online_env_steps == len(env.sent) == 12
    assert metrics.chunks == 3 and learner.critic_updates == 9
    assert len(replay) == 3 + extra and metrics.replay_overlap_chunks == extra
    assert [item.start_step for item in replay.buffer] == [0, 4, 8] + ([2, 6] if stride == 2 else [])
    td = replay.sample(256, td_chunks=3)
    assert td["td_steps"].tolist() == [12]
    assert td["td_next_state_vec"][0, 0] == 12
    replay_encodes = [(step, sent) for thread, step, sent in env.encodes if thread == "MainThread"]
    assert replay_encodes == ([(2, 12), (6, 12), (10, 12)] if stride == 2 else [])
    if stride == 2:
        assert env.stops and env.stops[-1] == 12
        assert records[-1]["replay_overlap_added"] == 2
        for item, start in zip(list(replay.buffer)[3:], (2, 6)):
            assert item.state_vec[0] == start and item.next_state_vec[0] == start + 4
            assert item.ref_chunk[0, 0] == 1000 + start
            torch.testing.assert_close(item.exec_chunk, torch.full((4, 6), .25))
            assert not item.done  # Online step budget is not a fabricated episode terminal.


def test_warmup_finishes_episode_and_augments_before_initialization(monkeypatch):
    config = ACTStage2Config(stage1_checkpoint="/tmp/stage1", act_checkpoint="/tmp/act",
                            device="cpu", batch_size=1, warmup_steps=4, total_env_steps=8,
                            replay_stride=2, utd_ratio=2)
    env, policy = training_env(monkeypatch, config, episode_steps=8)
    learner, replay = Learner(initialized=False), ReplayBuffer()
    try:
        metrics = run_online_stage2(policy, learner, env, replay, config)
    finally:
        env.close()
    assert metrics.warmup_env_steps == metrics.online_env_steps == 8
    assert metrics.env_steps == len(env.sent) == 16
    assert learner.initial_replay is not None and len(learner.initial_replay) == 3
    assert all(int(item.source) == 1 for item in learner.initial_replay)
    assert len(replay) == 6 and metrics.replay_overlap_chunks == 2
    assert metrics.episodes == 2 and metrics.successes == 2
    assert learner.critic_updates == 4  # Warmup and added windows consume no UTD budget.
    assert {int(item.episode_id) for item in replay.buffer} == {0, 1}


def test_real_camera_cache_supplies_intermediate_observations(monkeypatch):
    config = ACTStage2Config(stage1_checkpoint="/tmp/stage1", act_checkpoint="/tmp/act",
                            device="cpu", batch_size=1, warmup_steps=1, total_env_steps=8,
                            replay_stride=2)
    select_replay = FixedChunkRuntime._replay_observation
    env, policy = training_env(monkeypatch, config)
    monkeypatch.setattr(FixedChunkRuntime, "_replay_observation", select_replay)
    env.observation_selection = "latest"
    env.period_s = .01
    original_send = env.robot.send_action

    def send(command):
        original_send(command)
        # The simulated robot acknowledges only once its camera has published
        # this step, making the latest-mode integration test deterministic.
        runtime = env._runtime
        if runtime is not None:
            with runtime.observation_ready:
                assert runtime.observation_ready.wait_for(
                    lambda: runtime.observation_history
                    and runtime.observation_history[-1][3]["step"] == len(env.sent), timeout=1,
                )

    env.robot.send_action = send
    replay, records = ReplayBuffer(), []
    try:
        run_online_stage2(policy, Learner(), env, replay, config,
                          initial_metrics=OnlineStage2Metrics(warmup_env_steps=1),
                          on_chunk=lambda metrics, result: records.append(result.info.copy()))
    finally:
        env.close()
    assert [record["replay_observations_captured"] for record in records] == [1, 1]
    assert all(record["replay_observation_offsets"] == [2] for record in records)
    assert len(replay) == 3
    assert replay.buffer[-1].state_vec[0] == 2
    assert replay.buffer[-1].next_state_vec[0] == 6


def test_frozen_inference_never_caches_overlap_even_with_training_stride(monkeypatch):
    config = ACTStage2Config(stage1_checkpoint="/tmp/stage1", act_checkpoint="/tmp/act", replay_stride=2)
    env, policy = training_env(monkeypatch, config)
    env._episode_start = time.monotonic()
    env._monitor = SimpleNamespace(poll=lambda: None)
    runtime = FixedChunkRuntime(env, policy, {"step": 0}, warmup=False, step_budget=4, deterministic=True)
    try:
        result = runtime.next_result()
    finally:
        runtime.close()
    assert runtime.replay_stride == 4
    assert not result.replay_observations
    assert result.info["replay_observations_captured"] == 0


@pytest.mark.parametrize("refuse_at", [6, 8])
def test_workspace_end_flushes_without_counting_unsent_actions(monkeypatch, refuse_at):
    config = ACTStage2Config(stage1_checkpoint="/tmp/stage1", act_checkpoint="/tmp/act",
                            device="cpu", batch_size=1, warmup_steps=1, total_env_steps=12,
                            replay_stride=2)
    env, policy = training_env(monkeypatch, config, refuse_at=refuse_at)
    replay, records = ReplayBuffer(), []

    def on_chunk(metrics, result):
        records.append(result)
        if result.done:
            result.stop_requested = True  # End the test after the first workspace refusal.

    try:
        metrics = run_online_stage2(policy, Learner(), env, replay, config, on_chunk=on_chunk,
                                   initial_metrics=OnlineStage2Metrics(env_steps=1, warmup_env_steps=1))
    finally:
        env.close()
    assert metrics.online_env_steps == len(env.sent) == refuse_at
    assert len(replay) == 3 and metrics.chunks == 2 and metrics.replay_overlap_chunks == 1
    assert records[-1].actual_steps == (2 if refuse_at == 6 else 0)
    assert records[-1].info["workspace_violation"]
    extra = replay.buffer[-1]
    assert extra.next_state_vec[0] == 6
    assert bool(extra.truncated) == (refuse_at == 6)
    assert not extra.reward_seq.any()


def test_timeout_success_is_labelled_before_terminal_overlap_is_built(monkeypatch):
    config = ACTStage2Config(stage1_checkpoint="/tmp/stage1", act_checkpoint="/tmp/act",
                            device="cpu", batch_size=1, warmup_steps=1, total_env_steps=12,
                            replay_stride=2)
    env, policy = training_env(monkeypatch, config)
    original_send = env.robot.send_action

    def send(command):
        original_send(command)
        if len(env.sent) == 6:
            env._episode_start = time.monotonic() - 1000

    env.robot.send_action = send
    env._prompt_outcome_after_timeout = lambda: "s"
    replay = ReplayBuffer()
    try:
        metrics = run_online_stage2(
            policy, Learner(), env, replay, config,
            initial_metrics=OnlineStage2Metrics(env_steps=1, warmup_env_steps=1),
            on_chunk=lambda metrics, result: setattr(result, "stop_requested", result.done),
        )
    finally:
        env.close()
    assert metrics.online_env_steps == 6 and metrics.successes == 1
    extra = replay.buffer[-1]
    assert bool(extra.done) and bool(extra.terminated) and not extra.truncated
    assert extra.reward_seq.tolist() == [0., 0., 0., 1.]


@pytest.mark.parametrize("saved_stride,new_stride", [(None, 2), (2, 4), (4, 2)])
def test_checkpoint_stride_override_preserves_replay_and_counters(tmp_path, monkeypatch, saved_stride, new_stride):
    class SavedLearner:
        def state_dict(self):
            return {"learner_version": 3, "sentinel": 123}

        def load_state_dict(self, value):
            assert value["sentinel"] == 123

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    values = dict(stage1_checkpoint="/tmp/stage1", act_checkpoint="/tmp/act", device="cpu")
    saved_config = ACTStage2Config(**values, replay_stride=saved_stride)
    original = ReplayBuffer()
    original.add(transition(0))
    save_checkpoint(tmp_path, saved_config, SavedLearner(), original,
                    OnlineStage2Metrics(env_steps=4, warmup_env_steps=4, replay_overlap_chunks=7))
    path = tmp_path / "checkpoints/latest.pt"
    if saved_stride is None:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        del saved["config"]["replay_stride"]
        del saved["metrics"]["replay_overlap_chunks"]
        torch.save(saved, path)
    replay = ReplayBuffer()
    metrics = load_checkpoint(path, ACTStage2Config(**values, replay_stride=new_stride), SavedLearner(), replay)
    assert len(replay) == 1
    torch.testing.assert_close(replay.buffer[0].exec_chunk, original.buffer[0].exec_chunk)
    assert metrics.env_steps == metrics.warmup_env_steps == 4
    assert metrics.replay_overlap_chunks == (0 if saved_stride is None else 7)


def test_cli_and_config_validate_stride_and_preserve_legacy_default():
    parser = build_parser()
    base = ["--stage1-checkpoint", "/tmp/stage1", "--act-checkpoint", "/tmp/act"]
    assert parser.parse_args(base).replay_stride == 4
    assert parser.parse_args(base + ["--replay-stride", "2"]).replay_stride == 2
    with pytest.raises(SystemExit):
        parser.parse_args(base + ["--replay-stride", "3"])
    assert ACTStage2Config(stage1_checkpoint="a", act_checkpoint="b", chunk_length=3).replay_stride == 3
    with pytest.raises(ValueError, match="replay_stride"):
        ACTStage2Config(stage1_checkpoint="a", act_checkpoint="b", replay_stride=3)


@pytest.mark.parametrize('stride', [2, 4])
@pytest.mark.parametrize('first_outcome', ['s', 'f', 'd'])
def test_episode_replay_decision_never_trains_or_commits_pending_data(monkeypatch, stride, first_outcome):
    config = ACTStage2Config(stage1_checkpoint='/tmp/stage1', act_checkpoint='/tmp/act',
                            device='cpu', batch_size=1, warmup_steps=1, total_env_steps=8,
                            replay_stride=stride, utd_ratio=3)
    env, policy = training_env(monkeypatch, config)
    env.review_episode_replay = True
    decisions = iter([first_outcome, 's', 's'])

    def monitor():
        start, outcome = len(env.sent), next(decisions)
        emitted = [False]
        def poll():
            if not emitted[0] and len(env.sent) - start >= 8:
                emitted[0] = True
                return outcome
        return SimpleNamespace(poll=poll, close=lambda: None)

    monkeypatch.setattr('act_rlt.train_stage2.OutcomeMonitor', monitor)
    learner, replay, records = Learner(), ReplayBuffer(), []
    def record(metrics, result):
        records.append((result.done, result.info.copy(), len(replay), learner.critic_updates))
        if not result.done:
            assert len(replay) == 0 and learner.critic_updates == 0
    try:
        metrics = run_online_stage2(policy, learner, env, replay, config, on_chunk=record,
                                   initial_metrics=OnlineStage2Metrics(env_steps=1, warmup_env_steps=1))
    finally:
        env.close()
    deleted = first_outcome == 'd'
    assert metrics.online_env_steps == 8
    assert len(env.sent) == (16 if deleted else 8)
    assert learner.critic_updates == 6
    assert len(replay) == (3 if stride == 2 else 2)
    assert {int(t.episode_id) for t in replay.buffer} == {1 if deleted else 0}
    if deleted:
        first_terminal = next(row for row in records if row[0])
        assert first_terminal[1]['episode_replay_discarded'] is True
        assert first_terminal[2:] == (0, 0)
    else:
        assert metrics.successes == int(first_outcome == 's')


def test_timeout_can_delete_episode_without_second_confirmation(monkeypatch):
    env = object.__new__(FrankaInsertionStage2Env)
    env.review_episode_replay = True
    replies = iter(['d'])
    monkeypatch.setattr('builtins.input', lambda prompt: next(replies))
    assert env._prompt_outcome_after_timeout() == 'd'


def test_deleted_warmup_does_not_advance_initialization(monkeypatch):
    config = ACTStage2Config(stage1_checkpoint='/tmp/stage1', act_checkpoint='/tmp/act',
                            device='cpu', batch_size=1, warmup_steps=8, total_env_steps=8,
                            replay_stride=2, utd_ratio=2)
    env, policy = training_env(monkeypatch, config)
    env.review_episode_replay = True
    decisions = iter(['d', 's', 's', 's'])
    def monitor():
        start, outcome = len(env.sent), next(decisions)
        emitted = [False]
        def poll():
            if not emitted[0] and len(env.sent) - start >= 8:
                emitted[0] = True
                return outcome
        return SimpleNamespace(poll=poll, close=lambda: None)
    monkeypatch.setattr('act_rlt.train_stage2.OutcomeMonitor', monitor)
    learner, replay = Learner(initialized=False), ReplayBuffer()
    try:
        metrics = run_online_stage2(policy, learner, env, replay, config)
    finally:
        env.close()
    assert metrics.warmup_env_steps == metrics.online_env_steps == 8
    assert len(env.sent) == metrics.env_steps == 24
    assert {int(t.episode_id) for t in learner.initial_replay} == {1}
    assert {int(t.episode_id) for t in replay.buffer} == {1, 2}
    assert learner.critic_updates == 4
