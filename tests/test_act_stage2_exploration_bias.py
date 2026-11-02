from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest
import torch

from act_rlt.exploration_bias import (
    BIAS_STEP_M, BIAS_STEP_YAW_RAD, BIAS_KEYS, MAX_BIAS_LEVEL,
    EpisodeDownwardBias, EpisodeExplorationBias,
)
from act_rlt.stage2 import ACTStage2Config, OnlineStage2Metrics, run_online_stage2
from act_rlt.stage2_runtime import FixedChunkRuntime
from act_rlt.train_stage2 import FrankaInsertionStage2Env, OutcomeMonitor
from evo_rlt.core.replay_buffer import ReplayBuffer


@pytest.mark.parametrize("key,axis,sign", [(k, *v) for k, v in BIAS_KEYS.items()])
def test_levels_cap_combine_cancel_and_clear(key, axis, sign):
    bias = EpisodeExplorationBias()
    bias.prepare(warmup=False)
    for _ in range(MAX_BIAS_LEVEL + 2):
        bias.adjust(key)
    assert bias.pending[axis] == sign * MAX_BIAS_LEVEL
    other = "KEY_LEFT" if axis == 0 else "KEY_UP"
    other_axis = BIAS_KEYS[other][0]
    bias.adjust(other)
    assert bias.pending[other_axis] == -1
    opposite = next(k for k, v in BIAS_KEYS.items() if v == (axis, -sign))
    bias.adjust(opposite)
    assert bias.pending[axis] == sign * (MAX_BIAS_LEVEL - 1)
    bias.begin()
    expected = np.zeros(3)
    expected[axis], expected[other_axis] = sign * (MAX_BIAS_LEVEL - 1), -1
    expected *= [BIAS_STEP_M, BIAS_STEP_M, BIAS_STEP_YAW_RAD]
    np.testing.assert_allclose(bias.step_m, expected[:2])
    assert bias.step_yaw_rad == pytest.approx(expected[2])
    assert not bias.pending.any()
    result = SimpleNamespace(done=True, info={
        "exploration_bias_requested_xy_m": (expected[:2] * 160).tolist(),
        "exploration_bias_applied_xy_m": (expected[:2] * 150).tolist(),
        "exploration_bias_requested_yaw_rad": expected[2] * 160,
        "exploration_bias_applied_yaw_rad": expected[2] * 150,
    })
    bias.record(result)
    np.testing.assert_allclose(result.info["episode_exploration_bias_requested_xy_m"], expected[:2] * 160)
    np.testing.assert_allclose(result.info["episode_exploration_bias_applied_xy_m"], expected[:2] * 150)
    assert result.info["episode_exploration_bias_requested_yaw_rad"] == pytest.approx(expected[2] * 160)
    assert result.info["episode_exploration_bias_applied_yaw_rad"] == pytest.approx(expected[2] * 150)
    assert not bias.active.any()
    bias.prepare(warmup=False)
    bias.adjust(key)
    bias.adjust("KEY_0")
    assert not bias.pending.any()


def test_warmup_cannot_queue_bias_for_later():
    bias = EpisodeExplorationBias()
    bias.prepare(warmup=True)
    bias.adjust("KEY_UP")
    bias.adjust("KEY_M")
    bias.begin()
    assert not bias.step_m.any()
    assert bias.step_yaw_rad == 0
    bias.prepare(warmup=False)
    bias.begin()
    assert not bias.step_m.any()


class Actor(torch.nn.Module):
    def __init__(self, x=.25, yaw=0.):
        super().__init__()
        self.x = x
        self.yaw = yaw

    def forward(self, state, reference, training=False):
        return torch.zeros_like(reference), torch.ones_like(reference)

    def sample(self, state, reference, training=False):
        mean, _ = self.forward(state, reference, training)
        sampled = mean.reshape(-1, 4, 6).clone()
        sampled[:, :, 0] = self.x
        sampled[:, :, 5] = self.yaw
        return sampled.flatten(1), mean


def make_env(*, x=.25, yaw=0., yaw_level=0):
    bias = EpisodeExplorationBias()
    bias.prepare(warmup=False)
    bias.adjust("KEY_DOWN")
    bias.adjust("KEY_RIGHT")
    for _ in range(abs(yaw_level)):
        bias.adjust("KEY_M" if yaw_level > 0 else "KEY_N")
    bias.begin()
    env = object.__new__(FrankaInsertionStage2Env)
    env.config = SimpleNamespace(chunk_length=4, action_dim=6, proprio_dim=0)
    env._exploration_bias = bias
    env.post = lambda action: action * .001
    env._normalize_executed_action = lambda action: action[:6] / .001
    env.max_step_m, env.max_step_rad = .002, .02
    env.period_s, env.episode_time_s = .002, 10.
    env._episode_start = time.monotonic()
    env.pre = lambda observation: observation
    env._observation_frame = lambda observation: observation
    env.observation_selection = "latest"
    env._control_mode = "policy"
    env._monitor = SimpleNamespace(poll=lambda: None)
    env.sent = []
    env.robot = SimpleNamespace(
        get_observation=lambda: (time.sleep(.001) or {"state": torch.zeros(1, 1)}),
        resync_command_pose=lambda: None,
        send_action=lambda command: env.sent.append(command),
        robot=SimpleNamespace(stop_servo=lambda: True),
    )
    policy = SimpleNamespace(actor=Actor(x, yaw), encode_and_reference=lambda batch: (
        torch.zeros(1, 1), torch.zeros(1, 4, 6), torch.zeros(1, 16, 6),
    ))
    return env, policy


@pytest.mark.parametrize("mode", ["actor", "warmup", "human", "deterministic"])
@pytest.mark.parametrize("yaw_level", [-5, 5])
def test_physical_bias_after_denormalization_only_for_actor(mode, yaw_level):
    env, policy = make_env(yaw_level=yaw_level)
    runtime = object.__new__(FixedChunkRuntime)
    runtime.env, runtime.actor = env, policy.actor
    runtime.warmup, runtime.deterministic = mode == "warmup", mode == "deterministic"
    runtime.actor_lock, runtime.actor_state = threading.Lock(), None
    prediction = runtime._prediction(*policy.encode_and_reference({}), human=mode == "human")
    expected = torch.zeros(4, 6)
    if mode == "actor":
        expected[:, 0] = .25
        expected[:, :2] += BIAS_STEP_M / .001
        expected[:, 5] = yaw_level * BIAS_STEP_YAW_RAD / .001
    torch.testing.assert_close(prediction.executed, expected)
    torch.testing.assert_close(prediction.reference, torch.zeros(1, 4, 6))
    assert all(command["dx"] == 0 for command in prediction.mean_commands)
    if mode != "actor":
        assert not prediction.bias_delta.any()


@pytest.mark.parametrize("refuse_after", [None, 2, 0])
@pytest.mark.parametrize("yaw_level", [-5, 0, 5])
def test_runtime_replay_actions_and_bias_totals_count_only_executed_steps(monkeypatch, refuse_after, yaw_level):
    env, policy = make_env(x=1.999, yaw=19.999, yaw_level=yaw_level)
    monkeypatch.setattr("act_rlt.stage2_runtime.observation_frame", lambda obs: obs)
    original_send = env.robot.send_action

    def send(command):
        if refuse_after is not None and len(env.sent) == refuse_after:
            raise RuntimeError("refusing TCP target outside workspace: test")
        original_send(command)

    env.robot.send_action = send
    runtime = FixedChunkRuntime(env, policy, {}, warmup=False, step_budget=4)
    try:
        result = runtime.next_result()
    finally:
        runtime.close()
    count = 4 if refuse_after is None else refuse_after
    assert result.actual_steps == count
    if count:
        commands = torch.tensor([list(c.values()) for c in env.sent], dtype=torch.float32)
        torch.testing.assert_close(result.exec_chunk[:count], commands / .001)
        assert float(commands[:, :3].norm(dim=-1).max()) <= .00200001
        baseline = torch.tensor([.001999, 0.])
        effective = (commands[:, :2] - baseline).sum(0)
    else:
        effective = torch.zeros(2)
    np.testing.assert_allclose(result.info["exploration_bias_requested_xy_m"],
                               np.full(2, BIAS_STEP_M * count), atol=1e-10)
    np.testing.assert_allclose(result.info["exploration_bias_applied_xy_m"], effective, atol=1e-9)
    assert result.info["exploration_bias_requested_yaw_rad"] == pytest.approx(
        yaw_level * BIAS_STEP_YAW_RAD * count, abs=1e-9,
    )
    effective_yaw = sum(command["drz"] - .019999 for command in env.sent)
    assert result.info["exploration_bias_applied_yaw_rad"] == pytest.approx(effective_yaw, abs=1e-8)
    assert all(abs(command["drz"]) <= .02000001 for command in env.sent)
    assert not result.exec_chunk[count:].any()


def test_idle_bias_keys_stay_in_menu_and_do_not_move_tcp(monkeypatch):
    from act_rlt.idle_xy import read_idle_xy_choice
    env, _ = make_env()
    env._exploration_bias.prepare(warmup=False)
    env.teleop_input_backend, env.teleop_keyboard = "pynput", None
    env.workspace_min, env.workspace_max = np.full(3, -1.), np.full(3, 1.)
    env.robot.robot.get_tool_pose = lambda: np.zeros(6)
    env.robot.robot.servo_tool = lambda pose: pytest.fail("arrow keys must not move TCP")
    choices = iter(["KEY_UP", "KEY_LEFT", "KEY_N", "r", "KEY_DOWN", "KEY_0", "KEY_RIGHT", "KEY_M", ""])
    monitor = SimpleNamespace(poll=lambda: next(choices), pressed_keys=lambda: set(), close=lambda: None)
    monkeypatch.setattr("act_rlt.idle_xy.HumanInputMonitor", lambda *a, **kw: monitor)
    monkeypatch.setattr("act_rlt.idle_xy.time.sleep", lambda duration: None)
    assert read_idle_xy_choice(env, "Ready?") == "r"
    assert env._exploration_bias.pending.tolist() == [-1, -1, -1]
    assert read_idle_xy_choice(env, "Ready again?") == ""
    assert env._exploration_bias.pending.tolist() == [0, 1, 1]


def test_human_execution_has_no_bias_and_records_human_commands():
    env, policy = make_env(yaw_level=5)
    env._control_mode = "human"
    env.teleop_speed_m_s = .01
    env._monitor.direction = lambda: np.array([-1., 0., 0.])
    runtime = FixedChunkRuntime(env, policy, {}, warmup=False, step_budget=4)
    try:
        result = runtime.next_result()
    finally:
        runtime.close()
    assert result.intervention
    expected = torch.zeros(4, 6)
    expected[:, 0] = -.01 * env.period_s / .001
    torch.testing.assert_close(result.exec_chunk, expected)
    torch.testing.assert_close(result.bc_target_chunk, expected)
    assert result.info["exploration_bias_requested_xy_m"] == [0., 0.]
    assert result.info["exploration_bias_applied_xy_m"] == [0., 0.]
    assert result.info["exploration_bias_requested_yaw_rad"] == 0.
    assert result.info["exploration_bias_applied_yaw_rad"] == 0.


def test_legacy_execution_applies_yaw_and_records_the_sent_action():
    env, _ = make_env(yaw_level=-5)
    env._processed_observation = lambda: {}
    result = env.execute_chunk(torch.zeros(1, 4, 6), torch.zeros(1, 4, 6))
    assert result.actual_steps == 4
    sent = torch.tensor([list(command.values()) for command in env.sent], dtype=torch.float32)
    torch.testing.assert_close(result.exec_chunk, sent / .001)
    torch.testing.assert_close(sent[:, 5], torch.full((4,), -5 * BIAS_STEP_YAW_RAD))
    assert result.info["exploration_bias_requested_yaw_rad"] == pytest.approx(-20 * BIAS_STEP_YAW_RAD)
    assert result.info["exploration_bias_applied_yaw_rad"] == pytest.approx(-20 * BIAS_STEP_YAW_RAD)


def test_training_reset_locks_bias_and_replay_saves_biased_commands(monkeypatch):
    env, policy = make_env()
    env._downward_bias = EpisodeDownwardBias()
    env.config = ACTStage2Config(stage1_checkpoint="/tmp/stage1", act_checkpoint="/tmp/act",
                                device="cpu", proprio_dim=0, warmup_steps=1, total_env_steps=4)
    env._runtime, env._monitor = None, None
    env.enable_human_intervention = False
    env._processed_observation = lambda: {"state": torch.zeros(1, 1)}
    env._episode_sampled_tcp_command_sum = np.zeros(6)
    env._episode_reference_tcp_command_sum = np.zeros(6)
    env._episode_mean_tcp_command_sum = np.zeros(6)
    selections = iter([True, False])

    def select_bias():
        if next(selections):
            env._exploration_bias.adjust("KEY_UP")
            env._exploration_bias.adjust("KEY_N")

    env._move_to_accepted_workspace_sample = select_bias
    monkeypatch.setattr("act_rlt.train_stage2.OutcomeMonitor", lambda: SimpleNamespace(
        poll=lambda: None, close=lambda: None,
        consume_downward_bias=lambda: len(env.sent) >= 1,
    ))
    policy.eval = lambda: policy
    replay = ReplayBuffer()
    try:
        metrics = run_online_stage2(policy, SimpleNamespace(initialized=True), env, replay, env.config,
                                    initial_metrics=OnlineStage2Metrics(warmup_env_steps=1))
        assert metrics.online_env_steps == 4
        assert len(replay) == 1
        stored = replay.buffer[0]
        assert int(stored.actual_steps) == 4
        sent = torch.tensor([list(c.values()) for c in env.sent], dtype=torch.float32)
        torch.testing.assert_close(stored.exec_chunk, sent / .001)
        torch.testing.assert_close(stored.exec_chunk[:, 0], torch.full((4,), .25 - BIAS_STEP_M / .001))
        torch.testing.assert_close(stored.exec_chunk[:, 5], torch.full((4,), -BIAS_STEP_YAW_RAD / .001))
        torch.testing.assert_close(stored.exec_chunk[:, 2], torch.tensor([0., -.15, -.15, -.15]))
        assert not env._exploration_bias.cancelled
        torch.testing.assert_close(stored.ref_chunk, torch.zeros(4, 6))
        assert env._exploration_bias.active.tolist() == [-1, 0, -1]
        # A fresh physical episode does not inherit the previous selection.
        env.reset(episode_id=1, warmup=False)
        assert not env._exploration_bias.active.any()
        assert not env._exploration_bias.cancelled
        assert not env._downward_bias.enabled
    finally:
        env.close()


def test_downward_latch_is_idempotent_and_clears_at_episode_end_and_reset():
    bias = EpisodeDownwardBias()
    assert bias.step_m == 0
    bias.enable()
    bias.enable()
    assert bias.step_m == -.00015
    result = SimpleNamespace(done=False, info={
        "downward_bias_requested_z_m": -.0002,
        "downward_bias_applied_xyz_m": [.00001, 0., -.0001],
        "downward_bias_steps": 2,
    })
    bias.record(result)
    assert bias.enabled
    result.done = True
    bias.record(result)
    assert result.info["episode_downward_bias_steps"] == 4
    assert result.info["episode_downward_bias_requested_z_m"] == pytest.approx(-.0004)
    np.testing.assert_allclose(result.info["episode_downward_bias_applied_xyz_m"], [.00002, 0., -.0002])
    assert bias.step_m == 0
    bias.enable()
    bias.prepare()
    assert bias.step_m == 0
    assert bias.steps == 0 and bias.requested_z == 0 and not bias.applied_xyz.any()


@pytest.mark.parametrize("mode", ["actor", "warmup", "human", "deterministic"])
@pytest.mark.parametrize("full_dim", [6, 7])
def test_downward_bias_starts_mid_chunk_and_persists_across_chunks(mode, full_dim, capsys):
    env, policy = make_env(yaw_level=2)
    if full_dim == 7:
        encode = policy.encode_and_reference

        def with_gripper(batch):
            state, reference, full = encode(batch)
            return state, reference, torch.cat([full, torch.full((1, 16, 1), .3)], dim=-1)

        policy.encode_and_reference = with_gripper
    normalize = env._normalize_executed_action

    def checked_normalize(physical):
        if mode != "human":
            assert threading.current_thread().name != "stage2-servo"
        assert physical.numel() == full_dim
        if full_dim == 7:
            assert float(physical[6]) == pytest.approx(.0003)
        return normalize(physical)

    env._normalize_executed_action = checked_normalize
    env._downward_bias = EpisodeDownwardBias()
    env._monitor.consume_downward_bias = lambda: len(env.sent) >= 2
    if mode == "human":
        env._control_mode = "human"
        env.teleop_speed_m_s = .01
        env._monitor.direction = lambda: np.array([1., 0., 0.])
    runtime = FixedChunkRuntime(env, policy, {}, warmup=mode == "warmup", step_budget=8,
                                deterministic=mode == "deterministic")
    try:
        results = [runtime.next_result(), runtime.next_result()]
    finally:
        runtime.close()
    sent = torch.tensor([list(command.values()) for command in env.sent], dtype=torch.float32)
    torch.testing.assert_close(torch.cat([result.exec_chunk for result in results]), sent / .001)
    expected = torch.tensor([0., 0., -.00015, -.00015, -.00015, -.00015, -.00015, -.00015])
    if mode in {"human", "deterministic"}:
        expected.zero_()
    torch.testing.assert_close(sent[:, 2], expected)
    assert sum(result.info["downward_bias_steps"] for result in results) == (6 if mode in {"warmup", "actor"} else 0)
    assert sum(result.info["downward_bias_requested_z_m"] for result in results) == pytest.approx(float(expected.sum()))
    output = capsys.readouterr().out
    assert output.count("Downward bias enabled:") == (0 if mode == "deterministic" else 1)
    assert "XY/yaw directional bias cancelled" not in output
    if mode == "actor":
        torch.testing.assert_close(sent[:, 0], torch.full((8,), .00025 + BIAS_STEP_M))
        torch.testing.assert_close(sent[:, 1], torch.full((8,), BIAS_STEP_M))
        torch.testing.assert_close(sent[:, 5], torch.full((8,), 2 * BIAS_STEP_YAW_RAD))
        for result in results:
            np.testing.assert_allclose(result.info["exploration_bias_requested_xy_m"], np.full(2, 4 * BIAS_STEP_M))


@pytest.mark.parametrize("refuse_after", [None, 3, 0])
def test_downward_bias_clips_raw_action_and_counts_only_sent_steps(refuse_after):
    from act_rlt.infer import bounded_action

    env, policy = make_env(x=4., yaw_level=2)
    encode = policy.encode_and_reference

    def large_reference(batch):
        state, reference, full = encode(batch)
        reference[:, :, 0] = 4.
        full[:, :, 0] = 4.
        return state, reference, full

    policy.encode_and_reference = large_reference
    env._downward_bias = EpisodeDownwardBias()
    env._monitor.consume_downward_bias = lambda: True
    original_send = env.robot.send_action

    def send(command):
        if refuse_after is not None and len(env.sent) == refuse_after:
            raise RuntimeError("refusing TCP target outside workspace: test")
        original_send(command)

    env.robot.send_action = send
    runtime = FixedChunkRuntime(env, policy, {}, warmup=True, step_budget=4)
    try:
        result = runtime.next_result()
    finally:
        runtime.close()
    count = 4 if refuse_after is None else refuse_after
    raw = np.array([.004, 0., -.00015, 0., 0., 0.], dtype=np.float32)
    expected = np.array(list(bounded_action(raw, .002, .02).values()))
    raw[2] = 0
    baseline = np.array(list(bounded_action(raw, .002, .02).values()))
    assert result.actual_steps == count
    assert result.info["downward_bias_steps"] == count
    assert result.info["downward_bias_requested_z_m"] == pytest.approx(-.00015 * count)
    np.testing.assert_allclose(result.info["downward_bias_applied_xyz_m"], (expected[:3] - baseline[:3]) * count,
                               atol=1e-9)
    for command in env.sent:
        np.testing.assert_allclose(list(command.values()), expected, atol=1e-9)
    if count:
        torch.testing.assert_close(result.exec_chunk[:count], torch.tensor(expected, dtype=torch.float32).expand(count, -1) / .001)
    assert not result.exec_chunk[count:].any()


def test_legacy_downward_bias_begins_inside_chunk():
    env, _ = make_env()
    env._episode_warmup = True
    env._downward_bias = EpisodeDownwardBias()
    env._monitor.consume_downward_bias = lambda: len(env.sent) >= 1
    env._processed_observation = lambda: {}
    result = env.execute_chunk(torch.zeros(1, 4, 6), torch.zeros(1, 4, 6))
    torch.testing.assert_close(result.exec_chunk[:, 2], torch.tensor([0., -.15, -.15, -.15]))
    assert result.info["downward_bias_steps"] == 3
    assert result.info["downward_bias_requested_z_m"] == pytest.approx(-.00045)


def test_actor_b_adds_z_before_clipping_and_preserves_directional_offsets():
    from act_rlt.infer import bounded_action

    env, policy = make_env(x=4., yaw=19.999, yaw_level=5)
    env._downward_bias = EpisodeDownwardBias()
    env._monitor.consume_downward_bias = lambda: len(env.sent) >= 2
    runtime = FixedChunkRuntime(env, policy, {}, warmup=False, step_budget=4)
    try:
        result = runtime.next_result()
    finally:
        runtime.close()
    baseline = list(bounded_action([.004 + BIAS_STEP_M, BIAS_STEP_M, -.00015,
                                    0., 0., .019999 + 5 * BIAS_STEP_YAW_RAD], .002, .02).values())
    for command in env.sent[2:]:
        np.testing.assert_allclose(list(command.values()), baseline, atol=1e-9)
    assert result.info["exploration_bias_requested_yaw_rad"] == pytest.approx(20 * BIAS_STEP_YAW_RAD)
    assert result.info["downward_bias_steps"] == 2
    sent = torch.tensor([list(command.values()) for command in env.sent], dtype=torch.float32)
    torch.testing.assert_close(result.exec_chunk, sent / .001)


def test_tty_downward_key_does_not_consume_or_delay_outcome_keys(monkeypatch):
    import os
    import pty

    master, slave = pty.openpty()
    monitor = None
    try:
        monkeypatch.setattr("act_rlt.train_stage2.sys.stdin", SimpleNamespace(isatty=lambda: True, fileno=lambda: slave))
        monitor = OutcomeMonitor()
        os.write(master, b"bBbsfq")
        assert monitor.consume_downward_bias()
        assert not monitor.consume_downward_bias()
        assert [monitor.poll() for _ in range(4)] == ["s", "f", "q", None]
        os.write(master, b"sb")
        assert monitor.poll() == "s"
        assert monitor.consume_downward_bias()
    finally:
        if monitor is not None:
            monitor.close()
        os.close(master)
        os.close(slave)
