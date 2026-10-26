from contextlib import nullcontext
import io
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from act_rlt import infer_coarse as coarse


def config(state=7, action=7, chunk=10):
    return SimpleNamespace(
        type="act", chunk_size=chunk, n_action_steps=4, temporal_ensemble_coeff=.01,
        input_features={"observation.state": SimpleNamespace(shape=[state]),
                        "observation.images.wrist": SimpleNamespace(shape=[3, 480, 640])},
        output_features={"action": SimpleNamespace(shape=[action])},
    )


def test_reset_moves_directly_to_fixed_dataset_joints():
    calls = []
    robot = SimpleNamespace(
        robot=SimpleNamespace(
            move_joint=lambda target, speed: calls.append((target, speed)),
            get_joint_angles=lambda: list(coarse.COARSE_START_JOINTS),
        ),
        resync_command_pose=lambda: None,
    )
    assert coarse.return_to_dataset_start(robot, SimpleNamespace(dry_run=False, reset_joint_speed=.2))
    assert calls == [(list(coarse.COARSE_START_JOINTS), .2)]


def test_dry_run_reset_sends_no_motion():
    assert not coarse.return_to_dataset_start(
        SimpleNamespace(), SimpleNamespace(dry_run=True),
    )


@pytest.mark.parametrize("name,state,action,mode", [
    ("act_coarse_1020_rgb_joint_abs_joint", 7, 7, "joint_absolute"),
    ("act_coarse_1020_rgb_z_delta_tcp", 1, 6, "tcp_delta"),
    ("coarse_pose_joint_hold1s", 13, 6, "tcp_delta"),
    ("coarse_z_hold1s", 1, 6, "tcp_delta"),
])
def test_saved_training_contract_selects_mode_and_overrides_four_steps(tmp_path, name, state, action, mode):
    (tmp_path / "train_config.json").write_text(json.dumps({"job_name": name}))
    cfg = config(state, action)
    selected, cameras = coarse.checkpoint_action_mode(cfg, tmp_path)
    assert selected == mode and cameras == {"wrist": (480, 640, 3)}
    assert cfg.n_action_steps == 10 and cfg.temporal_ensemble_coeff is None


def test_pose_joint_hold_checkpoint_cannot_be_used_as_absolute_joint_actions(tmp_path):
    coarse.checkpoint_action_mode(config(13, 6), tmp_path, "tcp_delta")
    with pytest.raises(ValueError, match="does not match"):
        coarse.checkpoint_action_mode(config(13, 6), tmp_path, "joint_absolute")
    with pytest.raises(ValueError, match="does not match"):
        coarse.checkpoint_action_mode(config(13, 7), tmp_path, "tcp_delta")


def test_ambiguous_seven_outputs_require_explicit_joint_semantics(tmp_path):
    with pytest.raises(ValueError, match="Unknown action semantics"):
        coarse.checkpoint_action_mode(config(), tmp_path)
    coarse.checkpoint_action_mode(config(), tmp_path, "joint_absolute")
    with pytest.raises(ValueError, match="does not match"):
        coarse.checkpoint_action_mode(config(), tmp_path, "tcp_delta")
    with pytest.raises(ValueError, match="chunk_size=10"):
        coarse.checkpoint_action_mode(config(chunk=16), tmp_path, "joint_absolute")
    cfg = config()
    cfg.input_features["observation.images.front"] = SimpleNamespace(shape=[3, 480, 640])
    with pytest.raises(ValueError, match="only 640x480 wrist"):
        coarse.checkpoint_action_mode(cfg, tmp_path, "joint_absolute")


@pytest.fixture
def runtime(monkeypatch):
    initial = np.array([0., 0., 0., -1.5, 0., 1.5, 0.])
    calls = SimpleNamespace(joint=[], tcp=[], stopped=0, predictions=0, resets=0, observations=0,
                            prediction_steps=[], observation_steps=[])
    measured = initial.copy()

    def servo_joint(target):
        measured[:] = target
        calls.joint.append(target)

    def stop_servo():
        calls.stopped += 1
        return True

    def observe(robot, cfg):
        calls.observations += 1
        calls.observation_steps.append(len(calls.joint) + len(calls.tcp))
        return {"sequence": calls.observations}

    def predict(policy, pre, post, observation, cameras):
        calls.predictions += 1
        calls.prediction_steps.append(len(calls.joint) + len(calls.tcp))
        if policy.config.output_features["action"].shape == [7]:
            return np.stack([initial + (calls.predictions * 10 + i) * .0001 for i in range(10)])
        return np.stack([np.array([.0001 * (i + 1), 0, 0, 0, 0, 0]) for i in range(10)])

    def reset():
        calls.resets += 1

    robot = SimpleNamespace(
        robot=SimpleNamespace(get_joint_angles=lambda: measured.copy(), servo_joint=servo_joint,
                              stop_servo=stop_servo,
                              get_tool_pose=lambda: [0.55, 0.05, .402, 0., 0., 0.]),
        is_connected=True, resync_command_pose=lambda: None,
        send_action=lambda action: calls.tcp.append(action),
    )
    policy = SimpleNamespace(config=config(), reset=reset)
    args = SimpleNamespace(chunks=2, duration=1e9, dry_run=False, fps=150, prefetch_steps=5,
                           max_joint_step_rad=.04, max_step_m=.01, max_step_rad=.05,
                           checkpoint="model", z_gate=False)
    def wait_until(deadline):
        time.sleep(max(0, deadline - time.monotonic()))
        return controls.poll()

    controls = SimpleNamespace(poll=lambda: None, wait_until=wait_until)
    monkeypatch.setattr(coarse, "policy_observation", observe)
    monkeypatch.setattr(coarse, "_predict_full_action_chunk", predict)
    return SimpleNamespace(robot=robot, policy=policy, args=args, controls=controls,
                           calls=calls, initial=initial)


def run(runtime, mode, trace=None):
    return coarse.run_episode(runtime.robot, runtime.policy, None, None, runtime.args, mode,
                              {"wrist": (480, 640, 3)},
                              SimpleNamespace(inference_mode=nullcontext), runtime.controls, trace)


def test_joint_mode_prefetches_but_executes_all_ten_in_order(runtime):
    trace = io.StringIO()
    result = run(runtime, "joint_absolute", trace)
    assert result == dict(reason="chunks", chunks=2, actions=20)
    assert len(runtime.calls.joint) == 20 and runtime.calls.tcp == []
    assert runtime.calls.predictions == 2 and runtime.calls.observations >= 2
    assert runtime.calls.prediction_steps[0] == 0
    assert 5 <= runtime.calls.prediction_steps[1] < 10
    assert any(0 < step < 10 for step in runtime.calls.observation_steps)
    np.testing.assert_allclose(runtime.calls.joint[0], runtime.initial + .001)
    np.testing.assert_allclose(runtime.calls.joint[9], runtime.initial + .0019)
    np.testing.assert_allclose(runtime.calls.joint[10], runtime.initial + .002)
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    assert [row["index"] for row in rows if row["event"] == "action"] == list(range(10)) * 2
    assert runtime.calls.stopped == 1 and runtime.calls.resets == 2


def test_tcp_mode_uses_body_delta_interface_without_joint_or_gripper_commands(runtime):
    runtime.policy.config = config(1, 6)
    run(runtime, "tcp_delta")
    assert len(runtime.calls.tcp) == 20 and runtime.calls.joint == []
    assert list(runtime.calls.tcp[0]) == ["dx", "dy", "dz", "drx", "dry", "drz"]
    assert runtime.calls.tcp[9]["dx"] == pytest.approx(.001)


@pytest.mark.parametrize("rtc,temporal", [(True, False), (False, False), (True, True)])
def test_pose_joint_hold_model_uses_tcp_execution_in_all_scheduling_modes(runtime, rtc, temporal):
    runtime.policy.config = config(13, 6)
    runtime.args.rtc = rtc
    runtime.args.temporal_ensemble = temporal
    result = run(runtime, "tcp_delta")
    assert result == dict(reason="chunks", chunks=2, actions=20)
    assert len(runtime.calls.tcp) == 20 and runtime.calls.joint == []
    assert runtime.calls.stopped == 1


def enable_z_gate(runtime, *, z_offset=.001):
    runtime.policy.config = config(13, 6)
    runtime.args.z_gate = True
    runtime.args.chunks = 1
    pose = np.array([.55, .05, .40207 + z_offset, 0., 0., 0.])
    runtime.robot.robot.get_fresh_tool_pose = lambda **kwargs: pose.copy()
    runtime.calls.tool = []
    runtime.robot.robot.servo_tool = lambda target: runtime.calls.tool.append(target)
    return pose


@pytest.mark.parametrize('rtc,temporal', [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize('crossing_offset', [0., -.001])
def test_z_gate_stops_all_motion_mid_chunk_without_finishing(runtime, rtc, temporal, crossing_offset):
    pose = enable_z_gate(runtime)
    runtime.args.rtc = rtc
    runtime.args.temporal_ensemble = temporal
    send = runtime.robot.send_action

    def arrive_after_three_commands(action):
        send(action)
        if len(runtime.calls.tcp) == 3:
            pose[2] = .40207 + crossing_offset

    runtime.robot.send_action = arrive_after_three_commands
    trace = io.StringIO()
    result = run(runtime, 'tcp_delta', trace)
    assert result == dict(reason='z_gate', actions=3, chunks=0)
    assert len(runtime.calls.tcp) == 3 and runtime.calls.stopped == 1
    assert runtime.calls.tool == []
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    gate_index = next(i for i, row in enumerate(rows) if row['event'] == 'z_gate_reached')
    assert not any(row['event'] in {'action', 'refill_requested'} for row in rows[gate_index + 1:])
    assert rows[-1]['reason'] == 'z_gate'


@pytest.mark.parametrize('z_offset', [0., -.001])
def test_z_gate_at_start_sends_no_motion_or_episode_predictions(runtime, z_offset):
    enable_z_gate(runtime, z_offset=z_offset)
    result = run(runtime, 'tcp_delta')
    assert result == dict(reason='z_gate', actions=0, chunks=0)
    assert runtime.calls.predictions == runtime.calls.observations == 0
    assert runtime.calls.tcp == [] and runtime.calls.stopped == 1


def test_z_gate_without_arrival_has_no_extra_convergence_after_budget(runtime):
    enable_z_gate(runtime)
    result = run(runtime, 'tcp_delta')
    assert result == dict(reason='chunks', actions=10, chunks=1)
    assert len(runtime.calls.tcp) == 10 and runtime.calls.stopped == 1


def test_z_gate_disabled_uses_vanilla_interface_without_fresh_state_api(runtime):
    runtime.policy.config = config(13, 6)
    result = run(runtime, 'tcp_delta')
    assert result == dict(reason='chunks', chunks=2, actions=20)
    assert len(runtime.calls.tcp) == 20


def test_z_gate_stale_feedback_stops_and_preserves_error(runtime):
    pose = enable_z_gate(runtime)

    def read(**kwargs):
        if len(runtime.calls.tcp) >= 3:
            raise RuntimeError('Fresh streamed TCP pose unavailable or stale')
        return pose.copy()

    runtime.robot.robot.get_fresh_tool_pose = read
    trace = io.StringIO()
    with pytest.raises(RuntimeError, match='stale'):
        run(runtime, 'tcp_delta', trace)
    assert len(runtime.calls.tcp) == 3 and runtime.calls.stopped == 1
    assert json.loads(trace.getvalue().splitlines()[-1])['reason'] == 'error'


def test_z_gate_dry_run_sends_no_motion(runtime):
    enable_z_gate(runtime, z_offset=-.001)
    runtime.args.dry_run = True
    result = run(runtime, 'tcp_delta')
    assert result == dict(reason='chunks', actions=10, chunks=1)
    assert runtime.calls.tcp == [] and runtime.calls.stopped == 0


@pytest.mark.parametrize("mode,state,action", [("joint_absolute", 7, 7), ("tcp_delta", 1, 6)])
def test_two_step_prefetch_starts_after_action_eight_and_executes_both_full_chunks(
        runtime, mode, state, action):
    runtime.policy.config = config(state, action)
    runtime.args.prefetch_steps = 2
    runtime.args.fps = 50
    trace = io.StringIO()
    result = run(runtime, mode, trace)
    assert result == dict(reason="chunks", chunks=2, actions=20)
    assert 8 <= runtime.calls.prediction_steps[1] < 10
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    refill = [row for row in rows if row["event"] == "refill_requested" and row["chunk"] == 1]
    assert len(refill) == 1 and refill[0]["step"] == 8
    assert [row["index"] for row in rows if row["event"] == "action"] == list(range(10)) * 2


@pytest.mark.parametrize("mode,state,action", [("joint_absolute", 7, 7), ("tcp_delta", 1, 6)])
def test_no_rtc_observes_and_predicts_only_after_all_ten_control_periods(runtime, mode, state, action):
    runtime.args.rtc = False
    runtime.policy.config = config(state, action)
    trace = io.StringIO()
    result = run(runtime, mode, trace)
    assert result == dict(reason="chunks", chunks=2, actions=20)
    assert runtime.calls.observation_steps == [0, 10]
    assert runtime.calls.prediction_steps == [0, 10]
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    assert rows[0]["rtc_enabled"] is False
    assert rows[0]["camera_mode"] == "post_chunk" and rows[0]["prefetch_steps"] == 0
    actions = [row for row in rows if row["event"] == "action"]
    assert [row["index"] for row in actions] == list(range(10)) * 2
    refill = next(row for row in rows if row["event"] == "refill_requested" and row["chunk"] == 1)
    assert refill["step"] == 10
    assert refill["elapsed_s"] >= actions[9]["elapsed_s"] + 1 / runtime.args.fps - .001
    assert runtime.calls.stopped == 1


def test_no_rtc_stop_during_boundary_prediction_stops_before_joining_worker(runtime, monkeypatch):
    runtime.args.rtc = False
    started = threading.Event()
    stopped = threading.Event()
    prediction = coarse._predict_full_action_chunk
    stop = runtime.robot.robot.stop_servo

    def delayed_prediction(*args):
        if runtime.calls.predictions == 1:
            started.set()
            assert stopped.wait(1), "Stop must remain responsive while waiting for inference"
        return prediction(*args)

    def stop_servo():
        stopped.set()
        return stop()

    runtime.robot.robot.stop_servo = stop_servo
    runtime.controls.poll = lambda: "operator_stop" if started.is_set() else None
    monkeypatch.setattr(coarse, "_predict_full_action_chunk", delayed_prediction)
    result = run(runtime, "joint_absolute")
    assert result == dict(reason="operator_stop", chunks=1, actions=10)
    assert len(runtime.calls.joint) == 10 and runtime.calls.stopped == 1


def test_no_rtc_rechecks_duration_after_boundary_prediction(runtime, monkeypatch):
    runtime.args.rtc = False
    prediction = coarse._predict_full_action_chunk

    def expired_prediction(*args):
        if runtime.calls.predictions == 1:
            runtime.args.duration = 1e-9
        return prediction(*args)

    monkeypatch.setattr(coarse, "_predict_full_action_chunk", expired_prediction)
    result = run(runtime, "joint_absolute")
    assert result == dict(reason="duration", chunks=1, actions=10)
    assert runtime.calls.predictions == 2 and len(runtime.calls.joint) == 10


def test_no_rtc_observation_failure_stops_without_reusing_previous_chunk(runtime, monkeypatch):
    runtime.args.rtc = False
    observation = coarse.policy_observation

    def failing_observation(*args):
        if runtime.calls.predictions:
            raise RuntimeError("post-chunk camera disconnected")
        return observation(*args)

    monkeypatch.setattr(coarse, "policy_observation", failing_observation)
    with pytest.raises(RuntimeError, match="post-chunk camera disconnected"):
        run(runtime, "joint_absolute")
    assert len(runtime.calls.joint) == 10 and runtime.calls.stopped == 1


@pytest.mark.parametrize("dimension", [6, 7])
@pytest.mark.parametrize("coefficient", [0., .2, 1e6])
def test_temporal_ensemble_aligns_the_same_control_step_and_weights_oldest_first(dimension, coefficient):
    ensemble = coarse.CoarseTemporalEnsembler(dimension, coefficient)
    first = np.repeat(np.arange(10.)[:, None], dimension, axis=1)
    second = first + 100
    ensemble.add(0, first)
    ensemble.add(2, second)
    # Step 3 uses offset 3 of forecast 0 and offset 1 of forecast 2.
    action, contributors = ensemble.action_for(3)
    newer_weight = np.exp(-coefficient)
    np.testing.assert_allclose(action, (3 + 101 * newer_weight) / (1 + newer_weight))
    assert [row["offset"] for row in contributors] == [3, 1]
    assert sum(row["weight"] for row in contributors) == pytest.approx(1)
    # Once the first forecast expires, the second contributes on its own.
    action, contributors = ensemble.action_for(10)
    np.testing.assert_allclose(action, 108)
    assert len(contributors) == 1 and 0 not in ensemble.chunks
    with pytest.raises(RuntimeError, match="covers control step"):
        ensemble.action_for(12)


@pytest.mark.parametrize("mode,state,action", [("joint_absolute", 7, 7), ("tcp_delta", 1, 6)])
def test_temporal_ensemble_requests_each_step_and_blends_before_safety_limits(runtime, mode, state, action):
    runtime.args.temporal_ensemble = True
    runtime.args.temporal_ensemble_coeff = 0.
    runtime.args.fps = 50
    runtime.policy.config = config(state, action)
    trace = io.StringIO()
    result = run(runtime, mode, trace)
    assert result == dict(reason="chunks", chunks=2, actions=20)
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    assert rows[0]["temporal_ensemble"] is True and rows[0]["prefetch_steps"] == 0
    actions = [row for row in rows if row["event"] == "action"]
    assert len(actions) == 20 and max(row["ensemble_votes"] for row in actions) > 1
    assert [row["step"] for row in rows if row["event"] == "refill_requested"] == list(range(20))
    predictions = {row["start_step"]: row["predicted"] for row in rows if row["event"] == "chunk_ready"}
    for row in actions:
        contributors = row["ensemble_contributors"]
        expected = np.mean([predictions[v["start_step"]][row["step"] - v["start_step"]]
                            for v in contributors], axis=0)
        np.testing.assert_allclose(row["ensemble_action"], expected)
    assert runtime.calls.predictions > 2 and runtime.calls.stopped == 1


def test_temporal_ensemble_late_results_keep_original_step_offsets_and_skip_pending_requests(runtime, monkeypatch):
    runtime.args.temporal_ensemble = True
    runtime.args.fps = 50
    runtime.policy.config = config(1, 6)
    prediction = coarse._predict_full_action_chunk

    def delayed_prediction(*args):
        time.sleep(.07)
        return prediction(*args)

    monkeypatch.setattr(coarse, "_predict_full_action_chunk", delayed_prediction)
    trace = io.StringIO()
    result = run(runtime, "tcp_delta", trace)
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    assert result["actions"] == 20
    assert any(row["event"] == "refill_superseded" for row in rows)
    assert any(row["accepted_at_step"] > row["start_step"]
               for row in rows if row["event"] == "prediction_accepted")
    for row in rows:
        if row["event"] == "action":
            assert all(v["offset"] == row["step"] - v["start_step"] for v in row["ensemble_contributors"])


def test_temporal_ensemble_stops_when_no_forecast_covers_the_next_step(runtime, monkeypatch):
    runtime.args.temporal_ensemble = True
    runtime.policy.config = config(1, 6)
    stopped = threading.Event()
    prediction = coarse._predict_full_action_chunk
    stop = runtime.robot.robot.stop_servo

    def delayed_prediction(*args):
        if runtime.calls.predictions:
            assert stopped.wait(1), "Stop must precede joining the stalled inference worker"
        return prediction(*args)

    def stop_servo():
        stopped.set()
        return stop()

    runtime.robot.robot.stop_servo = stop_servo
    monkeypatch.setattr(coarse, "_predict_full_action_chunk", delayed_prediction)
    with pytest.raises(RuntimeError, match="covers control step 10"):
        run(runtime, "tcp_delta")
    assert len(runtime.calls.tcp) == 10 and runtime.calls.stopped == 1


def test_temporal_ensemble_limits_the_fused_tcp_action(runtime, monkeypatch):
    runtime.args.temporal_ensemble = True
    runtime.policy.config = config(1, 6)
    raw = np.tile([.02, .02, 0, .1, .1, 0], (10, 1))
    monkeypatch.setattr(coarse, "_predict_full_action_chunk", lambda *args: raw)
    run(runtime, "tcp_delta")
    for command in runtime.calls.tcp:
        assert np.linalg.norm([command[k] for k in ("dx", "dy", "dz")]) == pytest.approx(runtime.args.max_step_m)
        assert np.linalg.norm([command[k] for k in ("drx", "dry", "drz")]) == pytest.approx(runtime.args.max_step_rad)


def test_temporal_ensemble_rejects_invalid_joint_forecasts_before_averaging(runtime, monkeypatch):
    runtime.args.temporal_ensemble = True
    monkeypatch.setattr(coarse, "_predict_full_action_chunk", lambda *args: np.full((10, 7), 10.))
    with pytest.raises(ValueError, match="joint limits"):
        run(runtime, "joint_absolute")
    assert runtime.calls.joint == [] and runtime.calls.stopped == 1


@pytest.mark.parametrize("mode,state,action", [("joint_absolute", 7, 7), ("tcp_delta", 1, 6)])
def test_temporal_ensemble_dry_run_never_sends_motion(runtime, mode, state, action):
    runtime.args.temporal_ensemble = True
    runtime.args.dry_run = True
    runtime.policy.config = config(state, action)
    assert run(runtime, mode)["actions"] == 20
    assert runtime.calls.joint == runtime.calls.tcp == [] and runtime.calls.stopped == 0


@pytest.mark.parametrize("rtc", [True, False])
@pytest.mark.parametrize("mode,state,action", [("joint_absolute", 7, 7), ("tcp_delta", 1, 6)])
def test_dry_run_sends_no_servo_or_stop_commands(runtime, mode, state, action, rtc):
    runtime.args.rtc = rtc
    runtime.args.dry_run = True
    runtime.policy.config = config(state, action)
    result = run(runtime, mode)
    assert result["actions"] == 20
    assert runtime.calls.joint == runtime.calls.tcp == [] and runtime.calls.stopped == 0


def test_duration_expiry_finishes_full_chunk(runtime):
    runtime.args.duration = 1e-9
    result = run(runtime, "joint_absolute")
    assert result == dict(reason="duration", chunks=1, actions=10)


def test_endpoint_report_reads_after_stop_and_logs_absolute_z_and_intrinsic_euler(runtime, capsys):
    from scipy.spatial.transform import Rotation

    angles = [179.7, 1.4, 90.6]
    rotvec = Rotation.from_euler("XYZ", angles, degrees=True).as_rotvec()

    def get_pose():
        assert runtime.calls.stopped == 1, "Endpoint must be read after stop acknowledgement"
        return [.55, .05, .401234, *rotvec]

    runtime.robot.robot.get_tool_pose = get_pose
    trace = io.StringIO()
    run(runtime, "joint_absolute", trace)
    terminal = capsys.readouterr().out
    assert "Stopped TCP: Z=0.401234 m (401.234 mm)" in terminal
    assert "Euler intrinsic XYZ [deg]=[179.700, 1.400, 90.600]" in terminal
    end = json.loads(trace.getvalue().splitlines()[-1])
    assert end["stop_acknowledged"] is True and end["final_tcp_error"] is None
    assert end["final_tcp"]["z_m"] == .401234
    np.testing.assert_allclose(end["final_tcp"]["euler_xyz_deg"], angles)


def test_endpoint_read_failure_does_not_mask_control_error_or_prevent_cleanup(runtime, capsys):
    def fail_pose():
        raise RuntimeError("endpoint state unavailable")

    runtime.robot.robot.get_tool_pose = fail_pose
    runtime.controls.wait_until = lambda deadline: (_ for _ in ()).throw(RuntimeError("original control failure"))
    trace = io.StringIO()
    with pytest.raises(RuntimeError, match="original control failure"):
        run(runtime, "joint_absolute", trace)
    end = json.loads(trace.getvalue().splitlines()[-1])
    assert end["reason"] == "error" and end["final_tcp"] is None
    assert end["final_tcp_error"] == "endpoint state unavailable"
    assert "End TCP unavailable" in capsys.readouterr().out
    assert runtime.calls.stopped == 1 and runtime.calls.resets == 2


def test_dry_run_endpoint_is_labeled_as_observation_without_stopping(runtime, capsys):
    runtime.args.dry_run = True
    run(runtime, "joint_absolute")
    terminal = capsys.readouterr().out
    assert "Dry-run end observation: Z=" in terminal and "Stopped TCP" not in terminal
    assert runtime.calls.stopped == 0


def test_operator_stop_interrupts_chunk_and_stops_robot(runtime):
    runtime.controls.wait_until = lambda deadline: "operator_stop" if len(runtime.calls.joint) == 3 else None
    result = run(runtime, "joint_absolute")
    assert result == dict(reason="operator_stop", chunks=0, actions=3)
    assert runtime.calls.stopped == 1


@pytest.mark.parametrize("mode,state,action", [("joint_absolute", 7, 7), ("tcp_delta", 1, 6)])
def test_slow_next_prediction_overlaps_current_commands_and_keeps_boundary_cadence(
        runtime, monkeypatch, mode, state, action):
    runtime.policy.config = config(state, action)
    runtime.args.fps = 50
    next_started = threading.Event()
    current_progressed = threading.Event()
    next_finished = threading.Event()
    prediction = coarse._predict_full_action_chunk

    def slow_prediction(*args):
        if runtime.calls.predictions == 1:
            next_started.set()
            assert current_progressed.wait(1), "Servo stalled while inference was running"
            next_finished.set()
        return prediction(*args)

    def poll():
        commands = runtime.calls.joint if mode == "joint_absolute" else runtime.calls.tcp
        if next_started.is_set() and len(commands) >= 8:
            current_progressed.set()

    monkeypatch.setattr(coarse, "_predict_full_action_chunk", slow_prediction)
    runtime.controls.poll = poll
    # Every action timestamp is recorded by the asynchronous trace writer.
    trace = io.StringIO()
    run(runtime, mode, trace)
    assert next_finished.is_set()
    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
    action_rows = [row for row in rows if row["event"] == "action"]
    refill = [row for row in rows if row["event"] == "refill_requested" and row["chunk"] == 1]
    assert len(action_rows) == 20 and refill[0]["step"] == 5
    boundary_gap = action_rows[10]["elapsed_s"] - action_rows[9]["elapsed_s"]
    # At 50 Hz the boundary has one 20 ms control interval, with no 60+ ms
    # model latency added to it (prediction was blocked until action eight).
    assert .015 <= boundary_gap < .05


def test_late_next_prediction_stops_on_underrun_without_repeating_delta(runtime, monkeypatch):
    runtime.policy.config = config(1, 6)
    prediction = coarse._predict_full_action_chunk
    stopped = threading.Event()
    stop = runtime.robot.robot.stop_servo

    def stop_servo():
        stopped.set()
        return stop()

    def delayed_prediction(*args):
        if runtime.calls.predictions == 1:
            assert stopped.wait(1), "Hardware must stop before waiting for the model worker"
        return prediction(*args)

    runtime.robot.robot.stop_servo = stop_servo
    monkeypatch.setattr(coarse, "_predict_full_action_chunk", delayed_prediction)
    with pytest.raises(RuntimeError, match="queue underrun"):
        run(runtime, "tcp_delta")
    assert len(runtime.calls.tcp) == 10 and runtime.calls.stopped == 1


def test_camera_failure_during_chunk_stops_control(runtime, monkeypatch):
    observation = coarse.policy_observation

    def failing_camera(*args):
        if len(runtime.calls.joint) >= 3:
            raise RuntimeError("wrist disconnected")
        return observation(*args)

    monkeypatch.setattr(coarse, "policy_observation", failing_camera)
    with pytest.raises(RuntimeError, match="wrist disconnected"):
        run(runtime, "joint_absolute")
    assert len(runtime.calls.joint) < 10 and runtime.calls.stopped == 1


def test_ctrl_c_stops_servo_and_logs_operator_stop(runtime):
    trace = io.StringIO()

    def interrupt(deadline):
        raise KeyboardInterrupt

    runtime.controls.wait_until = interrupt
    with pytest.raises(KeyboardInterrupt):
        run(runtime, "joint_absolute", trace)
    assert runtime.calls.stopped == 1 and runtime.calls.joint == []
    assert json.loads(trace.getvalue().splitlines()[-1])["reason"] == "operator_stop"


def test_failed_stop_acknowledgement_surfaces_as_error(runtime):
    runtime.robot.robot.stop_servo = lambda: False
    trace = io.StringIO()
    with pytest.raises(RuntimeError, match="acknowledge stop_servo"):
        run(runtime, "joint_absolute", trace)
    assert json.loads(trace.getvalue().splitlines()[-1])["reason"] == "error"
    assert runtime.calls.resets == 2


@pytest.mark.parametrize("bad", [np.zeros((4, 7)), np.full((10, 7), np.nan)])
def test_bad_chunk_cannot_reach_robot_and_error_stops_servo(runtime, monkeypatch, bad):
    monkeypatch.setattr(coarse, "_predict_full_action_chunk", lambda *args: bad)
    with pytest.raises(ValueError, match="finite full action chunk"):
        run(runtime, "joint_absolute")
    assert runtime.calls.joint == [] and runtime.calls.stopped == 1


def test_joint_limits_and_measured_lead_are_enforced(runtime):
    q = runtime.initial
    result = coarse.bounded_joint_target(q + .2, q, q + .01, .04)
    np.testing.assert_allclose(result, q + .04)
    with pytest.raises(ValueError, match="joint limits"):
        coarse.bounded_joint_target(np.ones(7) * 10, q, None, .04)
    with pytest.raises(RuntimeError, match="diverged"):
        coarse.bounded_joint_target(q, q, q + .2, .04)


def test_cli_defaults_and_invalid_limits():
    parser = coarse.build_parser()
    args = parser.parse_args(["--checkpoint", "model"])
    coarse.validate_args(parser, args)
    assert args.fps == 15 and args.action_mode == "auto" and args.prefetch_steps == 2
    assert args.rtc is True
    assert args.z_gate is True and args.z_gate_m == .40207
    disabled = parser.parse_args(["--checkpoint", "model", "--no-rtc"])
    coarse.validate_args(parser, disabled)
    assert disabled.rtc is False
    assert args.temporal_ensemble is False and args.temporal_ensemble_coeff == .01
    enabled = parser.parse_args(["--checkpoint", "model", "--temporal-ensemble"])
    coarse.validate_args(parser, enabled)
    assert enabled.temporal_ensemble is True
    for extra in (["--fps", "nan"], ["--chunks", "0"], ["--dry-run-episodes", "1"],
                  ["--prefetch-steps", "0"], ["--prefetch-steps", "10"],
                  ["--temporal-ensemble", "--no-rtc"],
                  ["--temporal-ensemble-coeff", "nan"],
                  ["--temporal-ensemble-coeff", "-1"],
                  ["--temporal-ensemble-coeff", "inf"],
                  ["--z-gate-m", "nan"], ["--z-gate-state-max-age-s", "0"]):
        with pytest.raises(SystemExit):
            coarse.validate_args(parser, parser.parse_args(["--checkpoint", "model", *extra]))


@pytest.mark.parametrize("state,action", [(7, 7), (1, 6), (13, 6)])
def test_full_chunk_predictor_uses_correct_state_and_unnormalizes_all_ten(state, action):
    import torch

    from act_rlt.infer import _predict_full_action_chunk

    seen = {}
    observation = {"wrist": np.zeros((480, 640, 3), dtype=np.uint8),
                   "ee_x": .51, "ee_y": -.12, "ee_z": .55,
                   "ee_rx": 2.1, "ee_ry": -2.2, "ee_rz": .19,
                   **{f"joint_{i}": i * .1 for i in range(7)}}

    def pre(frame):
        seen.update(frame)
        return {key: tensor.unsqueeze(0) for key, tensor in frame.items()}

    raw = torch.arange(10 * action, dtype=torch.float32).reshape(1, 10, action)
    # No select_action method: using the four-action cache would fail this test.
    policy = SimpleNamespace(config=config(state, action), predict_action_chunk=lambda batch: raw)
    result = _predict_full_action_chunk(policy, pre, lambda value: value * 2 + 1,
                                        observation, {"wrist": (480, 640, 3)})
    np.testing.assert_allclose(result, (raw * 2 + 1).numpy()[0])
    assert seen["observation.state"].shape == (state,)
    expected = ([.55] if state == 1 else np.arange(7) * .1 if state == 7 else
                [*list(np.arange(7) * .1), .51, -.12, .55, 2.1, -2.2, .19])
    np.testing.assert_allclose(seen["observation.state"].numpy(), expected)
    assert seen["observation.images.wrist"].shape == (3, 480, 640)


@pytest.mark.parametrize("bad_field", ["joint_6", "ee_x", "ee_rz"])
def test_pose_joint_state_rejects_missing_or_nonfinite_measured_fields(bad_field):
    from act_rlt.infer import observation_frame

    observation = {**{f"joint_{i}": .1 for i in range(7)},
                   **{f"ee_{axis}": .2 for axis in ("x", "y", "z", "rx", "ry", "rz")}}
    observation.pop(bad_field)
    with pytest.raises(ValueError, match="requires measured fields"):
        observation_frame(observation, {}, "joints_pose")
    observation[bad_field] = float("nan")
    with pytest.raises(ValueError, match="Invalid joints_pose state"):
        observation_frame(observation, {}, "joints_pose")
