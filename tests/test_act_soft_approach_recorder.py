from contextlib import nullcontext
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from act_rlt.data_collection import record_soft_approach as collector


def arguments(tmp_path, *extra):
    return collector.build_parser().parse_args([
        "--dataset", "test/soft", "--root", str(tmp_path / "dataset"), *extra,
    ])


def test_defaults_reuse_latest_soft_approach_and_original_recording_schema(tmp_path):
    args = arguments(tmp_path)
    collector.validate_args(args)
    approach = collector.motion_args(args)
    assert args.fps == 15 and approach.fps == 30
    assert args.move_speed == 0.016 and approach.speed == 0.04
    assert approach.lateral_gate_distance == 0.06 and approach.vertical_speed_floor == 0.15
    assert approach.acceleration == 0.05 and approach.jerk == 0.2
    assert approach.finite_approach
    assert approach.slowdown_distance == 0.008
    assert not hasattr(args, "insertion_speed")
    assert not hasattr(args, "insertion_slowdown_distance")
    assert args.insert_minus_z == 0.01
    assert collector.EpisodeRecorder._add_frame is collector.base.EpisodeRecorder._add_frame
    assert not hasattr(args, "depth")
    assert collector.motion_args(args, remaining=0.3).timeout == 0.3


def test_short_sampled_approach_curves_downward_with_gentler_acceleration(tmp_path):
    args = arguments(tmp_path)
    start = np.array([0.0, 0.02, 0.402, 0, 0, 0])
    target = np.array([0.0, 0.0, 0.381, 0, 0, 0])
    parameters = collector.motion_args(args, start_pose=start, target_pose=target)
    assert parameters.lateral_gate_distance == pytest.approx(0.02)
    assert parameters.lateral_tolerance == pytest.approx(0.003)
    planner = collector.soft.OnlineApproach(start, target, parameters)
    diagnostics = planner.step(start, 1 / parameters.fps)
    assert diagnostics["vertical_gate"] == pytest.approx(0.15)
    half_xy = None
    previous_acceleration = planner.acceleration.copy()
    for _ in range(600):
        planner.step(planner.command.copy(), 1 / parameters.fps)
        assert np.linalg.norm(planner.acceleration) <= parameters.acceleration + 1e-12
        assert np.linalg.norm(planner.acceleration - previous_acceleration) <= parameters.jerk / parameters.fps + 1e-12
        previous_acceleration = planner.acceleration.copy()
        if half_xy is None and planner.command[1] <= 0.01:
            half_xy = planner.command.copy()
        if planner.at_goal():
            break
    assert planner.at_goal()
    # A straight line would have descended by half the total Z at half XY.
    # Prefer lateral alignment first, with descent continuously allowed.
    assert half_xy is not None
    z_fraction = (start[2] - half_xy[2]) / (start[2] - target[2])
    assert 0.05 < z_fraction < 0.4
    assert collector.soft.build_parser().parse_args([]).acceleration == 0.1


def test_stage_specific_gate_keeps_explicit_caps(tmp_path):
    args = arguments(tmp_path, "--lateral-gate-distance", "0.015", "--lateral-tolerance", "0.002",
                     "--acceleration", "0.03", "--jerk", "0.1")
    start, target = [0, 0.03, 0.02, 0, 0, 0], [0]*6
    parameters = collector.motion_args(args, start_pose=start, target_pose=target)
    assert parameters.lateral_gate_distance == 0.015 and parameters.lateral_tolerance == 0.002
    assert parameters.acceleration == 0.03 and parameters.jerk == 0.1


@pytest.mark.parametrize("options", [
    ["--control-fps", "1"], ["--approach-speed", "nan"], ["--vertical-speed-floor", "2"],
    ["--insert-minus-z", "0"], ["--slowdown-distance", "0"],
])
def test_invalid_motion_settings_are_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        collector.validate_args(arguments(tmp_path, *options))


def test_new_dataset_marker_separates_legacy_waypoint_collection(tmp_path):
    args = arguments(tmp_path, "--resume")
    with pytest.raises(ValueError, match="created by record_soft_approach"):
        collector.verify_dataset_mode(args)
    collector.write_mode_marker(args)
    collector.verify_dataset_mode(args)
    reference = [0.4, 0.1, 0.39, 0.1, 0.2, 0.3]
    directory = collector.record_run_metadata(args, reference)
    metadata = json.loads((directory / "run.json").read_text())
    np.testing.assert_allclose(metadata["final_pose"], [0.4, 0.1, 0.38, 0.1, 0.2, 0.3])
    assert reference[2] == 0.39


def test_two_stage_dataset_is_not_silently_resumed_with_single_stage_motion(tmp_path):
    args = arguments(tmp_path, "--resume")
    collector.write_mode_marker(args)
    marker = args.root / "meta" / "soft_approach_collection.json"
    metadata = json.loads(marker.read_text())
    assert metadata["recorded_stages"] == 1
    assert metadata["goal_offset_base_z_m"] == -0.01
    metadata["trajectory_mode"] = "online_soft_approach_then_minus_z_v1"
    marker.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="single-stage mode"):
        collector.verify_dataset_mode(args)


class FakeClient:
    def __init__(self):
        self.pose = [0.4, 0.12, 0.402, 0, 0, 0]
        self.sent = []
        self.stops = 0
        self.seq = 0

    def get_tool_pose(self): return list(self.pose)
    def _send(self, command):
        assert command == "get_state"
        self.seq += 1
        return {"success": True, "state": {
            "pose": list(self.pose), "joints": [0]*7, "ts": time.monotonic(),
            "seq": self.seq, "joint_speeds": [0]*7, "robot_mode": "Move",
            "servo_fault_seq": 0, "state_stale": False,
        }}

    def servo_tool(self, pose):
        self.sent.append(list(pose))
        self.pose = list(pose)

    def stop_move(self):
        self.stops += 1
        return True

    def stop_servo(self):
        self.stops += 1
        return True


class FakeRecorder:
    def __init__(self, *, abort=False):
        self.client = FakeClient()
        self.events = []
        self.robot = SimpleNamespace(
            robot=self.client, is_connected=True, config=SimpleNamespace(enable_fci_keepalive=True),
            _stop_keepalive=lambda: self.events.append("keepalive_off"),
            _start_keepalive=lambda: self.events.append("keepalive_on"),
            resync_command_pose=lambda: self.events.append("resync"),
        )
        self.started_at = time.perf_counter()
        self.last_sample_started_at = None
        self.fps = 15
        self.frames = 0
        self.capture_threads = []
        self.commands = []
        self.abort = abort

    def capture(self):
        if self.abort and self.frames >= 2:
            raise collector.EpisodeAborted("fake abort key")
        self.last_sample_started_at = time.perf_counter()
        self.commands.append(self.command_action.copy())
        self.capture_threads.append(threading.current_thread())
        self.frames += 1
        time.sleep(0.003)


def test_single_recorded_motion_targets_offset_goal_and_preserves_teaching(tmp_path, monkeypatch):
    recorder = FakeRecorder()
    main_thread = threading.current_thread()
    stages = []
    reference = [0.4, 0.1, 0.39, 0.1, 0.2, 0.3]
    final = collector.insertion_pose_minus_z(reference, 0.01)

    def run(proxy, start, target, config, emit):
        assert threading.current_thread() is not main_thread
        assert recorder.frames >= 1
        stages.append((list(target), config.speed, config.slowdown_distance))
        for _ in range(5):
            proxy.servo_tool(target)
            emit("sample", command_pose=target, reference_velocity_m_s=[0.01, 0, 0])
            time.sleep(0.003)
        emit("settled", state=start)

    monkeypatch.setattr(collector.soft, "run_online_move", run)
    path = tmp_path / "trace.jsonl"
    collector.run_recorded_episode(recorder, reference, final, arguments(tmp_path), path)
    assert stages == [(final, 0.04, 0.008)]
    assert reference == [0.4, 0.1, 0.39, 0.1, 0.2, 0.3]
    assert all(thread is main_thread for thread in recorder.capture_threads)
    assert recorder.events == ["keepalive_off", "resync", "keepalive_on"]
    assert any(command["target_z"] == final[2] for command in recorder.commands)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["phase"] for event in events if event["event"] == "stage_start"] == ["approach"]
    assert [event["target"] for event in events if event["event"] == "stage_start"] == [final]
    assert events[-1]["event"] == "episode_motion_finished"
    assert not any(thread.name == "act-soft-approach-control" for thread in threading.enumerate())


def test_failed_approach_stops_before_return(tmp_path, monkeypatch):
    recorder = FakeRecorder()
    calls = []

    def fail(proxy, start, target, config, emit):
        calls.append(target)
        raise RuntimeError("actual target not reached")

    monkeypatch.setattr(collector.soft, "run_online_move", fail)
    reference = [0.4, 0.1, 0.39, 0, 0, 0]
    with pytest.raises(RuntimeError, match="not reached"):
        collector.run_recorded_episode(recorder, reference,
                                       collector.insertion_pose_minus_z(reference, 0.01),
                                       arguments(tmp_path), tmp_path / "trace.jsonl")
    assert calls == [collector.insertion_pose_minus_z(reference, 0.01)]
    assert recorder.client.stops >= 1
    assert not any(thread.name == "act-soft-approach-control" for thread in threading.enumerate())


def test_real_online_planner_reaches_offset_goal_once_while_capture_continues(tmp_path):
    recorder = FakeRecorder()
    args = arguments(tmp_path, "--approach-speed", "0.1",
                     "--acceleration", "0.5", "--jerk", "5", "--slowdown-distance", "0.002",
                     "--settle-time", "0.05", "--stage-timeout", "10")
    reference = [0.4, 0.1, 0.39, 0, 0, 0]
    final = collector.insertion_pose_minus_z(reference, 0.01)
    path = tmp_path / "real-planner.jsonl"
    collector.run_recorded_episode(recorder, reference, final, args, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["phase"] for row in rows if row["event"] == "settled"] == ["approach"]
    np.testing.assert_allclose(recorder.client.pose, final, atol=1e-9)
    assert recorder.frames > 10
    assert recorder.client.stops == 1


def test_abort_cancels_control_and_joins_before_episode_can_be_discarded(tmp_path, monkeypatch):
    recorder = FakeRecorder(abort=True)

    def control(proxy, start, target, config, emit):
        for _ in range(1000):
            proxy.servo_tool(target)
            emit("sample", command_pose=target, reference_velocity_m_s=[0.01, 0, 0])
            time.sleep(0.001)

    monkeypatch.setattr(collector.soft, "run_online_move", control)
    reference = [0.4, 0.1, 0.39, 0, 0, 0]
    with pytest.raises(collector.EpisodeAborted):
        collector.run_recorded_episode(recorder, reference,
                                       collector.insertion_pose_minus_z(reference, 0.01),
                                       arguments(tmp_path), tmp_path / "trace.jsonl")
    assert recorder.client.stops >= 1
    assert not any(thread.name == "act-soft-approach-control" for thread in threading.enumerate())
    sent = len(recorder.client.sent)
    time.sleep(0.01)
    assert len(recorder.client.sent) == sent


def test_main_records_only_task_motion_then_retreats_outside_dataset(tmp_path, monkeypatch):
    args = arguments(tmp_path, "--episodes", "1")
    reference = [0.4, 0.1, 0.39, 0.1, 0.2, 0.3]
    robot = SimpleNamespace(is_connected=True, robot=object(), connect=lambda: None,
                            get_observation=lambda: {}, disconnect=lambda: None,
                            save_depth_calibration=lambda path: None)
    dataset = SimpleNamespace(num_episodes=0, save_episode=lambda: None)
    recorder = SimpleNamespace(start=lambda: None, clear_command=lambda: None, finish=lambda: 3,
                               abort_monitor=None, discard_depth=lambda: None,
                               prepare_depth_save=lambda: None, mark_depth_saved=lambda: None)
    moves = []
    monkeypatch.setattr(collector, "build_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(collector, "build_dataset_and_robot", lambda args: (dataset, robot, None, []))
    monkeypatch.setattr(collector.soft, "require_idle", lambda client: None)
    monkeypatch.setattr(collector, "teach_reference", lambda robot: reference.copy())
    monkeypatch.setattr(collector, "EpisodeRecorder", lambda **kwargs: recorder)
    monkeypatch.setattr(collector, "VideoEncodingManager", lambda dataset: nullcontext())
    monkeypatch.setattr(collector, "EpisodeAbortMonitor", lambda key: nullcontext())
    monkeypatch.setattr(collector, "robot_is_executable", lambda robot: True)
    monkeypatch.setattr(collector, "nonrecorded_arm_move",
                        lambda robot, target, **kwargs: moves.append(("outside", list(target))) or True)
    monkeypatch.setattr(collector, "run_recorded_episode",
                        lambda rec, target, final, args, path: moves.append(("recorded", final)))
    monkeypatch.setattr(collector, "read_key_during_delay", lambda seconds: None)
    monkeypatch.setattr(collector, "normalize_lerobot_scalar_buffer", lambda dataset: None)
    assert collector.main() == 0
    assert [kind for kind, _ in moves] == ["outside", "recorded", "outside"]
    assert moves[1] == ("recorded", collector.insertion_pose_minus_z(reference, 0.01))
    assert moves[2] == ("outside", reference)
