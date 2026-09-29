from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from act_rlt.data_collection import record_sample_space as collector
from act_rlt.data_collection.sampling import insertion_pose_minus_z, sample_z_insertion_pose
from act_rlt.data_collection.record_sample_space import (
    COMMAND_ACTION_KEY,
    COMMAND_ACTION_NAMES,
    EpisodeRecorder,
    build_parser,
    normalize_lerobot_scalar_buffer,
    run_recorded_move,
    validate_args,
)


def test_continuous_collection_defaults_are_slow_and_include_settle_delays():
    args = build_parser().parse_args(["--dataset", "test/act_rlt", "--root", "/tmp/act-rlt"])

    validate_args(args)
    assert args.move_speed == pytest.approx(0.02)
    assert args.insertion_speed == pytest.approx(0.01)
    assert args.pre_episode_sleep == pytest.approx(1.0)
    assert args.post_episode_sleep == pytest.approx(1.0)
    assert args.fps == 15


@pytest.mark.parametrize("fps", [15, 30])
def test_collection_fps_accepts_the_supported_rates(fps):
    args = build_parser().parse_args([
        "--dataset", "test/act_rlt", "--root", "/tmp/act-rlt", "--fps", str(fps),
    ])
    validate_args(args)
    assert args.fps == fps


def test_collection_fps_rejects_unsupported_rate():
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "--dataset", "test/act_rlt", "--root", "/tmp/act-rlt", "--fps", "50",
        ])


def test_single_camera_collection_modes_select_only_one_camera():
    wrist = build_parser().parse_args([
        "--dataset", "test/act_rlt", "--root", "/tmp/act-rlt", "--wrist_only",
    ])
    validate_args(wrist)
    assert wrist.wrist_only is True
    assert wrist.front_only is False

    front = build_parser().parse_args([
        "--dataset", "test/act_rlt", "--root", "/tmp/act-rlt", "--front_only",
    ])
    validate_args(front)
    assert front.front_only is True
    assert front.wrist_only is False

    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "--dataset", "test/act_rlt", "--root", "/tmp/act-rlt",
            "--wrist_only", "--front_only",
        ])


def test_z_insertion_is_the_default_collection_mode():
    args = build_parser().parse_args(["--dataset", "test/act_rlt", "--root", "/tmp/act-rlt"])
    validate_args(args)
    assert args.task is None
    assert args.insert_minus_z == pytest.approx(0.01)

    custom_depth = build_parser().parse_args([
        "--dataset", "test/act_rlt", "--root", "/tmp/act-rlt",
        "--insert-minus-z", "0.015",
    ])
    validate_args(custom_depth)
    assert custom_depth.insert_minus_z == pytest.approx(0.015)


@pytest.mark.parametrize("removed_option", ["--z-insertion-mode", "--insert-minus-y"])
def test_legacy_collection_mode_options_are_rejected(removed_option):
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "--dataset", "test/act_rlt", "--root", "/tmp/act-rlt", removed_option,
        ])


def test_collection_returns_to_taught_orientation_before_insertion(monkeypatch):
    taught = [0.561755, -0.225792, 0.387321, 0.1, 0.2, 0.3]
    reference = taught.copy()
    moves = []
    saved = []
    robot = SimpleNamespace(
        is_connected=True,
        connect=lambda: None,
        get_observation=lambda: {},
        disconnect=lambda: None,
    )
    dataset = SimpleNamespace(save_episode=lambda: saved.append(True))
    recorder = SimpleNamespace(
        start=lambda: None,
        capture=lambda: None,
        clear_command=lambda: None,
        finish=lambda: 3,
        abort_monitor=None,
    )

    def move_outside(_robot, target, **kwargs):
        moves.append(("outside", target.copy()))
        return True

    def move_recorded(_recorder, target, **kwargs):
        moves.append(("recorded", target.copy()))

    monkeypatch.setattr(collector.sys, "argv", [
        "record_sample_space.py", "--dataset", "test/act_rlt", "--root", "/tmp/act-rlt",
        "--episodes", "1", "--seed", "42",
    ])
    monkeypatch.setattr(collector, "build_dataset_and_robot", lambda args: (dataset, robot, None, []))
    monkeypatch.setattr(collector, "teach_reference", lambda robot: reference)
    monkeypatch.setattr(collector, "EpisodeRecorder", lambda **kwargs: recorder)
    monkeypatch.setattr(collector, "VideoEncodingManager", lambda dataset: nullcontext())
    monkeypatch.setattr(collector, "EpisodeAbortMonitor", lambda key: nullcontext())
    monkeypatch.setattr(collector, "robot_is_executable", lambda robot: True)
    monkeypatch.setattr(collector, "nonrecorded_arm_move", move_outside)
    monkeypatch.setattr(collector, "run_recorded_move", move_recorded)
    monkeypatch.setattr(collector, "read_key_during_delay", lambda seconds: None)
    monkeypatch.setattr(collector, "normalize_lerobot_scalar_buffer", lambda dataset: None)

    assert collector.main() == 0
    assert reference == taught
    assert moves == [
        ("outside", sample_z_insertion_pose(taught, np.random.default_rng(42))),
        ("recorded", taught),
        ("recorded", insertion_pose_minus_z(taught, 0.01)),
        ("outside", taught),
    ]
    assert moves[0][1][3:] != taught[3:]
    assert saved == [True]


def test_non_execution_mode_returns_to_reference_teaching_once(monkeypatch, capsys):
    first_reference = [0.5, -0.2, 0.38, 0.1, 0.2, 0.3]
    second_reference = [0.51, -0.21, 0.39, 0.4, 0.5, 0.6]
    taught_references = iter([first_reference, second_reference])
    executable = iter([False, True])
    saved = []
    robot = SimpleNamespace(
        is_connected=True,
        connect=lambda: None,
        get_observation=lambda: {},
        disconnect=lambda: None,
    )
    dataset = SimpleNamespace(save_episode=lambda: saved.append(True))
    recorder = SimpleNamespace(
        start=lambda: None,
        capture=lambda: None,
        clear_command=lambda: None,
        finish=lambda: 3,
        abort_monitor=None,
    )

    monkeypatch.setattr(collector.sys, "argv", [
        "record_sample_space.py", "--dataset", "test/act_rlt", "--root", "/tmp/act-rlt",
        "--episodes", "1", "--seed", "42",
    ])
    monkeypatch.setattr(collector, "build_dataset_and_robot", lambda args: (dataset, robot, None, []))
    monkeypatch.setattr(collector, "teach_reference", lambda robot: next(taught_references))
    monkeypatch.setattr(collector, "EpisodeRecorder", lambda **kwargs: recorder)
    monkeypatch.setattr(collector, "VideoEncodingManager", lambda dataset: nullcontext())
    monkeypatch.setattr(collector, "EpisodeAbortMonitor", lambda key: nullcontext())
    monkeypatch.setattr(collector, "robot_is_executable", lambda robot: next(executable))
    monkeypatch.setattr(collector, "nonrecorded_arm_move", lambda *args, **kwargs: True)
    monkeypatch.setattr(collector, "run_recorded_move", lambda *args, **kwargs: None)
    monkeypatch.setattr(collector, "read_key_during_delay", lambda seconds: None)
    monkeypatch.setattr(collector, "normalize_lerobot_scalar_buffer", lambda dataset: None)

    assert collector.main() == 0
    assert saved == [True]
    output = capsys.readouterr().out
    assert "Waiting for Franka Execution mode" not in output
    assert output.count("REFERENCE p0:") == 2


def test_command_action_tracks_exact_waypoint_and_realigns_pending_frame():
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.command_action = {name: 0.0 for name in COMMAND_ACTION_NAMES}
    recorder.pending = ({"observation.state": "frame"}, [0.0] * 6, 0.04, {})
    target = [0.1, 0.2, 0.3, 1.0, 2.0, 3.0]

    recorder.set_command(target, 0.02)

    expected = dict(zip(COMMAND_ACTION_NAMES, [*target, 0.02], strict=True))
    assert recorder.command_action == expected
    assert recorder.pending[-1] == expected

    recorder.clear_command()
    assert recorder.command_action == {name: 0.0 for name in COMMAND_ACTION_NAMES}
    assert recorder.pending[-1] == recorder.command_action


def test_recorded_move_uses_tracked_trajectory_rpc_until_server_idle():
    class FakeClient:
        def __init__(self):
            self.calls = []
            self.running = iter([True, False])

        def move_tool_traj(self, path, *, is_async):
            self.calls.append((path, is_async))

        def is_running(self):
            return next(self.running)

        def _wait_idle(self, timeout):
            assert timeout == pytest.approx(1.0 / 15)
            return {"success": True}

        def stop_move(self):
            raise AssertionError("successful move must not be stopped")

    class FakeRobot:
        def __init__(self):
            self.robot = FakeClient()
            self.config = SimpleNamespace(enable_fci_keepalive=False)
            self.is_connected = True
            self.keepalive_stops = 0
            self.resyncs = 0

        def _stop_keepalive(self):
            self.keepalive_stops += 1

        def resync_command_pose(self):
            self.resyncs += 1

    class FakeRecorder:
        def __init__(self):
            self.robot = FakeRobot()
            self.fps = 15
            self.commands = []
            self.captures = 0

        def set_command(self, target, speed):
            self.commands.append((target, speed))

        def capture(self):
            self.captures += 1

    recorder = FakeRecorder()
    target = [0.1, 0.2, 0.3, 1.0, 2.0, 3.0]

    run_recorded_move(recorder, target, speed=0.02, label="test")

    assert recorder.robot.robot.calls == [([[*target, 0.02, 0.0, 0.0]], True)]
    assert recorder.commands == [(target, 0.02)]
    assert recorder.captures == 3
    assert recorder.robot.keepalive_stops == 1
    assert recorder.robot.resyncs == 1


def test_frame_stores_actual_and_command_actions_separately():
    actual_names = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper_target_width"]

    class FakeDataset:
        features = {
            "action": {"dtype": "float32", "shape": (7,), "names": actual_names},
            COMMAND_ACTION_KEY: {
                "dtype": "float32",
                "shape": (7,),
                "names": COMMAND_ACTION_NAMES,
            },
            "complementary_info.policy_action": {
                "dtype": "float32",
                "shape": (7,),
                "names": actual_names,
            },
        }

        def add_frame(self, frame):
            self.frame = frame

    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.dataset = FakeDataset()
    recorder.action_names = actual_names
    recorder.task = "test task"
    recorder.frames = 0
    actual = dict(zip(actual_names, range(7), strict=True))
    command = dict(zip(COMMAND_ACTION_NAMES, range(10, 17), strict=True))

    recorder._add_frame({}, actual, command)

    assert recorder.dataset.frame["action"].tolist() == list(range(7))
    assert recorder.dataset.frame[COMMAND_ACTION_KEY].tolist() == list(range(10, 17))


def test_scalar_buffer_normalization_leaves_vector_actions_unchanged():
    scalar_values = [np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)]
    command_values = [np.arange(7, dtype=np.float32), np.arange(7, dtype=np.float32) + 1]

    class Writer:
        episode_buffer = {
            "complementary_info.phase": scalar_values.copy(),
            COMMAND_ACTION_KEY: command_values.copy(),
        }

    class Dataset:
        writer = Writer()
        features = {
            "complementary_info.phase": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["phase"],
            },
            COMMAND_ACTION_KEY: {
                "dtype": "float32",
                "shape": (7,),
                "names": COMMAND_ACTION_NAMES,
            },
        }

    normalize_lerobot_scalar_buffer(Dataset())

    assert all(np.asarray(value).shape == () for value in Writer.episode_buffer["complementary_info.phase"])
    assert all(
        actual is expected
        for actual, expected in zip(
            Writer.episode_buffer[COMMAND_ACTION_KEY], command_values, strict=True
        )
    )
