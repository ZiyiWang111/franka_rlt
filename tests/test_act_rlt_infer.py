import gc
import json
import numpy as np
from scipy.spatial.transform import Rotation
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from act_rlt.infer_gc import EpisodeGarbageCollection

from act_rlt.infer import (
    RESIDUAL_LIMIT_M,
    ResidualMotionAccumulator,
    TemporalActionEnsembler,
    bounded_action,
    bounded_action_with_triggers,
    build_parser,
    checkpoint_camera_shapes,
    centered_bounds,
    confirm_centered_bounds,
    move_to_next_sample,
    observation_frame,
    resolve_checkpoint,
    robot_camera_kwargs,
    run_inference_episode,
    sample_workspace_pose,
    validate_args,
)


def test_state_selection_and_rgb_conversion():
    obs = {f"joint_{i}": i / 10 for i in range(7)}
    obs.update(ee_x=99, joint_vel_0=99)
    obs["wrist"] = np.full((480, 640, 3), [255, 128, 0], dtype=np.uint8)
    obs["front"] = obs["wrist"].copy()
    frame = observation_frame(obs)
    np.testing.assert_allclose(frame["observation.state"], np.arange(7) / 10)
    assert frame["observation.images.wrist"].shape == (3, 480, 640)
    np.testing.assert_allclose(frame["observation.images.wrist"][:, 0, 0], [1, 128/255, 0])
    assert len(frame) == 3


def test_single_wrist_camera_uses_checkpoint_resolution_without_front():
    config = SimpleNamespace(
        input_features={
            "observation.state": SimpleNamespace(shape=(7,)),
            "observation.images.wrist": SimpleNamespace(shape=(3, 720, 1280)),
        },
        output_features={"action": SimpleNamespace(shape=(7,))},
    )
    cameras = checkpoint_camera_shapes(config)
    assert cameras == {"wrist": (720, 1280, 3)}
    kwargs = robot_camera_kwargs(
        cameras, SimpleNamespace(wrist_camera="wrist-serial", front_camera="front-serial")
    )
    assert kwargs["camera_width"] == 1280
    assert kwargs["camera_height"] == 720
    assert kwargs["enable_wrist_camera"] is True
    assert kwargs["enable_front_camera"] is False

    observation = {f"joint_{i}": 0.0 for i in range(7)}
    observation["wrist"] = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame = observation_frame(observation, cameras)
    assert set(frame) == {"observation.state", "observation.images.wrist"}
    assert frame["observation.images.wrist"].shape == (3, 720, 1280)


def test_checkpoint_camera_shapes_preserves_legacy_two_camera_contract():
    config = SimpleNamespace(
        input_features={
            "observation.state": SimpleNamespace(shape=(7,)),
            "observation.images.wrist": SimpleNamespace(shape=(3, 480, 640)),
            "observation.images.front": SimpleNamespace(shape=(3, 480, 640)),
        },
        output_features={"action": SimpleNamespace(shape=(7,))},
    )
    assert checkpoint_camera_shapes(config) == {
        "wrist": (480, 640, 3), "front": (480, 640, 3)
    }


def test_fps_cli_accepts_30_and_rejects_faster_camera_rate():
    parser = build_parser()
    valid = parser.parse_args(["--checkpoint", "model", "--fps", "30", "--dry-run"])
    validate_args(parser, valid)
    assert valid.fps == 30
    assert parser.parse_args(["--checkpoint", "model"]).fps == 30
    assert parser.parse_args(["--checkpoint", "model"]).residual_control is False
    assert parser.parse_args(
        ["--checkpoint", "model", "--residual-control"]
    ).residual_control is True
    invalid = parser.parse_args(["--checkpoint", "model", "--fps", "31", "--dry-run"])
    with TestCase().assertRaises(SystemExit):
        validate_args(parser, invalid)


def test_automatic_episodes_require_dry_run():
    parser = build_parser()
    args = parser.parse_args([
        "--checkpoint", "model", "--dry-run", "--dry-run-episodes", "8", "--profile-timing",
    ])
    validate_args(parser, args)
    assert args.dry_run_episodes == 8 and args.profile_timing
    for extra in (["--dry-run-episodes", "8"], ["--dry-run", "--dry-run-episodes", "0"]):
        with TestCase().assertRaises(SystemExit):
            validate_args(parser, parser.parse_args(["--checkpoint", "model", *extra]))


def test_motion_cli_uses_confirmed_center_instead_of_workspace_arguments():
    parser = build_parser()
    validate_args(parser, parser.parse_args(["--checkpoint", "model"]))
    with TestCase().assertRaises(SystemExit):
        parser.parse_args(["--checkpoint", "model", "--workspace-min", "0", "0", "0"])


def test_centered_xy_with_fixed_absolute_sample_and_workspace_z():
    pose = [0.4, -0.2, 0.3, 1, 2, 3]
    sample_min, sample_max, workspace_min, workspace_max = centered_bounds(pose)
    np.testing.assert_allclose(sample_min, [0.38, -0.22, 0.402])
    np.testing.assert_allclose(sample_max, [0.42, -0.18, 0.402])
    np.testing.assert_allclose(workspace_min, [-np.inf, -np.inf, 0.367])
    np.testing.assert_allclose(workspace_max, [np.inf, np.inf, 0.412])
    rng = np.random.default_rng(0)
    for _ in range(20):
        sampled = sample_workspace_pose(sample_min, sample_max, [1, 2, 3], 0.0, rng)
        assert np.all(sampled[:3] >= sample_min)
        assert np.all(sampled[:3] <= sample_max)
        assert sampled[2] == 0.402
        np.testing.assert_allclose(sampled[3:], [1, 2, 3])
    with TestCase().assertRaisesRegex(ValueError, "invalid measured TCP pose"):
        centered_bounds([0, 0, float("nan"), 0, 0, 0])


def test_next_sample_transition_lifts_measured_tcp_one_cm_in_base_z():
    calls = []
    current = [0.40, -0.20, 0.30, 1.0, 2.0, 3.0]

    def move_tool(pose, *, speed):
        calls.append((pose, speed))
        return pose

    robot = SimpleNamespace(
        robot=SimpleNamespace(get_tool_pose=lambda: current, move_tool=move_tool),
        resync_command_pose=lambda: None,
    )
    target = [0.42, -0.18, 0.305, 4.0, 5.0, 6.0]

    measured = move_to_next_sample(robot, target, 0.02)

    assert calls == [
        ([0.40, -0.20, 0.31, 1.0, 2.0, 3.0], 0.02),
        (target, 0.02),
    ]
    np.testing.assert_allclose(measured, target)


def test_confirm_centered_bounds_only_enables_limits_after_enter():
    config = SimpleNamespace(workspace_min_xyz=None, workspace_max_xyz=None)
    robot = SimpleNamespace(
        config=config,
        robot=SimpleNamespace(get_tool_pose=lambda: [0.4, -0.2, 0.3, 1, 2, 3]),
    )
    with patch("builtins.input", return_value="q"):
        assert confirm_centered_bounds(robot) is None
    assert config.workspace_min_xyz is None
    assert config.workspace_max_xyz is None

    with patch("builtins.input", return_value=""):
        sample_min, sample_max, orientation = confirm_centered_bounds(robot)
    np.testing.assert_allclose(sample_min, [0.38, -0.22, 0.402])
    np.testing.assert_allclose(sample_max, [0.42, -0.18, 0.402])
    expected_rotation = Rotation.from_euler("XYZ", [179.7, 1.4, 90.6], degrees=True)
    np.testing.assert_allclose(Rotation.from_rotvec(orientation).as_matrix(), expected_rotation.as_matrix())
    np.testing.assert_allclose(config.workspace_min_xyz, [-np.inf, -np.inf, 0.367])
    np.testing.assert_allclose(config.workspace_max_xyz, [np.inf, np.inf, 0.412])


def test_enter_captures_latest_xy_but_samples_use_fixed_collection_z_and_orientation():
    config = SimpleNamespace(workspace_min_xyz=None, workspace_max_xyz=None)
    poses = iter([
        [0.4, -0.2, 0.3, 1, 2, 3],  # preview before Enter
        [0.5, -0.1, 0.4, 4, 5, 6],  # measured when Enter is pressed
    ])
    robot = SimpleNamespace(config=config, robot=SimpleNamespace(get_tool_pose=lambda: next(poses)))
    with patch("builtins.input", return_value=""):
        sample_min, sample_max, orientation = confirm_centered_bounds(robot)
    np.testing.assert_allclose(sample_min, [0.48, -0.12, 0.402])
    np.testing.assert_allclose(sample_max, [0.52, -0.08, 0.402])
    expected_rotation = Rotation.from_euler("XYZ", [179.7, 1.4, 90.6], degrees=True)
    rng = np.random.default_rng(0)
    for _ in range(20):
        sampled = sample_workspace_pose(sample_min, sample_max, orientation, 0.0, rng)
        assert np.all(sampled[:2] >= sample_min[:2])
        assert np.all(sampled[:2] <= sample_max[:2])
        assert sampled[2] == 0.402
        np.testing.assert_allclose(
            Rotation.from_rotvec(sampled[3:]).as_matrix(), expected_rotation.as_matrix()
        )
    np.testing.assert_allclose(config.workspace_min_xyz, [-np.inf, -np.inf, 0.367])
    np.testing.assert_allclose(config.workspace_max_xyz, [np.inf, np.inf, 0.412])


def test_reset_orientation_is_not_a_cli_argument():
    parser = build_parser()
    with TestCase().assertRaises(SystemExit):
        parser.parse_args(["--checkpoint", "model", "--reset-orientation", "1", "2", "3"])


def test_checkpoint_resolution():
    with TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "checkpoints/last/pretrained_model"
        checkpoint.mkdir(parents=True)
        for name in ("config.json", "model.safetensors", "policy_preprocessor.json",
                     "policy_postprocessor.json"):
            (checkpoint / name).touch()
        assert resolve_checkpoint(tmp) == checkpoint


def test_action_limits_preserve_direction_and_hold_gripper():
    action = bounded_action([3, 4, 0, 0, 0, -2, 0.04], 0.002, 0.02)
    np.testing.assert_allclose(list(action.values()), [0.0012, 0.0016, 0, 0, 0, -0.02])
    assert "gripper_target_width" not in action


def test_action_limit_triggers_are_reported_independently():
    action, translation_triggered, rotation_triggered = bounded_action_with_triggers(
        [0.003, 0.004, 0, 0, 0, 0.01, 0.04], 0.002, 0.02
    )
    assert translation_triggered is True
    assert rotation_triggered is False
    np.testing.assert_allclose(list(action.values()), [0.0012, 0.0016, 0, 0, 0, 0.01])

    _, translation_triggered, rotation_triggered = bounded_action_with_triggers(
        [0.001, 0, 0, 0.03, 0.04, 0, 0.04], 0.002, 0.02
    )
    assert translation_triggered is False
    assert rotation_triggered is True


def test_invalid_action_is_rejected():
    with TestCase().assertRaises(ValueError):
        bounded_action([0, 0, float("nan"), 0, 0, 0, 0], 0.002, 0.02)
    with TestCase().assertRaises(ValueError):
        bounded_action([0] * 6, 0.002, 0.02)


def test_workspace_sample_has_optional_xy_padding_and_fixed_orientation():
    lower = np.array([0.3, -0.3, 0.1])
    upper = np.array([0.7, 0.2, 0.5])
    sampled = sample_workspace_pose(
        lower,
        upper,
        [1, 2, 3],
        0.03,
        np.random.default_rng(0),
    )
    assert np.all(sampled[:2] >= lower[:2] + 0.03)
    assert np.all(sampled[:2] <= upper[:2] - 0.03)
    assert lower[2] <= sampled[2] <= upper[2]
    np.testing.assert_allclose(sampled[3:], [1, 2, 3])


def test_workspace_sample_rejects_box_too_narrow_for_padding():
    with TestCase().assertRaisesRegex(ValueError, "must each exceed"):
        sample_workspace_pose(
            [0, 0, 0],
            [0.06, 0.2, 0.3],
            [1, 2, 3],
            0.03,
            np.random.default_rng(0),
        )


class _InferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeTorch:
    @staticmethod
    def inference_mode():
        return _InferenceMode()


class _FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _FakeRobot:
    def __init__(self):
        self.robot = self
        self.is_connected = True
        self.sent_at = []
        self.stop_calls = 0
        self.resync_calls = 0
        self.pose = np.zeros(6, dtype=float)
        self._cmd_pose = None

    def get_observation(self):
        time.sleep(0.002)
        return {"frame": len(self.sent_at)}

    def resync_command_pose(self):
        self.resync_calls += 1
        self._cmd_pose = None

    def get_tool_pose(self):
        return self.pose.tolist()

    def get_joint_angles(self):
        return [0.0] * 7

    def get_tool_force_raw(self):
        return [0.0] * 6

    def send_action(self, action, *, position_residual_xyz=None):
        translation = (
            np.array(list(action.values())[:3])
            if position_residual_xyz is None else np.asarray(position_residual_xyz)
        )
        self._cmd_pose = self.pose.copy()
        self._cmd_pose[:3] += translation
        self.pose = self._cmd_pose.copy()
        self.sent_at.append(time.monotonic())

    def stop_servo(self):
        self.stop_calls += 1
        return True


class _FakePolicy:
    def __init__(self, n_action_steps, slow_calls=None, chunk_size=None):
        self.config = SimpleNamespace(
            n_action_steps=n_action_steps,
            chunk_size=chunk_size or n_action_steps,
        )
        self.calls = 0
        self.slow_calls = slow_calls or {}

    def reset(self):
        pass

    def select_action(self, batch):
        self.calls += 1
        time.sleep(self.slow_calls.get(self.calls, 0))
        return _FakeTensor([0.001, 0, 0, 0, 0, 0, 0.04])

    def predict_action_chunk(self, batch):
        self.calls += 1
        time.sleep(self.slow_calls.get(self.calls, 0))
        values = np.zeros((1, self.config.chunk_size, 7), dtype=float)
        values[..., 0] = 0.001
        values[..., 6] = 0.04
        return _FakeTensor(values)


def _threaded_args(
    duration=0.32, dry_run=False, temporal_ensemble_coeff=None, fps=15,
    residual_control=False,
):
    return SimpleNamespace(
        duration=duration,
        dry_run=dry_run,
        max_step_m=0.002,
        max_step_rad=0.02,
        temporal_ensemble_coeff=temporal_ensemble_coeff,
        fps=fps,
        residual_control=residual_control,
    )


def _xyz_action(dx=0.0, dy=0.0, dz=0.0):
    return dict(dx=dx, dy=dy, dz=dz, drx=0.0, dry=0.0, drz=0.0)


def test_residual_accumulates_unexecuted_small_actions_and_subtracts_motion():
    controller = ResidualMotionAccumulator()
    measured = np.zeros(6)
    first, stalled = controller.step(measured, _xyz_action(dz=-0.000058), 0.0)
    second, _ = controller.step(measured, _xyz_action(dz=-0.000058), 1 / 30)
    np.testing.assert_allclose(first, [0, 0, -0.000058])
    np.testing.assert_allclose(second, [0, 0, -0.000116])
    assert not stalled

    measured[2] = -0.00008
    third, _ = controller.step(measured, _xyz_action(dz=-0.000058), 2 / 30)
    np.testing.assert_allclose(third, [0, 0, -0.000094])


def test_residual_clips_each_axis_to_three_tenths_millimetre():
    controller = ResidualMotionAccumulator()
    residual, _ = controller.step(
        np.zeros(6), _xyz_action(dx=0.002, dy=-0.002, dz=-0.002), 0.0
    )
    np.testing.assert_allclose(
        residual, [RESIDUAL_LIMIT_M, -RESIDUAL_LIMIT_M, -RESIDUAL_LIMIT_M]
    )


def test_z_stall_clears_downward_residual_and_latches_for_episode():
    controller = ResidualMotionAccumulator()
    measured = np.zeros(6)
    triggered = []
    for tick in range(10):
        residual, newly_stalled = controller.step(
            measured, _xyz_action(dz=-0.000058), tick / 30
        )
        triggered.append(newly_stalled)
    assert sum(triggered) == 1
    assert controller.z_stalled
    assert residual[2] == 0

    residual, newly_stalled = controller.step(
        measured, _xyz_action(dz=-0.000058), 10 / 30
    )
    assert residual[2] == 0 and not newly_stalled
    residual, _ = controller.step(measured, _xyz_action(dz=0.00012), 11 / 30)
    assert residual[2] > 0  # Retraction is still allowed.
    assert controller.z_stalled
    residual, _ = controller.step(measured, _xyz_action(dz=-0.000058), 12 / 30)
    assert residual[2] >= 0  # A later negative command cannot re-arm insertion.


def test_z_stall_does_not_trigger_when_measured_tcp_descends():
    controller = ResidualMotionAccumulator()
    for tick in range(15):
        measured = np.array([0, 0, -tick * 0.000058, 0, 0, 0])
        residual, triggered = controller.step(
            measured, _xyz_action(dz=-0.000058), tick / 30
        )
        assert not triggered
    assert not controller.z_stalled
    np.testing.assert_allclose(residual[2], -0.000058)


def test_z_stall_window_also_triggers_at_15_hz():
    controller = ResidualMotionAccumulator()
    triggered = []
    for tick in range(5):
        _, newly_stalled = controller.step(
            np.zeros(6), _xyz_action(dz=-0.000058), tick / 15
        )
        triggered.append(newly_stalled)
    assert triggered == [False, False, False, False, True]


def test_franka_residual_target_uses_offset_and_preserves_workspace_guard():
    from evo_rlt.adapters.lerobot.franka_robot.franka_robot import FrankaRobot

    sent = []
    hardware = SimpleNamespace(
        get_tool_pose=lambda: [0, 0, 0, 0, 0, 0],
        servo_tool=lambda pose: sent.append(pose),
    )
    robot = SimpleNamespace(
        robot=hardware,
        config=SimpleNamespace(
            workspace_min_xyz=(-0.001, -0.001, -0.001),
            workspace_max_xyz=(0.001, 0.001, 0.001),
        ),
        _cmd_pose=None,
    )
    action = _xyz_action(dz=-0.000058)
    FrankaRobot.send_action(
        robot, action, position_residual_xyz=[0, 0, -0.000116]
    )
    np.testing.assert_allclose(sent[-1][:3], [0, 0, -0.000116])
    FrankaRobot.send_action(robot, action)
    np.testing.assert_allclose(sent[-1][:3], [0, 0, -0.000174])

    robot._cmd_pose = None
    robot.config.workspace_min_xyz = (-0.001, -0.001, -0.0001)
    with TestCase().assertRaisesRegex(RuntimeError, "outside workspace"):
        FrankaRobot.send_action(
            robot, action, position_residual_xyz=[0, 0, -0.0002]
        )
    assert len(sent) == 2


def test_infer_workspace_allows_far_xy_but_rejects_z_outside_bounds():
    from evo_rlt.adapters.lerobot.franka_robot.franka_robot import FrankaRobot

    _, _, lower, upper = centered_bounds([0.4, -0.2, 0.402, 0, 0, 0])
    sent = []
    robot = SimpleNamespace(
        robot=SimpleNamespace(
            get_tool_pose=lambda: [0.8, 0.3, 0.402, 0, 0, 0],
            servo_tool=lambda pose: sent.append(pose),
        ),
        config=SimpleNamespace(workspace_min_xyz=tuple(lower), workspace_max_xyz=tuple(upper)),
        _cmd_pose=None,
    )
    FrankaRobot.send_action(robot, _xyz_action(dx=0.001, dy=0.001))
    np.testing.assert_allclose(sent[-1][:3], [0.801, 0.301, 0.402])
    robot.robot.get_tool_pose = lambda: [0.8, 0.3, 0.369, 0, 0, 0]
    robot._cmd_pose = None
    FrankaRobot.send_action(robot, _xyz_action())
    np.testing.assert_allclose(sent[-1][:3], [0.8, 0.3, 0.369])
    for z in (0.366, 0.413):
        robot.robot.get_tool_pose = lambda z=z: [0.8, 0.3, z, 0, 0, 0]
        robot._cmd_pose = None
        with TestCase().assertRaisesRegex(RuntimeError, "outside workspace"):
            FrankaRobot.send_action(robot, _xyz_action())
    assert len(sent) == 2


def test_temporal_ensemble_uses_paper_log_weights_for_same_timestep():
    ensembler = TemporalActionEnsembler(chunk_size=3, coefficient=np.log(2))
    oldest = np.zeros((3, 7), dtype=float)
    newest = np.zeros((3, 7), dtype=float)
    oldest[:, 0] = 0.001
    newest[:, 0] = 0.002
    ensembler.add_chunk(0, oldest)
    ensembler.add_chunk(1, newest)

    prepared, votes = ensembler.action_for(1, max_translation=1, max_rotation=1)

    assert votes == 2
    # log(w)=[0, -log(2)] -> normalized weights=[2/3, 1/3].
    np.testing.assert_allclose(prepared.values[0], 0.001 * 2 / 3 + 0.002 / 3)


def test_temporal_ensemble_pipeline_keeps_using_overlapping_chunks():
    robot = _FakeRobot()
    policy = _FakePolicy(n_action_steps=4, chunk_size=4, slow_calls={2: 0.08})

    with patch("act_rlt.infer.observation_frame", side_effect=lambda obs, shapes: obs):
        run_inference_episode(
            robot,
            policy,
            lambda frame: frame,
            lambda action: action,
            _threaded_args(temporal_ensemble_coeff=0.01),
            _FakeTorch,
        )

    assert len(robot.sent_at) >= 4
    assert robot.stop_calls == 1
    assert np.all(np.diff(robot.sent_at) < 0.09)


def test_full_trace_records_every_step_chunks_and_tcp_targets(tmp_path):
    robot = _FakeRobot()
    _, _, lower, upper = centered_bounds([0.4, -0.2, 0.402, 0, 0, 0])
    robot.config = SimpleNamespace(workspace_min_xyz=tuple(lower), workspace_max_xyz=tuple(upper))
    policy = _FakePolicy(n_action_steps=4, chunk_size=4)
    args = _threaded_args(duration=0.14, fps=30, temporal_ensemble_coeff=0.01)
    args.profile_timing = True
    args.log_dir = tmp_path
    args.checkpoint = Path("test-checkpoint")

    with patch("act_rlt.infer.observation_frame", side_effect=lambda obs, shapes: obs):
        run_inference_episode(
            robot, policy, lambda frame: frame, lambda action: action,
            args, _FakeTorch, episode_index=6,
        )

    paths = list(tmp_path.glob("*.jsonl"))
    assert len(paths) == 1
    events = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert events[0]["event"] == "episode_start"
    assert events[0]["episode_index"] == 6
    assert events[0]["workspace_min_xyz"] == [None, None, 0.367]
    np.testing.assert_allclose(events[0]["workspace_max_xyz"][2], 0.412)
    assert events[0]["workspace_max_xyz"][:2] == [None, None]
    assert events[-1]["event"] == "episode_end"
    chunks = [event for event in events if event["event"] == "model_chunk"]
    steps = [event for event in events if event["event"] == "control_step"]
    assert chunks and all(len(event["actions"]) == 4 for event in chunks)
    assert all("policy_first" in event["phase_timings"] for event in chunks)
    assert all(event["policy_cuda_elapsed_ms"] is None for event in chunks)
    assert len(steps) == len(robot.sent_at) == events[-1]["stats"]["ticks"]
    assert [event["step"] for event in steps] == list(range(len(steps)))
    assert all(event["ensemble_votes"] >= 1 for event in steps)
    assert all(len(event["measured_tcp_before"]) == 6 for event in steps)
    assert all(len(event["measured_joint_before"]) == 7 for event in steps)
    assert all(len(event["external_wrench_base_before"]) == 6 for event in steps)
    assert all(len(event["target_tcp_sent"]) == 6 for event in steps)
    assert all(len(event["action_unbounded"]) == 7 for event in steps)
    assert all(len(event["action_sent"]) == 6 for event in steps)
    assert events[0]["residual_control"] is False
    assert all(event["residual_xyz"] is None for event in steps)
    assert all(event["z_stall_latched"] is None for event in steps)
    assert robot.resync_calls >= len(steps)


def test_inference_stall_guard_clears_target_and_logs_latch(tmp_path):
    class StuckRobot(_FakeRobot):
        def send_action(self, action, *, position_residual_xyz=None):
            self._cmd_pose = self.pose.copy()
            self._cmd_pose[:3] += position_residual_xyz
            self.sent_at.append(time.monotonic())

    class DownPolicy(_FakePolicy):
        def select_action(self, batch):
            self.calls += 1
            return _FakeTensor([0, 0, -0.000058, 0, 0, 0, 0.04])

    robot = StuckRobot()
    args = _threaded_args(duration=0.42, fps=30, residual_control=True)
    args.log_dir = tmp_path
    policy = DownPolicy(n_action_steps=4)
    with patch("act_rlt.infer.observation_frame", side_effect=lambda obs, shapes: obs):
        run_inference_episode(
            robot, policy, lambda frame: frame, lambda action: action,
            args, _FakeTorch,
        )

    path, = tmp_path.glob("*.jsonl")
    events = [json.loads(line) for line in path.read_text().splitlines()]
    steps = [event for event in events if event["event"] == "control_step"]
    assert events[0]["residual_control"] is True
    assert events[-1]["stats"]["z_stall_events"] == 1
    assert sum(event["z_stall_triggered"] for event in steps) == 1
    assert min(event["target_tcp_sent"][2] for event in steps) >= -RESIDUAL_LIMIT_M
    latched = [event for event in steps if event["z_stall_latched"]]
    assert latched and all(event["target_tcp_sent"][2] == 0 for event in latched)


def test_three_worker_pipeline_hides_chunk_inference_latency():
    robot = _FakeRobot()
    policy = _FakePolicy(n_action_steps=4, slow_calls={1: 0.08, 5: 0.08})
    pre_calls = []

    with patch("act_rlt.infer.observation_frame", side_effect=lambda obs, shapes: obs):
        run_inference_episode(
            robot,
            policy,
            lambda frame: pre_calls.append(frame) or frame,
            lambda action: action,
            _threaded_args(),
            _FakeTorch,
        )

    assert len(robot.sent_at) >= 4
    assert robot.stop_calls == 1
    # Preprocessing happens once per generated chunk, not once per selected action.
    assert len(pre_calls) < policy.calls
    intervals = np.diff(robot.sent_at)
    assert np.all(intervals < 0.09), intervals


def test_action_queue_underrun_stops_instead_of_repeating_delta():
    robot = _FakeRobot()
    policy = _FakePolicy(n_action_steps=1, slow_calls={2: 0.15})

    with patch("act_rlt.infer.observation_frame", side_effect=lambda obs, shapes: obs):
        with TestCase().assertRaisesRegex(RuntimeError, "queue underrun"):
            run_inference_episode(
                robot,
                policy,
                lambda frame: frame,
                lambda action: action,
                _threaded_args(),
                _FakeTorch,
            )

    assert len(robot.sent_at) == 1
    assert robot.stop_calls == 1


def test_profiled_dry_run_identifies_gc_and_slow_phase_without_motion(tmp_path):
    callbacks_before = list(gc.callbacks)
    robot = _FakeRobot()
    args = _threaded_args(duration=0.32, dry_run=True, fps=30)
    args.profile_timing = True
    args.log_dir = tmp_path
    policy = _FakePolicy(n_action_steps=1, slow_calls={2: 0.12})

    def pre(frame):
        gc.collect(0)
        return frame

    with patch("act_rlt.infer.observation_frame", side_effect=lambda obs, shapes: obs):
        run_inference_episode(robot, policy, pre, lambda action: action, args, _FakeTorch)

    assert gc.callbacks == callbacks_before
    assert robot.sent_at == [] and robot.stop_calls == 0
    path, = tmp_path.glob("*.jsonl")
    events = [json.loads(line) for line in path.read_text().splitlines()]
    chunks = [row for row in events if row["event"] == "model_chunk"]
    slow = next(row for row in chunks if row["chunk_index"] == 1)
    assert slow["phase_timings"]["policy_first"]["wall_s"] >= 0.12
    assert slow["phase_timings"]["policy_first"]["thread_cpu_s"] < 0.08
    assert slow["policy_cuda_elapsed_ms"] is None
    assert any(row["event"] == "gc_pause" and row["inference_phase"] == "preprocess"
               for row in events)
    assert any(row["event"] == "chunk_enqueued" for row in events)
    assert any(row["event"] == "refill_requested" for row in events)
    assert any(row["event"] == "queue_underrun" for row in events)
    assert events[-1]["stats"]["queue_underruns"] > 0


def test_profile_gc_callback_is_removed_when_inference_fails(tmp_path):
    callbacks_before = list(gc.callbacks)
    args = _threaded_args()
    args.profile_timing = True
    args.log_dir = tmp_path
    with patch("act_rlt.infer.observation_frame", side_effect=ValueError("bad image")):
        with TestCase().assertRaisesRegex(ValueError, "bad image"):
            run_inference_episode(
                _FakeRobot(), _FakePolicy(4), lambda frame: frame, lambda action: action,
                args, _FakeTorch,
            )
    assert gc.callbacks == callbacks_before


def test_gc_runs_only_before_workers_and_after_servo_shutdown(tmp_path):
    robot = _FakeRobot()
    args = _threaded_args(duration=0.14, fps=30)
    args.profile_timing = True
    args.log_dir = tmp_path
    gc_was_enabled = gc.isenabled()
    boundary_states = []
    worker_states = []

    def collect(generation):
        assert generation == 2
        boundary_states.append((
            gc.isenabled(), robot.stop_calls,
            [t.name for t in threading.enumerate() if t.name.startswith("act-")],
        ))
        return 0

    def pre(frame):
        worker_states.append(gc.isenabled())
        return frame

    with patch("act_rlt.infer_gc.gc.collect", side_effect=collect):
        with patch("act_rlt.infer.observation_frame", side_effect=lambda obs, shapes: obs):
            run_inference_episode(robot, _FakePolicy(4), pre, lambda a: a, args, _FakeTorch)

    assert worker_states and not any(worker_states)
    assert boundary_states == [(False, 0, []), (False, 1, [])]
    assert gc.isenabled() == gc_was_enabled
    path, = tmp_path.glob("*.jsonl")
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[0]["gc_policy"] == "between_episodes"
    assert events[0]["gc_before_episode"]["rss_after_bytes"] > 0
    assert events[-1]["gc_after_episode"]["rss_after_bytes"] > 0


def test_gc_state_is_restored_after_worker_failure_even_if_previously_disabled():
    originally_enabled = gc.isenabled()
    try:
        for enabled in (True, False):
            (gc.enable if enabled else gc.disable)()
            with patch("act_rlt.infer_gc.gc.collect", return_value=0) as collect:
                with patch("act_rlt.infer.observation_frame", side_effect=ValueError("bad image")):
                    with TestCase().assertRaisesRegex(ValueError, "bad image"):
                        run_inference_episode(
                            _FakeRobot(), _FakePolicy(4), lambda f: f, lambda a: a,
                            _threaded_args(), _FakeTorch,
                        )
            assert collect.call_count == 2
            assert gc.isenabled() == enabled
    finally:
        (gc.enable if originally_enabled else gc.disable)()


def test_partial_thread_start_failure_stops_workers_and_restores_gc(tmp_path):
    robot = _FakeRobot()
    args = _threaded_args()
    args.profile_timing = True
    args.log_dir = tmp_path
    callbacks_before = list(gc.callbacks)
    gc_was_enabled = gc.isenabled()
    original_start = threading.Thread.start

    def start(thread):
        if thread.name == "act-inference":
            raise RuntimeError("thread startup failed")
        original_start(thread)

    with patch("act_rlt.infer_gc.gc.collect", return_value=0):
        with patch("act_rlt.infer.threading.Thread.start", new=start):
            with TestCase().assertRaisesRegex(RuntimeError, "thread startup failed"):
                run_inference_episode(
                    robot, _FakePolicy(4), lambda f: f, lambda a: a, args, _FakeTorch,
                )
    assert robot.stop_calls == 1
    assert not any(t.name.startswith("act-") for t in threading.enumerate())
    assert gc.isenabled() == gc_was_enabled
    assert gc.callbacks == callbacks_before


def test_gc_is_restored_when_boundary_collection_itself_fails():
    was_enabled = gc.isenabled()
    with patch("act_rlt.infer_gc.gc.collect", side_effect=RuntimeError("collection failed")):
        with TestCase().assertRaisesRegex(RuntimeError, "collection failed"):
            with EpisodeGarbageCollection():
                raise AssertionError("must not enter episode")
    assert gc.isenabled() == was_enabled
    with patch("act_rlt.infer_gc.gc.collect", side_effect=[0, RuntimeError("collection failed")]):
        with TestCase().assertRaisesRegex(RuntimeError, "collection failed"):
            with EpisodeGarbageCollection():
                assert not gc.isenabled()
    assert gc.isenabled() == was_enabled


def test_30_hz_servo_timing_with_prepared_action_chunk():
    robot = _FakeRobot()
    policy = _FakePolicy(n_action_steps=8)

    with patch("act_rlt.infer.observation_frame", side_effect=lambda obs, shapes: obs):
        run_inference_episode(
            robot, policy, lambda frame: frame, lambda action: action,
            _threaded_args(duration=0.14, fps=30), _FakeTorch,
        )

    assert len(robot.sent_at) >= 4
    assert robot.stop_calls == 1
    intervals = np.diff(robot.sent_at)
    assert np.all(np.abs(intervals - 1 / 30) < 0.025), intervals


def test_30_hz_servo_timing_with_temporal_ensemble():
    robot = _FakeRobot()
    policy = _FakePolicy(n_action_steps=4, chunk_size=16, slow_calls={2: 0.07})

    with patch("act_rlt.infer.observation_frame", side_effect=lambda obs, shapes: obs):
        run_inference_episode(
            robot, policy, lambda frame: frame, lambda action: action,
            _threaded_args(duration=0.14, fps=30, temporal_ensemble_coeff=0.01),
            _FakeTorch,
        )

    assert len(robot.sent_at) >= 4
    assert robot.stop_calls == 1
    intervals = np.diff(robot.sent_at)
    assert np.all(np.abs(intervals - 1 / 30) < 0.025), intervals


if __name__ == "__main__":
    test_state_selection_and_rgb_conversion()
    test_checkpoint_resolution()
    test_action_limits_preserve_direction_and_hold_gripper()
    test_invalid_action_is_rejected()
    print("3 checks passed")
