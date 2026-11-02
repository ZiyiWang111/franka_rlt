import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from act_rlt import train_stage2 as train
from lerobot.processor import NormalizerProcessorStep


BASE = ["--stage1-checkpoint", "/tmp/stage1", "--act-checkpoint", "/tmp/act"]


@pytest.mark.parametrize("extra,expected_fps", [([], 30.), (["--fps", "15"], 15.)])
def test_training_default_needs_no_mode_or_workspace_options(extra, expected_fps):
    parser = train.build_parser()
    args = parser.parse_args(BASE + ["--allow-motion", *extra])
    train.validate_args(parser, args)
    assert args.fps == expected_fps
    assert not hasattr(args, "randomize_reset_xy")
    assert not hasattr(args, "z_insertion_mode")
    assert "--z-insertion-mode" not in parser.format_help()


@pytest.mark.parametrize("option", ["--z-insertion-mode", "--randomize-reset-xy"])
def test_removed_mode_option_is_rejected(option):
    with pytest.raises(SystemExit):
        train.build_parser().parse_args(BASE + ["--dry-run", option])


@pytest.mark.parametrize("option", ["--workspace-min", "--workspace-max", "--sample-min", "--sample-max"])
def test_legacy_bounds_cannot_silently_override_default_geometry(option):
    parser = train.build_parser()
    args = parser.parse_args(BASE + ["--allow-motion", option, ".1", ".2", ".3"])
    with pytest.raises(SystemExit):
        train.validate_args(parser, args)


@pytest.mark.parametrize("half_range", [0., .01, .025])
@pytest.mark.parametrize("enforce_workspace", [True, False])
def test_environment_defaults_to_fixed_height_orientation_and_positive_z_retreat(half_range, enforce_workspace):
    reference = np.array([.55, -.18, .405, .1, .2, .3])
    span = .05 if enforce_workspace else .001
    lower, upper = reference[:3] - span, reference[:3] + span
    robot = SimpleNamespace(
        config=SimpleNamespace(workspace_min_xyz=lower, workspace_max_xyz=upper),
        robot=SimpleNamespace(get_tool_pose=lambda: reference.copy()),
    )
    env = train.FrankaInsertionStage2Env(
        robot=robot,
        preprocessor=SimpleNamespace(steps=[object.__new__(NormalizerProcessorStep)]),
        postprocessor=None, config=SimpleNamespace(), fps=30., episode_time_s=5.,
        max_step_m=.002, max_step_rad=.02, reference_pose=reference,
        reset_xy_half_range_m=half_range,
        enforce_workspace=enforce_workspace,
    )
    moves, prompts = [], []
    env._move_reset_pose = lambda target, **kwargs: moves.append(np.asarray(target).copy())
    env._read_idle_reset_choice = lambda prompt: prompts.append(prompt) or ""
    env._move_to_accepted_workspace_sample()
    assert len(moves) == 1
    assert len(prompts) == 2 and all("adjust next reset range" in prompt for prompt in prompts)
    assert (np.abs(moves[0][:2] - reference[:2]) <= half_range + 1e-12).all()
    assert moves[0][2] == train.SAMPLE_Z_M
    np.testing.assert_allclose(moves[0][3:], train.SAMPLE_ROTATION_VECTOR)
    env._retreat_after_outcome()
    np.testing.assert_allclose(moves[1], reference + [0., 0., .01, 0., 0., 0.])


@pytest.mark.parametrize("scenario", ["fresh", "resume", "resume-disabled", "dry-run"])
@pytest.mark.parametrize("three_chunk_td", [False, True])
def test_training_main_captures_or_restores_workspace_without_mode_flag(monkeypatch, tmp_path, scenario, three_chunk_td):
    from evo_rlt.adapters.lerobot import franka_robot
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import factory

    output = tmp_path / "run"
    reference = np.array([.55, -.18, .405, .1, .2, .3])
    lower, upper = reference[:3] - .05, reference[:3] + .05
    saved = {
        "version": 1, "mode": "z_insertion", "frame": "base", "position_unit": "m",
        "orientation": "rotation_vector_rad", "reference_pose": reference.tolist(),
        "workspace_min_xyz": lower.tolist(), "workspace_max_xyz": upper.tolist(),
        "reset_xy_half_range_m": .025,
    }
    if scenario == "resume-disabled":
        saved["workspace_min_xyz"] = saved["workspace_max_xyz"] = None
    events, robots, envs, checkpoints = [], [], [], []
    act_config = SimpleNamespace()
    monkeypatch.setattr(train, "resolve_pretrained_model", lambda path: Path(path))
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", lambda *args, **kwargs: act_config)
    monkeypatch.setattr(train, "checkpoint_state_mode", lambda config: "tcp_z")
    monkeypatch.setattr(train, "checkpoint_camera_shapes", lambda config: {})
    monkeypatch.setattr(train, "robot_camera_kwargs", lambda *args: {})
    monkeypatch.setattr(factory, "make_pre_post_processors", lambda *args, **kwargs: (None, None))
    policy = SimpleNamespace(stage1=SimpleNamespace(_act=SimpleNamespace(config=act_config)), act_action_dim=6)
    policy.to = lambda device: policy
    monkeypatch.setattr(train, "ACTStage2Policy", lambda config: policy)
    monkeypatch.setattr(train, "Stage2Learner", lambda *args: SimpleNamespace(configure_action_projection=lambda post: None))
    monkeypatch.setattr("act_rlt.human_input.validate_human_input", lambda *args: events.append("keyboard"))
    monkeypatch.setattr("builtins.input", lambda prompt: events.append("capture") or "")
    monkeypatch.setattr(train, "save_checkpoint", lambda *args, **kwargs: checkpoints.append(kwargs["startup_reference"]))

    class Robot:
        def __init__(self, config):
            self.config, self.is_connected = config, False
            self.robot = SimpleNamespace(get_tool_pose=lambda: reference.copy(), stop_move=lambda: None)
            robots.append(self)

        def enable_timestamped_observations(self):
            pass

        def connect(self):
            events.append("connect")
            self.is_connected = True

        def disconnect(self):
            self.is_connected = False

    monkeypatch.setattr(franka_robot, "FrankaRobot", Robot)

    def environment(**kwargs):
        envs.append(kwargs)
        return SimpleNamespace(reset_xy_half_range_m=kwargs["reset_xy_half_range_m"],
                               _reference_pose=kwargs["reference_pose"].copy(), close=lambda: None)

    monkeypatch.setattr(train, "FrankaInsertionStage2Env", environment)
    monkeypatch.setattr(train, "run_online_stage2", lambda *args, **kwargs: kwargs["initial_metrics"])
    extra = ["--dry-run"] if scenario == "dry-run" else ["--allow-motion"]
    if three_chunk_td:
        extra += ["--three-chunk-td"]
    if scenario.startswith("resume"):
        (output / "checkpoints").mkdir(parents=True)
        (output / "checkpoints/latest.pt").touch()

        def load(*args, startup_metadata, **kwargs):
            startup_metadata["startup_reference"] = saved
            return train.OnlineStage2Metrics()

        monkeypatch.setattr(train, "load_checkpoint", load)
        extra += ["--resume"]
    monkeypatch.setattr("sys.argv", ["train_stage2", *BASE, "--output", str(output), "--device", "cpu", *extra])
    assert train.main() == 0
    if scenario == "dry-run":
        assert not robots and not envs and not checkpoints and not events
        assert not output.exists()
        return
    assert events.index("keyboard") < events.index("connect")
    assert events.count("capture") == (scenario == "fresh")
    assert not robots[0].is_connected
    assert robots[0].config.workspace_min_xyz is None
    assert robots[0].config.workspace_max_xyz is None
    assert envs[0]["enforce_workspace"] is False
    assert envs[0]["fps"] == 30.
    assert envs[0]["config"].td_chunks == (3 if three_chunk_td else 1)
    assert "z_insertion_mode" not in envs[0]
    assert envs[0]["reset_xy_half_range_m"] == (.025 if scenario.startswith("resume") else 0.)
    np.testing.assert_allclose(envs[0]["reference_pose"], reference)
    metadata = json.loads((output / "startup_reference.json").read_text())
    assert metadata["mode"] == "z_insertion"
    assert metadata["workspace_enabled"] is False
    assert metadata["workspace_min_xyz"] is None and metadata["workspace_max_xyz"] is None
    assert metadata["reset_z_m"] == .402
    assert metadata["initial_reference"]["reference_pose"] == reference.tolist()
    assert checkpoints and all(value["mode"] == "z_insertion" for value in checkpoints)
