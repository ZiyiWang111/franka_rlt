import numpy as np
import time
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from act_rlt.infer import (
    DEFAULT_RESET_XY_PADDING_M,
    TemporalActionEnsembler,
    TRAINING_TCP_ORIENTATION_MEAN_RAD,
    bounded_action,
    bounded_action_with_triggers,
    observation_frame,
    resolve_checkpoint,
    run_inference_episode,
    sample_workspace_pose,
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


def test_workspace_sample_has_three_centimetre_xy_padding_and_mean_orientation():
    lower = np.array([0.3, -0.3, 0.1])
    upper = np.array([0.7, 0.2, 0.5])
    sampled = sample_workspace_pose(
        lower,
        upper,
        TRAINING_TCP_ORIENTATION_MEAN_RAD,
        DEFAULT_RESET_XY_PADDING_M,
        np.random.default_rng(0),
    )
    assert np.all(sampled[:2] >= lower[:2] + 0.03)
    assert np.all(sampled[:2] <= upper[:2] - 0.03)
    assert lower[2] <= sampled[2] <= upper[2]
    np.testing.assert_allclose(sampled[3:], [-2.19841545, 2.22703212, -0.03082950])


def test_workspace_sample_rejects_box_too_narrow_for_padding():
    with TestCase().assertRaisesRegex(ValueError, "must each exceed"):
        sample_workspace_pose(
            [0, 0, 0],
            [0.06, 0.2, 0.3],
            TRAINING_TCP_ORIENTATION_MEAN_RAD,
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

    def get_observation(self):
        time.sleep(0.002)
        return {"frame": len(self.sent_at)}

    def resync_command_pose(self):
        pass

    def send_action(self, action):
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


def _threaded_args(duration=0.32, dry_run=False, temporal_ensemble_coeff=None):
    return SimpleNamespace(
        duration=duration,
        dry_run=dry_run,
        max_step_m=0.002,
        max_step_rad=0.02,
        temporal_ensemble_coeff=temporal_ensemble_coeff,
    )


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

    with patch("act_rlt.infer.observation_frame", side_effect=lambda obs: obs):
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


def test_three_worker_pipeline_hides_chunk_inference_latency():
    robot = _FakeRobot()
    policy = _FakePolicy(n_action_steps=4, slow_calls={1: 0.08, 5: 0.08})
    pre_calls = []

    with patch("act_rlt.infer.observation_frame", side_effect=lambda obs: obs):
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

    with patch("act_rlt.infer.observation_frame", side_effect=lambda obs: obs):
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


if __name__ == "__main__":
    test_state_selection_and_rgb_conversion()
    test_checkpoint_resolution()
    test_action_limits_preserve_direction_and_hold_gripper()
    test_invalid_action_is_rejected()
    print("3 checks passed")
