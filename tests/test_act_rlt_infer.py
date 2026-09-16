import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from act_rlt.infer import bounded_action, observation_frame, resolve_checkpoint


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


def test_invalid_action_is_rejected():
    with TestCase().assertRaises(ValueError):
        bounded_action([0, 0, float("nan"), 0, 0, 0, 0], 0.002, 0.02)
    with TestCase().assertRaises(ValueError):
        bounded_action([0] * 6, 0.002, 0.02)


if __name__ == "__main__":
    test_state_selection_and_rgb_conversion()
    test_checkpoint_resolution()
    test_action_limits_preserve_direction_and_hold_gripper()
    test_invalid_action_is_rejected()
    print("3 checks passed")
