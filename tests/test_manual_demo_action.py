import pytest

from evo_rlt.adapters.lerobot.record.loop import _build_manual_demo_action


TCP_ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz"]


def _fake_delta(_from_pose, _to_pose):
    return dict(zip(TCP_ACTION_NAMES, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3], strict=True))


def test_manual_demo_action_adds_next_gripper_width():
    action = _build_manual_demo_action(
        [*TCP_ACTION_NAMES, "gripper_target_width"],
        [0.0] * 6,
        [1.0] * 6,
        0.019,
        compute_delta_action_fn=_fake_delta,
    )

    assert list(action) == [*TCP_ACTION_NAMES, "gripper_target_width"]
    assert action["gripper_target_width"] == pytest.approx(0.019)
    assert action["dz"] == pytest.approx(3.0)


def test_manual_demo_final_action_keeps_gripper_width():
    action = _build_manual_demo_action(
        [*TCP_ACTION_NAMES, "gripper_target_width"],
        [0.0] * 6,
        None,
        0.019,
    )

    assert [action[name] for name in TCP_ACTION_NAMES] == [0.0] * 6
    assert action["gripper_target_width"] == pytest.approx(0.019)


def test_manual_demo_action_supports_legacy_six_dimensional_schema():
    action = _build_manual_demo_action(
        TCP_ACTION_NAMES,
        [0.0] * 6,
        [1.0] * 6,
        0.019,
        compute_delta_action_fn=_fake_delta,
    )

    assert list(action) == TCP_ACTION_NAMES
