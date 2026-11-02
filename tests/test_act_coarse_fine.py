from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from act_rlt.infer_coarse_fine import check_fine_entry


@pytest.mark.parametrize("timeout", [False, True])
def test_fine_entry_waits_for_post_stop_state_before_allowing_motion(timeout):
    pose = [.6, -.1, .40187, 0., 0., 0.]
    controller = SimpleNamespace(
        get_fresh_tool_pose=Mock(side_effect=RuntimeError("stale cache")),
        get_tool_pose_after_move=Mock(
            return_value=pose,
            side_effect=RuntimeError("post-stop state timeout") if timeout else None,
        ),
        move_tool=Mock(),
    )
    robot = SimpleNamespace(robot=controller, resync_command_pose=Mock())
    args = SimpleNamespace(
        fine_entry_rotvec_rad=[0., 0., 0.], fine_entry_z_m=.40187,
        fine_entry_z_tolerance_m=.0002, fine_entry_angle_tolerance_deg=1.,
        fine_entry_max_adjustments=1,
    )
    emit = Mock()
    if timeout:
        with pytest.raises(RuntimeError, match="post-stop state timeout"):
            check_fine_entry(robot, args, emit)
        emit.assert_not_called()
        robot.resync_command_pose.assert_not_called()
    else:
        np.testing.assert_array_equal(check_fine_entry(robot, args, emit), pose)
        assert emit.call_args.kwargs["passed"] is True
    controller.get_tool_pose_after_move.assert_called_once_with(timeout_s=2.0)
    controller.get_fresh_tool_pose.assert_not_called()
    controller.move_tool.assert_not_called()
