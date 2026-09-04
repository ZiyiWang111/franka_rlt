"""6D delta-action math shared by the Franka manual-demo recorder and its
consistency-verification script.

The delta convention matches robotLab's ``BaseArmAdapter.get_action`` (used by
the FR3 control stack) and the forward integration in
``FrankaRobot.send_action``:

    action = (pos_cur - pos_prev, R_prev.inv() * R_cur  as rotvec)      # R2 = R1 * R_delta
    apply  = (pos_prev + action.pos, R_prev * R_action)

Units: position in **metres**, rotation as a body-frame axis-angle
(rotvec) in **radians**. A pose is ``[x, y, z, rx, ry, rz]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

_ACTION_KEYS = ["dx", "dy", "dz", "drx", "dry", "drz"]


def _as_vec(pose: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(pose, dtype=float).reshape(-1)[:6]


def _as_action_vec(
    action: dict[str, float] | Sequence[float] | np.ndarray,
) -> np.ndarray:
    if hasattr(action, "keys"):
        return np.asarray([action[k] for k in _ACTION_KEYS], dtype=float)
    return np.asarray(action, dtype=float).reshape(-1)[:6]


def compute_delta_action(
    pose_prev: Sequence[float] | np.ndarray,
    pose_cur: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Return the 6D delta ``{dx..drz}`` that carries ``pose_prev`` to ``pose_cur``.

    Matches robotLab ``BaseArmAdapter.get_action``: position is the plain
    difference; rotation is the body-frame delta ``R_prev.inv() * R_cur``.
    """
    s1 = _as_vec(pose_prev)
    s2 = _as_vec(pose_cur)

    delta_pos = s2[:3] - s1[:3]
    delta_rot = (Rotation.from_rotvec(s1[3:6]).inv() * Rotation.from_rotvec(s2[3:6])).as_rotvec()

    return {
        "dx": float(delta_pos[0]),
        "dy": float(delta_pos[1]),
        "dz": float(delta_pos[2]),
        "drx": float(delta_rot[0]),
        "dry": float(delta_rot[1]),
        "drz": float(delta_rot[2]),
    }


def apply_delta_action(
    pose: Sequence[float] | np.ndarray,
    action: dict[str, float] | Sequence[float] | np.ndarray,
) -> list[float]:
    """Apply a delta action to ``pose`` (the inverse of :func:`compute_delta_action`).

    ``R2 = R1 * R_action`` and ``pos2 = pos1 + action[:3]`` — the same forward
    convention used by ``FrankaRobot.send_action``.
    """
    s = _as_vec(pose)
    da = _as_action_vec(action)

    pos = s[:3] + da[:3]
    rot = (Rotation.from_rotvec(s[3:6]) * Rotation.from_rotvec(da[3:6])).as_rotvec()

    return [float(v) for v in np.concatenate([pos, rot])]
