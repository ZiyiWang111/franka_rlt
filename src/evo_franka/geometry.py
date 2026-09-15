"""Pure geometric/representation helpers (no robot, no franky dependency)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa: N817

ROTVEC_PI_BAND_RAD = 0.2  # canonical_rotvec ambiguity band around pi


def pose_to_matrix(pose_vec: Sequence[float]) -> np.ndarray:
    """[x, y, z, rx, ry, rz] (axis-angle) -> 4x4 homogeneous matrix."""
    T = np.eye(4)
    T[:3, 3] = pose_vec[:3]
    T[:3, :3] = R.from_rotvec(pose_vec[3:6]).as_matrix()
    return T


def matrix_to_pose(T: np.ndarray) -> list[float]:
    """4x4 homogeneous matrix -> [x, y, z, rx, ry, rz] (axis-angle)."""
    return [*T[:3, 3].tolist(), *R.from_matrix(T[:3, :3]).as_rotvec().tolist()]


def affine_to_matrix(affine: object) -> np.ndarray:
    """franky Affine (translation + xyzw quaternion) -> 4x4 matrix."""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(affine.quaternion, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(affine.translation, dtype=float)
    return T


def twist_to_list(twist: object) -> list[float]:
    """franky twists/accelerations -> [linear(3); angular(3)] list.

    Accepts objects with .linear/.angular (3-vectors), plain 6-arrays, or
    None (-> zeros)."""
    if twist is None:
        return [0.0] * 6
    try:
        return np.asarray(twist, dtype=float).reshape(6).tolist()
    except Exception:
        return [
            *np.asarray(twist.linear, dtype=float).reshape(3).tolist(),
            *np.asarray(twist.angular, dtype=float).reshape(3).tolist(),
        ]


def canonical_rotvec(rotvec: Sequence[float], band: float = ROTVEC_PI_BAND_RAD) -> list[float]:
    """Resolve the rotation-vector sign ambiguity near pi.

    Tool-down orientations sit at ~pi rotation, where scipy's rotvec can flip
    sign between reads. Raw flips poison recorded state features and pose
    deltas. Within the band around pi, remap theta@axis -> (2pi-theta)@-axis
    (identical rotation) so the first significant component is non-negative.
    """
    v = np.asarray(rotvec, dtype=np.float64)
    theta = float(np.linalg.norm(v))
    if theta > np.pi - band:
        lead = v[0] if abs(v[0]) > 1e-6 else (v[1] if abs(v[1]) > 1e-6 else v[2])
        if lead < 0:
            v = -v * (2 * np.pi - theta) / theta
    return v.tolist()
