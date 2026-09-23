"""Pure sample-space geometry shared by the recorder and unit tests."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_X_HALF_RANGE_M = 0.01
DEFAULT_Y_RANGE_M = 0.02
DEFAULT_Z_HALF_RANGE_M = 0.01
DEFAULT_INSERT_MINUS_Y_M = 0.01
DEFAULT_Z_ROTATION_DEG = 5.0


def _apply_sampled_base_z_rotation(
    sample: list[float], rng: np.random.Generator, max_degrees: float
) -> None:
    """Left-compose a random base-frame Z yaw onto a 6D rotvec TCP pose."""
    if max_degrees < 0 or not np.isfinite(max_degrees):
        raise ValueError(f"max_degrees must be finite and non-negative, got {max_degrees}")
    if max_degrees == 0:
        return
    yaw_rad = float(rng.uniform(-max_degrees, max_degrees) * np.pi / 180.0)
    yaw = Rotation.from_rotvec([0.0, 0.0, yaw_rad])
    orientation = Rotation.from_rotvec(np.asarray(sample[3:6], dtype=float))
    sample[3:6] = (yaw * orientation).as_rotvec().tolist()


def sample_pose(
    reference: list[float],
    rng: np.random.Generator,
    *,
    x_half_range: float = DEFAULT_X_HALF_RANGE_M,
    y_range: float = DEFAULT_Y_RANGE_M,
    z_half_range: float = DEFAULT_Z_HALF_RANGE_M,
    z_rotation_degrees: float = 0.0,
) -> list[float]:
    """Uniformly sample XYZ and optionally rotate TCP about robot-base Z.

    The robot-base-frame bounds are x0 +/- x_half_range,
    y0 <= y <= y0 + y_range, and z0 +/- z_half_range.
    """
    if len(reference) != 6 or not np.isfinite(reference).all():
        raise ValueError(f"reference must be a finite 6D TCP pose, got {reference}")
    sample = reference.copy()
    sample[0] += float(rng.uniform(-x_half_range, x_half_range))
    sample[1] += float(rng.uniform(0.0, y_range))
    sample[2] += float(rng.uniform(-z_half_range, z_half_range))
    _apply_sampled_base_z_rotation(sample, rng, z_rotation_degrees)
    return sample


def insertion_pose(reference: list[float], minus_y: float) -> list[float]:
    """Return the fixed final pose, offset along robot-base -Y."""
    if len(reference) != 6 or not np.isfinite(reference).all():
        raise ValueError(f"reference must be a finite 6D TCP pose, got {reference}")
    target = reference.copy()
    target[1] -= minus_y
    return target


def sample_z_insertion_pose(
    reference: list[float],
    rng: np.random.Generator,
    *,
    x_half_range: float = DEFAULT_X_HALF_RANGE_M,
    y_half_range: float = DEFAULT_X_HALF_RANGE_M,
    z_range: float = DEFAULT_Y_RANGE_M,
    z_rotation_degrees: float = 0.0,
) -> list[float]:
    """Sample for a -Z insertion: X/Y symmetric and Z from p0 upward."""
    if len(reference) != 6 or not np.isfinite(reference).all():
        raise ValueError(f"reference must be a finite 6D TCP pose, got {reference}")
    sample = reference.copy()
    sample[0] += float(rng.uniform(-x_half_range, x_half_range))
    sample[1] += float(rng.uniform(-y_half_range, y_half_range))
    sample[2] += float(rng.uniform(0.0, z_range))
    _apply_sampled_base_z_rotation(sample, rng, z_rotation_degrees)
    return sample


def insertion_pose_minus_z(reference: list[float], minus_z: float) -> list[float]:
    """Return the fixed final pose, offset along robot-base -Z."""
    if len(reference) != 6 or not np.isfinite(reference).all():
        raise ValueError(f"reference must be a finite 6D TCP pose, got {reference}")
    target = reference.copy()
    target[2] -= minus_z
    return target
