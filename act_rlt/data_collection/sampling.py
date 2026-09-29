"""Pure sample-space geometry shared by the recorder and unit tests."""

from __future__ import annotations

import numpy as np

from evo_franka.geometry import euler_xyz_deg_to_rotvec


DEFAULT_INSERT_MINUS_Z_M = 0.01

SAMPLE_X_HALF_RANGE_M = 0.02
SAMPLE_Y_MIN_OFFSET_M = 0.015
SAMPLE_Y_MAX_OFFSET_M = 0.025
SAMPLE_Z_M = 0.402
SAMPLE_EULER_XYZ_DEG = (179.7, 1.4, 90.6)
SAMPLE_ROTATION_VECTOR = tuple(euler_xyz_deg_to_rotvec(SAMPLE_EULER_XYZ_DEG))


def apply_fixed_sample_orientation(pose: list[float]) -> list[float]:
    """Return a copy with the required TCP XYZ-Euler orientation as a rotvec."""
    if len(pose) != 6 or not np.isfinite(pose).all():
        raise ValueError(f"pose must be a finite 6D TCP pose, got {pose}")
    fixed = pose.copy()
    fixed[3:6] = SAMPLE_ROTATION_VECTOR
    return fixed


def sample_z_insertion_pose(
    reference: list[float],
    rng: np.random.Generator,
) -> list[float]:
    """Sample the fixed workspace for the default -Z insertion trajectory."""
    if len(reference) != 6 or not np.isfinite(reference).all():
        raise ValueError(f"reference must be a finite 6D TCP pose, got {reference}")
    sample = apply_fixed_sample_orientation(reference)
    sample[0] += float(rng.uniform(-SAMPLE_X_HALF_RANGE_M, SAMPLE_X_HALF_RANGE_M))
    sample[1] += float(rng.uniform(SAMPLE_Y_MIN_OFFSET_M, SAMPLE_Y_MAX_OFFSET_M))
    sample[2] = SAMPLE_Z_M
    return sample


def insertion_pose_minus_z(reference: list[float], minus_z: float) -> list[float]:
    """Return the fixed final pose, offset along robot-base -Z."""
    if len(reference) != 6 or not np.isfinite(reference).all():
        raise ValueError(f"reference must be a finite 6D TCP pose, got {reference}")
    target = reference.copy()
    target[2] -= minus_z
    return target
