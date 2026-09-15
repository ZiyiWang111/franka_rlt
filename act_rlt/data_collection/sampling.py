"""Pure sample-space geometry shared by the recorder and unit tests."""

from __future__ import annotations

import numpy as np


DEFAULT_X_HALF_RANGE_M = 0.01
DEFAULT_Y_RANGE_M = 0.02
DEFAULT_Z_HALF_RANGE_M = 0.01
DEFAULT_INSERT_MINUS_Y_M = 0.01


def sample_pose(
    reference: list[float],
    rng: np.random.Generator,
    *,
    x_half_range: float = DEFAULT_X_HALF_RANGE_M,
    y_range: float = DEFAULT_Y_RANGE_M,
    z_half_range: float = DEFAULT_Z_HALF_RANGE_M,
) -> list[float]:
    """Uniformly sample XYZ and preserve the reference TCP orientation.

    The robot-base-frame bounds are x0 +/- x_half_range,
    y0 <= y <= y0 + y_range, and z0 +/- z_half_range.
    """
    if len(reference) != 6 or not np.isfinite(reference).all():
        raise ValueError(f"reference must be a finite 6D TCP pose, got {reference}")
    sample = reference.copy()
    sample[0] += float(rng.uniform(-x_half_range, x_half_range))
    sample[1] += float(rng.uniform(0.0, y_range))
    sample[2] += float(rng.uniform(-z_half_range, z_half_range))
    return sample


def insertion_pose(reference: list[float], minus_y: float) -> list[float]:
    """Return the fixed final pose, offset along robot-base -Y."""
    if len(reference) != 6 or not np.isfinite(reference).all():
        raise ValueError(f"reference must be a finite 6D TCP pose, got {reference}")
    target = reference.copy()
    target[1] -= minus_y
    return target
