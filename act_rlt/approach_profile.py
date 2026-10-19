"""Finite reference profiles for the soft-approach collection script."""
from __future__ import annotations

import math

import numpy as np


def stopping_distance(speed, acceleration, jerk):
    """Symmetric jerk-limited stop distance, starting with zero acceleration."""
    speed = max(0.0, float(speed))
    if speed <= acceleration * acceleration / jerk:
        return speed * math.sqrt(speed / jerk)
    return 0.5 * speed * (speed / acceleration + acceleration / jerk)


def cruise_speed(distance, speed_limit, acceleration, jerk):
    # Reserve roughly half of the initial chord for accelerating and braking.
    # On these short moves, trying to attain 40 mm/s would leave no cruise.
    low, high = 0.0, speed_limit
    for _ in range(40):
        middle = (low + high) * 0.5
        if stopping_distance(middle, acceleration, jerk) <= distance * 0.25:
            low = middle
        else:
            high = middle
    return low


def rest_duration(distance, speed, acceleration, jerk):
    """Duration bounding derivatives of the quintic rest-to-rest polynomial."""
    return max(1.875 * distance / speed,
               math.sqrt((10 / math.sqrt(3)) * distance / acceleration),
               (60 * distance / jerk) ** (1 / 3))


class FiniteStop:
    """Match the current position/velocity/acceleration and stop at a finite time.

    Candidate polynomials are checked for derivative limits and monotonic
    coordinates. This is a script reference, not a replacement for native OTG.
    """

    def __init__(self, coefficients, duration, target):
        self.coefficients = coefficients
        self.duration = float(duration)
        self.target = target.copy()

    @classmethod
    def plan(cls, position, velocity, acceleration, target, args):
        delta = target - position
        # Search a bounded family once at the braking transition. A 2% margin
        # covers extrema between the sampling points used for feasibility.
        maximum = max(3.0, 3 * math.sqrt(float(np.linalg.norm(delta)) / args.acceleration)
                      + 2 * float(np.linalg.norm(velocity)) / args.acceleration)
        durations = np.linspace(0.08, maximum, 240)
        duration = durations[:, None]
        vt, at2 = velocity * duration, acceleration * duration**2
        coefficients = np.stack([
            np.broadcast_to(position, vt.shape), vt, at2 * 0.5,
            10 * delta - 6 * vt - 1.5 * at2,
            -15 * delta + 8 * vt + 1.5 * at2,
            6 * delta - 3 * vt - 0.5 * at2,
        ], axis=1)
        s = np.linspace(0, 1, 81)
        derivatives = []
        values = coefficients
        for order in range(1, 4):
            values = values[:, 1:, :] * np.arange(1, values.shape[1])[None, :, None]
            powers = s[:, None] ** np.arange(values.shape[1])[None, :]
            derivatives.append(np.einsum("sk,nkj->nsj", powers, values)
                               / durations[:, None, None]**order)
        limits = (args.speed, args.acceleration, args.jerk)
        feasible = np.ones(len(durations), dtype=bool)
        for values, limit in zip(derivatives, limits):
            feasible &= np.max(np.linalg.norm(values, axis=2), axis=1) <= limit * 0.98
        # No reversal, overshoot, or positional clamp at the end of the stop.
        feasible &= np.all(derivatives[0] * np.sign(delta)[None, None, :] >= -1e-10, axis=(1, 2))
        indices = np.flatnonzero(feasible)
        if not len(indices):
            return None
        index = indices[0]
        return cls(coefficients[index], durations[index], target)

    def sample(self, elapsed):
        if elapsed >= self.duration:
            return self.target.copy(), np.zeros(3), np.zeros(3)
        s = max(0.0, elapsed / self.duration)
        values = self.coefficients
        result = []
        for order in range(3):
            result.append(np.polynomial.polynomial.polyval(s, values) / self.duration**order)
            values = values[1:] * np.arange(1, len(values))[:, None]
        return tuple(result)
