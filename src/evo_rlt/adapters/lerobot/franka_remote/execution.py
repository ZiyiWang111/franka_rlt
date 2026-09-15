"""Pure safety decisions used by the robot-host action executor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import ACTION_NAMES, ProtocolError


@dataclass(frozen=True)
class GripperDecision:
    command: str | None
    hold_arm: bool


class GripperStateMachine:
    """Turn absolute width predictions into debounced open/close commands."""

    def __init__(
        self,
        *,
        close_threshold_m: float = 0.040,
        open_threshold_m: float = 0.055,
        confirm_steps: int = 2,
    ) -> None:
        if not 0 <= close_threshold_m < open_threshold_m:
            raise ValueError("gripper thresholds must satisfy 0 <= close < open")
        if confirm_steps < 1:
            raise ValueError("confirm_steps must be positive")
        self.close_threshold_m = close_threshold_m
        self.open_threshold_m = open_threshold_m
        self.confirm_steps = confirm_steps
        self.current: str | None = None
        self.pending: str | None = None
        self.pending_count = 0

    def sync_from_observation(self, width_m: float, grasped: bool) -> None:
        if grasped or width_m <= self.close_threshold_m:
            observed = "closed"
        elif width_m >= self.open_threshold_m:
            observed = "open"
        else:
            return
        if observed != self.current:
            self.current = observed
            self.pending = None
            self.pending_count = 0

    def update(self, target_width_m: float) -> GripperDecision:
        if not np.isfinite(target_width_m):
            raise ProtocolError("gripper target is not finite")
        desired = None
        if target_width_m <= self.close_threshold_m:
            desired = "closed"
        elif target_width_m >= self.open_threshold_m:
            desired = "open"
        if desired is None or desired == self.current:
            self.pending = None
            self.pending_count = 0
            return GripperDecision(command=None, hold_arm=False)
        if desired != self.pending:
            self.pending = desired
            self.pending_count = 1
        else:
            self.pending_count += 1
        command = None
        if self.pending_count >= self.confirm_steps:
            command = "close" if desired == "closed" else "open"
        return GripperDecision(command=command, hold_arm=True)

    def mark_completed(self, command: str) -> None:
        if command not in {"open", "close"}:
            raise ValueError(command)
        self.current = "open" if command == "open" else "closed"
        self.pending = None
        self.pending_count = 0


def _limit_vector_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= limit or norm == 0.0:
        return vector
    return vector * (limit / norm)


class RelativeForceGuard:
    """Detect translational contact force relative to a stationary baseline."""

    def __init__(self, *, baseline_wrench: np.ndarray, threshold_n: float) -> None:
        baseline = np.asarray(baseline_wrench, dtype=np.float64)
        if baseline.shape != (6,) or not np.isfinite(baseline).all():
            raise ValueError(f"baseline wrench must be finite shape (6,), got {baseline.shape}")
        if threshold_n <= 0:
            raise ValueError("force threshold must be positive")
        self.baseline_wrench = baseline.copy()
        self.threshold_n = float(threshold_n)

    def check(self, wrench: np.ndarray) -> dict[str, object]:
        current = np.asarray(wrench, dtype=np.float64)
        if current.shape != (6,) or not np.isfinite(current).all():
            raise ProtocolError(f"wrench must be finite shape (6,), got {current.shape}")
        delta = current - self.baseline_wrench
        force_delta_n = float(np.linalg.norm(delta[:3]))
        return {
            "triggered": force_delta_n >= self.threshold_n,
            "threshold_n": self.threshold_n,
            "force_delta_n": force_delta_n,
            "baseline_wrench": self.baseline_wrench.tolist(),
            "current_wrench": current.tolist(),
            "delta_wrench": delta.tolist(),
        }


def clip_tcp_action(
    action: np.ndarray,
    *,
    max_translation_m: float,
    max_rotation_rad: float,
) -> dict[str, float]:
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ProtocolError(f"action must be finite shape (7,), got {values.shape}")
    clipped = values[:6].copy()
    for start, limit in ((0, max_translation_m), (3, max_rotation_rad)):
        clipped[start : start + 3] = _limit_vector_norm(clipped[start : start + 3], limit)
    return {name: float(clipped[index]) for index, name in enumerate(ACTION_NAMES[:6])}
