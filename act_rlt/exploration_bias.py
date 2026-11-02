"""Physical XY and tool-local yaw offsets selected between training episodes."""

from __future__ import annotations

import numpy as np


BIAS_STEP_M = 11.71875e-6
REFERENCE_EPISODE_STEPS = 1536 / 9
BIAS_STEP_YAW_RAD = np.deg2rad(2.0) / REFERENCE_EPISODE_STEPS
MAX_BIAS_LEVEL = 5
DOWNWARD_BIAS_STEP_M = -0.00015
BIAS_KEYS = {"KEY_UP": (0, -1), "KEY_DOWN": (0, 1),
             "KEY_LEFT": (1, -1), "KEY_RIGHT": (1, 1),
             "KEY_N": (2, -1), "KEY_M": (2, 1)}


class EpisodeDownwardBias:
    """One-way, episode-local latch; owned by the control thread during motion."""

    def __init__(self):
        self.prepare()

    def prepare(self):
        self.enabled = False
        self.requested_z = 0.0
        self.applied_xyz = np.zeros(3)
        self.steps = 0

    def enable(self):
        if not self.enabled:
            self.enabled = True
            print(f"Downward bias enabled: -Z {abs(DOWNWARD_BIAS_STEP_M) * 1000:g} mm per policy step until episode ends.", flush=True)

    @property
    def step_m(self):
        return DOWNWARD_BIAS_STEP_M if self.enabled else 0.0

    def record(self, result):
        self.requested_z += float(result.info.get("downward_bias_requested_z_m", 0.0))
        self.applied_xyz += np.asarray(result.info.get("downward_bias_applied_xyz_m", [0., 0., 0.]))
        self.steps += int(result.info.get("downward_bias_steps", 0))
        if result.done:
            result.info["episode_downward_bias_requested_z_m"] = self.requested_z
            result.info["episode_downward_bias_applied_xyz_m"] = self.applied_xyz.tolist()
            result.info["episode_downward_bias_steps"] = self.steps
            self.enabled = False


class EpisodeExplorationBias:
    def __init__(self):
        self.available = False
        self.cancelled = False
        self.pending = np.zeros(3, dtype=int)  # X, Y, tool-local yaw
        self.active = np.zeros(3, dtype=int)
        self.requested = np.zeros(2)
        self.applied = np.zeros(2)
        self.requested_yaw = 0.0
        self.applied_yaw = 0.0

    def prepare(self, *, warmup: bool):
        self.available = not warmup
        self.cancelled = False
        self.pending.fill(0)
        self.active.fill(0)
        self.requested.fill(0)
        self.applied.fill(0)
        self.requested_yaw = self.applied_yaw = 0.0

    def adjust(self, key: str):
        if not self.available:
            print("Exploration bias is disabled during ACT warmup.", flush=True)
            return
        if key == "KEY_0":
            self.pending.fill(0)
        else:
            axis, sign = BIAS_KEYS[key]
            self.pending[axis] = np.clip(self.pending[axis] + sign, -MAX_BIAS_LEVEL, MAX_BIAS_LEVEL)
        self.describe("Next episode", self.pending)

    @staticmethod
    def describe(label, levels):
        per_step_um = levels[:2] * BIAS_STEP_M * 1e6
        estimate_mm = levels[:2] * BIAS_STEP_M * REFERENCE_EPISODE_STEPS * 1000
        yaw_step_deg = float(np.rad2deg(levels[2] * BIAS_STEP_YAW_RAD))
        print(f"{label} exploration bias: X={levels[0]:+d}, Y={levels[1]:+d}, yaw={levels[2]:+d}; "
              f"per-step XY={per_step_um.tolist()} um; "
              f"per-step yaw={yaw_step_deg:+.8f} deg; "
              f"estimated cumulative XY={estimate_mm.tolist()} mm "
              f"and yaw={yaw_step_deg * REFERENCE_EPISODE_STEPS:+g} deg "
              f"at {REFERENCE_EPISODE_STEPS:.2f} steps", flush=True)

    def begin(self):
        self.active = self.pending.copy()
        self.pending.fill(0)
        if self.available:
            self.describe("Active episode", self.active)

    @property
    def step_m(self):
        return np.zeros(2) if self.cancelled else self.active[:2] * BIAS_STEP_M

    @property
    def step_yaw_rad(self):
        return 0.0 if self.cancelled else float(self.active[2] * BIAS_STEP_YAW_RAD)

    def cancel(self):
        if not self.cancelled:
            self.cancelled = True
            print("XY/yaw directional bias cancelled for the remaining episode steps.", flush=True)

    def record(self, result):
        result.info["exploration_bias_levels_xy"] = self.active[:2].tolist()
        result.info["exploration_bias_per_step_xy_m"] = self.step_m.tolist()
        result.info["exploration_bias_level_yaw"] = int(self.active[2])
        result.info["exploration_bias_cancelled"] = self.cancelled
        result.info["exploration_bias_per_step_yaw_rad"] = self.step_yaw_rad
        requested = np.asarray(result.info.get("exploration_bias_requested_xy_m", [0., 0.]))
        applied = np.asarray(result.info.get("exploration_bias_applied_xy_m", [0., 0.]))
        self.requested += requested
        self.applied += applied
        self.requested_yaw += float(result.info.get("exploration_bias_requested_yaw_rad", 0.0))
        self.applied_yaw += float(result.info.get("exploration_bias_applied_yaw_rad", 0.0))
        if result.done:
            result.info["episode_exploration_bias_levels_xy"] = self.active[:2].tolist()
            result.info["episode_exploration_bias_requested_xy_m"] = self.requested.tolist()
            result.info["episode_exploration_bias_applied_xy_m"] = self.applied.tolist()
            result.info["episode_exploration_bias_level_yaw"] = int(self.active[2])
            result.info["episode_exploration_bias_requested_yaw_rad"] = self.requested_yaw
            result.info["episode_exploration_bias_applied_yaw_rad"] = self.applied_yaw
            if self.available:
                print("Episode exploration bias cumulative XY: "
                      f"requested={(self.requested * 1000).tolist()} mm; "
                      f"after safety clipping={(self.applied * 1000).tolist()} mm; "
                      f"yaw requested={np.rad2deg(self.requested_yaw):+.6f} deg, "
                      f"after safety clipping={np.rad2deg(self.applied_yaw):+.6f} deg; "
                      "bias cleared for next episode", flush=True)
            self.pending.fill(0)
            self.active.fill(0)
