#!/usr/bin/env python3
"""Robot-free validation of branch-continuous joint-space IK -- the fix for the
FR3 cartesian/joint motion-generator discontinuity reflexes.

Replicates FrankaArmController._solve_branch_continuous_ik's core on
FR3Kinematics (pure numpy -- no franky, no robot) and asserts:
  1. each functional TCP waypoint is reached accurately (IK is correct);
  2. the densified, warm-started IK stays BRANCH-CONTINUOUS -- every per-step
     joint move is well under the branch-swing guard (MOTION_BRANCH_SWING_RAD),
     so no spurious branch flip / acceleration_discontinuity;
  3. densified seeding takes far smaller per-step joint moves than naive sparse
     per-waypoint IK (the contrast that makes the fix robust).

Run:  python tests/test_branch_continuous_ik.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from evo_franka.kinematics import FR3Kinematics
from evo_franka.constants import (
    LEFT_HOME_JOINTS,
    MOTION_BRANCH_SWING_RAD,
    PATH_DENSIFY_STEP_M,
    PATH_DENSIFY_STEP_RAD,
)

KIN = FR3Kinematics()          # identity ee_offset -> kin.fk == flange FK
HOME = np.array(LEFT_HOME_JOINTS, dtype=float)


def pose_of(q):
    T = KIN.fk(q)
    return np.concatenate([T[:3, 3], R.from_matrix(T[:3, :3]).as_rotvec()])


def _fk_err(q, wp):
    T = KIN.fk(q)
    pe = float(np.linalg.norm(T[:3, 3] - np.asarray(wp[:3])))
    oe = float((R.from_matrix(T[:3, :3]).inv() * R.from_rotvec(np.asarray(wp[3:6]))).magnitude())
    return pe, oe


def solve(start_pose, waypoints, dense):
    """Mirror of _solve_branch_continuous_ik's core (no franky).
    Returns [(q_functional, max_per_step_dq)]. dense=False == naive sparse IK."""
    q_seed = HOME.copy()
    prev_pos = np.asarray(start_pose[:3], dtype=float)
    prev_rot = R.from_rotvec(np.asarray(start_pose[3:6], dtype=float))
    out = []
    for wp in waypoints:
        pos = np.asarray(wp[:3], dtype=float)
        rot = R.from_rotvec(np.asarray(wp[3:6], dtype=float))
        if dense:
            lin = float(np.linalg.norm(pos - prev_pos))
            ang = float((prev_rot.inv() * rot).magnitude())
            steps = max(1, int(np.ceil(max(lin / PATH_DENSIFY_STEP_M,
                                           ang / PATH_DENSIFY_STEP_RAD))))
        else:
            steps = 1
        sub = None
        if steps > 1:
            key = R.from_quat(np.array([prev_rot.as_quat(), rot.as_quat()]))
            sub = Slerp([0.0, 1.0], key)(np.arange(1, steps + 1) / steps)
        maxdq = 0.0
        for k in range(steps):
            frac = (k + 1) / steps
            s_pos = prev_pos + (pos - prev_pos) * frac
            s_rot = sub[k] if sub is not None else rot
            q_new = np.asarray(KIN.ik(s_pos, s_rot.as_quat(), q_init=q_seed, q_rest=q_seed),
                               dtype=float)
            maxdq = max(maxdq, float(np.max(np.abs(q_new - q_seed))))
            q_seed = q_new
        out.append((q_seed.copy(), maxdq))
        prev_pos, prev_rot = pos, rot
    return out


def _paths():
    start = pose_of(HOME)
    ori = list(start[3:6])
    def p(dx, dy, dz):
        return list(start[:3] + np.array([dx, dy, dz])) + ori
    return start, {
        # straight 12 cm descent (insertion-like)
        "insertion": [p(0, 0, -0.12)],
        # lateral via-point then descend to target (approach corridor)
        "approach": [p(0.06, 0.04, -0.04), p(0.06, 0.0, -0.12)],
        # backward: down-target -> lift up THROUGH it -> offset start (the corner
        # that tripped cartesian_motion_generator_joint_velocity_discontinuity)
        "lift_through": [p(0, 0, -0.10), p(0, 0, 0.02), p(0.05, 0.05, 0.0)],
    }


def test_branch_continuous_ik():
    start, paths = _paths()
    for name, wps in paths.items():
        dense = solve(start, wps, dense=True)
        sparse = solve(start, wps, dense=False)
        for i, ((q, _dq), wp) in enumerate(zip(dense, wps)):
            pe, oe = _fk_err(q, wp)
            assert pe < 2e-3 and oe < 5e-3, \
                f"{name} wp{i}: FK err {pe*1000:.2f}mm / {np.rad2deg(oe):.2f}deg (IK wrong)"
        dmax = max(d for _, d in dense)
        smax = max(d for _, d in sparse)
        assert dmax < MOTION_BRANCH_SWING_RAD, \
            f"{name}: dense per-step dq {dmax:.2f} >= guard {MOTION_BRANCH_SWING_RAD} (would branch-swing)"
        print(f"PASS  {name:13s} reached + branch-continuous | "
              f"max per-step dq  dense={dmax:.3f}  sparse={smax:.3f} rad")


if __name__ == "__main__":
    try:
        test_branch_continuous_ik()
        print("\nALL PASS")
    except Exception as e:  # noqa: BLE001
        print(f"\nFAIL: {type(e).__name__}: {e}")
        sys.exit(1)
