"""FR3 kinematics: modified-DH model, forward kinematics, damped-least-squares
inverse kinematics. Pure numpy/scipy -- importable and testable without franky
or a robot.

The flange->EE transform is NOT assumed: `FR3Kinematics.calibrate()` derives
it from a live measured (q, EE pose) pair, so whatever end-effector the robot
is configured with is captured exactly (no F_T_EE convention assumptions).
Hardware-validated: sub-mm reach error across point-to-point, trajectory, and
servo execution.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa: N817

from evo_franka.constants import (
    IK_DLS_LAMBDA_MIN,
    IK_LIMIT_MARGIN_RAD,
    IK_MAX_ITER,
    IK_MAX_STEP_RAD,
    IK_NULLSPACE_GAIN,
    IK_ORI_TOL_RAD,
    IK_POS_TOL_M,
    JACOBIAN_EPS,
)

# FR3 geometry: modified-DH (Craig), one row per joint.
# Columns: a_{i-1} [m], d_i [m], alpha_{i-1} [rad].
DH_PARAMS = np.array([
    [0.0, 0.333, 0.0],
    [0.0, 0.0, -np.pi / 2],
    [0.0, 0.316, np.pi / 2],
    [0.0825, 0.0, np.pi / 2],
    [-0.0825, 0.384, -np.pi / 2],
    [0.0, 0.0, np.pi / 2],
    [0.088, 0.0, np.pi / 2],
])
FLANGE_D_M = 0.107

# FR3 joint limits (rad); IK keeps IK_LIMIT_MARGIN_RAD away from them.
Q_MIN = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
Q_MAX = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def _dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ])


def fk_flange(q: Sequence[float]) -> np.ndarray:
    """Base -> flange homogeneous transform for joint vector q (7,)."""
    T = np.eye(4)
    for i in range(7):
        a, d, alpha = DH_PARAMS[i]
        T = T @ _dh_transform(a, d, alpha, float(q[i]))
    return T @ _dh_transform(0.0, FLANGE_D_M, 0.0, 0.0)


class FR3Kinematics:
    def __init__(self) -> None:
        self.ee_offset: np.ndarray = np.eye(4)  # flange -> EE, set by calibrate()

    def calibrate(self, q: Sequence[float], ee_pos: Sequence[float],
                  ee_quat_xyzw: Sequence[float]) -> None:
        """Derive the flange->EE transform from a live measured (q, EE pose)."""
        T_ee = np.eye(4)
        T_ee[:3, :3] = R.from_quat(np.asarray(ee_quat_xyzw, dtype=float)).as_matrix()
        T_ee[:3, 3] = np.asarray(ee_pos, dtype=float)
        self.ee_offset = np.linalg.inv(fk_flange(np.asarray(q, dtype=float))) @ T_ee

    def fk(self, q: Sequence[float]) -> np.ndarray:
        """Base -> EE homogeneous transform."""
        return fk_flange(np.asarray(q, dtype=float)) @ self.ee_offset

    def jacobian(self, q: Sequence[float], eps: float = JACOBIAN_EPS) -> np.ndarray:
        """Geometric Jacobian (6x7): rows = [linear velocity; angular velocity],
        base frame, computed numerically from FK (~0.5 ms; fine for 30 Hz use)."""
        q = np.asarray(q, dtype=float)
        J = np.zeros((6, 7))
        T0 = self.fk(q)
        p0 = T0[:3, 3]
        R0 = R.from_matrix(T0[:3, :3])
        for i in range(7):
            dq = np.zeros(7)
            dq[i] = eps
            T1 = self.fk(q + dq)
            J[:3, i] = (T1[:3, 3] - p0) / eps
            J[3:, i] = (R.from_matrix(T1[:3, :3]) * R0.inv()).as_rotvec() / eps
        return J

    def _pose_error(self, T: np.ndarray, p_des: np.ndarray, R_des: np.ndarray) -> np.ndarray:
        e_p = p_des - T[:3, 3]
        e_o = R.from_matrix(R_des @ T[:3, :3].T).as_rotvec()
        return np.concatenate([e_p, e_o])

    def ik(self, p_des: Sequence[float], quat_des_xyzw: Sequence[float],
           q_init: Sequence[float], q_rest: Optional[Sequence[float]] = None,
           pos_tol: float = IK_POS_TOL_M, ori_tol: float = IK_ORI_TOL_RAD,
           max_iter: int = IK_MAX_ITER) -> np.ndarray:
        """Damped-least-squares IK seeded at q_init, with an optional nullspace
        pull toward q_rest. Tolerances default to 1 mm / ~0.3 deg (~1-3 ms per
        solve for nearby targets). Returns the joint vector or raises
        ValueError (likely unreachable)."""
        p_des = np.asarray(p_des, dtype=float)
        R_des = R.from_quat(np.asarray(quat_des_xyzw, dtype=float)).as_matrix()
        q = np.asarray(q_init, dtype=float).copy()
        for _ in range(max_iter):
            e = self._pose_error(self.fk(q), p_des, R_des)
            if np.linalg.norm(e[:3]) < pos_tol and np.linalg.norm(e[3:]) < ori_tol:
                return q
            J = self.jacobian(q)
            lam = IK_DLS_LAMBDA_MIN + float(np.linalg.norm(e))  # adaptive damping
            JJt = J @ J.T + (lam ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            if q_rest is not None:
                J_pinv = J.T @ np.linalg.inv(JJt)
                null_proj = np.eye(7) - J_pinv @ J
                dq = dq + null_proj @ (IK_NULLSPACE_GAIN * (np.asarray(q_rest, dtype=float) - q))
            dq = np.clip(dq, -IK_MAX_STEP_RAD, IK_MAX_STEP_RAD)
            q = np.clip(q + dq, Q_MIN + IK_LIMIT_MARGIN_RAD, Q_MAX - IK_LIMIT_MARGIN_RAD)
        e = self._pose_error(self.fk(q), p_des, R_des)
        raise ValueError(
            f"IK did not converge: pos_err={np.linalg.norm(e[:3]) * 1000:.2f}mm "
            f"ori_err={np.rad2deg(np.linalg.norm(e[3:])):.2f}deg {self._why_stuck(q)}"
        )

    @staticmethod
    def _why_stuck(q: np.ndarray) -> str:
        """WHICH joints the solve ended pinned against, or "target may be
        unreachable" when none are.

        The loop clips every step into [Q_MIN+margin, Q_MAX-margin], so a joint
        driven past its limit does not fail -- it STOPS, and the residual it was
        supposed to remove simply stays. `preflight._ik_reachable` already owns
        that reading ("a solution AT the raw limit means the clamp masked an
        unreachable target"), but only for solves that RETURN; a solve that raises
        said nothing about it, so the one message anybody sees named the one cause
        that was not the cause.

        The cost was a night. franka-right, 2026-08-26: `insert` aborted 124 ticks
        in with `pos_err=0.00mm ori_err=0.25deg (target may be unreachable)`, then
        0.32deg. A target 26 cm into a transit the arm had just driven is plainly
        reachable in position -- pos_err was ZERO -- and the hunt went to
        tolerances and iteration budgets. It was neither: with a joint on its clip
        the residual is immune to both (reproduced offline: identical 0.09 deg at
        300, 5 000 and 50 000 iterations, while the SAME target solves at the
        planning orientation tolerance). pos_err -> 0 with ori_err stuck is the
        signature, because six free joints have DOF to spare for three position
        constraints and none left for the last of the orientation.

        Naming the joint turns that into one line. It also says what will NOT
        help, because the two obvious knobs are both wrong here and both are
        reachable from this message."""
        lo, hi = Q_MIN + IK_LIMIT_MARGIN_RAD, Q_MAX - IK_LIMIT_MARGIN_RAD
        pinned = []
        for i in range(len(q)):
            if q[i] >= hi[i] - 1e-9:
                pinned.append(f"j{i + 1}={q[i]:.4f} at its MAX bound {hi[i]:.4f}")
            elif q[i] <= lo[i] + 1e-9:
                pinned.append(f"j{i + 1}={q[i]:.4f} at its MIN bound {lo[i]:.4f}")
        if not pinned:
            return "(target may be unreachable)"
        return (f"(JOINT LIMIT, not a solver shortfall: {', '.join(pinned)} -- the "
                f"clip stopped the joint the residual needed, so more iterations and "
                f"a looser tolerance both leave it exactly here; the target is "
                f"unreachable in ORIENTATION from this posture)")
