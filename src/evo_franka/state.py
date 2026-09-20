"""Franka state concern (F3 split of driver.py).

Owns every observation read -- joints, TCP pose, estimated wrench -- through the
tolerant _read_with_fallback (last-good cache + bounded fallback counters), the
bounded high-rate F/T sampler, the kinematics queries (fk / ik / manipulability),
the software FT tare, and the motion-state flag (is_running / robot_mode). The
franky MOTION boundary (finish_move / _classify_move_fault / _safe_move) lives in
motion.py. Methods live on StateMixin (combined by robots/franka/driver.py)."""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa: N817

from evo_franka._franky import _FRANKY_EXC, _STATE_READ_EXC
from evo_franka.errors import FrankaSessionLost
from evo_franka.constants import (
    STATE_FALLBACK_LOG_ENV,
    STATE_READ_MAX_CONSECUTIVE_FALLBACKS,
)
from evo_franka.geometry import matrix_to_pose
from evo_franka.ft_sampler import FTSampler

logger = logging.getLogger(__name__)

# Optional per-event state-read fallback log (item 1 observability): set the env
# var (STATE_FALLBACK_LOG_ENV) to log EVERY drop+reuse, else only the per-episode
# COUNT the collector logs. OFF by default so normal runs stay quiet.
_LOG_STATE_FALLBACK = os.environ.get(STATE_FALLBACK_LOG_ENV, "").strip().lower() in ("1", "true", "yes", "on")


class StateMixin:
    """State-reading / kinematics-query half of FrankaArmController."""


    # ---------------- state reading ----------------

    def reset_fallback_counters(self) -> int:
        """Reset the state-read fallback counters and return the running total
        since the last reset (for per-episode observability). The collector
        calls this at episode boundaries to log the per-episode fallback rate."""
        total = self.state_fallbacks_total
        self.state_fallbacks_total = 0
        self._state_fallback_streak.clear()
        return total

    def _read_with_fallback(self, key: str, reader):
        """Read a franky state signal tolerantly (item 1).

        libfranka forbids a blocking state read while a control loop runs; franky
        returns cached state during an async motion but does a synchronous
        readOnce() when idle, which can throw NetworkException / "UDP receive:
        Timeout" on a dropped datagram. On such a throw we reuse the LAST good
        value for this signal and continue -- NO retry, NO sleep, NO blocking
        (the deoxys `try/except: keep self._state_buffer[-1]` pattern).

        Bounding: consecutive fallbacks for a signal are counted; once they exceed
        STATE_READ_MAX_CONSECUTIVE_FALLBACKS we stop silently masking a real
        fault -- attempt recover_from_errors() once and, if there is still no
        cached value or recovery is impossible, raise a clear error so we never
        record frozen state indefinitely. A successful read clears the streak.

        `reader` is a zero-arg callable that performs the actual franky read; it
        is invoked under the franky lock here."""
        self._require()
        try:
            # ONE read, NO retry, NO sleep (the deoxys pattern): a transient FCI UDP
            # drop falls straight to the last-good cache below, and the cold-first-read
            # case is covered structurally by connect() priming the cache. The old
            # 1-retry+50ms-sleep only multiplied a ~1.5 s-per-read UDP timeout on a
            # persistently dead stream toward the watchdog window.
            with self._lock:
                value = reader()
            self._state_cache[key] = value
            self._state_fallback_streak[key] = 0
            return value
        except (*_STATE_READ_EXC, RuntimeError) as e:  # a read fault -> cache fallback / raise
            streak = self._state_fallback_streak.get(key, 0) + 1
            self._state_fallback_streak[key] = streak
            self.state_fallbacks_total += 1
            cached = self._state_cache.get(key)
            if streak > STATE_READ_MAX_CONSECUTIVE_FALLBACKS or cached is None:
                # Too many consecutive fallbacks (frozen state) OR no value was
                # ever read: do not keep masking. Try a one-shot recovery, then
                # surface the failure clearly.
                logger.error(
                    "[franka] state read '%s' failed %d consecutive times "
                    "(or no cached value); attempting reconnect then "
                    "raising: %s", key, streak, e,
                )
                # A DROPPED FCI link can't be fixed by recover_from_errors -> try a
                # full reconnect (rate-limited). If it comes back, retry the read and
                # carry on silently instead of failing the whole run / looping forever.
                if self._attempt_reconnect():
                    try:
                        with self._lock:
                            value = reader()
                        self._state_cache[key] = value
                        self._state_fallback_streak[key] = 0
                        return value
                    except (*_STATE_READ_EXC, RuntimeError):  # reconnected but still no read -> fall through
                        pass
                try:
                    if self.robot is not None and self.robot.has_errors:
                        self.robot.recover_from_errors()
                except (*_FRANKY_EXC, RuntimeError):  # best-effort recover before we raise; never mask the raise
                    pass
                raise FrankaSessionLost(
                    f"Franka state read '{key}' failed {streak} consecutive times "
                    f"({type(e).__name__}: {e}); "
                    "no usable cached value. The FCI CONNECTED but its realtime state "
                    "stream is NOT delivering -- this is not a software-fixable transient. "
                    "Fix: restart the FCI (Desk: deactivate then re-activate FCI / re-enable "
                    "the robot), check no other FCI client is connected, and verify the "
                    "realtime NIC link; then retry."
                ) from e
            # Normal fallback: reuse the last good value. Per-event logging is
            # OPTIONAL (env STATE_FALLBACK_LOG_ENV, off by default) so normal runs
            # stay quiet -- the per-episode fallback COUNT (logged by the collector)
            # is the always-on "never silently fail" signal. When the env flag is
            # on, log EVERY drop+reuse so you see exactly when and which signal
            # dropped.
            if _LOG_STATE_FALLBACK:
                logger.warning(
                    "[franka] state read '%s' DROP #%d (%s: %s); reused last good value",
                    key, streak, type(e).__name__, e,
                )
            return cached

    def get_joint_angles(self) -> list[float]:
        """Measured joint positions (rad). Tolerant: reuses the last good value
        on a transient state-read failure (see _read_with_fallback)."""
        return self._read_with_fallback(
            "joints", lambda: [float(x) for x in self.robot.state.q]   # cached, drop-tolerant
        )

    def get_joint_speeds(self) -> list[float]:
        """Measured joint velocities (rad/s)."""
        self._require()
        with self._lock:
            return [float(v) for v in self.robot.state.dq]   # cached, drop-tolerant

    def get_tool_pose(self) -> list[float]:
        """Measured TCP pose [x, y, z, rx, ry, rz] in the working frame."""
        return self._base_to_new(self._tool_pose_base())

    def get_tool_force(self) -> list[float]:
        """Estimated external TCP wrench [fx, fy, fz, mx, my, mz] (working
        frame, O_F_ext_hat_K -- model-based estimate, no dedicated F/T sensor),
        minus the bias captured by set_ft_zero(). Tolerant: reuses the last good
        wrench on a transient state-read failure (see _read_with_fallback)."""
        # ONE wrench reader: get_tool_force_raw() does the tolerant base-frame read;
        # this just applies the software bias and the working-frame rotation.
        wrench = np.asarray(self.get_tool_force_raw(), dtype=float)
        return self._rotate_to_new((wrench - self._ft_bias).tolist())

    def get_tool_force_raw(self) -> list[float]:
        """RAW estimated external TCP wrench [fx, fy, fz, mx, my, mz] in the base
        frame (O_F_ext_hat_K, no set_ft_zero bias, no working-frame rotation).
        Tolerant (shares the 'wrench' fallback cache with get_tool_force). Used
        by the data collector, which records the raw base-frame wrench."""
        wrench = self._read_with_fallback(
            "wrench",
            lambda: np.asarray(self.robot.state.O_F_ext_hat_K, dtype=float).reshape(6),
        )
        return [float(x) for x in wrench]

    # --------------------------------------------------------------------- #
    # Bounded high-rate F/T sampler (force-VLA force_buf). Franka has NO real
    # F/T sensor -- this is the model-based O_F_ext_hat_K estimate, untared base
    # frame, matching the recorded state wrench (low-confidence proxy). Runs at a
    # MODEST rate: franky reads share a C++ mutex with motion, so a fast thread
    # would add lock contention; reads use the tolerant fallback and never block
    # the caller. Check achieved_rate_hz() -- it can collapse during motion.
    # --------------------------------------------------------------------- #
    def start_force_sampler(self, rate_hz: float = 120.0) -> None:
        if self._ft_sampler is None:
            self._ft_sampler = FTSampler(self.get_tool_force_raw, rate_hz=rate_hz)
        self._ft_sampler.start()

    def stop_force_sampler(self) -> None:
        if self._ft_sampler is not None:
            self._ft_sampler.stop()

    def get_force_buf(self) -> np.ndarray:
        return (self._ft_sampler.get_force_buf() if self._ft_sampler
                else np.zeros((4, 6), dtype=np.float32))

    def begin_force_episode(self) -> None:
        if self._ft_sampler is not None:
            self._ft_sampler.begin_episode()

    def end_force_episode(self):
        return (self._ft_sampler.end_episode() if self._ft_sampler
                else (np.zeros((0,)), np.zeros((0, 6), dtype=np.float32)))

    def is_running(self) -> bool:
        """Whether a motion is currently controlling the robot."""
        if self.robot is None:
            return False
        with self._lock:
            return bool(getattr(self.robot, "is_in_control", False))

    def robot_mode(self) -> str:
        """Current libfranka robot mode as a string (e.g. 'Idle', 'Move',
        'Guiding', 'UserStopped'). Routed through _read_with_fallback so a transient
        FCI UDP drop (NetworkException / RuntimeError 'Net Exception') retries and
        falls back to the last-known mode instead of crashing preflight."""
        def read_mode():
            state = self.robot.state
            # Reuse this existing read; diagnostics must never initiate I/O.
            self._diagnostic_state = {
                "ts": time.monotonic(),
                "wall_time": time.time(),
                "mode": str(state.robot_mode),
                "ccsr": float(state.control_command_success_rate),
                "current_errors": str(state.current_errors),
                "last_motion_errors": str(state.last_motion_errors),
                "has_errors": bool(state.current_errors),
            }
            return state.robot_mode
        return str(self._read_with_fallback("robot_mode", read_mode))

    def is_user_stopped(self) -> bool:
        """True when the user-stop / enabling device is engaged. Motion is
        rejected in this mode -- callers should check before commanding moves."""
        return "UserStopped" in self.robot_mode()

    # ---------------- kinematics ----------------

    def fk(self, joint_angles: Optional[Sequence[float]] = None) -> list[float]:
        """TCP pose [x, y, z, rx, ry, rz] for the given joints (current joints
        if None). Kinematics is calibrated at connect()."""
        self._require_kin()
        q = self.get_joint_angles() if joint_angles is None else list(joint_angles)
        return self._base_to_new(matrix_to_pose(self.kin.fk(q)))

    def ik(self, pose: Sequence[float],
           q_init: Optional[Sequence[float]] = None) -> Optional[list[float]]:
        """Joint solution for a TCP pose [x, y, z, rx, ry, rz], seeded at
        q_init (current joints if None). Returns None if no solution (the only
        method that signals failure by return value; see module docstring).

        SEED-ROBUST: the damped-least-squares solver is seed-sensitive near
        singularities -- a genuinely reachable pose can fail to converge from up to
        ~half of seeds (measured on this cell: 18/40 random seeds failed for a pose the
        arm was physically AT). So try the requested seed, then a few DETERMINISTIC
        branch-exploring perturbations (elbow/wrist offsets that escape the local DLS
        basin; no Math.random -> reproducible), and return the FIRST convergent
        solution. A reachable pose is no longer reported unreachable on one unlucky
        seed -- the bug behind the recurring 'IK did not converge' on reachable poses."""
        self._require_kin()
        base_pose = self._new_to_base(pose)
        quat = R.from_rotvec(np.asarray(base_pose[3:6], dtype=float)).as_quat()
        primary = np.asarray(self.get_joint_angles() if q_init is None else list(q_init), dtype=float)
        seeds = [primary,
                 primary + np.array([0.0, 0.3, 0.0, 0.5, 0.0, -0.5, 0.0]),
                 primary + np.array([0.0, -0.3, 0.0, -0.5, 0.0, 0.5, 0.0]),
                 # ABSOLUTE, well-conditioned fallbacks -- perturbing a bad primary seed
                 # cannot leave its basin (measured: relative-only retries 18->14/40),
                 # but a fixed central config does: the Franka "ready" pose + an
                 # alternate elbow/wrist branch.
                 np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
                 np.array([0.4, 0.2, -0.3, -1.4, 0.3, 1.6, 0.8])]
        last = None
        for seed in seeds:
            try:
                return self.kin.ik(base_pose[:3], quat, q_init=seed, q_rest=seed).tolist()
            except ValueError as e:
                last = e
        print(f"[franka] no IK solution for pose {list(pose)} after {len(seeds)} seeds: {last}",
              file=sys.stderr, flush=True)
        return None

    def manipulability(self, q: Optional[Sequence[float]] = None) -> float:
        """Yoshikawa manipulability sqrt(det(J Jᵀ)) at q (current joints if None) -- a
        scalar that → 0 at a kinematic singularity. Used at teach time to WARN when the
        contact pose sits in an ill-conditioned region (where the DLS IK is fragile and
        the recorded approach struggles to reach the target), so the operator learns it
        at setup instead of 50 episodes in."""
        self._require_kin()
        qv = np.asarray(self.get_joint_angles() if q is None else list(q), dtype=float)
        J = self.kin.jacobian(qv)
        return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))

    # ---------------- configuration ----------------

    def set_ft_zero(self) -> None:
        """Tare get_tool_force(): subsequent readings are relative to the
        current wrench (software bias; the estimate cannot be hardware-tared)."""
        self._require()
        try:
            self._ft_bias = np.asarray(self.robot.state.O_F_ext_hat_K, dtype=float).reshape(6)
        except (*_STATE_READ_EXC, RuntimeError) as e:  # a direct state read -> translate the franky
            raise FrankaSessionLost(                    # comms fault so it never leaks raw (F5)
                f"set_ft_zero state read failed ({type(e).__name__}): {e}") from e

    def _tool_pose_base(self) -> list[float]:
        """Measured TCP pose [x,y,z,rx,ry,rz] in the raw base frame. Tolerant:
        reuses the last good pose on a transient state-read failure (see
        _read_with_fallback). The cached value is the finished pose list."""
        def _read() -> list[float]:
            # cached state (drop-tolerant); current_cartesian_state does a synchronous
            # read that times out under USB-NIC contention (camera streaming)
            ee = self.robot.state.O_T_EE
            pos = np.asarray(ee.translation, dtype=float).reshape(3)
            # franky quaternion is xyzw (scalar-last), matching scipy default.
            rotvec = R.from_quat(np.asarray(ee.quaternion, dtype=float)).as_rotvec()
            return [*pos.tolist(), *rotvec.tolist()]

        return list(self._read_with_fallback("pose_base", _read))
