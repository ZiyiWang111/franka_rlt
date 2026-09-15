"""Franka session concern (F3 split of driver.py).

Owns the FCI session lifecycle: connect (Robot + optional Hand, dynamics +
impedance + kinematics calibration), disconnect, the SINGLE recovery routine
(heal_session -> recover-in-place else reset_session), the idle-path reconnect,
readiness polling, the connection guards, and the working-frame transforms.
Methods live on SessionMixin; robots/franka/driver.py combines the mixins into
FrankaArmController, which owns the shared instance state built in __init__."""
from __future__ import annotations

import logging
import os as _os
import sys
import threading
import time as _time
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa: N817

from evo_franka._franky import (
    Gripper,
    RelativeDynamicsFactor,
    Robot,
    _FRANKY_EXC,
    _STATE_READ_EXC,
)
from evo_franka.errors import FrankaSessionLost, wrap_franky_exc
from evo_franka.constants import (
    ACCELERATION_DYNAMICS_FACTOR,
    CARTESIAN_IMPEDANCE,
    CONNECT_ATTEMPTS,
    CONNECT_RETRY_DELAY_S,
    DEFAULT_DYNAMICS_FACTOR,
    DEFAULT_ROBOT_IP,
    JERK_DYNAMICS_FACTOR,
    RECOVER_READY_POLL_S,
    RECOVER_READY_TIMEOUT_S,
)
from evo_franka.geometry import matrix_to_pose, pose_to_matrix
from evo_franka.kinematics import FR3Kinematics

logger = logging.getLogger(__name__)


class SessionMixin:
    """Connection / session-recovery / frames half of FrankaArmController."""

    def __init__(self, robot_ip: str = DEFAULT_ROBOT_IP,
                 relative_dynamics_factor: float = DEFAULT_DYNAMICS_FACTOR,
                 install_rad: float = 0.0, use_gripper: bool = True) -> None:
        """
        Args:
            robot_ip: FCI IP (172.16.0.2 = left arm on this laptop).
            relative_dynamics_factor: 0-1 scale on the robot's velocity /
                acceleration / jerk limits (lower = slower and safer).
            install_rad: yaw of the working frame about the base z-axis;
                poses/twists/wrenches in and out are expressed in this frame
                (0.0 = base frame).
            use_gripper: deployment flag. False -> do NOT open the Hand channel
                at connect, so has_gripper() is False and gripper-conditional
                callers run motion-only (franka-right, whose controller stalls
                every Hand command ~60 s under FCI).
        """
        self.robot_ip = robot_ip
        self.use_gripper = bool(use_gripper)
        self.relative_dynamics_factor = float(relative_dynamics_factor)
        self.robot: Optional[Robot] = None
        self.kin: Optional[FR3Kinematics] = None
        self.gripper: Optional[Gripper] = None  # set at connect when a Hand responds
        self._ft_bias: np.ndarray = np.zeros(6)
        self._servo_q: Optional[np.ndarray] = None  # persistent servo IK seed
        self._servo_q_rest: Optional[np.ndarray] = None  # servo nullspace anchor
        # Async-move hang deadline (F4): _safe_move arms it when it ISSUES an async
        # move; check_move_hang() (the async twin of _safe_move's blocking join cap)
        # fires past it. So the MOTION_JOIN_TIMEOUT_S cap lives in ONE place (the
        # driver's motion boundary) for both blocking and async issue -- the control
        # server no longer re-implements its own cap. None = no async move armed.
        self._async_move_deadline: Optional[float] = None
        # Serializes every entry into the franky Robot from Python. The recording
        # loop reads state (current_joint_state / O_F_ext_hat_K / ...) on the main
        # thread while a scripted stage runs a motion on a worker thread; franky's
        # state accessors and a synchronous move() share one C++ mutex, so without
        # this guard a state read can block for the entire motion (and a wedged
        # motion blocks it forever -- the episode-2 hang). Reentrant so a public
        # method can call another under the same lock.
        self._lock = threading.RLock()
        self._ft_sampler: "FTSampler | None" = None  # noqa: F821 -- quoted forward-ref (FTSampler), not evaluated at runtime; bounded high-rate F/T (force-VLA)
        # --- State-read tolerance (item 1): last-good cache + fallback counters.
        # Keyed by signal name ("joints" / "pose_base" / "wrench"); each holds the
        # last successfully read value. On a read exception we reuse it (no retry,
        # no sleep -- deoxys pattern). _state_fallback_streak tracks CONSECUTIVE
        # fallbacks per signal so the bound (STATE_READ_MAX_CONSECUTIVE_FALLBACKS)
        # can trip; state_fallbacks_total is a session/episode-resettable counter
        # for observability.
        self._state_cache: dict[str, object] = {}
        self._state_fallback_streak: dict[str, int] = {}
        self.state_fallbacks_total: int = 0
        self._last_reconnect_ts: float = 0.0    # rate-limit auto-reconnect on a dead FCI stream
        self.reconnects_total: int = 0
        self._prepare_installation(float(install_rad))

    # ---------------- connection ----------------

    def connect(self, attempts: int = CONNECT_ATTEMPTS) -> None:
        """Open the FCI session. Requires (Desk): brakes open, user-stop
        released, Execution mode, FCI active, no other FCI process attached.
        The Hand gripper is connected automatically when present."""
        self.robot = None  # never keep a stale handle through a retry
        robot = None
        last: Optional[Exception] = None
        # franky/libfranka (realtime enforce) elevates the CONSTRUCTING thread to
        # SCHED_FIFO prio 99 -- process-wide RT contamination: every thread python
        # spawns afterwards inherits FIFO:99, and librealsense's USB threads then
        # starve UVC transactions on this RT kernel (camera "No device connected"
        # ONLY when FCI is operational -- root-caused 2026-07-09 on host-009).
        # franky's own internal threads are spawned DURING construction and keep
        # the inherited RT priority, so confining the side effect afterwards does
        # not touch control-loop scheduling: restore the calling thread and let RT
        # live exactly where it belongs.
        _sched_ok = hasattr(_os, "sched_getscheduler")   # Linux-only API; cells are Linux
        if _sched_ok:
            _pol, _par = _os.sched_getscheduler(0), _os.sched_getparam(0)
        try:
            for _ in range(attempts):
                try:
                    robot = Robot(self.robot_ip)
                    break
                except (*_FRANKY_EXC, RuntimeError, OSError) as e:  # transient ProtocolException on a busy link
                    last = e
                    _time.sleep(CONNECT_RETRY_DELAY_S)
        finally:
            if _sched_ok and _os.sched_getscheduler(0) != _pol:
                _os.sched_setscheduler(0, _pol, _par)
                print("[franka] confined franky's RT elevation to its own threads "
                      "(main thread restored to normal scheduling)", flush=True)
        if robot is None:
            raise FrankaSessionLost(f"failed to connect to Franka at {self.robot_ip}: {last}")
        robot.recover_from_errors()
        self._apply_dynamics_factor(robot)
        robot.set_cartesian_impedance(list(CARTESIAN_IMPEDANCE))
        self.robot = robot
        if not getattr(self, "use_gripper", True):
            self.gripper = None
            print("[franka] gripper DISABLED for this cell (deployment use_gripper=false) "
                  "-- running without a Hand", flush=True)
        else:
            self._connect_hand()
        self._post_connect(robot)

    def _connect_hand(self) -> None:
        """Open the Hand channel. OPTIONAL: a cell can run without it."""
        try:
            self.gripper = Gripper(self.robot_ip)
            _ = float(self.gripper.width)  # probe the connection
            # A rebuilt Hand handle must never inherit a Future/state belonging
            # to the dead connection it replaced.
            for name in ("_gripper_future", "_gripper_motion", "_gripper_motion_seq"):
                if hasattr(self, name):
                    delattr(self, name)
        except (*_FRANKY_EXC, RuntimeError, OSError) as e:  # the Hand is OPTIONAL -> run without it
            self.gripper = None
            print(f"[franka] gripper unavailable ({type(e).__name__}: {str(e)[:80]}); "
                  "gripper methods will raise", file=sys.stderr, flush=True)

    def _post_connect(self, robot: Robot) -> None:
        """Kinematics calibration + state-cache priming, after the channels are up."""
        # Kinematics is UNCONDITIONAL: cartesian targets always execute as IK +
        # joint motions on this cell (the raw cartesian-pose path was dead -- it
        # reflexed on long moves), so there is one motion path. Calibrate the tool
        # offset from the FIRST realtime state read.
        self.kin = FR3Kinematics()
        # FIRST realtime state read. When a LEFTOVER process still holds the ONE FCI
        # session (a stale control_server, a collect, another panel's teleop), Robot()
        # above can construct but the realtime stream never delivers a datagram here --
        # franky raises NetworkException "UDP receive: Timeout" (root-caused on host-009,
        # 2026-07-09). Classify it honestly into the typed FrankaSessionLost with a hint
        # that names the likely holder, instead of leaking a raw franky exception. Only
        # the network-read family (NOT ControlException -- a reflex is a real fault);
        # no retry beyond connect()'s existing attempt loop.
        try:
            ee = robot.state.O_T_EE   # cached state, robust to USB-contention read timeouts
        except _STATE_READ_EXC as e:
            raise FrankaSessionLost(
                "FCI link timeout -- another process may hold the robot connection "
                "(control_server / collect / another panel's teleop?)") from e
        self.kin.calibrate(
            self.get_joint_angles(),   # also caches "joints" in _state_cache
            np.asarray(ee.translation, dtype=float),
            np.asarray(ee.quaternion, dtype=float),
        )
        # Prime the last-good state cache so the FIRST hot-path read (the 30 Hz
        # loop) is served from cache if its synchronous readOnce times out on a
        # still-settling FCI link -- the cold-first-read incident, covered
        # STRUCTURALLY here instead of by a per-read retry+sleep (see state.py).
        self._tool_pose_base()      # caches "pose_base"
        self.get_tool_force_raw()   # caches "wrench"
        print(f"[franka] connected {self.robot_ip}; mode={robot.state.robot_mode}")

    def _apply_dynamics_factor(self, robot: Robot) -> None:
        """Set the robot's dynamics scaling, decoupling JERK from velocity.

        franky scales the rated velocity/acceleration/jerk limits by a
        RelativeDynamicsFactor. The FR3 acceleration_discontinuity reflex is
        jerk-driven, so we keep jerk a low sub-rated fraction
        (JERK_DYNAMICS_FACTOR) independent of the velocity fraction (the
        caller-supplied self.relative_dynamics_factor) -- the Polymetis/deoxys/
        SERL consensus fix for the reflex (see constants.py). The velocity
        fraction is preserved so motion speed is unchanged.

        Graceful fallback: if this franky build lacks the 3-arg
        RelativeDynamicsFactor(velocity, acceleration, jerk) form, fall back to
        the uniform scalar (jerk then couples to velocity, as before).

        NOTE (item 4 -- limit_rate): libfranka has a `limit_rate` control flag
        (a per-command rate/discontinuity backstop, OFF by default on
        libfranka 0.10.0 / FR3). This franky build does NOT expose it (verified:
        absent from the Python API, the .pyi stubs, and the compiled _franky.so
        symbol table), and franky also does not expose the per-command
        cutoff_frequency. We therefore rely on franky's Ruckig online trajectory
        generator (which already produces jerk-limited, continuous setpoints) PLUS
        the low JERK_DYNAMICS_FACTOR above as the discontinuity backstop, rather
        than faking a flag that isn't there."""
        v = float(self.relative_dynamics_factor)
        try:
            rdf = RelativeDynamicsFactor(
                v, float(ACCELERATION_DYNAMICS_FACTOR), float(JERK_DYNAMICS_FACTOR)
            )
            robot.relative_dynamics_factor = rdf
            print(f"[franka] dynamics factor: velocity={v:.3f} "
                  f"acceleration={ACCELERATION_DYNAMICS_FACTOR:.3f} "
                  f"jerk={JERK_DYNAMICS_FACTOR:.3f} (decoupled jerk)")
        except (TypeError, ValueError, AttributeError) as e:  # older franky: no 3-arg RDF
            robot.relative_dynamics_factor = v
            print(f"[franka] RelativeDynamicsFactor(v,a,j) unavailable ({type(e).__name__}); "
                  f"using uniform scalar dynamics factor {v:.3f}", file=sys.stderr, flush=True)

    def set_speed(self, fraction: float) -> float:
        """Set the velocity dynamics factor at RUNTIME and push it to the live
        robot. Returns the clamped fraction applied.

        This exists so the operator's collection speed can reach the arm on the
        DECOUPLED control-server path. The server is launched with `--ip` only
        (tools/arm/control_server.sh), so it constructs its controller at
        DEFAULT_DYNAMICS_FACTOR; the client's constructor argument was kept "for
        API parity" and never transmitted, which meant the panel's speed number
        had NO effect at all on a remote Franka collection. A settable method is
        the fix, because the server is a long-lived process shared by successive
        runs -- a launch-time flag could not re-target it per run."""
        from evo_franka.speed import clamp_fraction
        self.relative_dynamics_factor = clamp_fraction(fraction)
        if self.robot is not None:
            self._apply_dynamics_factor(self.robot)
        return self.relative_dynamics_factor

    def disconnect(self) -> None:
        self.stop_force_sampler()
        self.stop_move()
        if self.gripper is not None:
            try:
                state = self.get_gripper_motion_state()
                if state.get("status") == "running":
                    self.stop_gripper()
            except Exception as e:  # best-effort teardown; never strand the arm session
                print(f"[franka] WARNING: could not stop Hand during disconnect: {e}",
                      file=sys.stderr, flush=True)
        self.robot = None
        self.gripper = None
        print("[franka] disconnected")

    def ensure_fci_lease(self) -> None:
        """Renew the FCI session lease. System 5.9.0 (franka-right) TERMINATES
        any FCI session without an active control loop after ~55 s -- proven
        bare-franky 2026-08-21: connect + sleep 90 s died, and connect + 2 Hz
        state reads STILL died at 55.6 s (reads do not renew; only a running
        control loop does; older firmware keeps idle sessions forever, so
        franka-left never showed this). robotLab moves in discrete motions, so
        every quiet window (model loading, waiting between demo rounds) expires
        the lease; the next op -- arm OR Hand, whichever comes first -- found a
        dead stream. Probe the arm stream here and rebuild the WHOLE session
        (arm + Hand handles, dynamics/impedance/calibration reapplied by
        connect) when it expired. Called at BOTH op-entry seams: motions
        (_warn_and_recover) and Hand ops (_hand)."""
        if self.robot is None:
            return
        try:
            _ = self.robot.state.robot_mode
        # RuntimeError is IN the caught set: franky raises the typed
        # NetworkException when a live session drops, but a bare
        # RuntimeError("Net Exception") when the handle is ALREADY dead (the
        # state the probe most often meets). Missing it made this very probe
        # crash the demo it exists to protect (2026-08-21).
        except (*_STATE_READ_EXC, RuntimeError):
            print("[franka] FCI session lease expired while idle (system 5.9.0) "
                  "-- rebuilding", file=sys.stderr, flush=True)
            self.reset_session()

    def reset_session(self, *args, **kwargs) -> None:
        """Force a fresh FCI session (stop + drop the handle + reconnect) -- the
        rebuild primitive, used after an aborted cartesian motion.

        On abort the robot freezes the commanded-pose registers (O_T_EE_c /
        O_dP_EE_c) at the abort point, and franky seeds the next cartesian
        motion from them -- while the arm actually stopped elsewhere, so the
        next cartesian command jumps and reflexes again (cascade). A fresh
        session re-initializes the registers from the measured state.
        `*args/**kwargs` forward to connect()."""
        self.stop_move()
        self.robot = None
        self.connect(*args, **kwargs)

    def heal_session(self, *args, **kwargs) -> None:
        """THE single session-recovery routine (item 3): bring the robot back to
        a READY state with the LEAST disruption. If it is present and error-free,
        do nothing; if it has latched errors, clear them IN PLACE
        (recover_from_errors); only if that is insufficient -- or the handle is
        gone -- rebuild the whole FCI session (reset_session). Both the driver
        itself and the control server call THIS, so 'session lost ->
        recover-or-reconnect' lives in exactly one place (the server used to
        re-implement it in _heal_session). `*args/**kwargs` forward to connect()
        on a rebuild."""
        r = self.robot
        if r is not None and not r.has_errors:
            return
        logger.info("[franka] heal_session: robot not ready (present=%s) -> healing", r is not None)
        if r is not None:
            try:
                self.recover()                       # clear latched reflex in place
                if self.robot is not None and not self.robot.has_errors:
                    logger.info("[franka] heal_session: recover cleared the error state")
                    return
            except (*_FRANKY_EXC, RuntimeError) as e:  # recover itself faulted -> full rebuild
                logger.error("[franka] heal_session: recover failed (%s: %s)", type(e).__name__, e)
        logger.info("[franka] heal_session: recover insufficient -> full reconnect")
        self.reset_session(*args, **kwargs)

    # cooldown so a persistently-dead stream doesn't re-connect on every read
    _RECONNECT_COOLDOWN_S = 8.0

    def _attempt_reconnect(self) -> bool:
        """The FCI realtime stream looks DEAD (persistent state-read failures).
        recover_from_errors() only clears reflexes -- it cannot re-establish a
        DROPPED FCI session, which is why a control-server whose link died would
        otherwise loop on 'Net Exception' forever. So rebuild the session: drop the
        stale handle and reconnect. Rate-limited; invoked only from the idle
        state-read path (no motion is running when reads are failing), so a
        non-motion reconnect is safe. Returns True if a fresh read works after."""
        now = _time.monotonic()
        if now - self._last_reconnect_ts < self._RECONNECT_COOLDOWN_S:
            return False
        self._last_reconnect_ts = now
        try:
            logger.warning("[franka] FCI stream dead -> full reconnect to %s "
                           "(recover_from_errors can't fix a dropped link)", self.robot_ip)
            self.reset_session()                # THE rebuild primitive: stop + drop handle + connect
            with self._lock:
                _ = self.robot.state.robot_mode  # verify the stream actually delivers now
            self._state_fallback_streak.clear()
            self.reconnects_total += 1
            logger.info("[franka] reconnect OK (total reconnects: %d)", self.reconnects_total)
            return True
        except Exception as e:  # noqa: BLE001 -- best-effort recovery: ANY failure to rebuild
            # the session returns the False sentinel (contract); the caller then raises the
            # clear FrankaSessionLost, so the fault is SURFACED, never masked.
            logger.error("[franka] reconnect failed: %s: %s", type(e).__name__, str(e)[:120])
            return False

    # ---------------- recovery ----------------

    def recover(self) -> None:
        """Clear latched robot errors (collision/reflex). Call ONLY when the
        motion is already stopped -- never pre-arm it before an expected error
        (pre-arming makes the next reflex fire instantly; franka_ros#316)."""
        if self.robot is None:
            return
        try:
            self.robot.recover_from_errors()
        except _FRANKY_EXC as e:  # direct franky call -> translate so it never leaks raw (F5)
            raise wrap_franky_exc(e, "recover") from e

    def wait_until_ready(self, timeout_s: float = RECOVER_READY_TIMEOUT_S,
                         poll_s: float = RECOVER_READY_POLL_S) -> bool:
        """Block until the robot is no longer in control (idle / ready), bounded
        by `timeout_s`. Returns True if the robot reached idle, False on timeout.

        The Polymetis/SERL post-recovery guard: recovery latency (the control
        thread winding down after a stop/recover) is unbounded in principle, so
        instead of a blind fixed sleep we POLL the robot's in-control flag until
        it is idle, capped by a timeout. Tolerant of a transient state-read error
        while polling (treats it as 'not yet ready')."""
        if self.robot is None:
            return True
        deadline = _time.perf_counter() + max(0.0, float(timeout_s))
        while True:
            try:
                with self._lock:
                    busy = bool(getattr(self.robot, "is_in_control", False))
                if not busy:
                    return True
            except Exception:  # noqa: BLE001 - a read hiccup just means 'poll again'
                pass
            if _time.perf_counter() >= deadline:
                print(f"[franka] WARNING: robot still in control after "
                      f"{timeout_s:.1f}s wait-for-ready", file=sys.stderr, flush=True)
                return False
            _time.sleep(max(0.0, float(poll_s)))

    # ---------------- internals ----------------

    def _require(self) -> None:
        if self.robot is None:
            raise RuntimeError("not connected; call connect() first")

    def _require_kin(self) -> None:
        self._require()
        if self.kin is None:
            raise RuntimeError("kinematics unavailable; call connect() first (calibrated there)")

    # ---------------- working-frame transforms ----------------

    def _prepare_installation(self, install_rad: float) -> None:
        # T_base_to_new maps working-frame coordinates into base-frame
        # coordinates (the working frame expressed in base); its inverse maps
        # base into the working frame.
        self.install_rad = install_rad
        self._has_install_transform = abs(install_rad) > 1e-12
        r = R.from_euler("z", install_rad).as_matrix()
        self.T_base_to_new: np.ndarray = np.eye(4)
        self.T_base_to_new[:3, :3] = r
        self.T_new_to_base: np.ndarray = np.linalg.inv(self.T_base_to_new)

    def _base_to_new(self, pose: Sequence[float]) -> list[float]:
        if not self._has_install_transform:
            return list(pose)
        return matrix_to_pose(self.T_new_to_base @ pose_to_matrix(pose))

    def _new_to_base(self, pose: Sequence[float]) -> list[float]:
        if not self._has_install_transform:
            return list(pose)
        return matrix_to_pose(self.T_base_to_new @ pose_to_matrix(pose))

    def _rotate_to_new(self, vec6: Sequence[float]) -> list[float]:
        """Rotate a base-frame twist/wrench [linear(3); angular(3)] into the
        working frame (no translation -- free vectors)."""
        if not self._has_install_transform:
            return list(vec6)
        rot = self.T_new_to_base[:3, :3]
        v = np.asarray(list(vec6), dtype=float).reshape(6)
        return np.concatenate([rot @ v[:3], rot @ v[3:]]).tolist()
