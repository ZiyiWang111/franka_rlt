"""Franka motion concern (F3 split of driver.py).

The ONLY motion entry points -- point-to-point (move_joint / move_tool), TCP
trajectory (move_tool_traj via the branch-continuous IK primitive), servo
(servo_joint / servo_tool), cartesian impedance, the guarded contact descent
(move_until_force), and the stops -- plus THE single franky motion boundary
(_safe_move + its async twin finish... no: finish_move/_classify_move_fault live
here too). Every motion is singularity-guarded / discontinuity-safe by
construction. Methods live on MotionMixin (combined by robots/franka/driver.py)."""
from __future__ import annotations

import contextlib
import logging
import math
import re
import sys
import time as _time
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp  # noqa: N817

from evo_franka.base_errors import JointSwingExceeded
from evo_franka._franky import (
    Affine,
    CartesianImpedanceMotion,
    CartesianMotion,
    CartesianPoseReaction,
    CartesianStopMotion,
    Duration,
    JointMotion,
    JointPositionReaction,
    JointState,
    JointStopMotion,
    JointWaypoint,
    JointWaypointMotion,
    Measure,
    ControlException,
    NetworkException,
    _FRANKY_EXC,
)
from evo_franka.errors import (
    FrankaError,
    FrankaMotionRefused,
    FrankaReflex,
    FrankaSessionLost,
    reflex_kind,
    wrap_franky_exc,
)
from evo_franka.constants import (
    ACCELERATION_DYNAMICS_FACTOR,
    CONTACT_RDF,
    DEFAULT_MAX_ANGULAR_VEL_RAD_S,
    DEFAULT_TOOL_SPEED_M_S,
    EXCURSION_FK_SAMPLES,
    FINAL_WAYPOINT_HOLD_MS,
    FR3_MIN_RATED_JOINT_ACCEL_RAD_S2,
    GUARD_FORCE_THRESHOLD_N,
    IMPEDANCE_ROTATIONAL_STIFFNESS,
    IMPEDANCE_TRANSLATIONAL_STIFFNESS,
    JOINT_AT_TARGET_RAD,
    MIN_SEGMENT_TIME_S,
    MOTION_BOW_OUT_M,
    MOTION_BRANCH_SWING_RAD,
    MOTION_JOIN_SLICE_S,
    MOTION_JOIN_TIMEOUT_S,
    PASSTHROUGH_BRAKE_MARGIN,
    PASSTHROUGH_VEL_HEADROOM,
    PATH_DENSIFY_STEP_M,
    PATH_DENSIFY_STEP_RAD,
    PATH_MIN_SPEED_FLOOR_M_S,
    REACTION_SETTLE_S,
    SERVO_IK_ORI_TOL_RAD,
    SERVO_IK_POS_TOL_M,
    STOP_JOIN_TIMEOUT_S,
)
from evo_franka.geometry import matrix_to_pose

logger = logging.getLogger(__name__)

# libfranka states the FCI control-command success rate in its own abort text.
# 1.0 is the only healthy value and it is not a tunable: the FCI accepts one
# command per 1 ms tick, so the rate IS the fraction of ticks this process met.
# Every control-server run on franka-right reports exactly 1.000.
_CCSR_RE = re.compile(r"control_command_success_rate:\s*([0-9]*\.?[0-9]+)")
_CCSR_HEALTHY = 1.0


def control_command_success_rate(exc: "BaseException | None") -> "float | None":
    """The FCI control-command success rate libfranka reports on an aborted
    motion, or None if this exception does not carry one.

    libfranka puts it in the abort text verbatim ("control_command_success_rate:
    0.82"), so it is read from the exception rather than by a state read after
    the fact -- the value AT THE FAULT is the one that explains the fault, and a
    later read has already moved on."""
    if exc is None:
        return None
    match = _CCSR_RE.search(str(exc))
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:      # a malformed number must never mask the real fault
        return None


def _control_rate_hint(exc: "BaseException | None", hosted: bool = False) -> str:
    """THE dropped-command diagnosis, and the reason it is worth a function.

    The FCI contract is one accepted command per 1 kHz tick, so a success rate
    below 1.0 means this process did not deliver commands in time -- the arm was
    starved, not obstructed. That distinction decides what an operator does next
    and the two look identical from a distance: both end in
    `joint_motion_generator_*_discontinuity`, and the generic reflex advice
    ("move away from the limit") is actively WRONG for the starved case.

    The number was in every log line all evening and nothing read it, which is how
    franka-right lost two evenings to the same cause (2026-08-20, and again
    2026-08-26 when a launcher change put the 1 kHz loop back in the policy's own
    process: rate 0.43-0.89 there against a flat 1.000 through the control
    server). It is the controller reporting on the health of its own control path,
    which makes it the driver's business to read and to say out loud.

    `hosted` = this controller ALREADY lives inside the dedicated control server
    (the server marks it at construction). The advice forks on it, because the
    un-hosted advice is "isolate the loop in the control server" and telling a
    control server to go run itself is worse than no hint: it sends the operator
    to re-apply a fix that is demonstrably in place (observed 2026-08-26, a
    rate-0.97 abort under 'cores=20-23 rt=yes' whose hint prescribed
    FRANKA_CONTROL_IPC). Inside the server a sub-1.0 rate means either the drops
    are REAL despite the isolation (something else on the pinned cores, load on
    the robot NIC, an SMT sibling) or the rate is the accounting of a motion
    aborted at its first ticks -- and the server's own periodic FCI-health lines
    (ccsr logged every ~5 s) tell those apart: a steady-state rate below 1.0
    between faults is real starvation; 1.000 between faults with a low rate only
    in the abort text is local to the aborted motion.

    Returns "" when the rate is healthy or absent, so callers can append freely."""
    rate = control_command_success_rate(exc)
    if rate is None or rate >= _CCSR_HEALTHY:
        return ""
    if hosted:
        return (f"THE ARM DROPPED COMMANDS: control_command_success_rate={rate:.3f} "
                f"(healthy is {_CCSR_HEALTHY:.3f}) -> it received only that fraction of "
                "its 1 kHz stream -- and this process IS the dedicated control server, "
                "so the process-isolation fix is already in place; do not re-apply it. "
                "Either the drops are real despite the isolation (check what ELSE can "
                "touch the pinned cores: another pinned process, an SMT sibling, load "
                "on the robot NIC) or this is the success-rate accounting of a motion "
                "aborted within its first ticks. This server's periodic FCI-health "
                "lines (ccsr every ~5 s) distinguish the two: below-1.0 between faults "
                "is real starvation; 1.000 between faults is abort-local accounting. "
                "Treat measurements from this run as suspect either way.")
    return (f"THE ARM DROPPED COMMANDS: control_command_success_rate={rate:.3f} "
            f"(healthy is {_CCSR_HEALTHY:.3f}) -> it received only that fraction of its "
            "1 kHz stream, so this is a CONTROL-PATH fault, not contact and not the "
            "policy. The commanded trajectory was broken by the gaps, which is what "
            "the discontinuity reflex is reporting. Look at what shares a core with "
            "the control loop: run it in its own pinned process (control_server + "
            "FRANKA_CONTROL_IPC) rather than beside cameras and inference, and treat "
            "any measurement taken during this run as void.")


def _mode_hint(mode: object, exc: "Exception | None" = None, hosted: bool = False) -> str:
    """Plain-English diagnosis of a rejected motion, keyed off the robot mode --
    so the log says WHAT to fix, not a generic checklist. The most common cause
    by far (and the easy-to-miss one) is the arm not being in Execution mode.

    A dropped-command rate OUTRANKS every mode-keyed hint below: those all assume
    the arm received what it was sent, and if it did not, they send the operator
    to the wrong place. `hosted` -- whether this controller already runs inside
    the dedicated control server -- forks that top hint, see _control_rate_hint."""
    starved = _control_rate_hint(exc, hosted)
    if starved:
        return starved
    # A PLANNER refusal outranks every mode-keyed hint below, because the mode is
    # not the story: ruckig validated franky's inputs and rejected them BEFORE any
    # command was generated, so the arm is (correctly) still Idle and the link is
    # (correctly) fine. Without this the Idle/Move branch below blames the FCI link
    # -- prose that is both wrong AND, because the collector's classifier reads
    # this line, the reason such a refusal was bucketed as 'comms'.
    if "motion planner failed" in str(exc or "").lower():
        return ("the MOTION PLANNER refused the inputs and commanded nothing (franky "
                "ruckig; code -100 = ErrorInvalidInput) -> not the link and not the "
                "mode. Either the target/limits are invalid for this motion (a "
                "velocity/accel above the configured max, a zero or negative limit, "
                "the dynamics factor) or the state it planned FROM is not finite -- "
                "franky seeds a first cartesian motion from libfranka's COMMANDED "
                "fields (O_T_EE_c/elbow_c/O_dP_EE_c), which are unset until a control "
                "loop has run. Same inputs = same refusal, so retrying is pointless: "
                "change the target, the limits, or run a joint motion first.")
    m = str(mode or "")
    if "UserStopped" in m:
        return ("ARM IS USER-STOPPED -> this is almost always Desk in PROGRAMMING "
                "mode or the user-stop button pressed. Switch Desk to EXECUTION mode "
                "AND release the user-stop, then retry. (motion is impossible until "
                "robot_mode is Idle/Move)")
    if "Guiding" in m:
        return ("arm is in GUIDING mode (hand-guiding button held / Programming mode) "
                "-> release it and switch Desk to Execution mode before commanding motion")
    if "Reflex" in m or (exc is not None and isinstance(exc, ControlException)):
        return ("a control REFLEX fired (joint/cartesian limit, singularity, or a "
                "discontinuous start) -> recover and move away from the limit; if it "
                "fires immediately on every move, check the arm isn't user-stopped")
    if "Idle" in m or "Move" in m:
        # The LAST branch reached when everything measurable has been ruled out. It used
        # to answer "likely the FCI link (NetworkException); check the realtime NIC" --
        # a cause invented from the ABSENCE of evidence, and the sentence whose word
        # "NetworkException" made the collector's classifier bucket a planner refusal as
        # 'comms' and spend a whole run telling the operator to close their browser. A
        # hint with no evidence must say so and hand the reader back to the fault text.
        return ("robot_mode is motion-ready and no other check matched, so THIS HINT HAS NO "
                "EVIDENCE for a cause and names none. Ruled out: the operating mode (the arm "
                "is ready), a planner refusal, and a dropped-command rate (measured above). "
                "Still open: the robot-net link, another FCI client holding the session, or "
                "something only the driver's own message names. Read the fault text on the "
                "line above this one first.")
    return (f"robot_mode={m or 'unknown'} is not one this hint recognises, so no cause is "
            "claimed. The blockers that stop motion at all are: Desk in Programming rather "
            "than Execution mode, the user-stop pressed, the brakes closed, and FCI not "
            "activated -- check those, then re-read the fault text above.")

# Servo backend: warm-started DLS-IK -> franky JointMotion streaming. WE do the IK
# (branch-continuous, warm-seeded, tracked to SERVO_IK_POS_TOL_M) and stream JOINT
# targets, so the command stays smooth in joint space regardless of the policy's
# Cartesian path. (A franky-native async CartesianMotion servo was tried and ABANDONED
# 2026-06-30: Ruckig's internal IK is NOT branch-continuous, so curved targets at
# irregular cadence tripped cartesian_motion_generator_*_discontinuity reflexes in the
# closed loop; the joint servo structurally avoids this via joint-space interpolation.)


class MotionMixin:
    """Motion / servo / stop / franky-boundary half of FrankaArmController."""

    def _require_hand_idle_for_arm_motion(self) -> None:
        """Enforce scheme 2 below the policy layer: Hand motion owns the pause."""
        poll = getattr(self, "_poll_gripper_motion", None)
        if poll is None or getattr(self, "gripper", None) is None:
            return
        state = poll()
        if state.get("status") == "running":
            raise FrankaMotionRefused(
                f"arm motion refused while gripper command {state.get('command_id')} "
                f"({state.get('command')}) is running")


    def finish_move(self) -> bool:
        """Non-blocking reap of an async move on the calling thread: returns True
        IFF the motion has actually finished (join_motion(0.0) joined), and
        surfaces any fault franky deferred (it stores an async-motion exception
        and re-raises on the next join_motion()/move()). Returns False if the
        motion is still running -- so the caller must treat is_running()==False as
        a HINT and only declare the move done once finish_move() returns True
        (guards a transient is_in_control glitch from mis-sequencing the move).
        This is the single-threaded replacement for the old move-helper thread's
        blocking join: the controller issues async, polls inline (state reads stay
        fresh -- same thread), and calls finish_move() to collect the outcome."""
        r = self.robot
        if r is None:
            return True
        try:
            with self._lock:
                joined = bool(r.join_motion(0.0))
        except FrankaError:
            raise                                        # already typed+classified
        except Exception as e:  # noqa: BLE001 -- the async twin of the _safe_move boundary
            # Classify an async-motion fault the SAME way _safe_move classifies a
            # blocking one, so the collector's classify_motion_failure buckets both
            # identically (network/comms vs control reflex).
            raise self._classify_move_fault(e, "Franka async move rejected") from e
        if joined:
            self._async_move_deadline = None             # move reaped -> disarm the hang cap
        return joined

    def _motion_hang_error(self) -> FrankaMotionRefused:
        """THE single motion-hang fault (F4): a move that does not finish within
        MOTION_JOIN_TIMEOUT_S is stopped and surfaced as this typed error. Built
        here so the BLOCKING join (_safe_move) and the ASYNC poll (check_move_hang,
        driven by the control server's loop) raise an IDENTICAL fault -- the hang
        cap is defined once, never re-implemented per caller."""
        # "motion hang" is the token the collector's classifier reads -- keep it verbatim.
        # What is NOT kept: the old "real-time loop starved; reduce host load", which
        # declared a cause the timeout alone does not establish (a wedged FCI call and a
        # state read that never returns produce exactly this too). The bucket's guidance
        # (collect_recovery.failure_guidance) lists the candidates honestly.
        return FrankaMotionRefused(
            f"Franka move timed out after {MOTION_JOIN_TIMEOUT_S:.0f}s "
            "(motion hang -- commanded, then no completion inside the join window; "
            "the timeout alone does not say why) -- aborted via stop_move")

    def check_move_hang(self) -> None:
        """Abort a wedged ASYNC move past the MOTION_JOIN_TIMEOUT_S cap -- the
        async twin of _safe_move's blocking hang guard.

        _safe_move arms a deadline when it ISSUES an async move; the control
        server's controller loop calls this each cycle (F4). It is checked
        regardless of is_running(), so a move wedged while still 'in control' is
        caught too. No-op until a deadline is armed and only fires past it, so a
        normally-completing move never touches it. On a hang it stops the motion
        and raises FrankaMotionRefused (the server reports it and clears busy).
        This is why the server no longer re-implements its own hang cap: the
        driver's motion boundary is the single owner of the MOTION_JOIN_TIMEOUT_S
        policy for both blocking and async issue."""
        dl = self._async_move_deadline
        if dl is None or _time.perf_counter() < dl:
            return
        self._async_move_deadline = None                 # disarm before stopping (no re-entry)
        self.stop_move()
        raise self._motion_hang_error()

    def _classify_move_fault(self, e: Exception, prefix: str) -> FrankaError:
        """THE motion-boundary classifier (item 3): map a franky exception caught
        in _safe_move / finish_move to a typed FrankaError, preserving the
        libfranka reason text and the [network/comms] / [control reflex] tag that
        classify_motion_failure greps for. A NetworkException means the FCI link
        dropped (FrankaSessionLost); a ControlException is a reflex/abort
        (FrankaReflex, kind inferred from the message); anything else is a generic
        rejection (FrankaMotionRefused)."""
        mode = None
        try:
            mode = self.robot.state.robot_mode
        except (*_FRANKY_EXC, RuntimeError):  # best-effort mode read for the message only;
            pass                              # never let a failed diagnostic read mask the fault
        # Whether this controller lives inside the dedicated control server (the
        # server stamps it at construction) -- the starved-control-path hint must
        # not prescribe an isolation this process already is.
        hosted = bool(getattr(self, "hosted_by_control_server", False))
        if isinstance(e, NetworkException):
            return FrankaSessionLost(
                f"{prefix} [network/comms] (robot_mode={mode}): {e}\n  {_mode_hint(mode, e, hosted)}")
        if isinstance(e, ControlException):
            return FrankaReflex(
                f"{prefix} [control reflex] (robot_mode={mode}): {e}\n  {_mode_hint(mode, e, hosted)}",
                kind=reflex_kind(str(e)))
        return FrankaMotionRefused(
            f"{prefix} (robot_mode={mode}): {e}\n  {_mode_hint(mode, e, hosted)}")

    # ---------------- point-to-point motion ----------------

    @contextlib.contextmanager
    def _limits_held(self, **overrides):
        """THE one way to override a franky robot limit for ONE motion.

        A franky limit (joint_velocity_limit, translation_velocity_limit, ...) is
        ROBOT-WIDE and outlives the call that set it, so every override needs a
        matching restore. Two sites did this by hand and BOTH leaked:

          * move_joint restored only `if not is_async and not self.is_running()`
            -- a restore conditional on a race, so a move that ended while any
            motion was in control left its limits applied to everything after it,
            the 30 Hz servo stream included;
          * move_until_force set the cartesian velocity limits through
            _set_velocity_limits and never restored them at all.

        Same defect class: a mutation whose lifetime was implicit and whose
        restore was not guaranteed. Here the lifetime IS the `with` block and the
        restore is in a finally, so it happens on every exit path -- including the
        reflex ones, which are exactly the paths the hand-written versions missed.

        `overrides` maps a franky limit ATTRIBUTE NAME to the value to set; a None
        value means "leave this one alone", so a caller with optional parameters
        does not have to branch. Restores run in reverse order and a restore that
        itself fails is SAID rather than raised: it must not replace the motion's
        own fault, and a limit stuck at an override is the operator's problem
        before the next motion."""
        saved: list = []
        try:
            for name, value in overrides.items():
                if value is None:
                    continue
                limit = getattr(self.robot, name)
                saved.append((name, limit, limit.get()))
                limit.set(value)
            yield
        finally:
            for name, limit, previous in reversed(saved):
                try:
                    limit.set(previous)
                except Exception as e:  # noqa: BLE001 -- must not replace the motion's fault
                    print(f"[franka] WARNING: could not restore {name} to {previous!r} "
                          f"({type(e).__name__}: {e}) -- it stays at this motion's "
                          f"override for every motion after it",
                          file=sys.stderr, flush=True)

    def move_joint(self, joint_values: Sequence[float], speed: Optional[float] = None,
                   acceleration: Optional[float] = None, is_async: bool = False) -> None:
        """Joint-space move. `speed`/`acceleration` set the joint velocity /
        acceleration limits (rad/s, rad/s^2; further scaled by the dynamics
        factor) FOR THE DURATION OF THIS MOTION, and are restored on every exit
        path (see _limits_held).

        They cannot be combined with is_async=True: the limits are robot-wide and
        their lifetime is the motion, so a caller that returns before the motion
        ends cannot own the restore. That combination used to apply the limits and
        never take them back."""
        self._require()
        self._require_hand_idle_for_arm_motion()
        q = [float(v) for v in joint_values]
        if len(q) != 7:
            raise ValueError(f"Franka needs 7 joint values, got {len(q)}")
        if is_async and (speed is not None or acceleration is not None):
            raise ValueError(
                "move_joint(is_async=True) cannot take speed/acceleration: they are "
                "ROBOT-WIDE limits held for the duration of the motion, and an async "
                "issue returns before the motion ends -- so this call cannot restore "
                "them and they would leak into every later motion, the servo stream "
                "included. Issue the move blocking, or hold the limits around your "
                "own join.")
        self._warn_and_recover()
        # franky refuses a limit change while a motion is in control, so a move
        # issued mid-motion keeps the limits already in force (unchanged behavior).
        per_motion: dict = {}
        if not self.is_running():
            per_motion["joint_velocity_limit"] = (
                None if speed is None else [float(speed)] * 7)
            per_motion["joint_acceleration_limit"] = (
                None if acceleration is None else [float(acceleration)] * 7)
        with self._limits_held(**per_motion):
            self._move_joint_reflex_robust(q, is_async)

    def _move_joint_reflex_robust(self, q: "list[float]", is_async: bool) -> None:
        """Run the joint motion, robust to the FR3 start-of-move discontinuity
        reflex. A move whose goal equals (or nearly equals) the current pose
        produces a degenerate franky trajectory that trips
        joint_motion_generator_*_discontinuity even though there is nothing to
        move; the same reflex also fires transiently at the start of a real move.
        On a control reflex we stop + recover (the order libfranka needs), then:
        (a) if the arm is already AT the commanded configuration the move was a
        no-op -> done; (b) otherwise retry ONCE from the now-clean, settled state.
        A reflex that still fails to reach the target is a real fault -> raise."""
        last = None
        for _ in range(2):
            try:
                self._safe_move(JointMotion(JointState(q)), is_async)
                return
            except FrankaError as e:
                if is_async or not isinstance(e, FrankaReflex):
                    raise                            # only a control reflex is retryable here
                last = e
                try:
                    self.stop_move()
                    self.robot.recover_from_errors()
                    cur = self.get_joint_angles()
                except (*_FRANKY_EXC, RuntimeError):  # can't stop/recover/read -> real fault
                    raise e
                if cur is not None and len(cur) == 7 \
                        and max(abs(a - b) for a, b in zip(q, cur)) < JOINT_AT_TARGET_RAD:
                    return  # reached the goal despite the reflex (incl. no-op moves)
        raise last

    def move_tool(self, pose: Sequence[float], speed: float = DEFAULT_TOOL_SPEED_M_S,
                  max_angular_vel: float = DEFAULT_MAX_ANGULAR_VEL_RAD_S,
                  acceleration: Optional[float] = None,
                  is_async: bool = False) -> list[float]:
        """Move the TCP to a pose [x, y, z, rx, ry, rz]. `speed` (m/s) paces the
        motion (average TCP speed ~= speed). Executed as IK + joint motion (one
        motion path); the path arcs slightly instead of being perfectly straight.
        `acceleration` is accepted for compatibility and ignored (the dynamics
        factor governs). Returns the measured TCP pose after the call (the
        reached pose for blocking calls; the in-motion pose when is_async)."""
        self._require()
        pose = list(pose)
        if len(pose) != 6:
            raise ValueError(f"tool pose must be [x,y,z,rx,ry,rz], got {len(pose)} values")
        return self.move_tool_traj(
            [pose + [float(speed), 0.0, 0.0]], is_async=is_async,
            max_angular_vel=max_angular_vel,
        )

    # ---------------- trajectory motion ----------------

    def _log_traj_excursion(self, solved_q, steps: int = EXCURSION_FK_SAMPLES) -> None:
        """Diagnostic for 'the arm moves too high'. A TCP waypoint path is
        executed as a JOINT-space interpolation, so between waypoints the TCP
        bows off the straight line -- and a near-singular/edge waypoint can make
        the IK jump branches, swinging the TCP far out of the sampled box (an
        excursion that is NOT in the recorded forward trace -- the reset move
        isn't recorded). FK-sample each leg's joint interpolation, report its
        TCP-z span, and flag BOW-OUT (TCP rises >5cm above both endpoints
        mid-leg) or BRANCH-SWING (a >~46deg single-joint jump). Logged to stderr
        so it lands in the collection log; never raises."""
        if self.kin is None or len(solved_q) < 2:
            return
        try:
            for i in range(len(solved_q) - 1):
                qa = np.asarray(solved_q[i], dtype=float)
                qb = np.asarray(solved_q[i + 1], dtype=float)
                d_joint = float(np.max(np.abs(qb - qa)))
                za = self._base_to_new(matrix_to_pose(self.kin.fk(qa)))[2]
                zb = self._base_to_new(matrix_to_pose(self.kin.fk(qb)))[2]
                zmax, zmin = max(za, zb), min(za, zb)
                for t in np.linspace(0.0, 1.0, steps):
                    z = self._base_to_new(matrix_to_pose(self.kin.fk(qa + (qb - qa) * t)))[2]
                    zmax = max(zmax, z); zmin = min(zmin, z)
                bow = zmax - max(za, zb)
                flags = ""
                if d_joint > MOTION_BRANCH_SWING_RAD:   # large single-joint jump -> IK branch swing
                    flags += " BRANCH-SWING"
                if bow > MOTION_BOW_OUT_M:              # TCP bows above both endpoints mid-leg
                    flags += " BOW-OUT"
                msg = (f"[motion] leg {i}: z {za:.3f}->{zb:.3f}  midpath z[{zmin:.3f},{zmax:.3f}]  "
                       f"bow=+{bow*100:.1f}cm  max|dq|={math.degrees(d_joint):.0f}deg{flags}")
                if flags:
                    print(msg, file=sys.stderr, flush=True)  # excursions visible in the log
                else:
                    logger.debug(msg)
        except Exception as e:  # noqa: BLE001 -- a diagnostic must NEVER break a motion
            logger.debug("[motion] excursion logging failed: %s", e)

    def _parse_tcp_waypoints(
        self, path: Sequence[Sequence[float]], who: str,
    ) -> list[tuple[np.ndarray, R, float, float]]:
        """Parse a TCP waypoint path [x,y,z,rx,ry,rz,speed,accel,blend] into
        base-frame (pos, rot, speed, blend) tuples (speed/accel/blend optional;
        accel ignored)."""
        if not path:
            raise ValueError(f"{who} requires at least one waypoint")
        parsed: list[tuple[np.ndarray, R, float, float]] = []
        for point in path:
            point = list(point)
            if len(point) < 6:
                raise ValueError(f"waypoint must start with [x,y,z,rx,ry,rz], got {len(point)} values")
            base_pose = self._new_to_base(point[:6])
            pos = np.asarray(base_pose[:3], dtype=float)
            rot = R.from_rotvec(np.asarray(base_pose[3:6], dtype=float))
            speed = float(point[6]) if len(point) > 6 else DEFAULT_TOOL_SPEED_M_S
            if speed <= 0:
                speed = DEFAULT_TOOL_SPEED_M_S
            blend = float(point[8]) if len(point) > 8 else 0.0
            parsed.append((pos, rot, speed, blend))
        return parsed

    def _segment_min_timer(self, max_angular_vel: float,
                           start_pose: Optional[Sequence[float]] = None):
        """Build a stateful per-segment minimum-time function seeded from the
        CURRENT measured TCP pose. Each call returns the minimum time for the
        leg from the previous waypoint to (pos, rot) at `speed`, pacing the
        segment so the average TCP speed matches the requested one (linear or
        angular, whichever is slower)."""
        start = list(start_pose) if start_pose is not None else self._tool_pose_base()
        state = {"pos": np.asarray(start[:3], dtype=float),
                 "rot": R.from_rotvec(np.asarray(start[3:6], dtype=float))}

        def segment_min_time(pos: np.ndarray, rot: R, speed: float) -> float:
            linear_dt = float(np.linalg.norm(pos - state["pos"])) / speed
            angular_dt = (float((state["rot"].inv() * rot).magnitude())
                          / max(max_angular_vel, PATH_MIN_SPEED_FLOOR_M_S))
            state["pos"], state["rot"] = pos, rot
            return max(linear_dt, angular_dt)

        return segment_min_time

    def _densify_steps(self, prev_pos: np.ndarray, prev_rot: R,
                       pos: np.ndarray, rot: R) -> int:
        """Sub-sample count for one leg so each Cartesian step stays under
        PATH_DENSIFY_STEP_M / PATH_DENSIFY_STEP_RAD (keeps warm-started IK on one
        branch)."""
        lin = float(np.linalg.norm(pos - prev_pos))
        ang = float((prev_rot.inv() * rot).magnitude())
        return max(1, int(math.ceil(max(lin / PATH_DENSIFY_STEP_M,
                                        ang / PATH_DENSIFY_STEP_RAD))))

    def _solve_branch_continuous_ik(self, start_pose: Sequence[float],
                                    parsed, max_angular_vel: float,
                                    max_joint_swing_rad: "float | None" = None):
        """IK a TCP waypoint path into BRANCH-CONTINUOUS joint waypoints.

        Each leg is densely sampled (linear position + slerp orientation) and
        every IK solve is warm-started from the previous DENSE solution, so
        adjacent solutions are a small step apart and cannot jump IK branches
        between far-apart functional waypoints -- the root cause of the
        joint_motion_generator_acceleration_discontinuity 'branch swing'.
        Returns a JointWaypoint list for the FUNCTIONAL waypoints, paced to each
        leg's TCP speed. Raises BEFORE any motion if a branch jump is
        unavoidable (near a singularity / unreachable branch), so it surfaces as
        a recoverable planning error rather than a runtime reflex. The dense
        samples are SEEDS only. A waypoint whose blend marker is 0 is a full
        stop; blend > 0 is passed THROUGH with a bounded target joint velocity
        (_passthrough_velocity), so Ruckig rounds the corner instead of halting
        -- the visible stop-then-crawl at a pre-grasp via."""
        segment_min_time = self._segment_min_timer(max_angular_vel, start_pose)
        q_seed = np.asarray(self.get_joint_angles(), dtype=float)
        prev_pos = np.asarray(start_pose[:3], dtype=float)
        prev_rot = R.from_rotvec(np.asarray(start_pose[3:6], dtype=float))
        legs: list = []          # (q_functional, min_time_s, blend) per waypoint
        solved_q = [q_seed]  # functional configs incl. the start, for excursion logging
        for pos, rot, speed, blend in parsed:
            steps = self._densify_steps(prev_pos, prev_rot, pos, rot)
            sub_rots = None
            if steps > 1:
                key = R.from_quat(np.array([prev_rot.as_quat(), rot.as_quat()]))
                sub_rots = Slerp([0.0, 1.0], key)(np.arange(1, steps + 1) / steps)
            for k in range(steps):
                frac = (k + 1) / steps
                s_pos = prev_pos + (pos - prev_pos) * frac
                s_rot = sub_rots[k] if sub_rots is not None else rot
                try:
                    q_new = np.asarray(
                        self.kin.ik(s_pos, s_rot.as_quat(), q_init=q_seed, q_rest=q_seed),
                        dtype=float)
                except ValueError as e:
                    raise FrankaMotionRefused(f"IK failed approaching {pos.tolist()}: {e}") from e
                dq = float(np.max(np.abs(q_new - q_seed)))
                if dq > MOTION_BRANCH_SWING_RAD:
                    raise FrankaMotionRefused(
                        f"IK branch discontinuity ({dq:.2f} rad > {MOTION_BRANCH_SWING_RAD}) "
                        f"approaching {pos.tolist()} -- near singularity / unreachable "
                        "branch; reshape the path or move the target")
                q_seed = q_new
            legs.append((q_seed, segment_min_time(pos, rot, speed), blend))
            solved_q.append(q_seed)
            prev_pos, prev_rot = pos, rot
        # NET joint-travel budget (the cartesian max-z analogue for rotation):
        # the dense warm-start keeps each STEP on one branch, but a start that is
        # smoothly reachable yet FAR still yields a large single-leg sweep that
        # passes every step check and then trips acceleration_discontinuity at the
        # robot. Reject it BEFORE motion so the caller resamples a closer start --
        # prevention, not post-hoc recovery. Recovery home moves use move_joint
        # (joint space), so they are NOT subject to this cap.
        if max_joint_swing_rad:
            for i in range(len(solved_q) - 1):
                dq = float(np.max(np.abs(np.asarray(solved_q[i + 1], dtype=float)
                                         - np.asarray(solved_q[i], dtype=float))))
                if dq > max_joint_swing_rad:
                    raise JointSwingExceeded(
                        f"leg {i} needs {math.degrees(dq):.0f}deg net joint travel "
                        f"> max rotation {math.degrees(max_joint_swing_rad):.0f}deg "
                        "(start too far / near a branch edge -- resample a closer start)")
        self._log_traj_excursion(solved_q)
        joint_waypoints: list = []
        v_lim = self._joint_velocity_ceiling()
        for i, (q, min_time_s, blend) in enumerate(legs):
            kwargs = {}
            if min_time_s > MIN_SEGMENT_TIME_S:
                kwargs["minimum_time"] = Duration(int(math.ceil(min_time_s * 1000.0)))
            if i + 1 == len(legs):
                kwargs["hold_target_duration"] = Duration(FINAL_WAYPOINT_HOLD_MS)
            state = JointState(q.tolist())
            if blend > 0.0 and i + 1 < len(legs):   # intermediate + marked: pass through
                v = self._passthrough_velocity(q, legs[i + 1][0], legs[i + 1][1], v_lim)
                if v is not None:
                    state = JointState(q.tolist(), v.tolist())
            joint_waypoints.append(JointWaypoint(state, **kwargs))
        return joint_waypoints

    def _joint_velocity_ceiling(self) -> np.ndarray:
        """The per-joint target-velocity ceiling RUCKIG will validate this motion
        against: franky scales the robot's own `joint_velocity_limit` by the
        session's velocity dynamics factor and refuses the whole motion
        (ErrorInvalidInput -> "Motion planner failed with error code -100")
        if ANY waypoint target velocity component is above it.

        Read from the live robot rather than hardcoded, because it is the one
        number here that MOVES: it tracks `set_speed`, so a run at speed 0.08 has
        a ceiling 47% lower than the default 0.15 and a frozen constant is wrong
        for every run that is not at the default."""
        lim = np.asarray(self.robot.joint_velocity_limit.get(), dtype=float).reshape(7)
        return float(self.relative_dynamics_factor) * lim

    @staticmethod
    def _passthrough_velocity(q_here: np.ndarray, q_next: np.ndarray,
                              t_next: float, v_lim: np.ndarray) -> "np.ndarray | None":
        """Target joint velocity for PASSING THROUGH a blend>0 waypoint: the next
        leg's average joint velocity, scaled (as one vector, preserving direction)
        so that per joint it can still brake to zero within that leg's own travel
        (|v_j| <= sqrt(2 * a_eff * d_j * PASSTHROUGH_BRAKE_MARGIN)) AND stays under
        `v_lim`, the caller's per-joint Ruckig ceiling (_joint_velocity_ceiling).
        a_eff uses the most conservative rated FR3 joint acceleration, so the
        braking bound holds for every joint. Returns None when the next leg is
        degenerate (no travel) -- the waypoint then keeps today's full stop."""
        d = np.asarray(q_next, dtype=float) - np.asarray(q_here, dtype=float)
        d_abs = np.abs(d)
        if float(d_abs.max()) < 1e-9:
            return None
        v = d / max(float(t_next), 1e-6)
        a_eff = FR3_MIN_RATED_JOINT_ACCEL_RAD_S2 * ACCELERATION_DYNAMICS_FACTOR
        v_eff = np.asarray(v_lim, dtype=float) * PASSTHROUGH_VEL_HEADROOM
        scale = 1.0
        for vj, dj, v_eff_j in zip(np.abs(v), d_abs, v_eff):
            if vj <= 0.0:
                continue
            cap = min(math.sqrt(2.0 * a_eff * dj * PASSTHROUGH_BRAKE_MARGIN), v_eff_j)
            scale = min(scale, cap / vj)
        return v * min(scale, 1.0)

    def move_tool_traj(self, path: Sequence[Sequence[float]], is_async: bool = False,
                       max_angular_vel: float = DEFAULT_MAX_ANGULAR_VEL_RAD_S,
                       max_joint_swing_rad: "float | None" = None) -> list[float]:
        """TCP waypoint path executed as ONE jerk-limited JOINT-space trajectory.

        Each waypoint is [x, y, z, rx, ry, rz, speed, acceleration, blend]
        (speed/accel/blend optional; accel ignored). Per-waypoint `speed` (m/s)
        paces its leg via a minimum segment time. blend == 0 -> full stop at the
        waypoint; blend > 0 on an intermediate waypoint -> passed through with a
        bounded joint velocity (the corner is rounded, no stop -- see
        _passthrough_velocity). The last waypoint always stops.

        Execution: the Cartesian path is densely sampled
        and IK'd with each solve warm-started from the previous DENSE solution
        (_solve_branch_continuous_ik), so the joint solutions stay on ONE IK
        branch; the functional waypoints then run as a single JointWaypointMotion
        (Ruckig joint OTG). This is the field-consensus robust primitive
        (polymetis/deoxys/frankapy/franka-control all plan in joint space) and
        deliberately does NOT stream CartesianPose: libfranka does not re-limit
        the joint velocities its internal Cartesian generator produces after IK,
        which is what trips cartesian_motion_generator_joint_velocity_discontinuity
        at corners/singularities. Joint-space Ruckig avoids BOTH that reflex and
        the branch-swing acceleration_discontinuity. An unreachable target or an
        unavoidable branch jump raises BEFORE any motion starts. The TCP arcs
        slightly between functional waypoints (densification keeps it near the
        straight line) and the arm stops at blend-0 waypoints. This is the only
        supported trajectory execution path. Returns the measured TCP
        pose after the call."""
        self._require()
        self._require_hand_idle_for_arm_motion()
        parsed = self._parse_tcp_waypoints(path, "move_tool_traj")
        self._require_kin()
        start_pose = self._tool_pose_base()
        joint_waypoints = self._solve_branch_continuous_ik(
            start_pose, parsed, max_angular_vel, max_joint_swing_rad=max_joint_swing_rad)
        self._warn_and_recover()
        self._safe_move(JointWaypointMotion(joint_waypoints), is_async)
        return self.get_tool_pose()

    # ---------------- servo control ----------------

    def servo_joint(self, joint_values: Sequence[float], speed: float = 0.0,
                    acceleration: float = 0.0, time: float = 0.008,
                    lookahead_time: float = 0.03, gain: int = 300) -> None:
        """Real-time joint servo: non-blocking, call in a loop with nearby
        targets. Each call sends an asynchronous JointMotion that PREEMPTS the
        previous one; franky replans online from the current commanded state,
        so consecutive targets blend smoothly. The trailing parameters are
        accepted for call-compatibility and ignored (the dynamics factor
        governs). Validated at ~30 Hz (~0.2 ms per call)."""
        self._require()
        self._require_hand_idle_for_arm_motion()
        q = [float(v) for v in joint_values]
        if len(q) != 7:
            raise ValueError(f"servo_joint needs 7 joint values, got {len(q)}")
        self._safe_move(JointMotion(JointState(q)), is_async=True)

    def servo_tool(self, pose: Sequence[float], speed: float = 0.0,
                   acceleration: float = 0.0, time: float = 0.008,
                   lookahead_time: float = 0.03, gain: int = 300) -> None:
        """Real-time TCP pose servo: non-blocking, call in a loop with nearby targets.

        A warm-started DLS-IK -> franky JointMotion servo whose IK seed persists
        (branch-continuous) at a tight tolerance (SERVO_IK_POS_TOL_M): WE do the IK and
        stream JOINT targets, so the command stays smooth in joint space regardless of the
        policy's Cartesian path. Raises RuntimeError if a target is unreachable."""
        pose = list(pose)
        if len(pose) != 6:
            raise ValueError(f"tool pose must be [x,y,z,rx,ry,rz], got {len(pose)} values")
        base_pose = self._new_to_base(pose)
        quat = R.from_rotvec(np.asarray(base_pose[3:6], dtype=float)).as_quat()
        self._require_kin()
        if self._servo_q is None:
            # FIRST tick of a NEW stream (stop_move resets the seed): a stream
            # entry is a motion entry, so it passes the same seam every other
            # motion entry does -- renew the lease, clear a latched fault loudly.
            # Without this, a reflex on the PREVIOUS stream's last tick (which
            # only the stop's drain saw -- _say_drained_reflex) stayed latched
            # into the next episode, and an episode whose reset mode issues no
            # motion (reset: "tare"/"none") burned its whole attempt on a
            # foregone `command not possible in the current mode ("Reflex")`
            # reject at its first tick (franka-right, 2026-08-26). ONCE per
            # stream, never per tick: a recover pre-armed before every tick is
            # what makes the following reflex fire instantly (franka_ros#316);
            # a mid-stream reflex must still surface as the fault it is.
            self._warn_and_recover()
            self._servo_q = np.asarray(self.get_joint_angles(), dtype=float)
            self._servo_q_rest = self._servo_q.copy()
        try:
            q_sol = self.kin.ik(base_pose[:3], quat, q_init=self._servo_q,
                                q_rest=self._servo_q_rest,
                                pos_tol=SERVO_IK_POS_TOL_M, ori_tol=SERVO_IK_ORI_TOL_RAD)
        except ValueError as e:
            raise FrankaMotionRefused(f"servo_tool IK failed for pose {pose}: {e}") from e
        self._servo_q = q_sol
        self.servo_joint(q_sol.tolist())

    # ---------------- force / contact ----------------

    def set_collision_behavior(self, torque_thresholds: float | Sequence[float],
                               force_thresholds: float | Sequence[float]) -> None:
        """Set the contact/collision reflex thresholds: torque (Nm, scalar or
        7 per-joint values) and force (N / Nm, scalar or 6 per-axis values).
        The robot REFLEX-STOPS when external loads exceed them -- raise the
        thresholds before deliberate contact work so pressing does not trip a
        reflex. Resets to franky's defaults (20 Nm / 30 N) on reconnect."""
        self._require()
        if self.is_running():
            raise RuntimeError("cannot set collision behavior while a motion is in control")
        try:
            self.robot.set_collision_behavior(torque_thresholds, force_thresholds)
        except _FRANKY_EXC as e:  # direct franky call (not a motion) -> translate so it never leaks raw (F5)
            raise wrap_franky_exc(e, "set_collision_behavior") from e

    def move_tool_impedance(self, pose: Sequence[float], duration_s: float,
                            translational_stiffness: float = IMPEDANCE_TRANSLATIONAL_STIFFNESS,
                            rotational_stiffness: float = IMPEDANCE_ROTATIONAL_STIFFNESS,
                            force_constraints: Optional[Sequence[Optional[float]]] = None,
                            is_async: bool = False) -> list[float]:
        """Move to / hold a TCP pose for `duration_s` under torque-based
        cartesian impedance control -- the force-control entry point:

        - stiffness sets how strongly position errors are corrected
          (F ~= stiffness x error), i.e. how hard the arm pushes back when
          obstructed;
        - `force_constraints` = [fx, fy, fz, tx, ty, tz] (None entries =
          unconstrained; base-frame axes) CAPS the exerted wrench on an axis:
          the arm applies up to that force there instead of tracking position.
          E.g. press ~5 N downward: target below the surface +
          force_constraints=[None, None, -5.0, None, None, None].

        Runs on the torque interface (independent of the cartesian pose
        generator). The pose is tracked softly, with no trajectory shaping --
        use it for contact phases, not free-space transit. Returns the
        measured TCP pose after the call."""
        self._require()
        self._require_hand_idle_for_arm_motion()
        pose = list(pose)
        if len(pose) != 6:
            raise ValueError(f"tool pose must be [x,y,z,rx,ry,rz], got {len(pose)} values")
        if force_constraints is not None:
            force_constraints = list(force_constraints)
            if len(force_constraints) != 6:
                raise ValueError("force_constraints must have 6 entries (None = unconstrained)")
            if self._has_install_transform:
                raise ValueError("force_constraints are base-frame axis caps and cannot be "
                                 "expressed in a rotated working frame (install_rad != 0)")
        base_pose = self._new_to_base(pose)
        target = Affine(np.asarray(base_pose[:3], dtype=float),
                        R.from_rotvec(np.asarray(base_pose[3:6], dtype=float)).as_quat())
        self._warn_and_recover()
        self._safe_move(
            CartesianImpedanceMotion(
                target, Duration(int(float(duration_s) * 1000)),
                translational_stiffness=float(translational_stiffness),
                rotational_stiffness=float(rotational_stiffness),
                force_constraints=force_constraints,
            ),
            is_async,
        )
        return self.get_tool_pose()

    def move_until_force(self, pose: Sequence[float], speed: float = DEFAULT_TOOL_SPEED_M_S,
                         force_threshold: float = GUARD_FORCE_THRESHOLD_N,
                         max_angular_vel: float = DEFAULT_MAX_ANGULAR_VEL_RAD_S,
                         ) -> tuple[list[float], bool]:
        """Guarded move: head toward `pose` and stop as soon as the magnitude of
        the measured external force on any axis exceeds `force_threshold` (N) --
        evaluated by the robot at 1 kHz on the raw force estimate (set_ft_zero bias
        does not apply). Blocking. Returns (measured_pose, triggered).

        Hybrid, chosen by on-robot comparison (2026-06-30):
        - PRIMARY: franky-native CartesianMotion (Ruckig plans a smooth straight-line
          descent, tiny per-step joint motion -> no branch-swing phantom; measured to
          reach real contact) carrying a Measure.FORCE CartesianPoseReaction.
        - FALLBACK: if franky rejects the start as singular (Ruckig's Cartesian IK is
          ill-conditioned at singularities -- "cannot start at singular pose"), descend
          with our warm-seeded DLS-IK -> JointWaypointMotion, which is damped and
          singularity-robust. This is why the pick no longer grabs air: a singular
          start no longer aborts the descent (it used to be swallowed by the caller).

        Typical contact search: raise set_collision_behavior first so the guard fires
        before the reflex does."""
        self._require()
        self._require_hand_idle_for_arm_motion()
        pose = list(pose)
        if len(pose) != 6:
            raise ValueError(f"tool pose must be [x,y,z,rx,ry,rz], got {len(pose)} values")
        base_pose = self._new_to_base(pose)
        pos = np.asarray(base_pose[:3], dtype=float)
        rot = R.from_rotvec(np.asarray(base_pose[3:6], dtype=float))
        speed = float(speed) if speed > 0 else DEFAULT_TOOL_SPEED_M_S

        t = abs(float(force_threshold))
        condition = (
            (Measure.FORCE_X < -t) | (Measure.FORCE_X > t)
            | (Measure.FORCE_Y < -t) | (Measure.FORCE_Y > t)
            | (Measure.FORCE_Z < -t) | (Measure.FORCE_Z > t)
        )

        # PRIMARY: franky-native CartesianMotion + Cartesian force reaction.
        hit = {"triggered": False}
        motion = CartesianMotion(Affine(pos, rot.as_quat()),
                                 relative_dynamics_factor=CONTACT_RDF)
        reaction = CartesianPoseReaction(condition, CartesianStopMotion())
        reaction.register_callback(lambda *_: hit.__setitem__("triggered", True))
        motion.add_reaction(reaction)
        self._warn_and_recover()
        # The cartesian speed limits belong to THIS guarded move and are restored
        # when it ends, on every path -- including the fallback leg and the reflex
        # that sends it there.
        with self._limits_held(**self._cartesian_speed_limits(speed, float(max_angular_vel))):
            try:
                self._safe_move(motion, is_async=False)
            except (ControlException, RuntimeError) as e:
                # Singular start (or a mid-motion Cartesian reflex): fall back to the
                # damped, singularity-robust joint descent so the guard still runs.
                print(f"[move_until_force] franky CartesianMotion rejected "
                      f"({type(e).__name__}); DLS-IK joint fallback", flush=True)
                self._require_kin()
                q_seed = np.asarray(self.get_joint_angles(), dtype=float)
                try:
                    q_sol = self.kin.ik(pos, rot.as_quat(), q_init=q_seed, q_rest=q_seed)
                except ValueError as ik_e:
                    raise FrankaMotionRefused(
                        f"move_until_force IK failed for pose {pose}: {ik_e}") from ik_e
                hit = self._joint_descent(q_sol, pos, rot, speed, max_angular_vel, condition)
            _time.sleep(REACTION_SETTLE_S)  # reaction callbacks fire asynchronously
        return self.get_tool_pose(), hit["triggered"]

    def _joint_descent(self, q_sol, pos: np.ndarray, rot: R, speed: float,
                       max_angular_vel: float, condition) -> dict:
        """Build + run move_until_force's fallback leg: a single damped joint
        descent to the DLS-IK solution `q_sol`, paced by the SHARED
        _segment_min_timer (identical MIN_SEGMENT_TIME_S / Duration(ceil(ms))
        rounding as move_tool_traj -- no re-derived segment math inline) and
        carrying the force `condition` reaction. Returns the reaction-hit dict
        (the caller reads it after REACTION_SETTLE_S). Numeric behavior is pinned
        by the franka golden traces (move_until_force_*_fallback)."""
        min_time_s = self._segment_min_timer(max_angular_vel)(pos, rot, speed)
        kwargs = {}
        if min_time_s > MIN_SEGMENT_TIME_S:
            kwargs["minimum_time"] = Duration(int(math.ceil(min_time_s * 1000.0)))
        hit = {"triggered": False}
        jmotion = JointWaypointMotion([JointWaypoint(JointState(q_sol.tolist()), **kwargs)])
        jreaction = JointPositionReaction(condition, JointStopMotion())
        jreaction.register_callback(lambda *_: hit.__setitem__("triggered", True))
        jmotion.add_reaction(jreaction)
        self._warn_and_recover()
        self._safe_move(jmotion, is_async=False)
        return hit

    # ---------------- stop ----------------

    def stop_joints(self, acceleration: float = 1.0, is_async: bool = False) -> None:
        """Stop a joint-interface motion. Kept as an interface member (control_server
        _INTERRUPT set + UR parity); on this backend every motion runs on the joint
        interface, so it is the same hard-stop as stop_move (which is safe any time).
        `acceleration`/`is_async` accepted for call-compat, ignored."""
        self.stop_move()

    def stop_tool(self, acceleration: float = 1.0, is_async: bool = False) -> None:
        """Stop a tool motion. Tool motions run on the joint interface here, so this
        is the same hard-stop as stop_move; kept as an interface member (control_server
        _INTERRUPT set + UR parity). `acceleration`/`is_async` accepted for call-compat,
        ignored."""
        self.stop_move()

    def stop_servo(self, acceleration: float = 1.0) -> None:
        """Stop servo motion. `acceleration` accepted for call-compat, ignored."""
        self.stop_move()

    def stop_move(self) -> None:
        """Hard-stop whatever motion is running (safe to call any time)."""
        self._servo_q = None
        self._servo_q_rest = None
        self._async_move_deadline = None                 # motion stopped -> disarm the hang cap
        if self.robot is None:
            return
        # Guard with the franky lock so stop() does not race a join_motion()
        # slice; a sliced join holds the lock at most MOTION_JOIN_SLICE_S, so a
        # concurrent abort (the collector's _abort_hung_stage on the main thread)
        # waits no longer than that before the hard-stop lands.
        with self._lock:
            try:
                self.robot.stop()
            except Exception:  # noqa: BLE001 -- teardown: a hard-stop must never itself raise
                pass
            # robot.stop() makes the control thread raise; franky stores that
            # exception and rethrows it on the NEXT move()/join_motion(). Drain it
            # here so the next motion command is not poisoned.
            try:
                joined = self.robot.join_motion(STOP_JOIN_TIMEOUT_S)
                if not joined:
                    print(f"[franka] WARNING: control thread still running "
                          f"{STOP_JOIN_TIMEOUT_S}s after stop", file=sys.stderr, flush=True)
            except Exception:  # noqa: BLE001 -- draining the INTENTIONAL stored stop-exception
                pass
            self._say_drained_reflex()

    def _say_drained_reflex(self) -> None:
        """Name a reflex that the stop's drain above destroyed.

        The drain discards whatever join_motion raises, and it has to: robot.stop()
        MAKES the control thread raise, and that bookkeeping exception must never
        propagate out of a hard stop. But it cannot tell that exception apart from a
        GENUINE reflex the arm suffered moments earlier, which franky had stored for
        the next move()/join_motion() to rethrow. Both arrive the same way, so a
        real fault is thrown out with the bookkeeping one.

        THAT IS THE SILENT HALF OF THE STREAMED-SERVO FAULT CHANNEL. A reflex on a
        servo tick that HAS a successor is rethrown by the successor's robot.move()
        and the control server logs `stream servo_tool failed`. A reflex on a
        stream's LAST tick has no successor: the next thing to touch franky is this
        stop, called from the episode loop's own finally. Nothing is logged, nothing
        is published, the fault count never moves, and the arm stays latched until
        some later motion's _warn_and_recover clears it and prints the only line
        anybody ever sees.

        franka-right, 2026-08-26: ONE `stream servo_tool failed` in the whole
        control-server log but TWO `clearing latched robot errors
        ["joint_motion_generator_acceleration_discontinuity"]`. The first clear had
        no fault line before it and three clean moves behind it, so an
        acceleration_discontinuity latched inside a ~2.9 s servo episode and left no
        trace anywhere. Faults we cannot see are faults we cannot count.

        So ASK THE ARM rather than try to classify an exception: errors latched
        after the drain are a fault the arm actually suffered. That is a state read,
        not a message match (which this backend forbids), and the same log shows the
        premise holds -- three consecutive moves printed no clearing line, so
        robot.stop() does not latch errors by itself.

        SAY IT ONLY. Recovering here would pre-arm a recover before the caller's
        next motion, which is what makes the following reflex fire instantly
        (franka_ros#316); _warn_and_recover remains the one place that clears."""
        try:
            if not self.robot.has_errors:
                return
            errors = self.robot.state.current_errors
        except (*_FRANKY_EXC, RuntimeError):
            return          # a diagnostic read must never turn a stop into a fault
        print(f"[franka] WARNING: a reflex latched during the motion this stop ended "
              f"and the stop's drain was the only thing that saw it: {errors} -- if "
              f"this was a servo stream, its last tick failed and no `stream ... "
              f"failed` line exists for it; the next motion clears the latch",
              file=sys.stderr, flush=True)

    def _cartesian_speed_limits(self, linear_speed: float, angular_speed: float) -> dict:
        """The rdf-compensated cartesian velocity limits for ONE guarded move, as
        a `_limits_held` override map -- so the effective speed (factor x limit)
        equals the requested one, clamped to the robot maxima.

        It COMPUTES rather than sets, because the setting has to be paired with a
        restore and `_limits_held` is the one place that pairs them. As a setter
        (`_set_velocity_limits`) it had no restore at all, so every guarded contact
        move left the robot's cartesian velocity limits wherever that move's
        contact speed put them, for the rest of the session."""
        rdf = max(self.relative_dynamics_factor, 1e-3)
        tv = self.robot.translation_velocity_limit
        rv = self.robot.rotation_velocity_limit
        return {"translation_velocity_limit": min(linear_speed / rdf, tv.max),
                "rotation_velocity_limit": min(angular_speed / rdf, rv.max)}

    def _warn_and_recover(self) -> None:
        """Clear latched errors before a motion, loudly: recovering silently
        would re-command motion after a collision/reflex with no operator-
        visible signal (stderr so it is not buried in a recording loop's
        stdout). No-op when the robot has no errors, so it is safe in paths
        that preempt a running motion.

        This is the MOTION entry seam for the 5.9.0 session lease: renew it
        before touching the robot (see session.ensure_fci_lease for the full
        story). ControlException is NOT caught -- a reflex is a real fault."""
        self.ensure_fci_lease()
        if self.robot.has_errors:
            print(
                f"[franka] WARNING: clearing latched robot errors before motion: "
                f"{self.robot.state.current_errors}",
                file=sys.stderr,
                flush=True,
            )
            self.robot.recover_from_errors()

    def _safe_move(self, motion: object, is_async: bool) -> None:
        """Dispatch a franky motion. Asynchronous moves return immediately. A
        BLOCKING (is_async=False) move is executed asynchronously and then joined
        in MOTION_JOIN_SLICE_S slices via join_motion(): franky releases its
        control/state mutex between slices, so a concurrent state read (the 30 Hz
        recording loop) interleaves instead of deadlocking against the in-control
        move. If the motion does not finish within MOTION_JOIN_TIMEOUT_S -- the
        real-time loop is starved (host overload) and the move is wedged -- it is
        stopped and surfaced as a FrankaMotionRefused ('motion hang'), which the
        collector classifies as 'hang' and recovers from, rather than freezing
        forever.

        This is THE single franky MOTION boundary (item 3): a franky reflex/abort/
        link-drop raised DURING the move is caught here and classified into a typed
        FrankaError (finish_move is its async twin). We do NOT recover_from_errors()
        here: the motion has not been stopped yet, and pre-arming a recover before
        the stop makes the next reflex fire instantly (franka_ros#316) -- the caller
        stops + recovers."""
        try:
            with self._lock:
                self.robot.move(motion, asynchronous=True)
            if is_async:
                # Arm the async-move hang deadline: the control server issues the
                # move async and reaps it via check_move_hang()/finish_move(), so
                # the SAME MOTION_JOIN_TIMEOUT_S cap the blocking join enforces
                # below is applied to async issue too -- from ONE place (F4).
                self._async_move_deadline = _time.perf_counter() + MOTION_JOIN_TIMEOUT_S
                return
            deadline = _time.perf_counter() + MOTION_JOIN_TIMEOUT_S
            while True:
                with self._lock:
                    joined = self.robot.join_motion(MOTION_JOIN_SLICE_S)
                if joined:
                    return
                if _time.perf_counter() >= deadline:
                    self.stop_move()
                    raise self._motion_hang_error()
        except FrankaError:
            raise                                        # already typed (incl. the motion-hang abort)
        except Exception as e:  # noqa: BLE001 -- THE single franky motion boundary: classify
            raise self._classify_move_fault(e, "Franka move rejected") from e
