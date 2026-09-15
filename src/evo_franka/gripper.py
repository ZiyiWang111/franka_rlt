"""Franka gripper concern (F3 split of driver.py).

The Franka Hand: open / close (force-controlled grasp) / move / home / width /
grasped, plus the connection guard and the P2 motion-serialization guard
(_require_gripper_idle -- a Hand op is refused while an arm motion is in control).
Methods live on GripperMixin (combined by robots/franka/driver.py)."""
from __future__ import annotations

import time
from typing import Optional

from evo_franka._franky import NetworkException, _FRANKY_EXC
from evo_franka.errors import FrankaGripperError
from evo_franka.constants import (
    GRIPPER_ASYNC_TIMEOUT_S,
    GRIPPER_EPSILON_INNER_M,
    GRIPPER_EPSILON_OUTER_M,
    GRIPPER_GRASP_FORCE_N,
    GRIPPER_GRASP_SPEED_M_S,
    GRIPPER_MAX_FORCE_N,
    GRIPPER_OPEN_SPEED_M_S,
)


class GripperMixin:
    """Franka Hand half of FrankaArmController."""


    # ---------------- gripper (Franka Hand) ----------------

    def _ensure_gripper_motion_state(self) -> None:
        """Lazily initialise the serialisable state for one async Hand command."""
        if hasattr(self, "_gripper_motion"):
            return
        self._gripper_future = None
        self._gripper_motion_seq = 0
        self._gripper_motion = {
            "command_id": 0,
            "command": None,
            "status": "idle",
            "result": None,
            "error": None,
            "started_at": None,
            "completed_at": None,
            "width": None,
            "grasped": None,
        }

    def _cache_gripper_measurement(self) -> None:
        """Refresh width/grasped only when no Hand command owns the connection."""
        self._gripper_motion["width"] = float(
            self._hand("width", lambda: self.gripper.width))
        self._gripper_motion["grasped"] = bool(
            self._hand("is_grasped", lambda: self.gripper.is_grasped))

    def _start_gripper_motion(self, command: str, future_factory) -> dict:
        self._require_gripper_idle(command)
        self._ensure_gripper_motion_state()
        self._poll_gripper_motion()
        if self._gripper_motion["status"] == "running":
            raise FrankaGripperError(
                f"cannot start gripper {command}: command "
                f"{self._gripper_motion['command_id']} is still running")
        if self._gripper_motion["width"] is None:
            self._cache_gripper_measurement()
        future = self._hand(f"{command}_async", future_factory)
        self._gripper_motion_seq += 1
        self._gripper_future = future
        self._gripper_motion = {
            "command_id": self._gripper_motion_seq,
            "command": command,
            "status": "running",
            "result": None,
            "error": None,
            "started_at": time.monotonic(),
            "completed_at": None,
            # A live read can race franky's async request on the same Hand
            # connection. Preserve the last reliable values and make staleness
            # explicit until the Future finishes.
            "width": self._gripper_motion.get("width"),
            "grasped": self._gripper_motion.get("grasped"),
        }
        return dict(self._gripper_motion)

    def _poll_gripper_motion(self) -> dict:
        self._ensure_gripper_motion_state()
        future = self._gripper_future
        if future is None or self._gripper_motion["status"] != "running":
            return dict(self._gripper_motion)
        try:
            started_at = self._gripper_motion["started_at"]
            if started_at is not None and time.monotonic() - started_at > GRIPPER_ASYNC_TIMEOUT_S:
                self._hand("stop timed-out command", lambda: self.gripper.stop())
                self._gripper_future = None
                self._gripper_motion["status"] = "error"
                self._gripper_motion["error"] = (
                    f"gripper command timed out after {GRIPPER_ASYNC_TIMEOUT_S:.1f}s")
                self._gripper_motion["completed_at"] = time.monotonic()
                self._cache_gripper_measurement()
                return dict(self._gripper_motion)
            if not future.wait(0.0):
                return dict(self._gripper_motion)
            result = bool(future.get())
            self._gripper_future = None
            self._gripper_motion["status"] = "finished"
            self._gripper_motion["result"] = result
            self._gripper_motion["completed_at"] = time.monotonic()
            self._cache_gripper_measurement()
        except Exception as e:  # deferred Franky/FCI failure from the Future
            self._gripper_future = None
            self._gripper_motion["status"] = "error"
            self._gripper_motion["error"] = f"{type(e).__name__}: {e}"
            self._gripper_motion["completed_at"] = time.monotonic()
        return dict(self._gripper_motion)

    def get_gripper_motion_state(self) -> dict:
        """Non-blocking state for the current/last asynchronous Hand command."""
        self._require_gripper()
        self._ensure_gripper_motion_state()
        state = self._poll_gripper_motion()
        if state["width"] is None and state["status"] != "running":
            self._cache_gripper_measurement()
            state = dict(self._gripper_motion)
        state["measurement_stale"] = state["status"] == "running"
        return state

    def start_open_gripper(self, width: Optional[float] = None,
                           speed: float = GRIPPER_OPEN_SPEED_M_S) -> dict:
        """Start opening/moving the Hand and return immediately."""
        if width is None:
            return self._start_gripper_motion(
                "open", lambda: self.gripper.open_async(float(speed)))
        return self._start_gripper_motion(
            "open", lambda: self.gripper.move_async(float(width), float(speed)))

    def start_close_gripper(self, width: float = 0.0,
                            force: float = GRIPPER_GRASP_FORCE_N,
                            speed: float = GRIPPER_GRASP_SPEED_M_S,
                            epsilon_inner: float = GRIPPER_EPSILON_INNER_M,
                            epsilon_outer: float = GRIPPER_EPSILON_OUTER_M) -> dict:
        """Start a force-controlled grasp and return immediately."""
        return self._start_gripper_motion(
            "close",
            lambda: self.gripper.grasp_async(
                float(width), float(speed), float(force),
                epsilon_inner=float(epsilon_inner), epsilon_outer=float(epsilon_outer)),
        )

    def stop_gripper(self) -> bool:
        """Stop an asynchronous Hand command and publish its terminal state."""
        self._require_gripper()
        self._ensure_gripper_motion_state()
        result = bool(self._hand("stop", lambda: self.gripper.stop()))
        self._gripper_future = None
        if self._gripper_motion["status"] == "running":
            self._gripper_motion["status"] = "stopped"
            self._gripper_motion["result"] = result
            self._gripper_motion["completed_at"] = time.monotonic()
            self._cache_gripper_measurement()
        return result

    def _hand(self, op: str, fn):
        """Run a Franka Hand op, translating any raw franky fault into the typed
        FrankaGripperError so the Hand boundary never leaks a franky exception
        (F5). The Hand is a separate FCI channel; a franky op there raises the
        same ControlException/NetworkException family the arm does.

        This is the HAND entry seam for the 5.9.0 session lease (see
        session.ensure_fci_lease): the Hand connection dies with the idle
        session, and Hand ops are the demo's FIRST touch after quiet windows
        (model load runs before setup's home/open), so renew the lease here
        exactly like the motion seam -- the rebuild refreshes self.gripper and
        the lambdas re-read it. The Hand handle can ALSO expire alone while the
        arm stays active (a long motion sequence with no Hand ops), which the
        arm-side probe cannot see: that surfaces as NetworkException from fn(),
        and the SAME lease logic rebuilds the handle and retries ONCE. Any
        other fault -- and a retry that fails -- raises the typed error."""
        ensure = getattr(self, "ensure_fci_lease", None)
        if ensure is not None:      # hand-only sessions have no arm session to probe
            ensure()
        try:
            return fn()
        # RuntimeError included for the same reason as in ensure_fci_lease: a
        # DEAD franky handle raises the bare RuntimeError("Net Exception"),
        # only a dying live one raises NetworkException.
        except (NetworkException, RuntimeError) as e:
            if self._rebuild_hand():
                try:
                    return fn()
                except (*_FRANKY_EXC, RuntimeError) as e2:
                    e = e2
            raise FrankaGripperError(f"gripper {op} failed ({type(e).__name__}): {e}") from e
        except _FRANKY_EXC as e:
            raise FrankaGripperError(f"gripper {op} failed ({type(e).__name__}): {e}") from e

    def _rebuild_hand(self) -> bool:
        """Rebuild the franky.Gripper handle after its lease expired. Returns
        True only when a LIVE replacement is in place (probed); on failure the
        old handle stays and the caller reports honestly."""
        ip = getattr(self, "robot_ip", None)
        if not ip:
            return False
        try:
            from evo_franka._franky import Gripper
            g = Gripper(ip)
            _ = float(g.width)      # probe: only swap in a LIVE handle
            self.gripper = g
            return True
        except Exception:  # noqa: BLE001
            return False

    def has_gripper(self) -> bool:
        """True if a Franka Hand is connected (set at connect when it responds). A
        METHOD (not a property) so it is callable over the control-server RPC, which
        dispatches client calls as getattr(ctrl, name)(...)."""
        return self.gripper is not None

    def open_gripper(self, width: Optional[float] = None,
                     speed: float = GRIPPER_OPEN_SPEED_M_S) -> None:
        """Open the gripper fully, or to `width` (m) if given. Blocking.
        Refused while an arm motion is in control (P2 serialization)."""
        self._require_gripper_idle("open")
        if width is None:
            result = self._hand("open", lambda: self.gripper.open(float(speed)))
        else:
            result = self._hand("open", lambda: self.gripper.move(float(width), float(speed)))
        self._ensure_gripper_motion_state()
        self._gripper_motion.update(
            command="open", status="finished", result=bool(result), error=None,
            completed_at=time.monotonic())
        self._cache_gripper_measurement()

    def close_gripper(self, width: float = 0.0,
                      force: float = GRIPPER_GRASP_FORCE_N,
                      speed: float = GRIPPER_GRASP_SPEED_M_S,
                      epsilon_inner: float = GRIPPER_EPSILON_INNER_M,
                      epsilon_outer: float = GRIPPER_EPSILON_OUTER_M) -> bool:
        """Grasp with a controlled force. Blocking; returns True if an object
        was grasped (final width within [width-epsilon_inner,
        width+epsilon_outer]).

        `force` (N) is the continuous grasp force the Hand holds the object
        with (20-70 N continuous; up to 140 N PEAK for short grasps -- sustained
        holds above 70 N thermally derate the Hand). For the success flag to be meaningful,
        epsilon_inner must be SMALLER than `width`: an empty close ends near
        0 m, which only counts as failure if 0 < width - epsilon_inner.
        E.g. a ~4 mm USB cable: close_gripper(width=0.004, force=20,
        epsilon_inner=0.003) -> True with the cable, False on empty air.
        Refused while an arm motion is in control (P2 serialization)."""
        self._require_gripper_idle("grasp")
        # NOTE 2026-07-18: the settle-pulse loop (re-grasp until width stabilizes) was
        # REVERTED same-day -- on a real thin object the re-grasp transitions visibly
        # relaxed the hold ("closes then loosens", owner). One clean grasp; off-center
        # placement is handled by teaching straight + the executor's search-grasp.
        result = bool(self._hand("grasp", lambda: self.gripper.grasp(
            float(width), float(speed), float(force),
            epsilon_inner=float(epsilon_inner), epsilon_outer=float(epsilon_outer))))
        self._ensure_gripper_motion_state()
        self._gripper_motion.update(
            command="close", status="finished", result=result, error=None,
            completed_at=time.monotonic())
        self._cache_gripper_measurement()
        return result

    def move_gripper(self, width: float, speed: float = GRIPPER_OPEN_SPEED_M_S) -> None:
        """Move the fingers to `width` (m) without force control. Blocking.
        Refused while an arm motion is in control (P2 serialization)."""
        self._require_gripper_idle("move")
        result = self._hand("move", lambda: self.gripper.move(float(width), float(speed)))
        self._ensure_gripper_motion_state()
        self._gripper_motion.update(
            command="move", status="finished", result=bool(result), error=None,
            completed_at=time.monotonic())
        self._cache_gripper_measurement()

    def get_gripper_width(self) -> float:
        """Current finger opening (m)."""
        self._require_gripper()
        self._ensure_gripper_motion_state()
        state = self._poll_gripper_motion()
        if state["status"] == "running" and state["width"] is not None:
            return float(state["width"])
        return float(self._hand("width", lambda: self.gripper.width))

    def is_grasped(self, expected_width_m: "float | None" = None) -> bool:
        """Whether the gripper currently holds an object (per the Hand)."""
        # ``expected_width_m`` is the contract's held convention (a part's
        # measured held width). This hand has no such measurement and no
        # width band; it is accepted so every adapter passes one shape.
        self._require_gripper()
        self._ensure_gripper_motion_state()
        state = self._poll_gripper_motion()
        if state["status"] == "running" and state["grasped"] is not None:
            return bool(state["grasped"])
        return bool(self._hand("is_grasped", lambda: self.gripper.is_grasped))

    def home_gripper(self) -> None:
        """Home the fingers (full open/close calibration sweep). Run once
        after a Hand power cycle or if width readings look wrong. Refused while
        an arm motion is in control (P2 serialization)."""
        self._require_gripper_idle("home")
        result = self._hand("home", lambda: self.gripper.homing())
        self._ensure_gripper_motion_state()
        self._gripper_motion.update(
            command="home", status="finished", result=bool(result), error=None,
            completed_at=time.monotonic())
        self._cache_gripper_measurement()

    def _require_gripper(self) -> None:
        self._require()
        if self.gripper is None:
            raise RuntimeError("gripper unavailable (Hand FCI connection failed at connect)")

    def _require_gripper_idle(self, op: str) -> None:
        """Gripper actuation preflight (P2 serialization): connected AND no arm
        motion in control. Hand ops are SERIALIZED against arm motion with the
        same wait-or-refuse guard the other mutating ops (set_collision_behavior)
        use: refuse rather than clobber.
        The Franka Hand is a separate FCI channel, but a Hand command issued
        mid-motion is exactly the dual-command hazard the incident history (the UR
        gripper-vs-control clobber) says to forbid -- so we make the serialization
        explicit and enforced, not accidental."""
        self._require_gripper()
        self._ensure_gripper_motion_state()
        state = self._poll_gripper_motion()
        if state["status"] == "running":
            raise FrankaGripperError(
                f"cannot {op} the gripper while Hand command "
                f"{state['command_id']} is running")
        if self.is_running():
            raise FrankaGripperError(f"cannot {op} the gripper while a motion is in control")


class FrankaGripperSession(GripperMixin):
    """A Franka HAND session with NO arm FCI -- the smallest honest seam for the panel's
    manual gripper path (owner override, S2 deferred).

    The Franka Hand is a SEPARATE FCI channel (``franky.Gripper``), independent of the
    arm's realtime control session. Opening ONLY the Hand means a manual open/close/move:
      * needs no ``franky.Robot`` and no arm kinematics -> no lerobot / camera imports and
        no ~1 s arm FCI connect (the SPEED win), and
      * is immune to Desk operating mode and to whoever holds the ARM's control session
        (the RELIABILITY win: a grasp no longer "holds then bounces back" because a fresh
        arm-FCI session raced it).

    It REUSES ``GripperMixin`` verbatim -- the same grasp / open / move / width / grasped
    logic (and the same force / epsilon defaults) the driver and collection use -- so the
    manual path can never drift from the collection path. Only the two ARM-coupled guards
    are re-pointed at the Hand-only session: ``_require`` checks the Hand (there is no arm
    to be "not connected"), and ``is_running`` is always False (no arm motion can be in
    control, so the P2 serialization never refuses a Hand op)."""

    def __init__(self, robot_ip: str) -> None:
        self.robot_ip = robot_ip
        self.gripper = None  # set in connect() to a franky.Gripper

    def connect(self) -> "FrankaGripperSession":
        """Open ONLY the Franka Hand channel (no ``franky.Robot``, no arm FCI). Raises
        FrankaGripperError if the Hand does not respond, so teleop surfaces one clean line."""
        from evo_franka._franky import Gripper, _FRANKY_EXC  # cell-only import (franky)
        try:
            g = Gripper(self.robot_ip)
            _ = float(g.width)  # probe the Hand connection so a dead link fails HERE
        except (*_FRANKY_EXC, RuntimeError, OSError) as e:
            self.gripper = None
            raise FrankaGripperError(
                f"Franka Hand unavailable at {self.robot_ip} ({type(e).__name__}): {e}") from e
        self.gripper = g
        return self

    def disconnect(self) -> None:
        """Release the Hand handle (best-effort; franky.Gripper has no explicit close)."""
        self.gripper = None

    # -- the two arm-coupled guards, re-pointed at the Hand-only session --------
    def _require(self) -> None:
        if self.gripper is None:
            raise FrankaGripperError("gripper session not connected; call connect() first")

    def is_running(self) -> bool:
        return False  # no arm session -> never "in control" -> a Hand op is never refused


class _HandOnlyArm:
    """The gripper-verb surface (tools/teleop's cmd_gripper / cmd_gripper_state) over a
    FrankaGripperSession -- COMPOSITION, not subclassing, because the adapter exposes
    ``has_gripper`` as a PROPERTY while the driver/session exposes it as a METHOD. Supplies
    the two adapter properties the verbs read (has_gripper, gripper_max_force); every
    actuation (close_gripper / move_gripper / get_gripper_width / is_grasped) delegates to
    the Hand-only session. Lives HERE (inside robots/) so teleop carries no backend import
    and no robot-type branch (the arch census)."""

    def __init__(self, session: "FrankaGripperSession") -> None:
        self._s = session
        self.cameras = {}  # a gripper op NEVER opens a camera

    @property
    def has_gripper(self) -> bool:
        try:
            return bool(self._s.has_gripper())
        except Exception:  # noqa: BLE001
            return False

    @property
    def gripper_max_force(self) -> float:
        return GRIPPER_MAX_FORCE_N  # 100% on the 0-100% force scale (Franka Hand)

    def connect(self) -> "_HandOnlyArm":
        self._s.connect()
        return self

    def disconnect(self) -> None:
        self._s.disconnect()

    def __getattr__(self, name):  # close_gripper / move_gripper / get_gripper_width / is_grasped
        return getattr(self._s, name)


def make_hand_only_arm(robot_ip: "str | None" = None) -> _HandOnlyArm:
    """Build the gripper-verb surface over a Franka HAND-ONLY session (no arm FCI, no
    cameras). robot_ip defaults to the franka-left cell. NOT connected -- call .connect()."""
    if robot_ip is None:
        robot_ip = DEFAULT_ROBOT_IP
    return _HandOnlyArm(FrankaGripperSession(robot_ip))
