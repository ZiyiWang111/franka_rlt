"""Typed franka-boundary errors.

The robot only ever sees the franky call stream; every past runtime fault
(discontinuity reflex, singular-start refusal, dropped FCI link) surfaced from
franky as an untyped ``ControlException`` / ``NetworkException`` or as a
``RuntimeError`` carrying a string tag. This module gives the driver ONE typed
vocabulary for those faults so callers can branch on TYPE, not substrings.

Every class inherits ``RuntimeError`` on purpose:

  * the driver's public contract (a rejected/aborted motion raises RuntimeError,
    see driver.py's module docstring) keeps holding, and every existing
    ``except RuntimeError`` site -- notably the collector's motion-failure retry
    (``data_collection/session/collect_recovery.classify_motion_failure``) and
    ``move_until_force``'s DLS-IK fallback -- keeps catching them unchanged;
  * the message TEXT is preserved verbatim by the driver's classifier: it still
    carries the libfranka reason plus the ``[network/comms]`` / ``[control
    reflex]`` tags that ``classify_motion_failure`` greps for.

So this is a strictly additive typing of the SAME faults -- no behaviour change
for anything that already caught RuntimeError.
"""
from __future__ import annotations

from evo_franka.base_errors import MotionRefused, RobotError


class FrankaError(RobotError):
    """Base for every typed fault raised at the franky boundary. Subclasses the
    robot-agnostic ``RobotError`` (which is a RuntimeError) so callers outside robots/
    can catch the whole robot-boundary hierarchy by TYPE without importing franka."""


class FrankaSessionLost(FrankaError):
    """The FCI session / realtime link is gone: a ``NetworkException`` raised
    mid-move, or the state stream stopped delivering datagrams past the
    last-good-value fallback bound. Recoverable only by rebuilding the session
    (``heal_session`` / ``reset_session``), not by ``recover_from_errors``."""

    operator_fix = (
        "in Desk switch to EXECUTION mode, unlock the joints / release the brakes, "
        "confirm FCI is activated, and make sure no other FCI client is connected.")


class FrankaMotionRefused(FrankaError, MotionRefused):
    """A motion could not run: the robot rejected the start (mode not ready, a
    generic control abort) or planning failed BEFORE any motion began (IK had no
    solution, a synthesized motion-hang abort). Also a robot-agnostic
    ``MotionRefused`` so generic callers can branch without importing franka."""


class FrankaReflex(FrankaError):
    """A control REFLEX fired -- a velocity/acceleration/joint discontinuity, a
    collision, or a singular-start refusal. ``kind`` is the libfranka reflex
    family when it can be inferred from the message (else "")."""

    def __init__(self, message: str, kind: str = ""):
        super().__init__(message)
        self.kind = kind


class FrankaGripperError(FrankaError):
    """A Franka Hand op failed or was refused -- including the motion-
    serialization guard: the Hand may not actuate while an arm motion is in
    control."""


# franky comms/link exception class NAMES (a dropped datagram, an RT hiccup, a
# protocol glitch) -> FrankaSessionLost. Matched by name so this module stays
# franky-free (errors.py must never import franky) and works identically for the
# real franky and the recording fake (both name their classes the same).
_SESSION_LOST_NAMES = frozenset({"NetworkException", "RealtimeException", "ProtocolException"})


def wrap_franky_exc(e: BaseException, context: str = "") -> "FrankaError":
    """Translate a raw franky exception into the typed FrankaError vocabulary (F5).

    For the few driver call sites that reach franky OUTSIDE the two classifying
    boundaries (motion's _safe_move / state's _read_with_fallback) -- namely
    set_collision_behavior and recover -- so a franky ControlException /
    NetworkException can never cross the driver (and therefore the adapter)
    boundary raw. An already-typed FrankaError passes through unchanged.
    NetworkException-family -> FrankaSessionLost; ControlException -> FrankaReflex;
    anything else franky raised -> the FrankaError base. Message text preserved."""
    if isinstance(e, FrankaError):
        return e
    name = type(e).__name__
    prefix = f"{context}: " if context else ""
    if name in _SESSION_LOST_NAMES:
        return FrankaSessionLost(f"{prefix}{name}: {e}")
    if name == "ControlException":
        return FrankaReflex(f"{prefix}{name}: {e}", kind=reflex_kind(str(e)))
    return FrankaError(f"{prefix}{name}: {e}")


def reflex_kind(message: str) -> str:
    """Best-effort libfranka reflex family from an exception message (for
    ``FrankaReflex.kind`` -- observability only, never control logic)."""
    m = (message or "").lower()
    if "singular" in m:
        return "singular_start"
    if "discontinuity" in m:
        return "discontinuity"
    if "limit" in m:
        return "limit"
    if "collision" in m or "reflex" in m:
        return "collision"
    return ""
