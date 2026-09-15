"""Robot-boundary typed errors -- the robot-AGNOSTIC base of every backend's typed
error hierarchy.

A backend's driver translates its SDK faults into subclasses of ``RobotError`` (see
robots/franka/errors.py: the FrankaError family, wrapped at the franky boundary by
``wrap_franky_exc``). Callers OUTSIDE robots/ (tools/arm/teleop.py, the panel) then branch
on TYPE, not on SDK error-string markers (the old ``"UDP"`` / ``"NetworkException"`` /
``"libfranka"`` substring matching). ``RobotError`` inherits ``RuntimeError`` on purpose
so every existing ``except RuntimeError`` site keeps catching these unchanged -- this is
a strictly additive typing.

Both backends are now typed (robots/franka/errors.py, robots/ur/errors.py), so a
caller's ``except RobotError`` branch catches EITHER arm. That is why the operator
remedy travels on the error (``operator_fix``) instead of being written into the
handler: a caller outside robots/ cannot know whether a Desk, a teach pendant, or
neither is involved, and one that guesses tells a UR operator to release Franka
brakes (measured on ur-tactile 2026-08-31).
"""
from __future__ import annotations


class RobotError(RuntimeError):
    """Base for every typed fault raised at a robot backend's boundary."""

    #: What the OPERATOR should do about this fault, in their arm's own vocabulary.
    #: Each backend's class overrides it; empty means "no arm-specific advice".
    operator_fix: str = ""


class MotionRefused(RobotError):
    """A motion was refused in PLANNING -- nothing was actuated (unreachable /
    non-converging IK, mode not ready, no branch-continuous solution). Safe to
    retry with a DIFFERENT target; each backend's refusal class subclasses this
    so robot-agnostic callers (the collector's start-resample loop) can branch
    on type without importing a specific backend."""


class JointSwingExceeded(RuntimeError):
    """A planned TCP leg exceeds the caller's net joint-travel budget.

    Raised before motion so the caller can resample a closer start; this is a
    planning/reachability signal, not a runtime robot fault.
    """


class ProtectiveStop(RobotError):
    """The arm's own SAFETY controller stopped it -- a protective / safeguard /
    emergency stop. Not a planning refusal: the arm WAS moving, something (the
    controller's force or position limit, an e-stop, a safeguard input) stopped it,
    and it will refuse every further motion until a person clears the stop at the
    arm's own panel.

    Robot-agnostic like ``MotionRefused`` above and for the same reason: a caller
    outside robots/ (the jog session, which must say WHY it stopped before its
    process exits) has to distinguish "the arm is latched and needs a human" from
    "that target was unreachable" without importing a backend. Each backend's
    protective-stop class subclasses this and carries the remedy in its own
    vocabulary on ``operator_fix`` (robots/ur/errors.py: the pendant unlock)."""
