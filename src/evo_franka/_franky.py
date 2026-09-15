"""THE single ``franky`` import boundary for the Franka backend (F3).

Every ``robots/franka`` concern module (session / state / motion / gripper, and
the ``driver`` façade) imports the franky names it needs from HERE, never
``import franky`` directly. Two payoffs:

  * ONE place does ``import franky`` -- the architecture test
    (``tests/test_franka_arch.py``) enforces it, so a franky type/exception can
    never leak in through a stray import elsewhere;
  * the F1 recording fake binds franky ONCE, at this module, instead of chasing a
    ``from franky import`` in every concern file. ``tests/franka_fake.
    load_isolated_driver`` installs the fake into ``sys.modules['franky']`` and
    re-imports this module (plus the concern modules) fresh, so ``import franky``
    below resolves to the fake and every re-export becomes a recording class.

The exception tuples ``_FRANKY_EXC`` / ``_STATE_READ_EXC`` live here too (they
were at driver module scope before the split; the state, motion and session
concerns all catch against them, so they must be shared).
"""
from __future__ import annotations

import franky
from franky import (
    Affine,
    CartesianImpedanceMotion,
    CartesianMotion,
    CartesianPoseReaction,
    CartesianStopMotion,
    Duration,
    Gripper,
    JointMotion,
    JointPositionReaction,
    JointState,
    JointStopMotion,
    JointWaypoint,
    JointWaypointMotion,
    Measure,
    RelativeDynamicsFactor,
    Robot,
)

# The RE-EXPORT surface: every concern module imports franky names from here
# (enforced by tests/test_franka_arch.py). __all__ declares that intent so an
# unused-import lint reads these as re-exports, not dead code.
__all__ = [
    "franky", "Affine", "CartesianImpedanceMotion", "CartesianMotion",
    "CartesianPoseReaction", "CartesianStopMotion", "Duration", "Gripper",
    "JointMotion", "JointPositionReaction", "JointState", "JointStopMotion",
    "JointWaypoint", "JointWaypointMotion", "Measure", "RelativeDynamicsFactor",
    "Robot", "_FRANKY_EXC", "_STATE_READ_EXC",
]

# franky's libfranka-backed exceptions. A dropped UDP datagram surfaces as
# NetworkException on a synchronous state read; a reflex/abort during a motion
# surfaces as ControlException. We catch these by TYPE (not message) where we
# fall back to cached state (item 1) and where we route an in-move failure into
# guarded recovery (item 3). franky may not export every name across versions,
# so resolve them defensively and fall back to the base Exception.
try:  # pragma: no cover - import shape varies by franky build
    from franky import ControlException, NetworkException
    # The TRANSIENT comms/read errors that warrant last-value fallback. franky's
    # exception hierarchy is flat (each inherits straight from Exception, with no
    # common franky base -- verified on the installed build), so we enumerate the
    # relevant ones explicitly: a dropped FCI datagram (NetworkException), a
    # realtime-loop hiccup (RealtimeException), and a protocol read glitch
    # (ProtocolException). NOTE we deliberately do NOT include ControlException
    # here: a control fault (reflex/abort) during a read is a REAL fault that must
    # propagate to be classified, not masked as a frozen-state fallback.
    _STATE_READ_EXC: tuple = tuple(
        e for e in (
            NetworkException,
            getattr(franky, "RealtimeException", None),
            getattr(franky, "ProtocolException", None),
        ) if isinstance(e, type)
    )
except Exception:  # noqa: BLE001 - keep the controller importable on odd builds
    ControlException = NetworkException = Exception  # type: ignore[assignment,misc]
    _STATE_READ_EXC = (Exception,)

# The franky exceptions we catch-and-classify at the boundaries (deduped; on an
# odd build where the imports above fell back, this collapses to (Exception,)).
_FRANKY_EXC: tuple = tuple(dict.fromkeys(
    (ControlException, NetworkException, *_STATE_READ_EXC)))
