"""Franka FR3 arm controller.

Built on franky (libfranka). One class, `FrankaArmController`, covers state
reading, kinematics, point-to-point / trajectory / velocity / servo control,
gripper, force/contact primitives, and failure recovery.

Conventions
-----------
- A tool pose is ``[x, y, z, rx, ry, rz]``: position in meters + axis-angle
  rotation (radians), in the working frame (= robot base frame unless
  ``install_rad`` is set). Twists/wrenches are returned in the same frame.
- Joint values are 7-vectors in radians.
- Motion methods block until the motion finishes unless ``is_async=True``
  (speed/servo methods are always non-blocking).

Errors
------
- ``RuntimeError``: motion rejected/aborted by the robot, or IK failure inside
  a motion method (raised BEFORE any motion starts).
- ``ValueError``: malformed arguments (wrong lengths, bad waypoints).
- ``ik()`` is the exception: it returns ``None`` on failure (call-compat).

Cell-specific facts (hardware-validated 2026-06-12)
---------------------------------------------------
- Cartesian targets are executed as IK + joint motions: the raw cartesian POSE
  interface aborts long (>~15 cm) moves with
  ``cartesian_motion_generator_*_discontinuity`` reflexes on this cell, while
  joint motions run reliably.
- Only one FCI process may connect to a robot at a time.
- After an aborted motion the robot freezes its commanded-pose registers; use
  ``reset_session()`` before commanding cartesian motion again.
"""
from __future__ import annotations

from evo_franka.session import SessionMixin
from evo_franka.state import StateMixin
from evo_franka.motion import MotionMixin
from evo_franka.gripper import GripperMixin


class FrankaArmController(SessionMixin, StateMixin, MotionMixin, GripperMixin):
    """Thin façade: the EXACT public surface of the pre-split monolith, assembled
    from the four concern mixins. Every method is defined in exactly one concern
    module (session / state / motion / gripper); this class only combines them so
    `self` remains a single object sharing the locks, caches and frames that
    __init__ (SessionMixin) builds. Callers -- the adapter, control_server,
    control_client, teleop, demos, tests -- keep importing FrankaArmController
    from here unchanged."""
