"""Franka FR3 dynamics / driver / kinematics / gripper / control-IPC constants.

Robot knowledge lives WITH the robot: these FR3-specific numbers (jerk/accel/velocity
reflex thresholds, FCI-robustness bounds, motion/IK tolerances, Hand limits, the ZMQ
control-server tunables, and the cell presets) moved here from core/constants.py so
``core`` keeps only robot-agnostic values. Unlike the old machine-local core/constants,
this module is SHIPPED with the robot (part of the client build id), so a cell's FR3
dynamics are uniform + versioned rather than drifting per host.

The values marked hardware-validated were established on this cell on 2026-06-12;
do not retune casually.

WHAT MAY LIVE HERE (the rule, applied by hand -- only half of it is testable)
----------------------------------------------------------------------------
A constant in this module is a claim that ONE number is right for EVERY run this
backend will ever serve. Before adding or changing one, ask the owner's question:
"is this generic for any run, or does it work for one run and fail for another?"

  HARDWARE / CONTROLLER FACT -- legitimate, and it must CITE ITS SOURCE: a rated
  limit, a datasheet figure, a libfranka/franky default, or a measurement of the
  ARM (not of a task). FR3 joint limits, the rated jerk, the reflex threshold,
  the Hand's force range.

  TASK-DERIVED -- does not belong here at ALL, however well argued. Any value
  whose justification names a demo, a dataset, a checkpoint, a manipulated
  object, or the band a policy trained in. It will be right for that run and
  wrong for the next. `SERVO_MAX_LAG_M` was exactly this: a servo bound tuned to
  the 5 mm band a VGA place demonstration happened to end in, and it is gone.
  Its replacement is the shape to copy -- `servo_lead_ceiling_m()` is the
  hardware ceiling and `FrankaAdapter.set_servo_lead_bound()` is the task's
  optional bound underneath it. A driver exposes the ceiling; a task spends it.

  ARBITRARY -- chosen once, never validated. Allowed, but SAY SO. "Arbitrary,
  never measured" is worth more than a rationale invented after the fact, because
  it tells the next person the number is free to move. Several here are exactly
  this and now say so.

`tests/robots/test_franka_arch.py` enforces the part a machine can see: no NUMBER
in robots/franka or robots/base may be justified by a task word, and the driver
never imports `demo`. It CANNOT see the harder half -- whether a number that
mentions nothing is nonetheless non-universal (right at one speed, one link
latency, one payload). That is applied by hand, here, with the question above.
Where a value cannot be made universal, the honest form is a declared parameter
with a hardware-enforced range, NOT a compromise value: a compromise is precisely
the thing that works for one run and fails for the next.
"""

DEFAULT_ROBOT_IP = "172.16.0.2"  # left arm on this laptop
# ARBITRARY, NEVER VALIDATED. A conservative fraction of the FR3's rated velocity
# limits, picked at bring-up because slow is safe and never revisited against any
# measurement. It is the speed of every run that does not set one, so it deserves
# better than that: what is missing is the fastest fraction at which this cell
# still tracks a 30 Hz servo stream with control_command_success_rate 1.000, which
# is a measurable number nobody has measured. Callers override it per run
# (relative_dynamics_factor / set_speed), so this binds only the unspecified case.
DEFAULT_DYNAMICS_FACTOR = 0.15

# --- Decoupled dynamics factors (FCI-robustness Phase 1, item 2) ---------------
# franky scales the robot's rated velocity/acceleration/JERK limits by a
# RelativeDynamicsFactor. The single scalar DEFAULT_DYNAMICS_FACTOR scales all
# three uniformly; that couples jerk to velocity, and the FR3
# `*_motion_generator_*_acceleration_discontinuity` reflex is JERK-driven -- so a
# velocity that is otherwise fine still trips the reflex if the commanded jerk is
# near the rated limit (FR3 rated jerk: 5000 rad/s^3 joint, 4500 m/s^3 Cartesian).
# We therefore decouple the three and keep jerk a LOW, sub-rated fraction,
# matching the Polymetis / deoxys / SERL consensus that jerk must be limited
# independently. These are applied via franky.RelativeDynamicsFactor(velocity,
# acceleration, jerk) at connect (see FrankaArmController.connect); the velocity
# fraction stays the caller-supplied relative_dynamics_factor.
# Start conservative; tune UP empirically once the discontinuity reflex is gone.
ACCELERATION_DYNAMICS_FACTOR = 0.15  # fraction of rated acceleration limit
JERK_DYNAMICS_FACTOR = 0.08  # fraction of rated jerk limit (sub-rated; the reflex driver)

# --- State-read tolerance (FCI-robustness Phase 1, item 1) ---------------------
# libfranka forbids a blocking state read while a control/motion loop runs.
# The FCI control loop rate: libfranka exchanges commands/state with the arm at
# 1 kHz (Franka Control Interface documentation; franky's Ruckig OTG runs its
# control thread at this rate -- see robots/franka/control_server.py). This is
# the rate the SERVO CONSUMER ticks at, and therefore the ceiling on how fast a
# setpoint stream can be meaningfully accepted (adapter.servo_rate_max_hz):
# targets arriving faster than the loop consumes them just overwrite each other.
FCI_CONTROL_LOOP_HZ = 1000.0

# franky returns CACHED state during an async motion (safe), but does a
# synchronous readOnce() when the robot is IDLE -- which can throw
# NetworkException / "UDP receive: Timeout" if a UDP datagram is dropped under
# host load. The deoxys-standard fix is to fall back to the LAST good cached
# value (no retry, no sleep, no blocking) and continue. We BOUND consecutive
# fallbacks so we never record frozen state indefinitely: after this many
# consecutive failed reads of a given signal, recovery is attempted / a clear
# error is raised. ~10 frames is ~0.3 s at 30 Hz.
STATE_READ_MAX_CONSECUTIVE_FALLBACKS = 10

# --- Bounded wait-for-ready after recovery (FCI-robustness Phase 1, item 3) ----
# Recovery latency (recover_from_errors + the control thread becoming idle) is
# unbounded in principle, so we do NOT blind-sleep-then-continue; we poll the
# robot until it is no longer in control (idle/ready), capped by a timeout. This
# is the Polymetis/SERL "wait on the robot being ready, bounded" pattern.
RECOVER_READY_TIMEOUT_S = 5.0  # max wait for the robot to return to idle after a stop/recover
RECOVER_READY_POLL_S = 0.02  # poll cadence while waiting for idle

# env var: log EVERY state-read drop+reuse (else only the per-episode count). OFF
# by default so normal runs stay quiet -- item-1 observability.
STATE_FALLBACK_LOG_ENV = "FRANKA_LOG_STATE_FALLBACK"

# Per-arm presets copied from the source deployment. The default IP can be
# overridden with ``--ip`` when starting the standalone control server.
LEFT_HOME_JOINTS = [-0.31777719, -0.03995929, -0.03193330, -1.65214336,
                    0.05245617, 1.73788381, 0.64199769]
RIGHT_HOME_JOINTS = [0.07841486, -0.07431660, 0.08570135, -1.57819736,
                     0.05896067, 1.59374964, 0.91257948]
LEFT_WRIST_CAMERA_SERIAL = "349622072679"
RIGHT_WRIST_CAMERA_SERIAL = "349622074867"
HOME_JOINT_SPEED_RAD_S = 0.5  # joint speed for home moves

# Stock-stiff cartesian impedance pinned at connect (impedance persists on the
# controller across FCI sessions; a soft teleop leftover causes reflexes).
CARTESIAN_IMPEDANCE = [3000.0, 3000.0, 3000.0, 300.0, 300.0, 300.0]

# The external cartesian force at which the controller fires `cartesian_reflex`
# and LATCHES the arm (franky's documented default, 20 Nm / 30 N -- see
# robots/franka/motion.set_collision_behavior). It is named here because it is
# not merely trivia: the servo's lead CEILING is derived from it
# (servo_lead_ceiling_m below), and nothing on the demo or eval path calls
# set_collision_behavior, so this is the threshold the arm actually runs at.
# Raise it deliberately for contact work; do not assume it has been raised.
CARTESIAN_REFLEX_FORCE_N = 30.0
# The torque half of the same default (the "20 Nm" in "20 Nm / 30 N"): the
# external cartesian TORQUE at which the same reflex fires. Named for the same
# reason -- the servo's ROTATIONAL lead ceiling is derived from it
# (servo_lead_ceiling_rad below).
CARTESIAN_REFLEX_TORQUE_NM = 20.0


def servo_lead_ceiling_m(stiffness_n_per_m: float = CARTESIAN_IMPEDANCE[0],
                         reflex_force_n: float = CARTESIAN_REFLEX_FORCE_N) -> float:
    """The lead at which a servo's commanded pose costs the arm its whole reflex
    budget: the HARDWARE CEILING on how far the integrating setpoint may run
    ahead of the measured pose. Force = stiffness x displacement, so the lead
    that reaches `reflex_force_n` is reflex_force_n / stiffness. At this cell's
    commanded 3000 N/m against franky's default 30 N that is 10 mm.

    WHY THIS IS A FUNCTION OF WHAT IS IN FORCE, NOT A FROZEN NUMBER. Both inputs
    are settable: a task doing contact work raises the reflex threshold
    (set_collision_behavior) and a task wanting compliance softens the impedance.
    A hard-coded 0.01 would be silently WRONG for both -- too tight for a soft
    arm, too loose for a stiff one. That is the "correct for one run, wrong for
    the next" defect, so the ceiling is computed from the values actually
    commanded rather than pinned at whatever this cell ran on one evening.

    THERE IS NO MARGIN TERM HERE, DELIBERATELY. The previous version multiplied
    this by a hand-picked fraction (0.5), and that fraction was chosen because it
    landed on 5 mm, the band a VGA place demonstration happened to end in. A
    driver cannot know what is being inserted, so it cannot own that choice: the
    ceiling is the hardware fact, and a TASK that wants to spend less of the
    budget asks for a tighter bound underneath it (FrankaAdapter.
    set_servo_lead_bound). No task setting one runs at the ceiling.

    HONEST LIMIT OF THE MODEL, which the previous comment stated as fact. This
    treats the commanded Cartesian stiffness as the gain from lead to force. That
    is exact only if the servo runs under the Cartesian impedance controller. The
    servo path is `servo_tool` -> IK -> `servo_joint` -> franky `JointMotion`, a
    JOINT motion generator, and `set_cartesian_impedance` configures the Cartesian
    one; under a joint impedance controller the true Cartesian stiffness is
    J^-T K_q J^-1, which varies with the arm's configuration and is not one
    number. Nothing in this repo has measured which controller franky selects for
    a JointMotion, so read the ceiling as a CONSERVATIVE ESTIMATE with the right
    units and the right monotonicity, not as a calibrated trip point. Measuring
    it (command a known lead against a blocked tool, read O_F_ext_hat_K) is the
    open item that would turn this from an estimate into a fact."""
    stiffness = float(stiffness_n_per_m)
    if stiffness <= 0.0:
        raise ValueError(f"stiffness must be positive, got {stiffness_n_per_m!r}")
    return float(reflex_force_n) / stiffness


def servo_lead_ceiling_rad(stiffness_nm_per_rad: float = CARTESIAN_IMPEDANCE[3],
                           reflex_torque_nm: float = CARTESIAN_REFLEX_TORQUE_NM) -> float:
    """The ROTATIONAL twin of servo_lead_ceiling_m: the angular lead at which the
    servo's commanded orientation costs the arm its whole torque-reflex budget.
    Torque = rotational stiffness x angular displacement, so the lead that reaches
    `reflex_torque_nm` is reflex_torque_nm / stiffness. At this cell's commanded
    300 Nm/rad against franky's default 20 Nm that is ~0.067 rad (~3.8 deg).

    IT EXISTS BECAUSE ITS ABSENCE WAS AN ASYMMETRY NOBODY CHOSE. send_action
    integrates translation AND rotation, but only the translational lead was
    bounded -- so when the wrist could no longer follow (joint 7 pinned on its
    limit, franka-right 2026-08-26), the integrator kept adding ~1 deg of commanded
    rotation per tick without limit until the servo IK gave up 124 ticks in, and
    the round hard-aborted mid-transit. The same run's TRANSLATIONAL lead was
    caught at its bound and merely warned.

    WHAT THE BOUND DOES AND DOES NOT BUY, stated here because the incident that
    motivated it is exactly the case it does not cure: bounding the rotational
    lead converts that hard abort into a stalled-but-alive stream -- the commanded
    orientation rides at the bound over the measured one, the clamp warning names
    it, and the episode's own guards keep deciding. It does NOT make an
    unreachable orientation reachable: an arm whose wrist is out of travel still
    cannot reach the pose, and where a policy's demonstrations end relative to a
    joint limit is a TASK question, not a driver one.

    Same shape as the translational ceiling in every other respect: a function of
    what is in force rather than a frozen number (both inputs are settable), no
    margin term (a task that wants a tighter leash bounds it underneath, see
    FrankaAdapter.set_servo_lead_bound), and the same honest limit of the model --
    the servo runs joint-space (servo_tool -> IK -> JointMotion) while the
    commanded stiffness is Cartesian, so this is a CONSERVATIVE ESTIMATE with the
    right units and the right monotonicity, not a calibrated trip point."""
    stiffness = float(stiffness_nm_per_rad)
    if stiffness <= 0.0:
        raise ValueError(f"rotational stiffness must be positive, got {stiffness_nm_per_rad!r}")
    return float(reflex_torque_nm) / stiffness


CONNECT_ATTEMPTS = 3
CONNECT_RETRY_DELAY_S = 0.5
STOP_JOIN_TIMEOUT_S = 2.0  # drain the control thread after robot.stop()
# Blocking moves are executed async then join_motion()'d in slices, so franky's
# internal control/state mutex is released between slices and a concurrent
# recording-thread state read cannot deadlock against an in-control move. The
# join slice bounds how long a single state read can be blocked; the overall cap
# turns a wedged motion (starved real-time loop) into a raised 'hang' the retry
# logic recovers from, instead of an indefinite freeze.
MOTION_JOIN_SLICE_S = 0.05  # join_motion() poll slice (also the max a concurrent
# state read waits for the lock; small so the 30 Hz recording cadence is barely
# perturbed, while abort latency stays low)
MOTION_JOIN_TIMEOUT_S = 45.0  # hard cap on a single blocking move before abort
# move-excursion diagnostic (controller._log_traj_excursion): flag a leg whose
# joint-space interpolation bows the TCP more than MOTION_BOW_OUT_M above both
# endpoints, or whose largest single-joint jump exceeds MOTION_BRANCH_SWING_RAD
# (an IK branch swing -> non-smooth joint path).
MOTION_BOW_OUT_M = 0.05
MOTION_BRANCH_SWING_RAD = 0.8
EXCURSION_FK_SAMPLES = 12  # FK samples per leg for the move-excursion diagnostic

DEFAULT_TOOL_SPEED_M_S = 0.1  # waypoints without an explicit speed
DEFAULT_MAX_ANGULAR_VEL_RAD_S = 0.4
MIN_SEGMENT_TIME_S = 0.02  # below this no minimum_time is sent
# Keep the final joint target active briefly after Ruckig reaches zero velocity.
# This lets residual motion settle before the motion controller releases control.
FINAL_WAYPOINT_HOLD_MS = 200

# --- Segment pacing floor (move_tool_traj / move_until_force) -------------------
# The lower floor that keeps the per-segment minimum-time math well-defined (named
# so the method body has no bare magic numbers).
PATH_MIN_SPEED_FLOOR_M_S = 1e-4   # speed/limit floor so a dt or a cap is never 1/0

# --- Branch-continuous joint-space execution (move_tool_traj) ------------------
# Scripted legs execute as ONE joint-space Ruckig trajectory (JointWaypointMotion),
# NOT through libfranka's Cartesian motion generator -- whose post-IK joint
# velocities libfranka does NOT re-limit, the cause of the
# cartesian_motion_generator_joint_velocity_discontinuity reflex (field consensus:
# polymetis/deoxys/frankapy/franka-control all plan in joint space). To stop the
# per-waypoint IK flipping branches between far-apart waypoints (the
# joint_motion_generator_acceleration_discontinuity 'branch swing'), each leg is
# densely sampled and every IK solve is warm-started from the previous DENSE
# solution -- so adjacent solutions are a small step apart and cannot jump
# branches. Smaller step = safer continuity, more IK solves; solves are
# warm-started and off the 1 kHz path, so the cost is negligible.
PATH_DENSIFY_STEP_M = 0.01      # max Cartesian position step between IK seeds (1 cm)
PATH_DENSIFY_STEP_RAD = 0.1     # max orientation step between IK seeds (~5.7 deg)

# --- Waypoint pass-through (move_tool_traj blend markers) -----------------------
# A waypoint whose blend marker is > 0 is passed THROUGH with a target joint
# velocity (the outgoing leg's average), so Ruckig rounds the corner instead of
# stopping dead (owner report 2026-07-13: smooth approach -> ~1 s stop at the
# pre-grasp via -> slow final 2 cm). blend == 0 keeps the full stop -- the
# relocate/backward legs rely on that. The pass-through velocity carries TWO
# bounds, and they answer different questions:
#
#  1. FEASIBILITY -- the NEXT leg must be able to brake to zero inside its own
#     travel: |v| <= sqrt(2 * a_eff * d * PASSTHROUGH_BRAKE_MARGIN), with a_eff
#     the most conservative rated FR3 joint acceleration scaled by
#     ACCELERATION_DYNAMICS_FACTOR. That factor is a constant, so this bound is
#     speed-independent and stays a constant here.
#  2. ADMISSIBILITY -- Ruckig VALIDATES every waypoint target velocity against
#     `relative_dynamics_factor.velocity() * robot.joint_velocity_limit` and
#     refuses the whole motion with ErrorInvalidInput (franky code -100) if any
#     component is over. That ceiling MOVES with the run's speed setting, so it
#     CANNOT live here as a constant: it is read per motion from the live robot
#     (MotionMixin._joint_velocity_ceiling). It used to be frozen at
#     `2.62 * DEFAULT_DYNAMICS_FACTOR = 0.393 rad/s`, which is 2.26x the real
#     ceiling of a run at speed 0.08 (0.08 * 2.175 = 0.174 rad/s) -- every reset
#     whose draw landed in that gap was refused, and the collect task burned
#     resamples on it (owner report 2026-08-28, place_fine_vertical_0_8).
FR3_MIN_RATED_JOINT_ACCEL_RAD_S2 = 7.5   # smallest FR3 rated joint acceleration (joint 2)
# Datasheet RATING, the input to the speed envelope robots/speed.py reports to the
# operator (its only consumer now). It is NOT the Ruckig ceiling and must not be
# used as one: franky's live joint_velocity_limit for this build is lower still.
FR3_MIN_RATED_JOINT_VEL_RAD_S = 2.62     # smallest FR3 rated joint velocity (joints 1-4)
PASSTHROUGH_BRAKE_MARGIN = 0.5           # fraction of the braking budget usable
# Ruckig's target-velocity check is a STRICT `>`, so a component scaled to sit
# exactly on the ceiling is one float rounding away from being refused. Hold a
# little back; 1% is invisible in the corner rounding and removes the edge.
PASSTHROUGH_VEL_HEADROOM = 0.99

# Force / contact defaults
IMPEDANCE_TRANSLATIONAL_STIFFNESS = 2000.0  # N/m (franky default)
IMPEDANCE_ROTATIONAL_STIFFNESS = 200.0  # Nm/rad (franky default)
GUARD_FORCE_THRESHOLD_N = 5.0  # default move_until_force trigger
REACTION_SETTLE_S = 0.05  # reaction callbacks run async; settle before reading
# per-motion rdf for the franky-native guarded descent (CartesianMotion). On-robot
# (2026-06-30) the Cartesian descent tracks a smooth straight line with tiny per-step
# joint motion (no branch-swing phantom), reaching real contact; contact velocity is
# bounded by _cartesian_speed_limits(speed), not by this rdf.
CONTACT_RDF = 0.4

# --- Franka driver tunables (moved out of driver.py module scope in F2) ---------
# Arrival tolerance (rad/joint) used as a joint move's SUCCESS criterion: if a
# control reflex fires mid-move but the arm is within this of the commanded
# configuration, the move effectively reached its goal. A (near-)zero or
# end-of-travel move trips the FR3 velocity/acceleration-discontinuity reflex even
# though there was nothing left to move -- that is benign, not a failure.
JOINT_AT_TARGET_RAD = 0.03
# Real-time servo IK tolerance. The planning IK default (IK_POS_TOL_M = 1 mm) is
# COARSER than a 30 Hz servo's per-step Cartesian move: at 1 mm the warm-started
# solver returns the PREVIOUS joint solution unchanged until the accumulated move
# crosses 1 mm, then double-jumps -> a 0/double stutter that trips the FR3 jerk
# reflex. Tracking the servo target to 0.1 mm keeps every sub-mm step in the joint
# command, so it stays continuous. (Distinct from planning IK, which stays at 1 mm.)
SERVO_IK_POS_TOL_M = 1e-4    # 0.1 mm
SERVO_IK_ORI_TOL_RAD = 5e-4  # ~0.03 deg

# Gripper (Franka Hand) defaults
GRIPPER_OPEN_SPEED_M_S = 0.05
GRIPPER_GRASP_SPEED_M_S = 0.02
# A native Franky Hand Future must not hold the arm in the transition state
# forever after a lost Hand connection or a mechanically wedged command.
GRIPPER_ASYNC_TIMEOUT_S = 10.0
# Off-center grasp SETTLE pulses (owner 2026-07-17: one finger touched the object's near
# side and the firmware stopped -- touched, not held). Re-grasping pushes the object
# toward center each pulse; settled = the measured width stops changing.
GRASP_SETTLE_PULSES = 3          # max extra grasp pulses after the first close
GRASP_SETTLE_TOL_M = 0.0015      # width change below this = settled (clamped, not sliding)
# HARDWARE FACT: the Franka Hand's continuous grasp force range is 20-70 N (spec).
# ARBITRARY WITHIN IT: 40 N is the midpoint, and nothing about the ARM picks it.
# How hard to squeeze depends entirely on WHAT is being held -- rigid parts slip
# below ~20 N under transit acceleration, deformable ones deform above it -- so no
# single number is right for every run and this one is only a starting point. The
# real answer is per-object and the TASK owns it: a recipe anchor sets
# `grasp_force_n`, and a step that does not is saying "anything in range will do".
GRIPPER_GRASP_FORCE_N = 40.0  # driver DEFAULT, mid-range; the task owns the real value
GRIPPER_MAX_FORCE_N = 120.0   # 100% on the 0-100% force scale (owner decision
                              # 2026-08-21, superseding the 140 N peak of 2026-07-29:
                              # cap at 120 N). The CONTINUOUS limit stays 70 N:
                              # hold above it only for short grasps or the Hand
                              # thermally derates.
GRIPPER_EPSILON_INNER_M = 0.005  # grasp success window below commanded width
GRIPPER_EPSILON_OUTER_M = 0.04  # grasp success window above commanded width
# NOTE: GRIPPER_MIN_HELD_M / DEFAULT_GRASP_FORCE_PCT live in demo/skills/pick.py, NOT
# here -- they are demo-skill knobs, not driver constants.

# IK solver parameters (hardware-validated: sub-mm reach error).
IK_LIMIT_MARGIN_RAD = 0.02
IK_POS_TOL_M = 1e-3
IK_ORI_TOL_RAD = 5e-3
IK_MAX_ITER = 300
IK_DLS_LAMBDA_MIN = 0.005
IK_NULLSPACE_GAIN = 0.02
IK_MAX_STEP_RAD = 0.2
JACOBIAN_EPS = 1e-6


# --- Decoupled control server / client (Phase 2: ZMQ control IPC) -------------
# The control server (robots/franka/control_server.py) hosts the franky
# FrankaArmController in its OWN process so the 1 kHz FCI loop never shares a
# GIL/CPU with the collector's recording + JPEG encoding (the root cause of the
# episode-N motion "hang"). The collector drives it through FrankaArmControllerClient
# (robots/franka/control_client.py), a drop-in for FrankaArmController. Transport
# mirrors ArrebolBlack/franka-control + the user's screw_demo: ZMQ DEALER/ROUTER
# for commands + PUSH/PULL for the state stream, msgpack bodies. Enabled by the
# FRANKA_CONTROL_IPC env var (data_collection/franka_robot.py). Generic tunables
# (ports/rates) -- safe to share across machines.
CONTROL_SERVER_HOST = "127.0.0.1"           # client connects here (server = same machine)
CONTROL_SERVER_CMD_PORT = 5555              # ROUTER: commands
CONTROL_SERVER_STATE_PORT = 5557            # PUSH: state stream
CONTROL_SERVER_STATE_POLL_HZ = 100.0        # state-snapshot publish rate (>= 3x the 30 Hz reader)
CONTROL_SERVER_SERVO_GAP_S = 0.3            # no servo for this long while active -> graceful stop_servo
CONTROL_SERVER_CMD_WAIT_S = 10.0            # controller-thread wait for a fast/interrupt command result
CONTROL_CLIENT_RCV_TIMEOUT_MS = 10000       # command round-trip cap (covers a stop that joins a move helper)
CONTROL_CLIENT_STATE_RCV_TIMEOUT_MS = 100   # state PULL recv timeout
CONTROL_CLIENT_IDLE_POLL_S = 0.05           # poll cadence while awaiting a blocking move (20 Hz)
CONTROL_CLIENT_CONNECT_TIMEOUT_S = 60.0     # connect / wait_until_ready can be slow
CONTROL_CLIENT_MOVE_TIMEOUT_S = 50.0        # blocking-move cap (>= MOTION_JOIN_TIMEOUT_S 45 + margin)
