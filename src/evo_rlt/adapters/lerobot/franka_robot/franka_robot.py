import logging
import os
import sys
import threading
import time
from pathlib import Path

import pyrealsense2 as rs

import numpy as np
from scipy.spatial.transform import Rotation
from lerobot.robots.robot import Robot

from .configuration_franka import FrankaRobotConfig

# Steady-state: if a camera delivers no new frame for this long, treat it as
# stalled (recorder aborts the take instead of writing stale/frozen frames).
# Warm-up frames at connect get a longer budget so a cold USB/sensor start is
# never mis-flagged.
CAM_FRAME_TIMEOUT_MS = 1000
CAM_WARMUP_TIMEOUT_MS = 5000
CAM_WARMUP_FRAMES = 10

# FCI session-lease renewal cadence. On firmware 5.9.0 a Franka FCI session with
# no active control loop is terminated after ~55 s; the robotLab control server's
# idle loop only *reads* state, and state reads do NOT renew the lease (only
# control commands do). robotLab is out of our hands, so Evo-RLT renews it from
# here: a background thread sends one fire-and-forget servo_joint to the *current*
# joint position this often while the arm is Idle, so the 1 kHz loop runs a few ms
# and resets the lease. (Validated 2026-09-07: single ticks every 20 s survived
# 120 s of idle.) During Guiding/UserStopped the pulse is skipped (no command is
# accepted then) -- safe because hand-guided stretches are < 55 s and the next
# Idle gap renews.
FCI_KEEPALIVE_INTERVAL_S = 20.0

# A Hand command is rejected while the arm still owns the FCI motion channel.
# stop_servo() is synchronous on the robotLab control client, but its PUSH state
# cache can trail the stop response by one controller cycle.  Bound the cache
# confirmation instead of relying on the server's 0.3 s servo-gap watchdog.
GRIPPER_ARM_IDLE_TIMEOUT_S = 2.0
GRIPPER_ARM_IDLE_POLL_S = 0.01


class CameraFrameTimeout(RuntimeError):
    """Raised when a camera stream does not deliver a new frame in time.

    Carries ``.camera`` (``"wrist"`` / ``"front"``) so the recorder's health
    watchdog can report *which* camera stalled without importing this module.
    """

    def __init__(self, camera: str, waited_ms: int):
        super().__init__(f"camera '{camera}' delivered no frame within {waited_ms} ms")
        self.camera = camera


class FrankaRobot(Robot):
    config_class = FrankaRobotConfig
    name = "franka"

    def __init__(self, config: FrankaRobotConfig):
        super().__init__(config)
        self.config = config
        robotlab_path = os.environ.get("ROBOTLAB_PATH", config.robotlab_path)
        if robotlab_path not in sys.path:
            sys.path.insert(0, robotlab_path)
        try:
            from robots.franka.control_client import FrankaArmControllerClient
        except ImportError as exc:
            raise ImportError(
                f"Cannot import robotLab Franka client from {robotlab_path!r}; "
                "set --robotlab-path or ROBOTLAB_PATH on the robot host"
            ) from exc
        self.robot = FrankaArmControllerClient(config.robot_ip)
        self._connected = False
        self._cmd_pose = None
        self._has_gripper = None  # cached from control-server RPC on first observation
        # LeRobot record backend expects a `cameras` attribute (e.g. for image
        # writer thread count). The wrist RealSense is managed internally.
        self.cameras = {}
        self.camera = rs.pipeline()
        self.camera_cfg = rs.config()
        self.camera_cfg.enable_device(config.camera_serial)
        self.camera_cfg.enable_stream(
            rs.stream.color, 640, 480, rs.format.rgb8, 30
        )
        # second camera: max resolution (1920x1080), center-cropped to 640x480
        # in get_observation (pure crop, no resize/interpolation)
        self.front_camera = rs.pipeline()
        self.front_camera_cfg = rs.config()
        self.front_camera_cfg.enable_device(config.front_camera_serial)
        self.front_camera_cfg.enable_stream(
            rs.stream.color, 1920, 1080, rs.format.rgb8, 30
        )
        # Collection-health bookkeeping (see CAM_FRAME_TIMEOUT_MS / .camera_frame_ages)
        self._cam_streams = {"wrist": self.camera, "front": self.front_camera}
        self._cam_last_frame_t = {"wrist": None, "front": None}
        self._keepalive_thread = None
        self._keepalive_stop = threading.Event()

    @property
    def observation_features(self):
        # Scalar-float ordering defines observation.state names in the dataset:
        #   [joint_0..6, ee_x..ee_rz, joint_vel_0..6, gripper_width, gripper_grasped]
        # (joint_vel / gripper slots are all zeros when the Hand is absent).
        return {
            **{f"joint_{i}": float for i in range(7)},
            **{k: float for k in ["ee_x", "ee_y", "ee_z", "ee_rx", "ee_ry", "ee_rz"]},
            **{f"joint_vel_{i}": float for i in range(7)},
            "gripper_width": float,
            "gripper_grasped": float,
            "wrist": (480, 640, 3),
            "front": (480, 640, 3),
        }
    

    @property
    def action_features(self):
        names = ["dx", "dy", "dz", "drx", "dry", "drz"]
        if self.config.include_gripper_action:
            names.append("gripper_target_width")
        return {name: float for name in names}

    @property
    def is_connected(self):
        return self._connected

    @property
    def is_calibrated(self):
        return True

    def calibrate(self):
        pass

    def configure(self):
        pass

    def connect(self, calibrate=True):
        self.robot.connect()
        try:
            self.camera.start(self.camera_cfg)
            self.front_camera.start(self.front_camera_cfg)
            # Warm both pipelines; a camera that cannot deliver its first frames is
            # a startup failure (raise), not a silent mid-take surprise.
            for _ in range(CAM_WARMUP_FRAMES):
                self._read_camera_frame("wrist", timeout_ms=CAM_WARMUP_TIMEOUT_MS)
                self._read_camera_frame("front", timeout_ms=CAM_WARMUP_TIMEOUT_MS)
        except BaseException:
            # The control connection was already acquired. Release it even
            # though `_connected` is not set until all camera warm-up succeeds.
            for pipeline in (self.camera, self.front_camera):
                try:
                    pipeline.stop()
                except Exception:
                    pass
            try:
                self.robot.disconnect()
            except Exception:
                logging.exception("failed to disconnect Franka after camera startup failure")
            raise
        self._connected = True
        self._cmd_pose = None
        if self.config.enable_fci_keepalive:
            self._start_keepalive()

    def disconnect(self):
        self._stop_keepalive()
        for pipeline in (self.camera, self.front_camera):
            try:
                pipeline.stop()
            except Exception:
                pass
        if self._connected:
            self.robot.disconnect()
        self._connected = False
        self._cmd_pose = None

    # -- FCI session-lease keep-alive (system 5.9.0) -------------------------
    # See FCI_KEEPALIVE_INTERVAL_S for the why. servo_joint is chosen over
    # move_tool_impedance because it is a *streaming* command: fire-and-forget,
    # neither blocking nor marking the server busy, so a pulse can never collide
    # with a concurrent gripper RPC ("worker busy"). A single tick is enough --
    # the arm is already at the target, so the motion completes in ~1-2 ms and
    # the lease watchdog (no control command for ~55 s) is reset.
    def _start_keepalive(self) -> None:
        self._stop_keepalive()          # no double threads if connect() re-runs
        self._keepalive_stop.clear()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop, daemon=True, name="fci-keepalive")
        self._keepalive_thread.start()
        logging.info("FCI session-lease keep-alive started (every %.0f s)",
                     FCI_KEEPALIVE_INTERVAL_S)

    def _stop_keepalive(self) -> None:
        self._keepalive_stop.set()
        thread, self._keepalive_thread = self._keepalive_thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def _keepalive_loop(self) -> None:
        while not self._keepalive_stop.is_set():
            self._keepalive_stop.wait(timeout=FCI_KEEPALIVE_INTERVAL_S)
            if self._keepalive_stop.is_set():
                break
            self._keepalive_pulse()

    def _keepalive_pulse(self) -> None:
        try:
            if not self._connected:
                return
            if "Idle" not in self.robot.robot_mode_nowait():
                return
            q = self.robot.get_joint_angles()
            # Guard the server's read-failure fallback ([0.0]*7): never servo the
            # arm to an all-zero target that is not a real measured pose.
            if len(q) != 7 or all(abs(v) < 1e-6 for v in q):
                return
            self.robot.servo_joint(q)
        except Exception:  # noqa: BLE001 - a renewal glitch must never kill the thread
            pass

    def get_observation(self):
        q = self.robot.get_joint_angles()
        dq = self.robot.get_joint_speeds()
        pose = self.robot.get_tool_pose()

        # Gripper state is a control-server RPC (not in the 30 Hz snapshot);
        # query once whether a Hand is attached, then only per frame when it is.
        if self._has_gripper is None:
            self._has_gripper = self.robot.has_gripper()
        if self._has_gripper:
            gripper_width = float(self.robot.get_gripper_width())
            gripper_grasped = 1.0 if self.robot.is_grasped() else 0.0
        else:
            gripper_width = 0.0
            gripper_grasped = 0.0

        frames = self._read_camera_frame("wrist")
        wrist = np.asanyarray(frames.get_color_frame().get_data())

        front_frames = self._read_camera_frame("front")
        front_full = np.asanyarray(front_frames.get_color_frame().get_data())
        # max resolution (1920x1080) -> center crop 640x480, no resize
        y0 = (front_full.shape[0] - 480) // 2
        x0 = (front_full.shape[1] - 640) // 2
        front = front_full[y0 : y0 + 480, x0 : x0 + 640]

        return {
            **{f"joint_{i}": float(q[i]) for i in range(7)},
            "ee_x": float(pose[0]),
            "ee_y": float(pose[1]),
            "ee_z": float(pose[2]),
            "ee_rx": float(pose[3]),
            "ee_ry": float(pose[4]),
            "ee_rz": float(pose[5]),
            **{f"joint_vel_{i}": float(dq[i]) for i in range(7)},
            "gripper_width": gripper_width,
            "gripper_grasped": gripper_grasped,
            "wrist": wrist,
            "front": front,
        }

    # -- collection health (camera + control-server liveness) ---------------
    def _read_camera_frame(self, camera: str, timeout_ms: int = CAM_FRAME_TIMEOUT_MS):
        """Wait (bounded) for one new frame from ``camera``.

        Never hangs silently: if the stream delivers nothing within
        ``timeout_ms`` we raise :class:`CameraFrameTimeout` so the recorder can
        abort the take instead of writing stale frames. On success the per-camera
        liveness timestamp used by the health heartbeat is updated.
        """
        pipeline = self._cam_streams[camera]
        try:
            frame = pipeline.wait_for_frames(timeout_ms=timeout_ms)
        except (RuntimeError, rs.error) as exc:
            raise CameraFrameTimeout(camera, timeout_ms) from exc
        self._cam_last_frame_t[camera] = time.perf_counter()
        return frame

    def camera_frame_ages(self) -> dict[str, float | None]:
        """Seconds since each camera last delivered a frame (``None`` if never)."""
        now = time.perf_counter()
        return {
            name: (None if last is None else now - last)
            for name, last in self._cam_last_frame_t.items()
        }

    def probe(self) -> dict:
        """Cheap control-server liveness probe (robotLab ZMQ ``ping``).

        Returns ``{"ok", "error", "ms"}`` and never raises. A dead/unreachable
        server is only confirmed after the robotLab client's command-socket
        RCVTIMEO (10 s by default), so detection latency for a hard drop is
        ~10 s -- the same bound the per-frame gripper RPCs already have.
        """
        t0 = time.perf_counter()
        try:
            ok = bool(self.robot.ping())
        except Exception as exc:  # noqa: BLE001 - a probe must never raise
            return {"ok": False, "error": str(exc), "ms": (time.perf_counter() - t0) * 1000.0}
        return {
            "ok": ok,
            "error": None if ok else "no reply (client RCVTIMEO)",
            "ms": (time.perf_counter() - t0) * 1000.0,
        }

    # -- gripper (manual-demo control) --------------------------------------
    # In manual_demo_mode the arm is hand-guided (Desk Guiding) and the
    # recorder never calls send_action; these wrappers let the operator command
    # the Franka Hand from the keyboard (default C = force-grasp, O = open).
    # They are blocking RPCs to the control server and must be called from the
    # recording thread (see record.loop.process_gripper_key_events), never from
    # the keyboard-listener thread: the client socket is single-threaded.
    def open_gripper(self):
        """Open the Franka Hand fully (blocking RPC). robotLab defaults apply."""
        return self.robot.open_gripper()

    def close_gripper(self, width: float | None = None):
        """Force-controlled grasp (blocking RPC). robotLab defaults apply
        (~40 N). When a policy supplies an absolute target width, preserve it
        so an empty close is not mistaken for a successful grasp at width 0."""
        if width is None:
            return bool(self.robot.close_gripper())
        return bool(self.robot.close_gripper(width=float(width)))

    def prepare_gripper_transition(self) -> None:
        """End arm servo ownership before issuing a blocking Hand command.

        robotLab deliberately refuses gripper actuation while ``is_running()``
        is true.  Merely omitting one or two 30 Hz targets is insufficient: its
        fallback servo-gap watchdog waits 0.3 s.  Explicitly stop the stream and
        wait for the same state predicate used by robotLab's gripper guard.
        """
        self._stop_keepalive()
        try:
            if not self.robot.stop_servo():
                raise RuntimeError("control server rejected stop_servo before gripper transition")
            deadline = time.monotonic() + GRIPPER_ARM_IDLE_TIMEOUT_S
            while self.robot.is_running():
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "arm did not become idle after stop_servo; gripper command was not sent"
                    )
                time.sleep(GRIPPER_ARM_IDLE_POLL_S)
            self._cmd_pose = None
        except BaseException:
            if self._connected and self.config.enable_fci_keepalive:
                self._start_keepalive()
            raise

    def finish_gripper_transition(self) -> None:
        """Restore idle-session renewal after the blocking Hand command."""
        self._cmd_pose = None
        if self._connected and self.config.enable_fci_keepalive:
            self._start_keepalive()

    def get_external_wrench_base(self) -> np.ndarray:
        """Return cached raw base-frame external wrench without a control RPC."""
        wrench = np.asarray(self.robot.get_tool_force_raw(), dtype=float)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise RuntimeError(f"invalid external wrench from control server: {wrench}")
        return wrench

    def stop_and_retreat_up(self, *, distance_m: float, speed_m_s: float) -> dict:
        """Stop policy servo and make one bounded deterministic base +Z retreat."""
        if distance_m <= 0 or speed_m_s <= 0:
            raise ValueError("retreat distance and speed must be positive")
        self._stop_keepalive()
        self._cmd_pose = None
        try:
            if not self.robot.stop_servo():
                raise RuntimeError("control server rejected stop_servo after force trigger")
            deadline = time.monotonic() + GRIPPER_ARM_IDLE_TIMEOUT_S
            while self.robot.is_running():
                if time.monotonic() >= deadline:
                    raise TimeoutError("arm did not become idle after force-triggered stop")
                time.sleep(GRIPPER_ARM_IDLE_POLL_S)

            start_pose = np.asarray(self.robot.get_tool_pose(), dtype=float)
            if start_pose.shape != (6,) or not np.isfinite(start_pose).all():
                raise RuntimeError(f"invalid measured pose before force retreat: {start_pose}")
            target_pose = start_pose.copy()
            target_pose[2] += float(distance_m)

            lower = self.config.workspace_min_xyz
            upper = self.config.workspace_max_xyz
            if lower is None or upper is None:
                raise RuntimeError("force retreat requires explicit workspace bounds")
            lower_xyz = np.asarray(lower, dtype=float)
            upper_xyz = np.asarray(upper, dtype=float)
            if np.any(target_pose[:3] < lower_xyz) or np.any(target_pose[:3] > upper_xyz):
                raise RuntimeError(
                    f"force retreat target outside workspace: target={target_pose[:3]}, "
                    f"min={lower_xyz}, max={upper_xyz}; arm remains stopped"
                )

            measured_pose = self.robot.move_tool(target_pose.tolist(), speed=float(speed_m_s))
            self._cmd_pose = None
            return {
                "start_pose": start_pose.tolist(),
                "target_pose": target_pose.tolist(),
                "measured_pose": list(measured_pose),
                "distance_m": float(distance_m),
                "speed_m_s": float(speed_m_s),
            }
        finally:
            self._cmd_pose = None
            if self._connected and self.config.enable_fci_keepalive:
                self._start_keepalive()

    def resync_command_pose(self):
        """Re-anchor the integrated TCP target to the next measured pose."""
        self._cmd_pose = None

    def send_action(self, action):
        measured = np.asarray(self.robot.get_tool_pose(), dtype=float)

        if self._cmd_pose is None:
            self._cmd_pose = measured.copy()

        self._cmd_pose[:3] += [
            action["dx"],
            action["dy"],
            action["dz"],
        ]

        delta_rot = Rotation.from_rotvec([
            action["drx"],
            action["dry"],
            action["drz"],
        ])

        self._cmd_pose[3:6] = (
            Rotation.from_rotvec(self._cmd_pose[3:6]) * delta_rot
        ).as_rotvec()
        # 防止 policy 积分目标离真实机械臂太远
        MAX_POS_LEAD = 0.010      # 10 mm
        MAX_ROT_LEAD = 0.10       # rad

        pos_lead = self._cmd_pose[:3] - measured[:3]
        dist = np.linalg.norm(pos_lead)
        if dist > MAX_POS_LEAD:
            self._cmd_pose[:3] = measured[:3] + pos_lead / dist * MAX_POS_LEAD

        bounds = (self.config.workspace_min_xyz, self.config.workspace_max_xyz)
        if (bounds[0] is None) != (bounds[1] is None):
            raise ValueError("workspace_min_xyz and workspace_max_xyz must be set together")
        if bounds[0] is not None and bounds[1] is not None:
            lower = np.asarray(bounds[0], dtype=float)
            upper = np.asarray(bounds[1], dtype=float)
            if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
                raise ValueError(f"invalid TCP workspace bounds: min={lower}, max={upper}")
            if np.any(self._cmd_pose[:3] < lower) or np.any(self._cmd_pose[:3] > upper):
                raise RuntimeError(
                    f"refusing TCP target outside workspace: target={self._cmd_pose[:3]}, "
                    f"min={lower}, max={upper}"
                )

        measured_rot = Rotation.from_rotvec(measured[3:6])
        cmd_rot = Rotation.from_rotvec(self._cmd_pose[3:6])
        rot_lead = (measured_rot.inv() * cmd_rot).as_rotvec()
        angle = np.linalg.norm(rot_lead)

        if angle > MAX_ROT_LEAD:
            self._cmd_pose[3:6] = (
                measured_rot
                * Rotation.from_rotvec(rot_lead / angle * MAX_ROT_LEAD)
            ).as_rotvec()
        self.robot.servo_tool(self._cmd_pose.tolist())
        return action
