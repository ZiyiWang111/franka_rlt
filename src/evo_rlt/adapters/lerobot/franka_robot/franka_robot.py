import sys
from pathlib import Path

import pyrealsense2 as rs

import numpy as np
from scipy.spatial.transform import Rotation
from lerobot.robots.robot import Robot

from .configuration_franka import FrankaRobotConfig

sys.path.insert(0, "/home/embint/robotLab")
from robots.franka.control_client import FrankaArmControllerClient


class FrankaRobot(Robot):
    config_class = FrankaRobotConfig
    name = "franka"

    def __init__(self, config: FrankaRobotConfig):
        super().__init__(config)
        self.config = config
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
        return {k: float for k in ["dx", "dy", "dz", "drx", "dry", "drz"]}

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
        self.camera.start(self.camera_cfg)
        self.front_camera.start(self.front_camera_cfg)
        for _ in range(10):
            self.camera.wait_for_frames()
            self.front_camera.wait_for_frames()
        self._connected = True
        self._cmd_pose = None

    def disconnect(self):
        for pipeline in (self.camera, self.front_camera):
            try:
                pipeline.stop()
            except Exception:
                pass
        if self._connected:
            self.robot.disconnect()
        self._connected = False
        self._cmd_pose = None

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

        frames = self.camera.wait_for_frames()
        wrist = np.asanyarray(frames.get_color_frame().get_data())

        front_frames = self.front_camera.wait_for_frames()
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

    def close_gripper(self):
        """Force-controlled grasp (blocking RPC). robotLab defaults apply
        (width 0, ~40 N). Returns True if the Hand reports holding an object."""
        return bool(self.robot.close_gripper())

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
