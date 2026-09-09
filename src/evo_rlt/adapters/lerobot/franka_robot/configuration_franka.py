from dataclasses import dataclass

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("franka")
@dataclass
class FrankaRobotConfig(RobotConfig):
    robot_ip: str = "172.16.0.2"
    robotlab_path: str = "/home/embint/robotLab"
    # wrist camera (640x480 native stream)
    camera_serial: str = "349622072679"
    # second camera: captured at max resolution, center-cropped to 640x480
    front_camera_serial: str = "233522075778"
    # New manual-demo datasets include the next measured gripper width as the
    # seventh action. Set false only when resuming a legacy 6D dataset.
    include_gripper_action: bool = True
    # Disable for strictly read-only live shadow inference. Motion-capable
    # sessions keep it enabled so firmware 5.9.0 does not drop the FCI lease.
    enable_fci_keepalive: bool = True
    # Optional absolute TCP position guard in the robot base frame (metres).
    workspace_min_xyz: tuple[float, float, float] | None = None
    workspace_max_xyz: tuple[float, float, float] | None = None
