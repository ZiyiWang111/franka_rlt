from dataclasses import dataclass

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("franka")
@dataclass
class FrankaRobotConfig(RobotConfig):
    robot_ip: str = "172.16.0.2"
    # wrist camera (640x480 native stream)
    camera_serial: str = "349622072679"
    # Second camera. Existing pipelines retain the 1920x1080 default and resize
    # to the dataset's 640x480 tensor; ACT-RLT overrides these to 640x480 so the
    # sensor output is stored directly without geometric resizing.
    front_camera_serial: str = "233522075778"
    front_camera_width: int = 1920
    front_camera_height: int = 1080
    # New manual-demo datasets include the next measured gripper width as the
    # seventh action. Set false only when resuming a legacy 6D dataset.
    include_gripper_action: bool = True
    # Disable for strictly read-only live shadow inference. Motion-capable
    # sessions keep it enabled so firmware 5.9.0 does not drop the FCI lease.
    enable_fci_keepalive: bool = True
    # Optional absolute TCP position guard in the robot base frame (metres).
    workspace_min_xyz: tuple[float, float, float] | None = None
    workspace_max_xyz: tuple[float, float, float] | None = None
