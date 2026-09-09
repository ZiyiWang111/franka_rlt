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
