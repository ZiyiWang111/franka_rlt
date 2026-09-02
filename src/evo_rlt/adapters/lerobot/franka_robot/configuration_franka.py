from dataclasses import dataclass

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("franka")
@dataclass
class FrankaRobotConfig(RobotConfig):
    robot_ip: str = "172.16.0.2"
    robotlab_path: str = "/home/embint/robotLab"
    camera_serial: str = "349622072679"