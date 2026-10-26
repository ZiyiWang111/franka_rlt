"""Stop coarse inference at a measured absolute Z threshold without shaping actions."""
import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZGateConfig:
    enabled: bool = True
    z_m: float = .40207
    state_max_age_s: float = .25

    @classmethod
    def from_args(cls, args):
        defaults = cls()
        return cls(enabled=getattr(args, 'z_gate', defaults.enabled),
                   z_m=getattr(args, 'z_gate_m', defaults.z_m),
                   state_max_age_s=getattr(args, 'z_gate_state_max_age_s', defaults.state_max_age_s))


def read_gate_pose(robot, config):
    pose = np.asarray(robot.robot.get_fresh_tool_pose(max_age_s=config.state_max_age_s), dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError('Invalid measured TCP pose for Z gate')
    return pose


def add_z_gate_arguments(parser):
    defaults = ZGateConfig()
    parser.add_argument('--z-gate', action=argparse.BooleanOptionalAction, default=defaults.enabled,
                        help='Stop inference and all servo motion at or below the measured Z threshold (default: enabled)')
    parser.add_argument('--z-gate-m', '--terminal-target-z', dest='z_gate_m', type=float,
                        default=defaults.z_m,
                        help='Absolute Z stop threshold in metres (default: 0.40207)')
    parser.add_argument('--z-gate-state-max-age-s', type=float, default=defaults.state_max_age_s,
                        help='Maximum accepted streamed state age in seconds (default: 0.25)')


def validate_z_gate_arguments(parser, args):
    if not np.isfinite(args.z_gate_m):
        parser.error('z-gate-m must be finite')
    if not np.isfinite(args.z_gate_state_max_age_s) or args.z_gate_state_max_age_s <= 0:
        parser.error('z-gate-state-max-age-s must be finite and positive')
