"""Use the keyboard teleop servo controller between Stage-2 episodes."""

from __future__ import annotations

import time

from act_rlt.human_input import HumanInputMonitor
from act_rlt.exploration_bias import BIAS_KEYS, MAX_BIAS_LEVEL
from scripts.fr3_keyboard_teleop import Teleop, advance_schedule, build_parser


def read_idle_xy_choice(env, prompt: str) -> str:
    # Keep the same cadence, speed, command lead bound and servo implementation
    # as standalone teleop. Only WASD and the reset menu keys are enabled here.
    args = build_parser().parse_args([])
    args.workspace_min = env.workspace_min
    args.workspace_max = env.workspace_max
    teleop = Teleop(env.robot.robot, args, waypoints=None)
    monitor = None
    moved = False
    completed = False
    bias = getattr(env, "_exploration_bias", None)
    print(
        f"{prompt}\nHold WASD: W=-X, S=+X, A=-Y, D=+Y "
        f"({args.speed_m_s * 100:g} cm/s, {args.fps:g} Hz); release to stop.",
        flush=True,
    )
    if bias is not None:
        if bias.available:
            print("Arrows: Up=-X, Down=+X, Left=-Y, Right=+Y exploration bias; "
                  "N=-yaw, M=+yaw (TCP local Z); "
                  f"one press per level, max {MAX_BIAS_LEVEL} per axis; 0: clear all.", flush=True)
            bias.describe("Next episode", bias.pending)
        else:
            print("Exploration bias is disabled during ACT warmup.", flush=True)
    try:
        monitor = HumanInputMonitor(env.teleop_input_backend, env.teleop_keyboard, idle_xy=True)
        while True:
            choice = monitor.poll()
            if choice in (*BIAS_KEYS, "KEY_0"):
                if bias is not None:
                    bias.adjust(choice)
                choice = None
            if choice is not None:
                completed = True
                return choice
            pressed = monitor.pressed_keys()
            now = time.perf_counter()
            for key in teleop.pressed - pressed:
                teleop.key_event(key, False, now)
            for key in pressed - teleop.pressed:
                teleop.key_event(key, True, now)
            if teleop.next_tick is not None and now >= teleop.next_tick:
                moved = moved or bool(teleop.pressed)
                advance_schedule(teleop, teleop.next_tick)
            time.sleep(.005)
    finally:
        try:
            if not env.robot.robot.stop_servo():
                raise RuntimeError("control server did not acknowledge idle XY stop_servo")
            update_reference = getattr(env, "_update_reference_xy_from_idle", None)
            if completed and moved and update_reference is not None:
                update_reference()
        finally:
            try:
                if monitor is not None:
                    monitor.close()
            finally:
                env.robot.resync_command_pose()
