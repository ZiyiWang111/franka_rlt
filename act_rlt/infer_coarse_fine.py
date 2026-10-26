"""Run coarse ACT -> fine ACT-RLT with one connection and global outcome labels.

python -m act_rlt.infer_coarse_fine --allow-motion
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import select
import sys
import time
from uuid import uuid4

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from act_rlt import infer_coarse as coarse
from act_rlt.infer import (
    _predict_full_action_chunk, checkpoint_camera_shapes, policy_observation,
    resolve_checkpoint, robot_camera_kwargs,
)
from act_rlt.infer_stage2 import load_frozen_policy, resolve_stage2_checkpoint
from act_rlt.stage2 import resolve_pretrained_model
from act_rlt.stage2_runtime import FixedChunkRuntime
from act_rlt.train_stage2 import FrankaInsertionStage2Env


FINE_WORKSPACE_HALF_RANGE_M = np.asarray([.05, .10, .05], dtype=float)


def fine_workspace(endpoint):
    """Base-frame bounds centred on this round's measured coarse endpoint."""
    center = np.asarray(endpoint, dtype=float)[:3]
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Invalid fine workspace centre")
    return center - FINE_WORKSPACE_HALF_RANGE_M, center + FINE_WORKSPACE_HALF_RANGE_M


class RoundClock:
    """Inference elapsed time, excluding explicit retry confirmation and recovery."""
    def __init__(self):
        self.started = time.monotonic()
        self.excluded = 0.0
        self.key = None
        self.label_elapsed = None

    def elapsed(self):
        return time.monotonic() - self.started - self.excluded

    def mark(self, key):
        if self.key is None:
            self.key = key
            self.label_elapsed = self.elapsed()

    def clear_label(self):
        self.key = self.label_elapsed = None


class GlobalKeys(coarse.EpisodeControls):
    """Same keys and clock in both phases; only the coarse reason is translated."""
    def __init__(self, clock):
        self.clock = clock

    def read_key(self):
        while self.fd is not None and select.select([self.fd], [], [], 0)[0]:
            key = os.read(self.fd, 1).decode(errors="ignore").lower()
            if key == "\x03":
                raise KeyboardInterrupt
            if key in {"s", "f"}:
                self.clock.mark(key)
                return key
            if not key or key in {"q", "x", "\x1b"}:
                return "q"
        return None

    def poll(self):
        return {"s": "operator_success", "f": "operator_failure", "q": "operator_stop"}.get(self.read_key())


class FineKeys:
    def __init__(self, keys):
        self.keys = keys

    def poll(self):
        return self.keys.read_key()

    def close(self):
        pass  # GlobalKeys owns the terminal mode.


def measured_pose(robot, *, after_move=False):
    if after_move:
        print("Waiting for fresh post-move TCP state...", flush=True)
        reader = getattr(robot.robot, "get_tool_pose_after_move", None)
        # The in-process driver reads directly; the IPC client must fence on
        # server snapshot sequence rather than merely the packet reception age.
        values = reader(timeout_s=2.0) if callable(reader) else robot.robot.get_tool_pose()
    else:
        values = robot.robot.get_fresh_tool_pose(max_age_s=.25)
    pose = np.asarray(values, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise RuntimeError("Invalid measured TCP pose")
    return pose


def check_fine_entry(robot, args, emit):
    """Validate measured Z and SO(3) orientation before allowing fine motion."""
    target_rotation = Rotation.from_rotvec(args.fine_entry_rotvec_rad)
    print(f"[FINE ENTRY TARGET] Z={args.fine_entry_z_m:.6f} m; "
          f"rotvec [rad]={args.fine_entry_rotvec_rad}; "
          f"tolerance Z={args.fine_entry_z_tolerance_m * 1000:.3f} mm, "
          f"orientation={args.fine_entry_angle_tolerance_deg:.3f} deg", flush=True)
    pose = measured_pose(robot)
    for adjustment in range(args.fine_entry_max_adjustments + 1):
        z_error = float(pose[2] - args.fine_entry_z_m)
        angle_error_deg = float(np.degrees(
            (target_rotation.inv() * Rotation.from_rotvec(pose[3:])).magnitude()
        ))
        passed = (abs(z_error) <= args.fine_entry_z_tolerance_m
                  and angle_error_deg <= args.fine_entry_angle_tolerance_deg)
        print(f"[FINE ENTRY {'PASS' if passed else 'FAIL'}] check={adjustment}; "
              f"Z={pose[2]:.6f} m; Z error={z_error * 1000:+.3f} mm; "
              f"orientation error={angle_error_deg:.3f} deg; "
              f"measured rotvec [rad]={pose[3:].tolist()}", flush=True)
        emit("fine_entry_gate", adjustment=adjustment, passed=bool(passed),
             measured_tcp=pose.tolist(), z_error_m=z_error, angle_error_deg=angle_error_deg,
             target_z_m=args.fine_entry_z_m, target_rotvec_rad=args.fine_entry_rotvec_rad,
             z_tolerance_m=args.fine_entry_z_tolerance_m,
             angle_tolerance_deg=args.fine_entry_angle_tolerance_deg)
        if passed:
            robot.resync_command_pose()
            return pose
        if adjustment == args.fine_entry_max_adjustments:
            raise RuntimeError("Fine entry gate still failed after adjustments; fine was not started")
        target = pose.copy()
        target[2] = args.fine_entry_z_m
        # Equivalent rotation vectors differ by 2*pi along their axis. Choose
        # the equivalent target nearest the measured vector, avoiding numeric
        # sign jumps near pi; the gate itself always uses SO(3) angular distance.
        rotvec = target_rotation.as_rotvec()
        angle = np.linalg.norm(rotvec)
        candidates = [rotvec]
        if angle > 1e-12:
            candidates += [rotvec + k * 2 * np.pi * rotvec / angle for k in (-1, 1)]
        target[3:] = min(candidates, key=lambda value: np.linalg.norm(value - pose[3:]))
        print(f"[FINE ENTRY ADJUST] Keep measured XY; move Z/orientation to target "
              f"at {args.fine_entry_adjust_speed:g} m/s", flush=True)
        robot.robot.move_tool(target.tolist(), speed=args.fine_entry_adjust_speed)
        robot.resync_command_pose()
        pose = measured_pose(robot, after_move=True)


def move_pose(robot, target, speed, label):
    print(f"[RECOVERY] {label}: {np.asarray(target).tolist()}", flush=True)
    robot.robot.move_tool(np.asarray(target).tolist(), speed=speed)
    robot.resync_command_pose()
    actual = measured_pose(robot, after_move=True)
    orientation_error = (Rotation.from_rotvec(actual[3:]).inv()
                         * Rotation.from_rotvec(np.asarray(target)[3:])).magnitude()
    if np.linalg.norm(actual[:3] - np.asarray(target)[:3]) > .003 or orientation_error > .02:
        raise RuntimeError(f"Recovery did not reach target: {actual.tolist()}")


def confirm(message):
    while True:
        answer = input(f"{message} | Enter: continue | q: quit\n> ").strip().lower()
        if not answer:
            return True
        if answer in {"q", "quit"}:
            return False


def run_fine(env, policy, keys, duration):
    """Use the original fine workers, without sampling, reset motion or retreat.

    We deliberately do not call env.reset/execute_policy_chunk: those implement
    standalone sampling, timeout labeling and unconditional outcome retreat.
    Recovery and labeling belong to the combined runner instead.
    """
    env.robot.resync_command_pose()
    env._episode_start = time.monotonic()
    env._monitor = FineKeys(keys)
    batch = env._processed_observation()
    steps = 0
    budget = max(1000, math.ceil(duration / env.period_s) + policy.config.chunk_length)
    try:
        env._runtime = FixedChunkRuntime(
            env, policy, batch, warmup=False, step_budget=budget, deterministic=True,
        )
        while True:
            result = env._runtime.next_result()
            steps += result.actual_steps
            if result.done or result.stop_requested or result.info.get("collector_paused"):
                if result.info.get("workspace_violation"):
                    raise RuntimeError(result.info.get("workspace_error", "Fine workspace violation"))
                if result.stop_requested:
                    return "stop", steps
                if result.info.get("timed_out"):
                    return "timeout", steps
                return ("s" if result.info.get("success") else "f"), steps
    finally:
        env.close()
        env.robot.resync_command_pose()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-checkpoint", type=Path, default=Path("outputs/ethernet_coarse/coarse_z_hold1s"))
    parser.add_argument("--fine-checkpoint", type=Path, default=Path("outputs/rlt_928_addon_rgb_z_action6"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=10, help="Number of complete coarse/fine rounds")
    parser.add_argument("--coarse-fps", type=float, default=15)
    parser.add_argument("--fine-fps", type=float, default=30)
    parser.add_argument("--coarse-duration", type=float, default=5)
    parser.add_argument("--fine-duration", type=float, default=15)
    parser.add_argument("--reset-joint-speed", type=float, default=.6)
    parser.add_argument("--recovery-speed", type=float, default=.04, help="TCP recovery speed in m/s")
    parser.add_argument("--z-gate-m", type=float, default=.40207)
    parser.add_argument("--fine-entry-z-m", type=float, default=.40187)
    parser.add_argument("--fine-entry-rotvec-rad", type=float, nargs=3,
                        default=[-2.19967, 2.22472, -.02892], metavar=("RX", "RY", "RZ"))
    parser.add_argument("--fine-entry-z-tolerance-m", type=float, default=.0002)
    parser.add_argument("--fine-entry-angle-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--fine-entry-adjust-speed", type=float, default=.01)
    parser.add_argument("--fine-entry-max-adjustments", type=int, default=3)
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal-ensemble", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=.01)
    parser.add_argument("--prefetch-steps", type=int, default=2)
    parser.add_argument("--robot-ip", default=os.environ.get("ROBOT_IP", "172.16.0.2"))
    parser.add_argument("--wrist-camera", default="349622072679")
    parser.add_argument("--front-camera", default="233522075778")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/act_coarse_fine"))
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Load both models and validate contracts; no robot connection")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("episodes must be positive")
    for name in ("coarse_fps", "fine_fps", "coarse_duration", "fine_duration", "reset_joint_speed", "recovery_speed",
                 "fine_entry_z_tolerance_m", "fine_entry_angle_tolerance_deg", "fine_entry_adjust_speed"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if not math.isfinite(args.fine_entry_z_m) or not np.isfinite(args.fine_entry_rotvec_rad).all():
        parser.error("fine entry target must be finite (Z in metres, rotation vector in radians)")
    if args.fine_entry_max_adjustments < 0:
        parser.error("fine-entry-max-adjustments must be nonnegative (0: check only)")
    if not args.allow_motion and not args.dry_run:
        parser.error("robot motion requires --allow-motion")
    if not args.dry_run and not sys.stdin.isatty():
        parser.error("combined inference requires an interactive terminal for s/f and round confirmation")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    coarse_args = coarse.build_parser().parse_args(["--checkpoint", str(args.coarse_checkpoint)])
    for name in ("device", "rtc", "temporal_ensemble", "temporal_ensemble_coeff", "prefetch_steps", "reset_joint_speed", "z_gate_m"):
        setattr(coarse_args, name, getattr(args, name))
    coarse_args.fps, coarse_args.duration = args.coarse_fps, args.coarse_duration
    coarse.validate_args(parser, coarse_args)
    checkpoint = resolve_checkpoint(args.coarse_checkpoint)
    config = PreTrainedConfig.from_pretrained(str(checkpoint), local_files_only=True)
    mode, coarse_cameras = coarse.checkpoint_action_mode(config, checkpoint)
    config.device = args.device
    coarse_policy = ACTPolicy.from_pretrained(str(checkpoint), config=config, local_files_only=True).to(args.device).eval()
    coarse_pre, coarse_post = make_pre_post_processors(
        config, pretrained_path=str(checkpoint), preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    fine_policy, fine_config = load_frozen_policy(
        resolve_stage2_checkpoint(args.fine_checkpoint), device=args.device,
    )
    fine_cameras = checkpoint_camera_shapes(fine_policy.stage1._act.config)
    # Runtime image contract is deliberately independent of training metadata.
    # Both stages consume native 720p wrist RGB, without resizing or cropping.
    # Keep checkpoint configs intact for their saved processors/model loading.
    print(f"Checkpoint camera shapes (HWC): coarse={coarse_cameras}; fine={fine_cameras}", flush=True)
    coarse_cameras = {name: ((720, 1280, 3) if name == "wrist" else shape)
                      for name, shape in coarse_cameras.items()}
    fine_cameras = {name: ((720, 1280, 3) if name == "wrist" else shape)
                    for name, shape in fine_cameras.items()}
    cameras = {**coarse_cameras, **fine_cameras}
    fine_pre, fine_post = make_pre_post_processors(
        fine_policy.stage1._act.config, pretrained_path=str(resolve_pretrained_model(fine_config.act_checkpoint)),
        preprocessor_overrides={"device_processor": {"device": fine_config.device}},
    )
    print(f"[READY] coarse={checkpoint}; fine={args.fine_checkpoint}; no fine random sampling", flush=True)
    print("[CAMERA] Fixed wrist 1280x720 RGB for coarse and fine; no resizing/cropping.", flush=True)
    print("Fine workspace: measured coarse endpoint +/- [0.05, 0.10, 0.05] m (base XYZ).", flush=True)
    if args.dry_run:
        print("Model/contract check complete; robot not connected.", flush=True)
        return 0

    from evo_rlt.adapters.lerobot.franka_robot import FrankaRobot, FrankaRobotConfig

    robot = FrankaRobot(FrankaRobotConfig(
        robot_ip=args.robot_ip, **robot_camera_kwargs(cameras, args),
        include_gripper_action=fine_policy.act_action_dim == 7, enable_fci_keepalive=False,
    ))
    robot.enable_timestamped_observations()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}.jsonl"

    def make_env():
        return FrankaInsertionStage2Env(
            robot=robot, preprocessor=fine_pre, postprocessor=fine_post, config=fine_config,
            fps=args.fine_fps, episode_time_s=args.fine_duration,
            max_step_m=fine_config.max_step_m, max_step_rad=fine_config.max_step_rad,
            camera_shapes=fine_cameras, z_insertion_mode=True, observation_selection="latest",
        )

    try:
        robot.connect()
        with torch.inference_mode():
            _predict_full_action_chunk(coarse_policy, coarse_pre, coarse_post, policy_observation(robot, config), coarse_cameras)
            coarse_policy.reset()
            # Warm up fine before the first round, without dispatching actions.
            # These temporary bounds are discarded; real bounds are set at Z gate.
            lower, upper = fine_workspace(measured_pose(robot))
            robot.config.workspace_min_xyz, robot.config.workspace_max_xyz = tuple(lower), tuple(upper)
            warm_env = make_env()
            try:
                state, ref, _ = fine_policy.encode_and_reference(warm_env._processed_observation())
                fine_policy.actor_chunk(state, ref, deterministic=True)
            finally:
                warm_env.close()
        robot.config.workspace_min_xyz = robot.config.workspace_max_xyz = None
        if not confirm("Models warmed up. Move to coarse start?"):
            return 0
        coarse.return_to_dataset_start(robot, coarse_args)
        with log_path.open("x") as trace, torch.inference_mode():
            for episode in range(1, args.episodes + 1):
                if not confirm(f"Round {episode}/{args.episodes}: start coarse -> fine?"):
                    break
                clock = RoundClock()

                def emit(event, **fields):
                    row = dict(event=event, round=episode, global_elapsed_s=clock.elapsed(), **fields)
                    trace.write(json.dumps(row, allow_nan=False) + "\n")
                    trace.flush()

                emit("round_start")
                print("[COARSE] s/f: end round | q/x/Esc: stop session", flush=True)
                with GlobalKeys(clock) as keys:
                    result = coarse.run_episode(robot, coarse_policy, coarse_pre, coarse_post, coarse_args,
                                                mode, coarse_cameras, torch, keys, trace)
                emit("coarse_end", **result)
                if result["reason"] == "operator_stop":
                    break
                if result["reason"] in {"z_gate", "duration"}:
                    if result["reason"] == "duration":
                        print("[COARSE TIME LIMIT] Entering fine entry gate before fine inference.", flush=True)
                    fine_start = check_fine_entry(robot, args, emit).copy()
                    lower, upper = fine_workspace(fine_start)
                    robot.config.workspace_min_xyz, robot.config.workspace_max_xyz = tuple(lower), tuple(upper)
                    emit("fine_workspace", center=fine_start[:3].tolist(),
                         workspace_min_xyz=lower.tolist(), workspace_max_xyz=upper.tolist())
                    print(f"[FINE WORKSPACE] {lower.tolist()} .. {upper.tolist()}", flush=True)
                    attempt = 0
                    while True:
                        attempt += 1
                        clock.clear_label()
                        if attempt > 1:
                            check_fine_entry(robot, args, emit)
                        emit("fine_start", attempt=attempt, tcp=fine_start.tolist())
                        print("[FINE] s: finish round | f: return to fine start and retry | q: stop", flush=True)
                        env = make_env()
                        with GlobalKeys(clock) as keys:
                            outcome, steps = run_fine(env, fine_policy, keys, args.fine_duration)
                        if outcome == "timeout":
                            print("[FINE] Time limit; motion stopped. Mark the global outcome.", flush=True)
                            while True:
                                outcome = input("s: success | f: retry fine | q: stop\n> ").strip().lower()
                                if outcome in {"s", "f", "q"}:
                                    break
                            if outcome in {"s", "f"}:
                                clock.mark(outcome)
                        if outcome in {"s", "f"} and clock.key is None:
                            clock.mark(outcome)
                        emit("fine_end", attempt=attempt, outcome=outcome, steps=steps,
                             label_elapsed_s=clock.label_elapsed)
                        if outcome in {"q", "stop"}:
                            return 0
                        print(f"[{'SUCCESS' if outcome == 's' else 'FAILURE'}] Round {episode} | "
                              f"global elapsed {clock.label_elapsed:.3f} s", flush=True)
                        if outcome == "s":
                            target = measured_pose(robot)
                            target[2] += .01
                            move_pose(robot, target, args.recovery_speed, "+Z 1 cm retreat")
                            break
                        paused = time.monotonic()
                        move_pose(robot, fine_start, args.recovery_speed, "return to this fine start")
                        if not confirm("At fine start: retry fine?"):
                            return 0
                        clock.excluded += time.monotonic() - paused
                    emit("round_end", outcome="s", label_elapsed_s=clock.label_elapsed)
                else:
                    # Operator outcomes and other stop reasons do not enter fine.
                    emit("round_end", outcome=clock.key or result["reason"], label_elapsed_s=clock.label_elapsed)
                    print(f"[COARSE END] {clock.key or result['reason']} | "
                          f"global elapsed {(clock.label_elapsed if clock.label_elapsed is not None else clock.elapsed()):.3f} s", flush=True)
                robot.config.workspace_min_xyz = robot.config.workspace_max_xyz = None
                coarse.return_to_dataset_start(robot, coarse_args)
    except KeyboardInterrupt:
        print("[STOP] Operator interrupted session.", flush=True)
    finally:
        if robot.is_connected:
            try:
                robot.robot.stop_servo()
            finally:
                robot.disconnect()
        print(f"Log: {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
