"""Verify that a manual-demo Franka dataset's stored TCP delta actions are
consistent with the measured TCP trajectory.

Convention checked (matching ``record_manual_demo_loop`` / the FR3 recorder):
for every frame ``t < N-1`` the stored ``action[t, :6]`` must equal the 6D delta
that carries the measured ``eef_pose[t]`` to ``eef_pose[t+1]``; the final frame
``N-1`` is stored with a zero TCP action (it has no successor). A seventh
``gripper_target_width`` action, when present, must equal the next measured
gripper width and retain the final measured width on the last frame. As a
second check the script forward-integrates
``pose[t+1] = apply(pose[t], action[t])`` and compares against the measured
pose, so accumulated drift is also reported.

Pure offline check: reads the parquet chunks directly (pandas/pyarrow), never
decodes video, and never touches the robot. Action/state conventions are
imported from ``evo_rlt/adapters/lerobot/franka_robot/pose_math.py`` -- the
single source of truth shared with the recorder.

Usage:
    evo-rlt-verify-manual-demo --repo_id franka_m1_manual_demo \
        --root /home/embint/Evo-RLT/datasets
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

_EE_ORDER = ["ee_x", "ee_y", "ee_z", "ee_rx", "ee_ry", "ee_rz"]


def _load_pose_math():
    """Load pose_math.py without importing the franka_robot package (which pulls
    in pyrealsense / robotLab client). pose_math itself only needs numpy+scipy."""
    path = Path(__file__).resolve().parents[1] / "adapters" / "lerobot" / "franka_robot" / "pose_math.py"
    spec = importlib.util.spec_from_file_location("evo_rlt._verify_pose_math", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.compute_delta_action, mod.apply_delta_action


def _rot_angle_rad(rotvec_a: np.ndarray, rotvec_b: np.ndarray) -> float:
    """Angular distance (rad) between two rotvec rotations."""
    return float(
        np.linalg.norm(
            (Rotation.from_rotvec(rotvec_a).inv() * Rotation.from_rotvec(rotvec_b)).as_rotvec()
        )
    )


def _resolve_dataset_dir(root: str | Path, repo_id: str) -> Path:
    """Resolve the dataset directory.

    LeRobot's ``create(root=...)`` treats ``root`` as the dataset directory
    itself, while record CLI invocations often pass a *container* directory with
    the dataset nested as ``<container>/<repo_id>``. Accept both, plus the
    org-less spelling ``<container>/<last-segment>`` when repo_id is "org/name".
    """
    root = Path(root)
    candidates = [root / repo_id]
    if "/" in repo_id:
        candidates.append(root / repo_id.split("/")[-1])
    candidates.append(root)  # root may itself be the dataset directory
    for cand in candidates:
        if (cand / "meta" / "info.json").exists():
            return cand
    raise FileNotFoundError(
        f"Could not find a LeRobot dataset for repo_id={repo_id!r} under {root}. "
        f"Looked at: {[str(c) for c in candidates]}"
    )


def _verify_episode(
    episode_index: int,
    pose: np.ndarray,
    action: np.ndarray,
    tol_pos: float,
    tol_rot: float,
    compute_delta_action,
    apply_delta_action,
    gripper_width: np.ndarray | None = None,
    gripper_action_index: int | None = None,
    tol_gripper: float = 1e-6,
) -> dict:
    """Per-episode consistency metrics (see module docstring for the convention)."""
    n = pose.shape[0]
    max_pos_err = 0.0
    max_rot_err = 0.0
    mean_pos_err = 0.0
    mean_rot_err = 0.0
    recon_drift_max = 0.0
    recon_drift_final = 0.0

    if n >= 2:
        # 1) Stored action vs the delta recomputed from the measured trajectory.
        frame_errs_pos = np.zeros(n - 1)
        frame_errs_rot = np.zeros(n - 1)
        recon = np.zeros_like(pose)
        recon[0] = pose[0]
        for t in range(n - 1):
            delta = compute_delta_action(pose[t], pose[t + 1])
            stored = action[t]
            frame_errs_pos[t] = float(
                np.linalg.norm(
                    np.asarray([delta["dx"], delta["dy"], delta["dz"]]) - stored[:3]
                )
            )
            frame_errs_rot[t] = _rot_angle_rad(
                np.asarray([delta["drx"], delta["dry"], delta["drz"]]), stored[3:6]
            )
            # 2) Forward reconstruction drift.
            recon[t + 1] = np.asarray(apply_delta_action(recon[t], stored), dtype=float)
        max_pos_err = float(frame_errs_pos.max())
        max_rot_err = float(frame_errs_rot.max())
        mean_pos_err = float(frame_errs_pos.mean())
        mean_rot_err = float(frame_errs_rot.mean())
        drift = np.linalg.norm(recon[1:] - pose[1:], axis=1)
        recon_drift_max = float(drift.max())
        recon_drift_final = float(drift[-1])

    max_gripper_err = None
    if gripper_action_index is not None:
        if gripper_width is None:
            raise ValueError("gripper action is present but gripper state is missing")
        expected = np.concatenate((gripper_width[1:], gripper_width[-1:]))
        max_gripper_err = float(np.max(np.abs(action[:, gripper_action_index] - expected)))

    passed = max_pos_err <= tol_pos and max_rot_err <= tol_rot
    if max_gripper_err is not None:
        passed = passed and max_gripper_err <= tol_gripper
    return {
        "episode_index": int(episode_index),
        "frames": int(n),
        "max_pos_err": max_pos_err,
        "mean_pos_err": mean_pos_err,
        "max_rot_err": max_rot_err,
        "mean_rot_err": mean_rot_err,
        "recon_drift_max": recon_drift_max,
        "recon_drift_final": recon_drift_final,
        "max_gripper_err": max_gripper_err,
        "pass": passed,
    }


def _format_episode(m: dict, tol_pos: float, tol_rot: float, tol_gripper: float) -> str:
    flag = "PASS" if m["pass"] else "FAIL"
    result = (
        f"ep {m['episode_index']:>3}  frames={m['frames']:>4}  "
        f"pos max={m['max_pos_err']:.6f} m (mean {m['mean_pos_err']:.6f}) | tol {tol_pos:g}  "
        f"rot max={m['max_rot_err']:.6f} rad (mean {m['mean_rot_err']:.6f}) | tol {tol_rot:g}  "
        f"recon-drift max={m['recon_drift_max']:.6f} m final={m['recon_drift_final']:.6f} m"
    )
    if m["max_gripper_err"] is not None:
        result += f"  gripper max={m['max_gripper_err']:.8f} m | tol {tol_gripper:g}"
    return f"{result}  => {flag}"


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass  # e.g. when stdout is replaced by a StringIO during tests
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="franka_m1_manual_demo")
    parser.add_argument("--root", default="/home/embint/Evo-RLT/datasets")
    parser.add_argument("--tol-pos", type=float, default=1e-3, help="position tolerance in metres")
    parser.add_argument("--tol-rot", type=float, default=1e-2, help="rotation tolerance in radians")
    parser.add_argument(
        "--tol-gripper",
        type=float,
        default=1e-6,
        help="gripper target-width tolerance in metres",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    compute_delta_action, apply_delta_action = _load_pose_math()

    ds_dir = _resolve_dataset_dir(args.root, args.repo_id)
    info = json.loads((ds_dir / "meta" / "info.json").read_text())

    features = info["features"]
    action_names = list(features["action"]["names"])
    state_names = list(features["observation.state"]["names"])
    state_dim = len(state_names)
    image_keys = [k for k in features if k.startswith("observation.images.")]
    expected_tcp_actions = ["dx", "dy", "dz", "drx", "dry", "drz"]
    if action_names[:6] != expected_tcp_actions:
        raise ValueError(
            f"First six action names must be {expected_tcp_actions}, got {action_names}"
        )

    # TCP pose slice (x,y,z,rx,ry,rz) inside observation.state.
    eef_positions = []
    for name in _EE_ORDER:
        if name not in state_names:
            raise ValueError(
                f"observation.state has no '{name}'; cannot extract eef_pose. "
                f"State names are: {state_names}"
            )
        eef_positions.append(state_names.index(name))

    print(f"Dataset : {ds_dir}")
    print(f"Task    : {info.get('total_tasks', '?')} task(s)")
    print(f"State   : dim={state_dim}, eef slice={eef_positions}")
    print(f"Action  : {action_names}")
    print(f"Images  : {image_keys}")
    has_joint_vel = any(n.startswith("joint_vel_") for n in state_names)
    has_gripper = any(n.startswith("gripper_") for n in state_names)
    gripper_action_index = (
        action_names.index("gripper_target_width")
        if "gripper_target_width" in action_names
        else None
    )
    gripper_state_index = (
        state_names.index("gripper_width") if gripper_action_index is not None else None
    )
    print(
        f"Schema  : joint_velocity={has_joint_vel}, gripper_state={has_gripper} "
        f"(True means this is the manual-demo 22-dim observation.state)"
    )

    chunk_files = sorted(glob.glob(str(ds_dir / "data" / "chunk-*" / "file-*.parquet")))
    if not chunk_files:
        print(f"FAIL: no parquet chunks found under {ds_dir / 'data'}")
        sys.exit(1)
    df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    episodes = []
    for ep, grp in df.groupby("episode_index", sort=True):
        if args.max_episodes is not None and len(episodes) >= args.max_episodes:
            break
        rows = grp.sort_values("frame_index")
        pose = np.stack(rows["observation.state"].to_numpy())[:, eef_positions].astype(float)
        state = np.stack(rows["observation.state"].to_numpy()).astype(float)
        action = np.stack(rows["action"].to_numpy()).astype(float)
        if pose.ndim != 2 or pose.shape[1] != 6 or action.shape[1] not in (6, 7):
            print(f"WARNING: ep {ep} pose={pose.shape} action={action.shape}, skipping")
            continue
        try:
            m = _verify_episode(
                ep, pose, action, args.tol_pos, args.tol_rot,
                compute_delta_action, apply_delta_action,
                gripper_width=(
                    state[:, gripper_state_index]
                    if gripper_state_index is not None
                    else None
                ),
                gripper_action_index=gripper_action_index,
                tol_gripper=args.tol_gripper,
            )
        except Exception as exc:  # structural problem in one episode
            print(f"ep {ep}: ERROR {exc}")
            continue
        episodes.append(m)
        print(_format_episode(m, args.tol_pos, args.tol_rot, args.tol_gripper))

    if not episodes:
        print("FAIL: no episodes could be verified.")
        sys.exit(1)

    n_pass = sum(1 for m in episodes if m["pass"])
    n_frames = sum(m["frames"] for m in episodes)
    print(f"RESULT: {n_pass}/{len(episodes)} episodes PASSED, total {n_frames} frames.")
    sys.exit(0 if n_pass == len(episodes) else 1)


if __name__ == "__main__":
    main()
