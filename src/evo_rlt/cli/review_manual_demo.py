"""Review and interactively remove local LeRobot dataset episodes.

By default, the latest episode is shown with ``lerobot-dataset-viz``. After an
explicit keep/delete decision, review continues toward older episodes until the
operator quits. The original suspicious-episode scanner remains available as
an optional mode.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "datasets" / "act_rlt_001"


@dataclass(frozen=True)
class Candidate:
    episode_index: int
    frames: int
    duration_s: float
    total_path_m: float
    moving_ratio: float
    tail_static_frames: int
    endpoint_distance_m: float
    post_grasp_path_m: float | None
    reasons: tuple[str, ...]


def _stack_column(group: pd.DataFrame, name: str) -> np.ndarray:
    return np.stack(group[name].to_numpy()).astype(float)


def _tail_true_count(values: np.ndarray) -> int:
    count = 0
    for value in values[::-1]:
        if not value:
            break
        count += 1
    return count


def analyze_dataframe(
    df: pd.DataFrame,
    *,
    state_names: list[str],
    fps: float,
    min_total_path_m: float = 0.10,
    min_moving_ratio: float = 0.20,
    static_translation_m: float = 3e-4,
    static_rotation_rad: float = 3e-3,
    min_tail_static_s: float = 2.0,
    max_endpoint_distance_m: float | None = 0.15,
    require_pick_sequence: bool = False,
    min_post_grasp_path_m: float = 0.25,
) -> list[Candidate]:
    """Return suspicious episodes and the metrics/reasons that flagged them."""
    required = {"episode_index", "frame_index", "action", "observation.state"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    ee_indices = []
    for name in ("ee_x", "ee_y", "ee_z"):
        if name not in state_names:
            raise ValueError(f"observation.state has no {name!r}; state names: {state_names}")
        ee_indices.append(state_names.index(name))
    grasp_index = state_names.index("gripper_grasped") if "gripper_grasped" in state_names else None
    if require_pick_sequence and grasp_index is None:
        raise ValueError("--require-pick-sequence needs gripper_grasped in observation.state")

    ordered = df.sort_values(["episode_index", "frame_index"])
    raw: list[dict] = []
    for episode_index, group in ordered.groupby("episode_index", sort=True):
        state = _stack_column(group, "observation.state")
        action = _stack_column(group, "action")
        if action.ndim != 2 or action.shape[1] < 6:
            raise ValueError(f"Episode {episode_index}: expected action shape (N, >=6), got {action.shape}")
        translation = np.linalg.norm(action[:, :3], axis=1)
        rotation = np.linalg.norm(action[:, 3:6], axis=1)
        static = (translation < static_translation_m) & (rotation < static_rotation_rad)
        xyz = state[:, ee_indices]

        post_grasp_path = None
        has_pick_sequence = None
        if grasp_index is not None:
            grasped = state[:, grasp_index] > 0.5
            rising = np.flatnonzero((~grasped[:-1]) & grasped[1:]) + 1
            has_pick_sequence = len(rising) > 0
            if has_pick_sequence:
                post_grasp_path = float(np.linalg.norm(np.diff(xyz[rising[0] :], axis=0), axis=1).sum())
            else:
                post_grasp_path = 0.0

        raw.append(
            {
                "episode_index": int(episode_index),
                "task_index": int(group["task_index"].iloc[0]) if "task_index" in group else 0,
                "frames": len(group),
                "duration_s": float(group["timestamp"].max()) if "timestamp" in group else len(group) / fps,
                "total_path_m": float(translation.sum()),
                "moving_ratio": float((~static).mean()),
                "tail_static_frames": _tail_true_count(static),
                "endpoint": xyz[-1],
                "post_grasp_path_m": post_grasp_path,
                "has_pick_sequence": has_pick_sequence,
            }
        )

    # Use only meaningfully moving episodes to estimate the dominant task endpoint.
    # A coordinate-wise median tolerates a minority of frozen/incomplete takes.
    endpoint_centers = {}
    for task_index in {r["task_index"] for r in raw}:
        task_rows = [r for r in raw if r["task_index"] == task_index]
        endpoint_source = [r["endpoint"] for r in task_rows if r["total_path_m"] >= min_total_path_m]
        if not endpoint_source:
            endpoint_source = [r["endpoint"] for r in task_rows]
        endpoint_centers[task_index] = np.median(np.stack(endpoint_source), axis=0)
    tail_threshold = max(1, int(round(min_tail_static_s * fps)))

    candidates = []
    for row in raw:
        endpoint_distance = float(
            np.linalg.norm(row["endpoint"] - endpoint_centers[row["task_index"]])
        )
        reasons = []
        if row["total_path_m"] < min_total_path_m:
            reasons.append(f"path<{min_total_path_m:g}m")
        if row["moving_ratio"] < min_moving_ratio:
            reasons.append(f"moving_ratio<{min_moving_ratio:g}")
        if row["tail_static_frames"] >= tail_threshold:
            reasons.append(f"static_tail>={min_tail_static_s:g}s")
        if max_endpoint_distance_m is not None and endpoint_distance > max_endpoint_distance_m:
            reasons.append(f"endpoint_distance>{max_endpoint_distance_m:g}m")
        if require_pick_sequence and (
            not row["has_pick_sequence"] or row["post_grasp_path_m"] < min_post_grasp_path_m
        ):
            reasons.append(f"incomplete_pick/post_grasp<{min_post_grasp_path_m:g}m")
        if reasons:
            candidates.append(
                Candidate(
                    episode_index=row["episode_index"],
                    frames=row["frames"],
                    duration_s=row["duration_s"],
                    total_path_m=row["total_path_m"],
                    moving_ratio=row["moving_ratio"],
                    tail_static_frames=row["tail_static_frames"],
                    endpoint_distance_m=endpoint_distance,
                    post_grasp_path_m=row["post_grasp_path_m"],
                    reasons=tuple(reasons),
                )
            )
    return candidates


def _load_dataset(root: Path) -> tuple[dict, pd.DataFrame, bytes]:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Not a LeRobot dataset root (missing {info_path})")
    before = info_path.read_bytes()
    info = json.loads(before)
    parquet_files = sorted(glob.glob(str(root / "data" / "chunk-*" / "file-*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    frames = []
    for path in parquet_files:
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:
            raise RuntimeError(
                f"Cannot read {path}. The recorder may still be writing this dataset; "
                "stop recording and retry. Original error: {exc}"
            ) from exc
    if info_path.read_bytes() != before:
        raise RuntimeError("meta/info.json changed during the scan; stop the recorder and retry")
    df = pd.concat(frames, ignore_index=True)
    expected = int(info.get("total_frames", len(df)))
    if len(df) != expected:
        raise RuntimeError(
            f"Metadata says {expected} frames but parquet contains {len(df)}; "
            "the dataset may be mid-write or inconsistent"
        )
    return info, df, before


def _print_candidate(candidate: Candidate, fps: float) -> None:
    post = "n/a" if candidate.post_grasp_path_m is None else f"{candidate.post_grasp_path_m:.3f}m"
    print(
        f"\nEpisode {candidate.episode_index}: frames={candidate.frames}, "
        f"duration={candidate.duration_s:.2f}s, path={candidate.total_path_m:.3f}m, "
        f"moving={candidate.moving_ratio:.1%}, static_tail={candidate.tail_static_frames} "
        f"({candidate.tail_static_frames / fps:.2f}s), endpoint_distance="
        f"{candidate.endpoint_distance_m:.3f}m, post_grasp_path={post}"
    )
    print("Reasons: " + ", ".join(candidate.reasons))


def _review_candidates(
    candidates: list[Candidate], *, repo_id: str, root: Path, fps: float, viz_command: str
) -> list[int] | None:
    rejected = []
    for position, candidate in enumerate(candidates, 1):
        print(f"\n--- Candidate {position}/{len(candidates)} ---")
        _print_candidate(candidate, fps)
        command = [
            viz_command,
            "--repo-id",
            repo_id,
            "--root",
            str(root),
            "--episode-index",
            str(candidate.episode_index),
            "--mode",
            "local",
        ]
        print("Running: " + " ".join(command))
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"WARNING: visualizer exited with status {result.returncode}.")
        while True:
            answer = input(
                "关闭可视化后，是否排除此 episode？ "
                "[y=排除 / n=保留 / q=退出且不修改] "
            ).strip().lower()
            if answer in {"y", "yes"}:
                rejected.append(candidate.episode_index)
                break
            if answer in {"n", "no", ""}:
                break
            if answer in {"q", "quit"}:
                return None
            print("请输入 y、n 或 q。")
    return rejected


def _visualize_episode(
    episode_index: int,
    *,
    repo_id: str,
    root: Path,
    viz_command: str,
) -> None:
    command = [
        viz_command,
        "--repo-id",
        repo_id,
        "--root",
        str(root),
        "--episode-index",
        str(episode_index),
        "--mode",
        "local",
    ]
    print("Running: " + " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"WARNING: visualizer exited with status {result.returncode}.")


def _resolve_executable(command: str) -> str:
    resolved = shutil.which(command)
    if resolved is not None:
        return resolved
    sibling = Path(sys.executable).parent / command
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    raise FileNotFoundError(f"Executable is not available in PATH or beside Python: {command}")


def _edit_command(repo_id: str, root: Path, output_root: Path, rejected: list[int]) -> list[str]:
    editor = _resolve_executable("lerobot-edit-dataset")
    return [
        editor,
        "--repo_id",
        repo_id,
        "--root",
        str(root),
        "--new_repo_id",
        repo_id,
        "--new_root",
        str(output_root),
        "--operation.type",
        "delete_episodes",
        "--operation.episode_indices",
        json.dumps(rejected),
    ]


def _apply_deletions(
    repo_id: str,
    root: Path,
    rejected: list[int],
    output_root: Path | None,
    source_info_snapshot: bytes,
    *,
    preserve_backup: bool = True,
) -> Path:
    info_path = root / "meta" / "info.json"
    if info_path.read_bytes() != source_info_snapshot:
        raise RuntimeError("Dataset changed during review; no episodes were removed. Scan again.")
    total_episodes = int(json.loads(source_info_snapshot)["total_episodes"])
    if len(set(rejected)) >= total_episodes:
        raise ValueError("Refusing to remove every episode from the dataset")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    final_root = output_root if output_root is not None else root
    staging = final_root.with_name(f".{final_root.name}.filtering-{timestamp}-{os.getpid()}")
    if final_root != root and final_root.exists():
        raise FileExistsError(f"Output path already exists: {final_root}")
    if staging.exists():
        raise FileExistsError(f"Staging path already exists: {staging}")

    command = _edit_command(repo_id, root, staging, rejected)
    print("\nRebuilding filtered dataset with LeRobot...")
    print("Running: " + " ".join(command))
    subprocess.run(command, check=True)
    if info_path.read_bytes() != source_info_snapshot:
        raise RuntimeError(
            f"Source dataset changed while the filtered copy was being built. "
            f"Source was not replaced; staging data is at {staging}"
        )
    new_info = json.loads((staging / "meta" / "info.json").read_text())
    expected = total_episodes - len(set(rejected))
    if int(new_info["total_episodes"]) != expected:
        raise RuntimeError(
            f"Filtered dataset validation failed: expected {expected} episodes, "
            f"got {new_info['total_episodes']}. Staging data kept at {staging}"
        )

    if output_root is not None:
        staging.rename(final_root)
        return final_root

    backup = root.with_name(f"{root.name}_backup_{timestamp}")
    if backup.exists():
        raise FileExistsError(f"Backup path already exists: {backup}")
    root.rename(backup)
    try:
        staging.rename(root)
    except Exception:
        backup.rename(root)
        raise
    if preserve_backup:
        print(f"Original dataset preserved at: {backup}")
    else:
        shutil.rmtree(backup)
    return root


def _review_from_tail(
    *,
    repo_id: str,
    root: Path,
    viz_command: str,
    start_episode: int | None = None,
) -> None:
    """Review newest-to-oldest and apply each confirmed deletion immediately."""
    cursor = start_episode
    while True:
        info, df, source_info_snapshot = _load_dataset(root)
        total_episodes = int(info["total_episodes"])
        if total_episodes <= 0:
            print("Dataset has no episodes.")
            return
        if cursor is None:
            cursor = total_episodes - 1
        if cursor >= total_episodes:
            raise ValueError(
                f"Episode {cursor} does not exist; valid range is 0..{total_episodes - 1}"
            )
        if cursor < 0:
            print("Reached the first episode; review complete.")
            return

        episode = df[df["episode_index"] == cursor]
        if episode.empty:
            raise RuntimeError(f"Episode {cursor} is missing from dataset parquet files")
        duration = (
            float(episode["timestamp"].max())
            if "timestamp" in episode
            else len(episode) / float(info["fps"])
        )
        print(
            f"\n--- Episode {cursor} / {total_episodes - 1} ---\n"
            f"frames={len(episode)}, duration={duration:.2f}s"
        )
        _visualize_episode(
            cursor,
            repo_id=repo_id,
            root=root,
            viz_command=viz_command,
        )

        while True:
            answer = input(
                "关闭可视化后，是否删除这个 episode？ "
                "[y=删除并查看上一条 / n=保留并查看上一条 / q=退出] "
            ).strip().lower()
            if answer in {"y", "yes"}:
                if total_episodes == 1:
                    print("不能删除数据集中的唯一一条 episode；请选择 n 或 q。")
                    continue
                _apply_deletions(
                    repo_id,
                    root,
                    [cursor],
                    None,
                    source_info_snapshot,
                    preserve_backup=False,
                )
                print(f"Episode {cursor} deleted.")
                cursor -= 1
                break
            if answer in {"n", "no", ""}:
                print(f"Episode {cursor} kept.")
                cursor -= 1
                break
            if answer in {"q", "quit"}:
                print("Review stopped.")
                return
            print("请输入 y、n 或 q。")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        help="Dataset id passed to lerobot-dataset-viz (default: dataset directory name)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Exact local dataset directory (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument("--viz-command", default="lerobot-dataset-viz")
    parser.add_argument(
        "--episode-index",
        type=int,
        help="First episode for descending review (default: latest episode)",
    )
    parser.add_argument(
        "--suspicious-only",
        action="store_true",
        help="Use the original suspicious-episode review instead of descending review",
    )
    parser.add_argument(
        "--scan-only", action="store_true", help="Print candidates without visualization/deletion"
    )
    parser.add_argument("--output-root", type=Path, help="Write a filtered copy instead of replacing --root")
    parser.add_argument("--require-pick-sequence", action="store_true")
    parser.add_argument("--min-total-path-m", type=float, default=0.10)
    parser.add_argument("--min-moving-ratio", type=float, default=0.20)
    parser.add_argument("--static-translation-m", type=float, default=3e-4)
    parser.add_argument("--static-rotation-rad", type=float, default=3e-3)
    parser.add_argument("--min-tail-static-s", type=float, default=2.0)
    parser.add_argument("--max-endpoint-distance-m", type=float, default=0.15)
    parser.add_argument("--min-post-grasp-path-m", type=float, default=0.25)
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = args.root.expanduser().resolve()
    repo_id = args.repo_id or root.name
    try:
        if not args.suspicious_only and not args.scan_only:
            if args.output_root is not None:
                raise ValueError("--output-root requires --suspicious-only or --scan-only")
            _review_from_tail(
                repo_id=repo_id,
                root=root,
                viz_command=_resolve_executable(args.viz_command),
                start_episode=args.episode_index,
            )
            return

        info, df, source_info_snapshot = _load_dataset(root)
        state_names = list(info["features"]["observation.state"]["names"])
        fps = float(info["fps"])
        candidates = analyze_dataframe(
            df,
            state_names=state_names,
            fps=fps,
            min_total_path_m=args.min_total_path_m,
            min_moving_ratio=args.min_moving_ratio,
            static_translation_m=args.static_translation_m,
            static_rotation_rad=args.static_rotation_rad,
            min_tail_static_s=args.min_tail_static_s,
            max_endpoint_distance_m=args.max_endpoint_distance_m,
            require_pick_sequence=args.require_pick_sequence,
            min_post_grasp_path_m=args.min_post_grasp_path_m,
        )
        print(
            f"Dataset: {root}\nEpisodes: {info['total_episodes']}, frames: {info['total_frames']}, "
            f"candidates: {len(candidates)}"
        )
        if not candidates:
            print("No suspicious episodes found.")
            return
        if args.scan_only:
            for candidate in candidates:
                _print_candidate(candidate, fps)
            return
        viz_command = _resolve_executable(args.viz_command)
        rejected = _review_candidates(
            candidates, repo_id=repo_id, root=root, fps=fps, viz_command=viz_command
        )
        if rejected is None:
            print("Review aborted; dataset was not modified.")
            return
        if not rejected:
            print("No episodes selected for removal; dataset was not modified.")
            return
        print(f"\nSelected for removal: {rejected}")
        answer = input("确认一次性重建并排除这些 episode？ [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled; dataset was not modified.")
            return
        destination = _apply_deletions(
            repo_id, root, rejected, args.output_root, source_info_snapshot
        )
        print(f"Done. Filtered dataset: {destination}")
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
