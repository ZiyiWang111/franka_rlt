#!/usr/bin/env python3
"""Add a gripper target-width action to a Franka manual-demo dataset.

The manual-demo recorder stores the 6D TCP delta from frame t to t+1, but it
only records the measured gripper state.  Its gripper RPC is blocking, so the
next observation is taken after an open/close command completes.  We therefore
use the next frame's measured gripper width as frame t's seventh action:

    [dx, dy, dz, drx, dry, drz, gripper_target_width]

The final frame of each episode keeps its current width.  Unchanged files are
hard-linked into the destination to avoid duplicating the videos.  Every file
that this script changes is atomically replaced first, so the source dataset is
never modified through those hard links.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats


ACTION_KEY = "action"
STATE_KEY = "observation.state"
GRIPPER_ACTION_NAME = "gripper_target_width"
STAT_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


def replace_parquet(path: Path, table: pa.Table) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def rewrite_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=4) + "\n")
    os.replace(temporary, path)


def update_huggingface_metadata(table: pa.Table, action_dim: int) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    raw = metadata.get(b"huggingface")
    if raw is None:
        return table
    huggingface = json.loads(raw)
    action_feature = huggingface["info"]["features"][ACTION_KEY]
    action_feature["length"] = action_dim
    metadata[b"huggingface"] = json.dumps(huggingface).encode()
    return table.replace_schema_metadata(metadata)


def as_fixed_size_list(array: np.ndarray) -> pa.FixedSizeListArray:
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got {array.shape}")
    values = pa.array(array.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(values, array.shape[1])


def json_stats(stats: dict[str, np.ndarray]) -> dict[str, list]:
    return {name: np.asarray(stats[name]).tolist() for name in STAT_NAMES}


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)

    info = json.loads((source / "meta/info.json").read_text())
    action_feature = info["features"][ACTION_KEY]
    state_feature = info["features"][STATE_KEY]
    action_names = list(action_feature["names"])
    state_names = list(state_feature["names"])
    if action_names != ["dx", "dy", "dz", "drx", "dry", "drz"]:
        raise ValueError(f"Expected the Franka 6D delta action, got {action_names}")
    if len(state_names) != 15 or "gripper_width" not in state_names:
        raise ValueError(
            "Expected the 15D state dataset with a gripper_width feature; "
            f"got {state_names}"
        )
    gripper_width_index = state_names.index("gripper_width")
    output_action_names = [*action_names, GRIPPER_ACTION_NAME]
    output_action_feature = {
        **action_feature,
        "names": output_action_names,
        "shape": [len(output_action_names)],
    }

    # Hard-link large, unchanged assets. Atomic replacement below breaks links
    # for every metadata/parquet file that is modified.
    shutil.copytree(source, destination, copy_function=os.link)

    destination_info = json.loads((destination / "meta/info.json").read_text())
    destination_info["features"][ACTION_KEY] = output_action_feature
    rewrite_json(destination / "meta/info.json", destination_info)

    parquet_paths = sorted((destination / "data").rglob("*.parquet"))
    tables = [pq.read_table(path) for path in parquet_paths]

    episode_rows: dict[int, list[tuple[int, int, int]]] = {}
    for table_index, table in enumerate(tables):
        episode_indices = table["episode_index"].combine_chunks().to_pylist()
        frame_indices = table["frame_index"].combine_chunks().to_pylist()
        for row_index, (episode_index, frame_index) in enumerate(zip(episode_indices, frame_indices, strict=True)):
            episode_rows.setdefault(int(episode_index), []).append(
                (int(frame_index), table_index, row_index)
            )

    seventh_by_table = [np.empty(len(table), dtype=np.float32) for table in tables]
    episode_actions: dict[int, np.ndarray] = {}
    transitions: dict[int, int] = {}
    for episode_index, locations in episode_rows.items():
        locations.sort()
        expected_frames = list(range(len(locations)))
        actual_frames = [frame for frame, _, _ in locations]
        if actual_frames != expected_frames:
            raise ValueError(
                f"Episode {episode_index} has non-contiguous frames: "
                f"{actual_frames[:3]}...{actual_frames[-3:]}"
            )

        widths = []
        six_dim_actions = []
        for _, table_index, row_index in locations:
            state = tables[table_index][STATE_KEY][row_index].as_py()
            widths.append(float(state[gripper_width_index]))
            six_dim_actions.append(tables[table_index][ACTION_KEY][row_index].as_py())
        widths_array = np.asarray(widths, dtype=np.float32)
        targets = np.concatenate((widths_array[1:], widths_array[-1:]))
        transitions[episode_index] = int(np.count_nonzero(np.diff(widths_array) < -1e-4))

        for target, (_, table_index, row_index) in zip(targets, locations, strict=True):
            seventh_by_table[table_index][row_index] = target
        episode_actions[episode_index] = np.column_stack(
            (np.asarray(six_dim_actions, dtype=np.float32), targets)
        )

    bad_transitions = {episode: count for episode, count in transitions.items() if count != 1}
    if bad_transitions:
        raise ValueError(f"Expected exactly one gripper closure per episode: {bad_transitions}")

    for path, table, seventh in zip(parquet_paths, tables, seventh_by_table, strict=True):
        action_index = table.schema.get_field_index(ACTION_KEY)
        six_dim_actions = np.asarray(table[ACTION_KEY].combine_chunks().to_pylist(), dtype=np.float32)
        seven_dim_actions = np.column_stack((six_dim_actions, seventh))
        table = table.set_column(action_index, ACTION_KEY, as_fixed_size_list(seven_dim_actions))
        table = update_huggingface_metadata(table, len(output_action_names))
        replace_parquet(path, table)

    episode_stats = {
        episode_index: compute_episode_stats(
            {ACTION_KEY: actions}, {ACTION_KEY: output_action_feature}
        )[ACTION_KEY]
        for episode_index, actions in episode_actions.items()
    }
    all_stats = aggregate_stats(
        [{ACTION_KEY: episode_stats[index]} for index in sorted(episode_stats)]
    )[ACTION_KEY]

    stats_path = destination / "meta/stats.json"
    stats_payload = json.loads(stats_path.read_text())
    stats_payload[ACTION_KEY] = json_stats(all_stats)
    rewrite_json(stats_path, stats_payload)

    for path in sorted((destination / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        episode_indices = table["episode_index"].combine_chunks().to_pylist()
        for stat_name in STAT_NAMES:
            column_name = f"stats/{ACTION_KEY}/{stat_name}"
            column_index = table.schema.get_field_index(column_name)
            if column_index < 0:
                raise ValueError(f"Missing {column_name} in {path}")
            values = [episode_stats[int(ep)][stat_name].tolist() for ep in episode_indices]
            table = table.set_column(
                column_index,
                column_name,
                pa.array(values, type=table.schema.field(column_index).type),
            )
        replace_parquet(path, table)

    widths = np.concatenate([actions[:, -1] for actions in episode_actions.values()])
    print(f"Created: {destination}")
    print(f"Episodes/frames: {len(episode_actions)}/{len(widths)}")
    print(f"Action: 6 -> 7 ({', '.join(output_action_names)})")
    print(f"Gripper target range: {widths.min():.6f} .. {widths.max():.6f} m")
    print("Verified exactly one >0.1 mm closing transition in every episode")


if __name__ == "__main__":
    main()
