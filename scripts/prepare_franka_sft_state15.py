#!/usr/bin/env python3
"""Create an SFT dataset copy with a selected Franka state layout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


STATE_KEY = "observation.state"
JOINT_NAMES = tuple(f"joint_{index}" for index in range(7))
VELOCITY_NAMES = {f"joint_vel_{index}" for index in range(7)}
EPISODE_STATE_STATS = (
    "min",
    "max",
    "mean",
    "std",
    "q01",
    "q10",
    "q50",
    "q90",
    "q99",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--state-layout",
        choices=("state15", "joint7"),
        default="state15",
        help=(
            "Output state layout: state15 removes joint velocities; joint7 keeps "
            "only joint_0..6 (default: state15)"
        ),
    )
    return parser.parse_args()


def select_state_indices(state_names: list[str], state_layout: str) -> list[int]:
    velocity_names = set(state_names) & VELOCITY_NAMES
    if len(state_names) != 22 or velocity_names != VELOCITY_NAMES:
        raise ValueError(
            "Expected the raw 22D Franka state with joint_vel_0..6; "
            f"got {len(state_names)} dimensions and velocities {sorted(velocity_names)}"
        )

    if state_layout == "state15":
        keep_indices = [
            index for index, name in enumerate(state_names) if name not in VELOCITY_NAMES
        ]
        if len(keep_indices) != 15:
            raise ValueError(
                f"Expected state 22 -> 15 dimensions, got 22 -> {len(keep_indices)}"
            )
        return keep_indices

    if state_layout == "joint7":
        missing = [name for name in JOINT_NAMES if name not in state_names]
        duplicates = [name for name in JOINT_NAMES if state_names.count(name) != 1]
        if missing or duplicates:
            raise ValueError(
                f"Expected exactly one joint_0..6; missing={missing}, duplicates={duplicates}"
            )
        return [state_names.index(name) for name in JOINT_NAMES]

    raise ValueError(f"Unsupported state layout: {state_layout}")


def replace_parquet(path: Path, table: pa.Table) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def update_huggingface_metadata(table: pa.Table, state_dim: int) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    raw = metadata.get(b"huggingface")
    if raw is None:
        return table
    huggingface = json.loads(raw)
    state_feature = huggingface["info"]["features"][STATE_KEY]
    state_feature["length"] = state_dim
    metadata[b"huggingface"] = json.dumps(huggingface).encode()
    return table.replace_schema_metadata(metadata)


def rewrite_frame_parquet(path: Path, keep_indices: list[int]) -> None:
    table = pq.read_table(path)
    column_index = table.schema.get_field_index(STATE_KEY)
    states = table.column(column_index).combine_chunks().to_pylist()
    trimmed = [[state[index] for index in keep_indices] for state in states]
    values = pa.array(
        [value for state in trimmed for value in state], type=pa.float32()
    )
    state_array = pa.FixedSizeListArray.from_arrays(values, len(keep_indices))
    table = table.set_column(column_index, STATE_KEY, state_array)
    table = update_huggingface_metadata(table, len(keep_indices))
    replace_parquet(path, table)


def rewrite_episode_parquet(path: Path, keep_indices: list[int]) -> None:
    table = pq.read_table(path)
    for statistic in EPISODE_STATE_STATS:
        key = f"stats/{STATE_KEY}/{statistic}"
        column_index = table.schema.get_field_index(key)
        if column_index < 0:
            continue
        column = table.column(column_index).combine_chunks()
        trimmed = [
            [values[index] for index in keep_indices]
            for values in column.to_pylist()
        ]
        table = table.set_column(
            column_index, key, pa.array(trimmed, type=column.type)
        )
    replace_parquet(path, table)


def rewrite_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=4) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)

    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    state_feature = info["features"][STATE_KEY]
    state_names = state_feature["names"]
    keep_indices = select_state_indices(state_names, args.state_layout)
    kept_names = [state_names[index] for index in keep_indices]

    shutil.copytree(source, destination)

    destination_info_path = destination / "meta/info.json"
    destination_info = json.loads(destination_info_path.read_text())
    destination_info["features"][STATE_KEY]["names"] = kept_names
    destination_info["features"][STATE_KEY]["shape"] = [len(kept_names)]
    rewrite_json(destination_info_path, destination_info)

    for parquet_path in sorted((destination / "data").rglob("*.parquet")):
        rewrite_frame_parquet(parquet_path, keep_indices)

    for parquet_path in sorted((destination / "meta/episodes").rglob("*.parquet")):
        rewrite_episode_parquet(parquet_path, keep_indices)

    stats_path = destination / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    for statistic, values in stats[STATE_KEY].items():
        if statistic == "count":
            continue
        if len(values) != len(state_names):
            raise ValueError(
                f"Unexpected {STATE_KEY} {statistic} length: {len(values)}"
            )
        stats[STATE_KEY][statistic] = [values[index] for index in keep_indices]
    rewrite_json(stats_path, stats)

    print(f"Created {destination}")
    print(f"Layout: {args.state_layout}")
    print(f"State: {len(state_names)} -> {len(kept_names)}")
    print("Kept:", ", ".join(kept_names))


if __name__ == "__main__":
    main()
