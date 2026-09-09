#!/usr/bin/env python3
"""Create an SFT dataset copy with joint velocities removed from state."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


STATE_KEY = "observation.state"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


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
    velocity_names = set(state_names) & VELOCITY_NAMES
    if velocity_names != VELOCITY_NAMES:
        raise ValueError(
            f"Expected joint_vel_0..6 in state, found {sorted(velocity_names)}"
        )

    keep_indices = [
        index for index, name in enumerate(state_names) if name not in VELOCITY_NAMES
    ]
    kept_names = [state_names[index] for index in keep_indices]
    if len(state_names) != 22 or len(kept_names) != 15:
        raise ValueError(
            f"Expected state 22 -> 15 dimensions, got {len(state_names)} -> {len(kept_names)}"
        )

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
    print(f"State: {len(state_names)} -> {len(kept_names)}")
    print("Kept:", ", ".join(kept_names))


if __name__ == "__main__":
    main()
