#!/usr/bin/env python3
"""Make a separate RGB/action6 dataset without changing the collected source."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz"]


def prepare(source: Path, destination: Path, *, wrench_as_state: bool = False):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if destination == source or source in destination.parents:
        raise ValueError("Destination must be separate from the source dataset")
    source_info = (source / "meta/info.json").read_bytes()
    info = json.loads(source_info)
    action = info["features"]["action"]
    if action["shape"] != [7] or action["names"] != ACTION_NAMES + ["gripper_target_width"]:
        raise ValueError("Expected the original seven-dimensional TCP delta + gripper dataset")
    wrench_feature = info["features"].get("observation.external_wrench")
    if wrench_as_state and (wrench_feature is None or wrench_feature.get("shape") != [6]
                            or wrench_feature.get("names") != ["fx", "fy", "fz", "mx", "my", "mz"]):
        raise ValueError("Expected six-dimensional external wrench in fx/fy/fz/mx/my/mz order")
    destination.parent.mkdir(parents=True, exist_ok=True)

    def copy_file(src, dst):
        # Only read-only video assets may share inodes. All rewritten tables
        # and metadata are independent copies, including on one filesystem.
        if Path(src).relative_to(source).parts[0] == "videos":
            try:
                return os.link(src, dst)
            except OSError:
                pass
        return shutil.copy2(src, dst)

    with TemporaryDirectory(prefix=".act-camera-only-", dir=destination.parent) as temporary:
        staging = Path(temporary) / destination.name
        shutil.copytree(source, staging, copy_function=copy_file)
        frames = 0
        for path in sorted((staging / "data").rglob("*.parquet")):
            table = pq.read_table(path)
            labels = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            if labels.ndim != 2 or labels.shape[1] != 7 or not np.isfinite(labels).all():
                raise ValueError(f"Invalid source actions in {path}")
            table = table.set_column(
                table.schema.get_field_index("action"), "action",
                pa.array(labels[:, :6].tolist(), type=pa.list_(pa.float32(), 6)),
            )
            if wrench_as_state:
                wrench = np.asarray(table["observation.external_wrench"].to_pylist(), dtype=np.float32)
                if wrench.shape != (len(table), 6) or not np.isfinite(wrench).all():
                    raise ValueError(f"Invalid wrench in {path}")
                values = pa.array(wrench.tolist(), type=pa.list_(pa.float32(), 6))
                if "observation.state" in table.column_names:
                    table = table.set_column(table.schema.get_field_index("observation.state"), "observation.state", values)
                else:
                    table = table.append_column("observation.state", values)
            elif "observation.state" in table.column_names:
                table = table.drop(["observation.state"])
            pq.write_table(table.replace_schema_metadata(None), path)
            frames += len(table)
        if frames != info["total_frames"]:
            raise ValueError(f"Frame count mismatch: {frames} != {info['total_frames']}")
        for path in sorted((staging / "meta/episodes").rglob("*.parquet")):
            table = pq.read_table(path)
            table = table.drop([name for name in table.column_names if name.startswith("stats/observation.state/")])
            if wrench_as_state:
                wrench_columns = [name for name in table.column_names if name.startswith("stats/observation.external_wrench/")]
                if not wrench_columns:
                    raise ValueError(f"Missing per-episode wrench statistics in {path}")
                for name in wrench_columns:
                    table = table.append_column(name.replace("observation.external_wrench", "observation.state"), table[name])
            for name in table.column_names:
                if name.startswith("stats/action/") and not name.endswith("/count"):
                    values = [row[:6] for row in table[name].to_pylist()]
                    table = table.set_column(table.schema.get_field_index(name), name, pa.array(values))
            pq.write_table(table.replace_schema_metadata(None), path)
        stats_path = staging / "meta/stats.json"
        stats = json.loads(stats_path.read_text())
        stats.pop("observation.state", None)
        if wrench_as_state:
            stats["observation.state"] = deepcopy(stats["observation.external_wrench"])
        for name, value in stats["action"].items():
            if name != "count":
                stats["action"][name] = value[:6]
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
        info["features"].pop("observation.state", None)
        if wrench_as_state:
            info["features"]["observation.state"] = deepcopy(wrench_feature)
        info["features"]["action"].update(shape=[6], names=ACTION_NAMES)
        (staging / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
        (staging / "meta/camera_only_conversion.json").write_text(json.dumps({
            "source": str(source), "source_info_sha256": hashlib.sha256(source_info).hexdigest(),
            "frames": frames, "action_names": ACTION_NAMES,
            "joint_pose_proprio_removed": True, "videos_unchanged": True,
            "state_source": "observation.external_wrench" if wrench_as_state else None,
        }, indent=2) + "\n")
        if (source / "meta/info.json").read_bytes() != source_info:
            raise RuntimeError("Source metadata changed during conversion")
        staging.rename(destination)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--wrench-as-state", action="store_true",
                        help="replace joint/TCP state with the six measured external wrench channels")
    args = parser.parse_args()
    info = prepare(args.source, args.destination, wrench_as_state=args.wrench_as_state)
    inputs = "RGB + wrench" if args.wrench_as_state else "RGB"
    print(f"Prepared {args.destination}: {info['total_frames']} frames, {inputs} input, six-dimensional TCP delta labels")


if __name__ == "__main__":
    main()
