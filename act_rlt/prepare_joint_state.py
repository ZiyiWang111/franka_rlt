"""Create a separate LeRobot dataset with only the seven joint angles in state."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    src, dst = args.root.resolve(), args.output.resolve()
    info_bytes = (src / "meta/info.json").read_bytes()
    info = json.loads(info_bytes)
    names = info["features"]["observation.state"]["names"]
    selected = [names.index(f"joint_{i}") for i in range(7)]
    if dst.exists():
        parser.error(f"Output already exists: {dst}")
    if src in dst.parents:
        parser.error("Output must be outside the source dataset")
    shutil.copytree(src, dst)
    rows = 0
    for path in sorted((dst / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        values = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        state = pa.array(values[:, selected].tolist(), type=pa.list_(pa.float32(), 7))
        table = table.set_column(table.schema.get_field_index("observation.state"), "observation.state", state)
        # Remove embedded HF schemas describing the old 22D array.
        table = table.replace_schema_metadata(None)
        pq.write_table(table, path)
        rows += len(table)
    assert rows == info["total_frames"], (rows, info["total_frames"])
    for path in sorted((dst / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        for name in table.column_names:
            if name.startswith("stats/observation.state/") and not name.endswith("/count"):
                values = table[name].to_pylist()
                sliced = [[v[i] for i in selected] for v in values]
                table = table.set_column(table.schema.get_field_index(name), name, pa.array(sliced))
        pq.write_table(table.replace_schema_metadata(None), path)
    stats_path = dst / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    for key, value in stats["observation.state"].items():
        if key != "count":
            stats["observation.state"][key] = [value[i] for i in selected]
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    info["features"]["observation.state"].update(shape=[7], names=[f"joint_{i}" for i in range(7)])
    (dst / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    if (src / "meta/info.json").read_bytes() != info_bytes:
        raise RuntimeError("Source changed during conversion; do not train on this copy")
    print(f"Prepared {dst}: {rows} frames, state=7 joint angles, action unchanged")


if __name__ == "__main__":
    main()
