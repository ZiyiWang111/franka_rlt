"""Make a separate joint7/TCP6 dataset with exact vector statistics for PI05."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


TASK = "insert the ethernet cable into the hole"
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz"]
STATE_NAMES = [f"joint_{i}" for i in range(7)]
QUANTILES = {"q01": .01, "q10": .1, "q50": .5, "q90": .9, "q99": .99}


def exact_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Statistics require a nonempty finite 2D array")
    result = {
        "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }
    result.update({key: np.quantile(values, q, axis=0).tolist() for key, q in QUANTILES.items()})
    return result


def set_column(table, name, values, dtype=None):
    index = table.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"Missing column: {name}")
    return table.set_column(index, name, pa.array(values, type=dtype))


def write_table(table, path):
    # Embedded HF feature metadata may still describe seven action dimensions.
    pq.write_table(table.replace_schema_metadata(None), path)


def prepare_dataset(source, destination, task=TASK):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if source == destination or source in destination.parents:
        raise ValueError("Output must be outside the source dataset")
    if not task.strip():
        raise ValueError("Task must not be empty")
    info = json.loads((source / "meta/info.json").read_text())
    if info["features"]["observation.state"]["names"] != STATE_NAMES:
        raise ValueError("Expected state joint_0 through joint_6, in that order")
    names = info["features"]["action"]["names"]
    if names != ACTION_NAMES + ["gripper_target_width"]:
        raise ValueError(f"Expected TCP delta + gripper action, got {names}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pi05-prepare-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "dataset"
        shutil.copytree(source, staging)
        vectors = {"observation.state": [], "action": [], "task_index": []}
        episode_ids = []
        for path in sorted((staging / "data").rglob("*.parquet")):
            table = pq.read_table(path)
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)[:, :6]
            states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            if states.shape != (len(table), 7):
                raise ValueError(f"Invalid state shape in {path}")
            tasks = np.zeros((len(table), 1), dtype=np.int64)
            table = set_column(table, "action", actions.tolist(), pa.list_(pa.float32(), 6))
            table = set_column(table, "task_index", tasks[:, 0].tolist(), table.schema.field("task_index").type)
            write_table(table, path)
            vectors["observation.state"].append(states)
            vectors["action"].append(actions)
            vectors["task_index"].append(tasks)
            episode_ids.extend(table["episode_index"].to_pylist())
        vectors = {key: np.concatenate(parts) for key, parts in vectors.items()}
        ids = np.asarray(episode_ids)
        if len(ids) != info["total_frames"] or len(np.unique(ids)) != info["total_episodes"]:
            raise ValueError("Frame/episode count mismatch")
        global_stats = {key: exact_stats(values) for key, values in vectors.items()}
        for key in ("observation.state", "action"):
            span = np.array(global_stats[key]["q99"]) - np.array(global_stats[key]["q01"])
            if np.any(span <= 0):
                raise ValueError(f"Degenerate global quantiles in {key}; choose a constant-dimension policy")
        per_episode = {
            int(ep): {key: exact_stats(values[ids == ep]) for key, values in vectors.items()}
            for ep in np.unique(ids)
        }
        seen = []
        for path in sorted((staging / "meta/episodes").rglob("*.parquet")):
            table = pq.read_table(path)
            eps = table["episode_index"].to_pylist()
            seen.extend(eps)
            for ep, length in zip(eps, table["length"].to_pylist(), strict=True):
                if per_episode[ep]["action"]["count"] != [length]:
                    raise ValueError(f"Episode {ep} length mismatch")
            table = set_column(table, "tasks", [[task] for _ in eps], table.schema.field("tasks").type)
            for key in vectors:
                for stat in global_stats[key]:
                    column = f"stats/{key}/{stat}"
                    old_type = table.schema.field(column).type
                    values = [per_episode[ep][key][stat] for ep in eps]
                    dtype = pa.list_(old_type.value_type, len(values[0])) if pa.types.is_fixed_size_list(old_type) else old_type
                    table = set_column(table, column, values, dtype)
            write_table(table, path)
        if sorted(seen) != sorted(per_episode):
            raise ValueError("Episode metadata is missing or duplicated")
        tasks_path = staging / "meta/tasks.parquet"
        task_table = pq.read_table(tasks_path)
        if set(task_table.column_names) != {"task_index", "task"}:
            raise ValueError("Unexpected task-table schema")
        task_table = pa.Table.from_pylist([{"task_index": 0, "task": task}], schema=task_table.schema)
        write_table(task_table, tasks_path)
        stats_path = staging / "meta/stats.json"
        stats = json.loads(stats_path.read_text())
        stats.update(global_stats)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
        info["features"]["action"].update(shape=[6], names=ACTION_NAMES)
        info["total_tasks"] = 1
        (staging / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
        provenance = {
            "source": str(source), "task": task, "action_names": ACTION_NAMES,
            "statistics": "Exact full-frame/episode numpy statistics; quantile method=linear, std ddof=0",
            "gripper": "Not predicted; deployment must independently maintain the existing grasp",
        }
        (staging / "meta/pi05_preparation.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--task", default=TASK)
    args = parser.parse_args()
    info = prepare_dataset(args.source, args.destination, args.task)
    print(f"Prepared {args.destination}: {info['total_episodes']} episodes, {info['total_frames']} frames, state7/action6")


if __name__ == "__main__":
    main()
