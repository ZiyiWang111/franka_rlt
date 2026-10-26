"""Copy a TCP-delta dataset, appending synthetic stationary frames to each episode."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset

av.logging.set_level(av.logging.ERROR)


def read_tables(directory):
    return pa.concat_tables([pq.read_table(p) for p in sorted(directory.rglob("*.parquet"))])


def append_tail(source, output, seconds):
    if output.exists():
        raise FileExistsError(output)
    info = json.loads((source / "meta/info.json").read_text())
    fps = info["fps"]
    count = round(seconds * fps)
    if not np.isfinite(seconds) or count < 1 or not np.isclose(count / fps, seconds):
        raise ValueError("Duration must be a positive whole number of frames")
    features = deepcopy(info["features"])
    assert features["action"]["names"] == ["dx", "dy", "dz", "drx", "dry", "drz"]
    assert features["observation.state"]["names"] == ["ee_z"]
    video_key = "observation.images.wrist"
    assert {k for k, v in features.items() if v["dtype"] == "video"} == {video_key}
    assert set(features) == {"action", "observation.state", video_key, "timestamp",
                             "frame_index", "episode_index", "index", "task_index"}
    for feature in features.values():
        feature["shape"] = tuple(feature["shape"])
    data = read_tables(source / "data").sort_by("index").to_pylist()
    episodes = read_tables(source / "meta/episodes").sort_by("episode_index").to_pylist()
    tasks = pq.read_table(source / "meta/tasks.parquet").to_pandas()
    task_names = dict(zip(tasks["task_index"], tasks.index))
    assert len(data) == info["total_frames"]
    assert len(episodes) == info["total_episodes"]
    writer = LeRobotDataset.create(
        repo_id=output.name, root=output, fps=fps, features=features,
        robot_type=info["robot_type"], vcodec="h264", streaming_encoding=True,
        encoder_threads=2,
    )
    # Lossless H.264 encoding avoids introducing compression drift during the hold.
    writer.writer._streaming_encoder.crf = 0
    container = None
    current_path = None
    decoded_index = -1
    last_image = None
    try:
        for e in episodes:
            ep = e["episode_index"]
            rows = data[e["dataset_from_index"]:e["dataset_to_index"]]
            assert len(rows) == e["length"] and all(r["episode_index"] == ep for r in rows)
            path = source / info["video_path"].format(
                video_key=video_key, chunk_index=e[f"videos/{video_key}/chunk_index"],
                file_index=e[f"videos/{video_key}/file_index"],
            )
            if path != current_path:
                if container is not None:
                    container.close()
                container = av.open(str(path))
                frames = iter(container.decode(video=0))
                current_path, decoded_index = path, -1
            first = round(e[f"videos/{video_key}/from_timestamp"] * fps)
            for i, row in enumerate(rows):
                wanted = first + i
                if wanted <= decoded_index:
                    raise ValueError("Source video episodes are not sequential")
                while decoded_index < wanted:
                    last_image = next(frames).to_ndarray(format="rgb24")
                    decoded_index += 1
                writer.add_frame({
                    "action": np.asarray(row["action"], dtype=np.float32),
                    "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                    video_key: last_image,
                    "task": task_names[row["task_index"]],
                })
            for _ in range(count):
                writer.add_frame({
                    "action": np.zeros(6, dtype=np.float32),
                    "observation.state": np.asarray(rows[-1]["observation.state"], dtype=np.float32),
                    video_key: last_image,
                    "task": task_names[rows[-1]["task_index"]],
                })
            # LeRobot maps shape=(1,) to a scalar in parquet (NumPy 2.x).
            buffer = writer.writer.episode_buffer
            buffer["observation.state"] = [value[0] for value in buffer["observation.state"]]
            writer.save_episode()
            if (ep + 1) % 10 == 0 or ep == episodes[-1]["episode_index"]:
                print(f"Saved {ep + 1}/{len(episodes)} episodes", flush=True)
    finally:
        if container is not None:
            container.close()
        writer.finalize()

    restore_state_column(output)
    verify_tail(source, output, seconds)


def restore_state_column(output):
    """Preserve the source's vector-of-length-one state for ACT training."""
    for path in sorted((output / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        state = table["observation.state"]
        if pa.types.is_floating(state.type):
            values = np.asarray(state.to_pylist(), dtype=np.float32).reshape(-1, 1)
            table = table.set_column(
                table.schema.get_field_index("observation.state"), "observation.state",
                pa.array(values.tolist(), type=pa.list_(pa.float32(), 1)),
            )
            pq.write_table(table.replace_schema_metadata(None), path)


def verify_tail(source, output, seconds):
    info = json.loads((source / "meta/info.json").read_text())
    fps = info["fps"]
    count = round(seconds * fps)
    video_key = "observation.images.wrist"
    data = read_tables(source / "data").sort_by("index").to_pylist()
    episodes = read_tables(source / "meta/episodes").sort_by("episode_index").to_pylist()
    # Verify every original numeric sample, every appended sample and all indexing.
    result_table = read_tables(output / "data").sort_by("index")
    assert result_table["observation.state"].type == pa.list_(pa.float32(), 1)
    result = result_table.to_pylist()
    for row in result:
        row["observation.state"] = np.asarray(row["observation.state"]).reshape(1).tolist()
    out_episodes = read_tables(output / "meta/episodes").sort_by("episode_index").to_pylist()
    assert len(result) == len(data) + len(episodes) * count
    for original, augmented in zip(episodes, out_episodes, strict=True):
        before = data[original["dataset_from_index"]:original["dataset_to_index"]]
        after = result[augmented["dataset_from_index"]:augmented["dataset_to_index"]]
        assert len(after) == len(before) + count
        for key in ("action", "observation.state", "task_index"):
            np.testing.assert_array_equal([r[key] for r in before], [r[key] for r in after[:len(before)]])
        np.testing.assert_array_equal([r["action"] for r in after[-count:]], np.zeros((count, 6)))
        np.testing.assert_array_equal([r["observation.state"] for r in after[-count:]],
                                      [before[-1]["observation.state"]] * count)
        np.testing.assert_array_equal([r["frame_index"] for r in after], np.arange(len(after)))
        np.testing.assert_allclose([r["timestamp"] for r in after], np.arange(len(after)) / fps, atol=1e-6)
    np.testing.assert_array_equal([r["index"] for r in result], np.arange(len(result)))

    # Decode all output videos and verify all 15 tail images match their endpoint.
    by_video = {}
    for e in out_episodes:
        path = output / info["video_path"].format(
            video_key=video_key, chunk_index=e[f"videos/{video_key}/chunk_index"],
            file_index=e[f"videos/{video_key}/file_index"],
        )
        by_video.setdefault(path, []).append(e)
    for path, group in by_video.items():
        checks = {}
        for e in group:
            start = round(e[f"videos/{video_key}/from_timestamp"] * fps)
            endpoint = start + e["length"] - count - 1
            for j in range(endpoint, endpoint + count + 1):
                checks[j] = endpoint
        reference = None
        with av.open(str(path)) as video:
            for j, frame in enumerate(video.decode(video=0)):
                if j in checks:
                    image = frame.to_ndarray(format="rgb24")
                    if j == checks[j]:
                        reference = image
                    else:
                        np.testing.assert_array_equal(image, reference)
                    del checks[j]
        assert not checks, f"Missing video frames in {path}"
    provenance = dict(
        source=str(source.resolve()),
        source_info_sha256=hashlib.sha256((source / "meta/info.json").read_bytes()).hexdigest(),
        synthetic=True, tail_seconds=seconds, tail_frames=count,
        state="repeat final state", action="zero TCP delta", video="repeat final decoded RGB frame",
        original_frames=len(data), total_frames=len(result), episodes=len(episodes),
        verification="all original numeric rows preserved; all tail states/actions/images verified",
    )
    (output / "meta/static_tail.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"VERIFIED: {len(episodes)} episodes, {len(result)} frames: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    (verify_tail if args.verify_only else append_tail)(args.source, args.output, args.seconds)
