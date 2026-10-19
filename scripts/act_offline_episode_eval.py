"""Replay a local ACT episode without robot imports and save action comparisons."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from act_rlt.act_encoder import enable_camera_only_act_forward
from act_rlt.infer import (
    ACTION_KEYS, TemporalActionEnsembler, _predict_full_action_chunk,
    bounded_action, checkpoint_camera_shapes, checkpoint_state_mode, resolve_checkpoint,
)


def metrics(prediction, target):
    error = prediction - target
    correlation = []
    for pred, truth in zip(prediction.T, target.T):
        correlation.append(float(np.corrcoef(pred, truth)[0, 1])
                           if pred.std() > 1e-12 and truth.std() > 1e-12 else None)
    return {
        "mae": np.abs(error).mean(axis=0).tolist(),
        "rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
        "correlation": correlation,
        "mean": prediction.mean(axis=0).tolist(),
        "std": prediction.std(axis=0).tolist(),
        "translation_error_norm_mean_m": float(np.linalg.norm(error[:, :3], axis=1).mean()),
        "rotation_error_norm_mean_rad": float(np.linalg.norm(error[:, 3:], axis=1).mean()),
        "translation_target_cosine": float(np.sum(prediction[:, :3] * target[:, :3]) /
            max(np.linalg.norm(prediction[:, :3]) * np.linalg.norm(target[:, :3]), 1e-20)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    args.output.mkdir(parents=True, exist_ok=True)
    info = json.loads((args.dataset / "meta/info.json").read_text())
    metadata = [row for path in sorted((args.dataset / "meta/episodes").rglob("*.parquet"))
                for row in pq.read_table(path).to_pylist() if row["episode_index"] == args.episode]
    if len(metadata) != 1:
        raise ValueError("episode must have exactly one metadata row")
    meta = metadata[0]
    data_path = args.dataset / info["data_path"].format(
        chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"])
    rows = [row for row in pq.read_table(data_path).to_pylist()
            if row["episode_index"] == args.episode]
    rows.sort(key=lambda row: row["frame_index"])
    assert len(rows) == meta["length"]
    assert info["features"]["action"]["names"] == list(ACTION_KEYS)
    checkpoint = str(resolve_checkpoint(args.checkpoint))
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = args.device
    config.pretrained_backbone_weights = None  # Complete trained weights are loaded below.
    policy = ACTPolicy.from_pretrained(checkpoint, config=config, local_files_only=True).to(args.device).eval()
    enable_camera_only_act_forward(policy.model)
    cameras = checkpoint_camera_shapes(config)
    if cameras != {"wrist": (720, 1280, 3)}:
        raise ValueError("this episode replay supports the two single-wrist 720p checkpoints")
    mode = checkpoint_state_mode(config)
    if mode == "wrench":
        assert info["features"]["observation.state"]["names"] == ["fx", "fy", "fz", "mx", "my", "mz"]
    pre, post = make_pre_post_processors(config, pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}})
    key = "observation.images.wrist"
    video = args.dataset / info["video_path"].format(video_key=key,
        chunk_index=meta[f"videos/{key}/chunk_index"], file_index=meta[f"videos/{key}/file_index"])
    offset = meta[f"videos/{key}/from_timestamp"]
    target_times = np.array([offset + row["timestamp"] for row in rows])
    predictions, decode_skews, durations = [], [], []
    cursor = 0
    with av.open(str(video)) as container, torch.inference_mode():
        # Decode in presentation order; only frames from the selected episode enter ACT.
        for frame in container.decode(video=0):
            timestamp = float(frame.pts * frame.time_base)
            if timestamp < target_times[cursor] - 0.5 / info["fps"]:
                continue
            skew = timestamp - target_times[cursor]
            if abs(skew) > 0.5 / info["fps"]:
                raise ValueError(f"video/data timestamp mismatch: {skew}")
            observation = {"wrist": frame.to_ndarray(format="rgb24")}
            if mode == "wrench":
                observation["external_wrench"] = np.asarray(rows[cursor]["observation.state"], dtype=np.float32)
                np.testing.assert_array_equal(observation["external_wrench"], rows[cursor]["observation.external_wrench"])
            started = time.monotonic()
            predictions.append(_predict_full_action_chunk(policy, pre, post, observation, cameras))
            durations.append(time.monotonic() - started)
            decode_skews.append(skew)
            cursor += 1
            if cursor % 10 == 0 or cursor == len(rows):
                print(f"{args.dataset.name}: {cursor}/{len(rows)} frames", flush=True)
            if cursor == len(rows):
                break
    assert cursor == len(rows), "video ended before episode finished"
    chunks = np.stack(predictions)
    truth = np.array([row["action"] for row in rows])
    first = chunks[:, 0]
    n_steps = config.n_action_steps
    queued = np.stack([chunks[t - t % n_steps, t % n_steps] for t in range(cursor)])
    ensemble = TemporalActionEnsembler(config.chunk_size, 0.01, action_dim=6)
    blended = []
    for t, chunk in enumerate(chunks):
        ensemble.add_chunk(t, chunk)
        action, _ = ensemble.action_for(t, 0.002, 0.02)
        blended.append(action.values)
    blended = np.stack(blended)
    clipped = np.array([list(bounded_action(row, 0.002, 0.02).values()) for row in blended])
    future_truth, future_prediction = [], []
    for t, chunk in enumerate(chunks):
        valid = min(config.chunk_size, cursor - t)
        future_truth.extend(truth[t:t + valid])
        future_prediction.extend(chunk[:valid])
    summary = {
        "checkpoint": checkpoint, "dataset": str(args.dataset.resolve()), "episode": args.episode,
        "frames": cursor, "fps": info["fps"], "state_mode": mode, "action_names": list(ACTION_KEYS),
        "units": ["m", "m", "m", "rad", "rad", "rad"], "n_action_steps": n_steps,
        "max_video_alignment_error_s": float(np.max(np.abs(decode_skews))),
        "mean_cpu_forward_s": float(np.mean(durations)),
        "target_mean": truth.mean(axis=0).tolist(), "target_std": truth.std(axis=0).tolist(),
        "first_action": metrics(first, truth), "queue4": metrics(queued, truth),
        "ensemble": metrics(blended, truth), "ensemble_clipped": metrics(clipped, truth),
        "zero_baseline": metrics(np.zeros_like(truth), truth),
        "all_valid_chunk_horizons": metrics(np.array(future_prediction), np.array(future_truth)),
        "limitations": "Training episode; recorded observations only, no closed-loop dynamics. "
                        "Synchronous replay with no inference latency; ensemble uses one chunk per frame.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "predictions.npz", target=truth, chunks=chunks,
                        first=first, queue4=queued, ensemble=blended, ensemble_clipped=clipped)
    with (args.output / "actions.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "timestamp"] + [f"{kind}_{axis}" for kind in
            ("target", "first", "queue4", "ensemble", "ensemble_clipped") for axis in ACTION_KEYS])
        for t in range(cursor):
            writer.writerow([rows[t]["frame_index"], rows[t]["timestamp"],
                             *truth[t], *first[t], *queued[t], *blended[t], *clipped[t]])
    print(f"Saved {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
