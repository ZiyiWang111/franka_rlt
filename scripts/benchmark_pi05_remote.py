"""Measure remote PI05 latency on recorded observations without robot hardware."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/pi05-remote-benchmark-cache")

import numpy as np
import torch

from evo_rlt.adapters.lerobot.franka_remote import protocol_state7 as wire
from scripts.fr3_remote_robot_client import RemoteInferenceClient


def summarize(values):
    values = np.asarray(values, dtype=float)
    return dict(mean_ms=float(values.mean()), median_ms=float(np.median(values)),
                p95_ms=float(np.quantile(values, .95)), min_ms=float(values.min()),
                max_ms=float(values.max()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", required=True)
    p.add_argument("--root", type=Path, default=Path("datasets/pi05_dataset_928_state7_action6"))
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--requests", type=int, default=20)
    p.add_argument("--wait-ready-s", type=float, default=300)
    p.add_argument("--output", type=Path, default=Path("outputs/pi05_remote_latency_multi.json"))
    args = p.parse_args()
    if args.warmup < 0 or args.requests < 1:
        p.error("warmup must be nonnegative and requests positive")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(args.root.name, root=args.root, video_backend="pyav")
    observations = []
    for index in np.linspace(0, len(dataset) - 1, args.requests, dtype=int):
        sample = dataset[int(index)]
        image = sample["observation.images.wrist"].mul(255).round().clamp(0,255).to(torch.uint8)
        observations.append((int(index), sample["observation.state"].numpy(),
                             image.permute(1,2,0).contiguous().numpy(), sample["task"]))
    probe = RemoteInferenceClient(args.server, 5, protocol=wire)
    deadline = time.monotonic() + args.wait_ready_s
    try:
        while True:
            try:
                ready = probe.ping()
                break
            except TimeoutError:
                if time.monotonic() >= deadline:
                    raise
    finally:
        probe.close()
    print("READY", ready["checkpoint"], flush=True)
    client = RemoteInferenceClient(args.server, 60, protocol=wire)
    records = []
    try:
        for i in range(args.warmup + args.requests):
            sample_id = 0 if i < args.warmup else i - args.warmup
            index, state, image, task = observations[sample_id]
            started = time.perf_counter()
            result = client.request(wire.encode_inference_request(
                request_id=i+1, timestamp_ns=time.time_ns(), task=task, state=state, wrist=image), i+1)
            elapsed = (time.perf_counter() - started) * 1000
            if not isinstance(result, wire.InferenceResponse) or result.checkpoint != ready["checkpoint"]:
                raise RuntimeError("Unexpected response/checkpoint changed during benchmark")
            record = dict(request=i+1, sample_index=index, warmup=i < args.warmup,
                          server_ms=result.inference_ms, end_to_end_ms=elapsed,
                          other_ms=elapsed-result.inference_ms)
            records.append(record)
            print(json.dumps(record), flush=True)
    finally:
        client.close()
    measured = [r for r in records if not r["warmup"]]
    summary = {key: summarize([r[key] for r in measured])
               for key in ("server_ms", "end_to_end_ms", "other_ms")}
    report = dict(checkpoint=ready["checkpoint"], source="recorded_dataset_samples",
                  warmup=args.warmup, measured_requests=args.requests,
                  excludes="dataset decoding and live hardware acquisition", summary=summary, requests=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
