#!/usr/bin/env python3
"""Serve the finetuned FR3 π0.5 policy over a validated ZMQ protocol."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/fr3_pi05_sft_60ep_state15_action7_bs4_30k"
    / "checkpoints/030000/pretrained_model"
)
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/fr3_pi05_remote_hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.policies.utils import prepare_observation_for_inference  # noqa: E402

from evo_rlt.adapters.lerobot.franka_remote.protocol import (  # noqa: E402
    ACTION_CHUNK_SHAPE,
    ACTION_NAMES,
    STATE_NAMES,
    InferenceRequest,
    ProtocolError,
    decode_request,
    encode_error_response,
    encode_inference_response,
    encode_ready_response,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--bind", default="tcp://0.0.0.0:5559")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-requests", type=int, default=0, help="0 means serve until interrupted")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class Pi05InferenceRuntime:
    def __init__(self, checkpoint: Path, device: torch.device, seed: int) -> None:
        self.checkpoint = checkpoint.resolve()
        self.device = device
        self.seed = seed
        require((self.checkpoint / "model.safetensors").is_file(), f"missing model: {self.checkpoint}")

        config = PreTrainedConfig.from_pretrained(self.checkpoint, local_files_only=True)
        require(config.type == "pi05", f"expected pi05 checkpoint, got {config.type!r}")
        input_shapes = {name: tuple(feature.shape) for name, feature in config.input_features.items()}
        require(
            input_shapes
            == {
                "observation.state": (15,),
                "observation.images.wrist": (3, 480, 640),
                "observation.images.front": (3, 480, 640),
            },
            f"unexpected checkpoint input schema: {input_shapes}",
        )
        require(
            tuple(config.output_features["action"].shape) == (7,),
            f"unexpected checkpoint action shape: {config.output_features}",
        )
        require(tuple(config.action_feature_names or ()) == ACTION_NAMES, "checkpoint action names differ")
        require(int(config.chunk_size) == ACTION_CHUNK_SHAPE[0], "checkpoint chunk size differs")

        config.device = str(device)
        config.pretrained_path = str(self.checkpoint)
        policy_class = get_policy_class(config.type)
        print(f"Loading checkpoint {self.checkpoint} on {device} ...", flush=True)
        self.policy = policy_class.from_pretrained(
            pretrained_name_or_path=self.checkpoint,
            config=config,
            local_files_only=True,
        )
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        self.config = config
        print("Checkpoint and processors loaded successfully", flush=True)

    def predict(self, request: InferenceRequest) -> tuple[np.ndarray, float]:
        observation = {
            "observation.state": request.state,
            "observation.images.wrist": request.wrist,
            "observation.images.front": request.front,
        }
        prepared = prepare_observation_for_inference(
            observation,
            device=self.device,
            task=request.task,
            robot_type="franka",
        )
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            processed = self.preprocessor(prepared)
            normalized = self.policy.predict_action_chunk(processed)
            action_chunk = self.postprocessor(normalized)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1000.0
        output = action_chunk[0].detach().cpu().to(torch.float32).numpy()
        require(output.shape == ACTION_CHUNK_SHAPE, f"model output shape {output.shape}")
        require(np.isfinite(output).all(), "model output contains NaN or Inf")
        return np.ascontiguousarray(output), inference_ms


def import_zmq():
    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError(
            "pyzmq is required; install the project with `pip install -e '.[lerobot]'`"
        ) from exc
    return zmq


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the π0.5 inference server requires an available CUDA device")
    runtime = Pi05InferenceRuntime(args.checkpoint.expanduser(), device, args.seed)

    zmq = import_zmq()
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.bind(args.bind)
    checkpoint_label = str(runtime.checkpoint)
    print(
        f"READY bind={args.bind} state={len(STATE_NAMES)} action={ACTION_CHUNK_SHAPE}",
        flush=True,
    )

    completed = 0
    try:
        while args.max_requests == 0 or completed < args.max_requests:
            try:
                frames = socket.recv_multipart()
            except zmq.Again:
                continue
            request_id = -1
            try:
                kind, request_id, request = decode_request(frames)
                if kind == "ping":
                    socket.send_multipart(encode_ready_response(request_id, checkpoint_label))
                    continue
                assert request is not None
                action_chunk, inference_ms = runtime.predict(request)
                socket.send_multipart(
                    encode_inference_response(
                        request_id=request_id,
                        action_chunk=action_chunk,
                        inference_ms=inference_ms,
                        checkpoint=checkpoint_label,
                    )
                )
                completed += 1
                gripper = action_chunk[:, 6]
                print(
                    f"request={request_id} inference_ms={inference_ms:.1f} "
                    f"gripper={gripper.min():.5f}..{gripper.max():.5f}",
                    flush=True,
                )
            except Exception as exc:  # keep REP state valid by always replying
                socket.send_multipart(encode_error_response(request_id, str(exc)))
                print(f"request={request_id} ERROR {type(exc).__name__}: {exc}", flush=True)
    except KeyboardInterrupt:
        print("Stopping inference server", flush=True)
    finally:
        socket.close(linger=0)


if __name__ == "__main__":
    main()
