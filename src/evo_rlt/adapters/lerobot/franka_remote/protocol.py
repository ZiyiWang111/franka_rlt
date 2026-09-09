"""Validated multipart wire format used by the FR3 remote inference pair."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


PROTOCOL_VERSION = 1
STATE_NAMES = (
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_rx",
    "ee_ry",
    "ee_rz",
    "gripper_width",
    "gripper_grasped",
)
ACTION_NAMES = (
    "dx",
    "dy",
    "dz",
    "drx",
    "dry",
    "drz",
    "gripper_target_width",
)
CAMERA_NAMES = ("wrist", "front")
IMAGE_SHAPE = (480, 640, 3)
ACTION_CHUNK_SHAPE = (50, 7)
MAX_TASK_UTF8_BYTES = 4096


class ProtocolError(ValueError):
    """A malformed or schema-incompatible remote inference message."""


@dataclass(frozen=True)
class InferenceRequest:
    request_id: int
    timestamp_ns: int
    task: str
    state: np.ndarray
    wrist: np.ndarray
    front: np.ndarray


@dataclass(frozen=True)
class InferenceResponse:
    request_id: int
    action_chunk: np.ndarray
    inference_ms: float
    checkpoint: str


def _json_frame(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _parse_json_frame(frame: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(frame.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid JSON header") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("JSON header must be an object")
    return payload


def _check_version(header: dict[str, Any]) -> None:
    if header.get("version") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol version {header.get('version')!r} != {PROTOCOL_VERSION}"
        )


def _as_exact_array(
    value: np.ndarray,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ProtocolError(f"{name} shape {array.shape} != {shape}")
    if array.dtype != dtype:
        raise ProtocolError(f"{name} dtype {array.dtype} != {dtype}")
    if not np.isfinite(array).all():
        raise ProtocolError(f"{name} contains NaN or Inf")
    return np.ascontiguousarray(array)


def encode_ping(request_id: int) -> list[bytes]:
    return [_json_frame({"kind": "ping", "version": PROTOCOL_VERSION, "request_id": request_id})]


def encode_inference_request(
    *,
    request_id: int,
    timestamp_ns: int,
    task: str,
    state: np.ndarray,
    wrist: np.ndarray,
    front: np.ndarray,
) -> list[bytes]:
    if not isinstance(task, str) or not task.strip():
        raise ProtocolError("task must be non-empty text")
    if len(task.encode("utf-8")) > MAX_TASK_UTF8_BYTES:
        raise ProtocolError("task text is too long")
    state = _as_exact_array(state, shape=(15,), dtype=np.dtype("float32"), name="state")
    wrist = _as_exact_array(wrist, shape=IMAGE_SHAPE, dtype=np.dtype("uint8"), name="wrist")
    front = _as_exact_array(front, shape=IMAGE_SHAPE, dtype=np.dtype("uint8"), name="front")
    header = {
        "kind": "infer",
        "version": PROTOCOL_VERSION,
        "request_id": int(request_id),
        "timestamp_ns": int(timestamp_ns),
        "task": task,
        "state_shape": list(state.shape),
        "state_dtype": str(state.dtype),
        "image_shape": list(IMAGE_SHAPE),
        "image_dtype": "uint8",
        "camera_names": list(CAMERA_NAMES),
    }
    return [_json_frame(header), state.tobytes(), wrist.tobytes(), front.tobytes()]


def decode_request(frames: Sequence[bytes]) -> tuple[str, int, InferenceRequest | None]:
    if not frames:
        raise ProtocolError("empty request")
    header = _parse_json_frame(frames[0])
    _check_version(header)
    kind = header.get("kind")
    request_id = int(header.get("request_id", -1))
    if request_id < 0:
        raise ProtocolError("request_id must be non-negative")
    if kind == "ping":
        if len(frames) != 1:
            raise ProtocolError("ping request must contain exactly one frame")
        return kind, request_id, None
    if kind != "infer":
        raise ProtocolError(f"unsupported request kind {kind!r}")
    if len(frames) != 4:
        raise ProtocolError(f"inference request has {len(frames)} frames; expected 4")
    if header.get("state_shape") != [15] or header.get("state_dtype") != "float32":
        raise ProtocolError("request state schema does not match 15D float32")
    if header.get("image_shape") != list(IMAGE_SHAPE) or header.get("image_dtype") != "uint8":
        raise ProtocolError("request image schema does not match 480x640 RGB uint8")
    if header.get("camera_names") != list(CAMERA_NAMES):
        raise ProtocolError(f"camera order must be {CAMERA_NAMES}")
    task = header.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ProtocolError("task must be non-empty text")
    state = np.frombuffer(frames[1], dtype=np.float32).copy()
    wrist = np.frombuffer(frames[2], dtype=np.uint8).copy().reshape(IMAGE_SHAPE)
    front = np.frombuffer(frames[3], dtype=np.uint8).copy().reshape(IMAGE_SHAPE)
    state = _as_exact_array(state, shape=(15,), dtype=np.dtype("float32"), name="state")
    wrist = _as_exact_array(wrist, shape=IMAGE_SHAPE, dtype=np.dtype("uint8"), name="wrist")
    front = _as_exact_array(front, shape=IMAGE_SHAPE, dtype=np.dtype("uint8"), name="front")
    request = InferenceRequest(
        request_id=request_id,
        timestamp_ns=int(header.get("timestamp_ns", 0)),
        task=task,
        state=state,
        wrist=wrist,
        front=front,
    )
    return kind, request_id, request


def encode_ready_response(request_id: int, checkpoint: str) -> list[bytes]:
    return [
        _json_frame(
            {
                "kind": "ready",
                "version": PROTOCOL_VERSION,
                "request_id": request_id,
                "checkpoint": checkpoint,
                "state_names": list(STATE_NAMES),
                "action_names": list(ACTION_NAMES),
                "action_chunk_shape": list(ACTION_CHUNK_SHAPE),
                "camera_names": list(CAMERA_NAMES),
            }
        )
    ]


def encode_error_response(request_id: int, message: str) -> list[bytes]:
    return [
        _json_frame(
            {
                "kind": "error",
                "version": PROTOCOL_VERSION,
                "request_id": request_id,
                "message": str(message),
            }
        )
    ]


def encode_inference_response(
    *,
    request_id: int,
    action_chunk: np.ndarray,
    inference_ms: float,
    checkpoint: str,
) -> list[bytes]:
    action_chunk = _as_exact_array(
        action_chunk,
        shape=ACTION_CHUNK_SHAPE,
        dtype=np.dtype("float32"),
        name="action_chunk",
    )
    header = {
        "kind": "result",
        "version": PROTOCOL_VERSION,
        "request_id": int(request_id),
        "action_shape": list(ACTION_CHUNK_SHAPE),
        "action_dtype": "float32",
        "action_names": list(ACTION_NAMES),
        "inference_ms": float(inference_ms),
        "checkpoint": checkpoint,
    }
    return [_json_frame(header), action_chunk.tobytes()]


def decode_response(frames: Sequence[bytes], expected_request_id: int) -> dict[str, Any] | InferenceResponse:
    if not frames:
        raise ProtocolError("empty response")
    header = _parse_json_frame(frames[0])
    _check_version(header)
    request_id = int(header.get("request_id", -1))
    if request_id != expected_request_id:
        raise ProtocolError(f"response request_id {request_id} != {expected_request_id}")
    kind = header.get("kind")
    if kind == "error":
        raise RuntimeError(f"inference server error: {header.get('message', 'unknown error')}")
    if kind == "ready":
        if len(frames) != 1:
            raise ProtocolError("ready response must contain exactly one frame")
        if header.get("state_names") != list(STATE_NAMES):
            raise ProtocolError("server state schema differs from client")
        if header.get("action_names") != list(ACTION_NAMES):
            raise ProtocolError("server action schema differs from client")
        if header.get("action_chunk_shape") != list(ACTION_CHUNK_SHAPE):
            raise ProtocolError("server action chunk shape differs from client")
        if header.get("camera_names") != list(CAMERA_NAMES):
            raise ProtocolError("server camera order differs from client")
        return header
    if kind != "result":
        raise ProtocolError(f"unsupported response kind {kind!r}")
    if len(frames) != 2:
        raise ProtocolError(f"result response has {len(frames)} frames; expected 2")
    if header.get("action_shape") != list(ACTION_CHUNK_SHAPE):
        raise ProtocolError("response action shape header is inconsistent")
    if header.get("action_dtype") != "float32":
        raise ProtocolError("response action dtype must be float32")
    if header.get("action_names") != list(ACTION_NAMES):
        raise ProtocolError("response action names differ from client schema")
    action_chunk = np.frombuffer(frames[1], dtype=np.float32).copy().reshape(ACTION_CHUNK_SHAPE)
    action_chunk = _as_exact_array(
        action_chunk,
        shape=ACTION_CHUNK_SHAPE,
        dtype=np.dtype("float32"),
        name="action_chunk",
    )
    return InferenceResponse(
        request_id=request_id,
        action_chunk=action_chunk,
        inference_ms=float(header["inference_ms"]),
        checkpoint=str(header["checkpoint"]),
    )
