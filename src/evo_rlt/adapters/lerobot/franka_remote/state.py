"""Conversion from the hardware adapter observation to the π0.5 15D state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .protocol import IMAGE_SHAPE, STATE_NAMES, ProtocolError


def observation_to_state15(observation: Mapping[str, Any]) -> np.ndarray:
    missing = [name for name in STATE_NAMES if name not in observation]
    if missing:
        raise ProtocolError(f"hardware observation is missing state fields: {missing}")
    state = np.asarray([observation[name] for name in STATE_NAMES], dtype=np.float32)
    if state.shape != (15,) or not np.isfinite(state).all():
        raise ProtocolError(f"invalid model state: shape={state.shape}, finite={np.isfinite(state).all()}")
    return state


def observation_image(observation: Mapping[str, Any], camera: str) -> np.ndarray:
    if camera not in observation:
        raise ProtocolError(f"hardware observation has no {camera!r} camera")
    image = np.asarray(observation[camera])
    if image.shape != IMAGE_SHAPE or image.dtype != np.uint8:
        raise ProtocolError(
            f"{camera} must be RGB uint8 with shape {IMAGE_SHAPE}; got {image.shape} {image.dtype}"
        )
    return np.ascontiguousarray(image)
