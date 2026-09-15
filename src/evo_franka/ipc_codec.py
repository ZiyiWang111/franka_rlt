"""Shared msgpack coercion for the control server + client.

``jsonable`` was duplicated verbatim in control_server.py and control_client.py;
it lives here once so both import it (the client must stay franky-free, so it
cannot import the server -- hence a neutral third module). numpy-only, no zmq /
msgpack / franky deps, so it is importable on the collector side too.
"""
from __future__ import annotations

import numpy as np


def jsonable(v):
    """Coerce controller return values / state into msgpack-friendly types."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    return v
