"""High-rate F/T ring-buffer sampler shared by the robot backends.

A background thread polls a backend `read_fn` (UR getActualTCPForce, Franka
tared wrench) into a time-stamped ring buffer; the collector pulls a per-frame
``force_buf`` (the fvla force window) and a raw stream for the sidecar.

Design notes (per codex review):
- The binning is a PURE function (bin_to_rows) so it is unit-tested offline; the
  thread just calls it.
- read_fn is wrapped: a failed/contended read reuses the last sample (never
  raises out of the loop), and achieved_rate() surfaces missed deadlines.
- UR runs at ~500 Hz (fvla parity: 16 raw -> 4 rows). Franka MUST run bounded/
  low-rate: its franky reads share a C++ mutex with motion, so a fast thread
  would add lock contention -- keep the rate modest and never block the caller.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

import numpy as np

WRENCH_DIM = 6


def bin_to_rows(window: np.ndarray, n_rows: int = 4, per_bin: int = 4) -> np.ndarray:
    """Bin the most recent ``n_rows*per_bin`` raw wrench samples into ``n_rows``
    averaged rows (n_rows, 6) -- the fvla force_buf (4 rows from ~16 raw @500 Hz).
    Pads by repeating the oldest available sample when short; zeros if empty."""
    wr = np.asarray(window, dtype=np.float32).reshape(-1, WRENCH_DIM) if np.size(window) \
        else np.zeros((0, WRENCH_DIM), dtype=np.float32)
    need = int(n_rows) * int(per_bin)
    if wr.shape[0] == 0:
        return np.zeros((n_rows, WRENCH_DIM), dtype=np.float32)
    wr = wr[-need:]
    if wr.shape[0] < need:  # left-pad with the oldest sample
        wr = np.vstack([np.repeat(wr[:1], need - wr.shape[0], axis=0), wr])
    return wr.reshape(n_rows, per_bin, WRENCH_DIM).mean(axis=1).astype(np.float32)


class FTSampler:
    """Threaded F/T ring buffer. start()/stop() bracket a session; get_force_buf()
    returns the latest (n_rows, 6) window; get_window() returns the raw (ts, wrench)
    for the audit sidecar."""

    def __init__(self, read_fn: "Callable[[], object]", rate_hz: float = 500.0,
                 window_s: float = 2.0, n_rows: int = 4, per_bin: int = 4) -> None:
        self.read_fn = read_fn
        self.dt = 1.0 / float(rate_hz)
        self.n_rows = int(n_rows)
        self.per_bin = int(per_bin)
        self._buf: "deque" = deque(maxlen=int(rate_hz * window_s) + 1)
        self._lock = threading.Lock()
        self._thread: "threading.Thread | None" = None
        self._stop = threading.Event()
        self._n_read = 0
        self._t0: "float | None" = None
        self._episode: "list | None" = None  # unbounded capture between begin/end_episode

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._n_read = 0
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _loop(self) -> None:
        last = np.zeros(WRENCH_DIM, dtype=np.float32)
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                w = np.asarray(self.read_fn(), dtype=np.float32).reshape(-1)[:WRENCH_DIM]
                if w.size == WRENCH_DIM:
                    last = w
            except Exception:  # noqa: BLE001 -- contended/failed read: reuse last
                w = last
            with self._lock:
                self._buf.append((t, last.copy()))
                if self._episode is not None:
                    self._episode.append((t, last.copy()))
                self._n_read += 1
            elapsed = time.perf_counter() - t
            if self.dt - elapsed > 0:
                time.sleep(self.dt - elapsed)

    def get_force_buf(self) -> np.ndarray:
        with self._lock:
            window = np.array([w for _, w in self._buf], dtype=np.float32) if self._buf \
                else np.zeros((0, WRENCH_DIM), dtype=np.float32)
        return bin_to_rows(window, self.n_rows, self.per_bin)

    def get_window(self) -> "tuple[np.ndarray, np.ndarray]":
        """Raw (timestamps (N,), wrench (N,6)) currently buffered, for the sidecar."""
        with self._lock:
            items = list(self._buf)
        if not items:
            return np.zeros((0,), np.float64), np.zeros((0, WRENCH_DIM), np.float32)
        ts = np.array([t for t, _ in items], dtype=np.float64)
        wr = np.array([w for _, w in items], dtype=np.float32)
        return ts, wr

    def begin_episode(self) -> None:
        """Start an unbounded raw capture (for the per-episode audit sidecar)."""
        with self._lock:
            self._episode = []

    def end_episode(self) -> "tuple[np.ndarray, np.ndarray]":
        """Stop the capture and return the whole episode's raw (ts (N,), wrench (N,6))."""
        with self._lock:
            items = self._episode or []
            self._episode = None
        if not items:
            return np.zeros((0,), np.float64), np.zeros((0, WRENCH_DIM), np.float32)
        ts = np.array([t for t, _ in items], dtype=np.float64)
        wr = np.array([w for _, w in items], dtype=np.float32)
        return ts, wr

    def achieved_rate_hz(self) -> float:
        """Measured sample rate since start (diagnostic; flags missed deadlines)."""
        if self._t0 is None or self._n_read == 0:
            return 0.0
        dur = time.perf_counter() - self._t0
        return self._n_read / dur if dur > 0 else 0.0
