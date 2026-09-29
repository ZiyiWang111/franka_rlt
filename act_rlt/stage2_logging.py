"""Batch Stage-2 metrics on a dedicated writer thread."""

import copy
import json
import math
import queue
import threading
import time
from pathlib import Path


_STOP = object()


class Stage2MetricsWriter:
    """Snapshot every chunk; serialize and flush batches outside the learner."""

    def __init__(self, path, *, format_terminal, interval_s=1.0, queue_capacity=1024):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("metrics interval must be finite and positive")
        if not isinstance(queue_capacity, int) or queue_capacity <= 0:
            raise ValueError("metrics queue capacity must be a positive integer")
        self.path = Path(path)
        self.format_terminal = format_terminal
        self.interval_s = interval_s
        self._queue = queue.Queue(maxsize=queue_capacity)
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._error = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="stage2-metrics", daemon=True,
        )
        self._thread.start()
        # Opening the file can fail. Detect that before connecting the robot.
        self._ready.wait()
        self._raise_if_failed()

    def _raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError(f"Stage-2 metrics writer failed: {self._error}") from self._error

    def append(self, record):
        with self._lock:
            self._raise_if_failed()
            if self._closed:
                raise RuntimeError("Stage-2 metrics writer is closed")
            # Snapshot mutable diagnostics before the learner changes them.
            # Serialization and all file/terminal I/O happen on the writer.
            snapshot = copy.deepcopy(record)
            try:
                self._queue.put_nowait(snapshot)
            except queue.Full as exc:
                raise RuntimeError(
                    "Stage-2 metrics queue is full; writer fell behind; stopping without dropping logs"
                ) from exc

    @staticmethod
    def _write_batch(stream, pending):
        if pending:
            stream.writelines(json.dumps(row, sort_keys=True) + "\n" for row in pending)
            stream.flush()
            pending.clear()

    def _run(self):
        try:
            with self.path.open("a", encoding="utf-8") as stream:
                self._ready.set()
                pending = []
                latest = None
                deadline = time.monotonic() + self.interval_s
                while True:
                    try:
                        record = self._queue.get(timeout=max(0.0, deadline - time.monotonic()))
                    except queue.Empty:
                        record = None
                    if record is _STOP:
                        self._write_batch(stream, pending)
                        break
                    if record is not None:
                        pending.append(record)
                        latest = record
                    if time.monotonic() >= deadline:
                        self._write_batch(stream, pending)
                        if latest is not None:
                            print(self.format_terminal(latest), flush=True)
                            latest = None
                        deadline = time.monotonic() + self.interval_s
        except BaseException as exc:
            self._error = exc
        finally:
            self._ready.set()

    def close(self):
        with self._lock:
            first_close = not self._closed
            self._closed = True
        if first_close:
            while self._thread.is_alive():
                try:
                    self._queue.put(_STOP, timeout=0.05)
                    break
                except queue.Full:
                    continue
        self._thread.join()
        self._raise_if_failed()
