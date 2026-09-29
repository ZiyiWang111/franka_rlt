"""Nonblocking Stage-2 timings; CUDA events never synchronize the collector."""
from contextlib import contextmanager
import time

import torch


class Stage2Timing:
    def __init__(self):
        self.values = {}
        self.events = {}

    def start_cuda(self, device):
        device = torch.device(device)
        if device.type != "cuda":
            return None
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
        return start, end, stream

    def finish_cuda(self, name, events):
        if events is not None:
            start, end, stream = events
            end.record(stream)
            self.events[name] = (start, end)

    @contextmanager
    def measure(self, name, device="cpu"):
        started = time.monotonic()
        events = self.start_cuda(device)
        try:
            yield
        finally:
            self.finish_cuda(name, events)
            self.values[name + "_wall_ms"] = (time.monotonic() - started) * 1000

    def resolve(self):
        values = dict(self.values)
        for name, (start, end) in self.events.items():
            # Terminal encodes may still be running. Report null, never stall servo.
            values[name + "_cuda_ms"] = start.elapsed_time(end) if end.query() else None
        return values
