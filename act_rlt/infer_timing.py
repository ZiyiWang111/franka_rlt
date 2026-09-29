"""Opt-in ACT timing probes without disk writes or extra CUDA synchronization."""

import gc
import threading
import time


class ChunkTiming:
    def __init__(self, torch, device, state):
        self.torch = torch
        self.device = device
        self.state = state
        self.phases = []
        self.gpu_events = None
        self.gc_count_before = gc.get_count()

    def call(self, phase, function, *args):
        self.state["phase"] = phase
        gpu_events = None
        if phase == "policy_first" and str(self.device).startswith("cuda"):
            stream = self.torch.cuda.current_stream(self.device)
            gpu_events = (
                self.torch.cuda.Event(enable_timing=True),
                self.torch.cuda.Event(enable_timing=True),
            )
            gpu_events[0].record(stream)
        started = time.monotonic()
        cpu_started = time.thread_time()
        try:
            return function(*args)
        finally:
            cpu_s = time.thread_time() - cpu_started
            finished = time.monotonic()
            if gpu_events is not None:
                gpu_events[1].record(stream)
                self.gpu_events = gpu_events
            self.phases.append({
                "phase": phase,
                "started_monotonic_s": started,
                "finished_monotonic_s": finished,
                "wall_s": finished - started,
                "thread_cpu_s": cpu_s,
            })
            self.state["phase"] = None

    def summary(self):
        totals = {}
        for span in self.phases:
            total = totals.setdefault(span["phase"], {"wall_s": 0.0, "thread_cpu_s": 0.0})
            for key in total:
                total[key] += span[key]
        gpu_ms = None
        # Postprocessing already copies actions to CPU. Only query completed
        # events: never insert synchronize() into the path being diagnosed.
        if self.gpu_events is not None:
            start, end = self.gpu_events
            if end.query():
                gpu_ms = start.elapsed_time(end)
        return {
            "phase_timings": totals,
            "phase_spans": self.phases,
            "policy_cuda_elapsed_ms": gpu_ms,
            "gc_count_before": self.gc_count_before,
            "gc_count_after": gc.get_count(),
        }


class GCPauseTracer:
    def __init__(self, trace, inference_state):
        self.trace = trace
        self.inference_state = inference_state
        self.active = {}

    def __call__(self, phase, info):
        generation = info["generation"]
        if phase == "start":
            self.active[generation] = (
                time.monotonic(), threading.current_thread().name,
                self.inference_state.copy(),
            )
        elif phase == "stop":
            started = self.active.pop(generation, None)
            if started is not None:
                start_s, thread_name, state = started
                end_s = time.monotonic()
                self.trace(
                    "gc_pause", generation=generation,
                    started_monotonic_s=start_s, finished_monotonic_s=end_s,
                    duration_s=end_s - start_s, thread_name=thread_name,
                    inference_chunk_index=state.get("chunk_index"),
                    inference_phase=state.get("phase"),
                    collected=info.get("collected"), uncollectable=info.get("uncollectable"),
                )

    def start(self):
        gc.callbacks.append(self)

    def close(self):
        gc.callbacks.remove(self)
