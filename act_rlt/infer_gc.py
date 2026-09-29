"""Schedule cyclic garbage collection outside the ACT control window."""

import gc
import os
import time
from pathlib import Path


def resident_memory_bytes():
    """Current Linux RSS, unlike the process lifetime high-water mark."""
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


class EpisodeGarbageCollection:
    """Preserve the caller's GC state even when startup or shutdown fails."""

    def __init__(self):
        self.was_enabled = None
        self.before = None
        self.after = None

    @staticmethod
    def collect():
        rss_before = resident_memory_bytes()
        started = time.monotonic()
        collected = gc.collect(2)
        finished = time.monotonic()
        return {
            "started_monotonic_s": started,
            "finished_monotonic_s": finished,
            "duration_s": finished - started,
            "collected": collected,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": resident_memory_bytes(),
        }

    def restore(self):
        if self.was_enabled:
            gc.enable()
        else:
            gc.disable()

    def __enter__(self):
        self.was_enabled = gc.isenabled()
        gc.disable()
        try:
            self.before = self.collect()
        except BaseException:
            self.restore()
            raise
        return self

    def collect_after_episode(self):
        if self.after is None:
            self.after = self.collect()
        return self.after

    def __exit__(self, *args):
        try:
            self.collect_after_episode()
        finally:
            self.restore()
        return False
