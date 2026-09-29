"""Conservative RealSense exposure times mapped to the host monotonic clock.

Only SDK global_time plus device exposure metadata is accepted. Host arrival
(system_time) is NOT evidence that an image was exposed after a boundary.
"""
from dataclasses import dataclass, field
import math


@dataclass
class TimedObservation:
    observation: dict
    fresh_after: float
    diagnostics: dict = field(default_factory=dict)


class ExposureClock:
    # Global time is an SDK clock estimate, not a hardware synchronization
    # guarantee. Keep a margin and reject implausible/discontinuous estimates.
    MARGIN_S = 0.002
    MAX_AGE_S = 0.250
    MAX_OFFSET_CHANGE_S = 0.005

    def __init__(self):
        self.offset = None
        self.previous = None
        self.previous_number = None
        self.valid_frames = 0
        self.previous_device_us = None
        self.previous_global_ms = None

    def invalidate(self, reason):
        self.valid_frames = 0
        return None, reason

    def convert(self, *, domain, timestamp_ms, frame_us, sensor_us, exposure_us,
                number, wall_s, monotonic_s):
        reject = self.invalidate

        if domain != "global_time":
            return reject("timestamp_domain_" + domain)
        values = (timestamp_ms, frame_us, sensor_us, exposure_us, wall_s, monotonic_s)
        if not all(math.isfinite(v) for v in values):
            return reject("nonfinite_timestamp")
        offset = wall_s - monotonic_s
        old_offset, self.offset = self.offset, offset
        if old_offset is not None and abs(offset - old_offset) > self.MAX_OFFSET_CHANGE_S:
            self.previous = self.previous_number = None
            return reject("host_clock_jump")
        # Metadata timestamps are microseconds on the device clock. Signed
        # modular subtraction handles the 32-bit metadata clock rollover.
        delta_us = (sensor_us - frame_us + 2**31) % 2**32 - 2**31
        if not 0 < exposure_us <= 100_000 or not -100_000 <= delta_us <= 0:
            return reject("invalid_exposure_metadata")
        midpoint = timestamp_ms / 1000 + delta_us / 1e6 - offset
        start = midpoint - exposure_us / 2e6
        if midpoint > monotonic_s or monotonic_s - start > self.MAX_AGE_S:
            return reject("implausible_frame_age")
        if self.previous is not None and (midpoint <= self.previous or number <= self.previous_number):
            self.previous, self.previous_number = midpoint, number
            return reject("nonmonotonic_frame")
        self.previous, self.previous_number = midpoint, number
        previous_device, previous_global = self.previous_device_us, self.previous_global_ms
        self.previous_device_us, self.previous_global_ms = frame_us, timestamp_ms
        if previous_device is not None:
            device_delta = (frame_us - previous_device + 2**31) % 2**32 - 2**31
            global_delta = (timestamp_ms - previous_global) * 1000
            if abs(global_delta - device_delta) > self.MARGIN_S * 1e6:
                return reject("camera_clock_mapping_jump")
        self.valid_frames += 1
        if self.valid_frames < 3:
            return None, "clock_warming_up"
        return (start - self.MARGIN_S, midpoint), None
