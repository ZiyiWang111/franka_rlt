"""Terminal presentation for frozen Stage-2 inference."""

from __future__ import annotations

import sys
import time
from collections import Counter

from act_rlt.data_collection.sampling import SAMPLE_EULER_XYZ_DEG, SAMPLE_Z_M


class InferenceTerminal:
    def __init__(self, episodes: int, duration: float, *, stream=None):
        self.stream = stream if stream is not None else sys.stdout
        self.episodes = episodes
        self.duration = duration
        self.outcomes = Counter()
        self._line_width = 0
        self._last_status_at = -float("inf")

    def clear_status(self):
        if self._line_width:
            self.stream.write("\r" + " " * self._line_width + "\r")
            self.stream.flush()
            self._line_width = 0

    def write(self, message=""):
        self.clear_status()
        print(message, file=self.stream, flush=True)

    def prompt(self, message):
        self.clear_status()
        return f"  {message}\n> "

    @staticmethod
    def pose(pose):
        xyz = "  ".join(f"{float(v) * 1000:+.1f}" for v in pose[:3])
        rotation = "  ".join(f"{float(v):+.4f}" for v in pose[3:])
        return f"XYZ [mm] {xyz} | rotvec [rad] {rotation}"

    def configuration(self, checkpoint, act_path, config, cameras, args):
        rows = [
            ("Checkpoint", str(checkpoint)),
            ("ACT", str(act_path)),
            ("RL Token", str(config.stage1_checkpoint)),
            ("Policy", "Actor mean | exploration off | weights frozen"),
            ("Device", config.device),
            ("Control", f"{args.fps:g} Hz | {getattr(config, 'chunk_length', 4)} actions/chunk | "
                        f"{args.episode_time:g} s/episode | {args.episodes} episodes"),
            ("Observation", f"{args.observation_selection} | camera mode: {args.camera_boundary_mode}"),
            ("Cameras", ", ".join(f"{name} {shape[1]}x{shape[0]}" for name, shape in cameras.items())),
            ("Step limits", f"{config.max_step_m * 1000:g} mm translation | {config.max_step_rad:g} rad rotation"),
        ]
        if args.z_insertion_mode:
            half_range_mm = getattr(args, "reset_xy_half_range_m", 0.01) * 1000
            rows.append(("Reset", f"random XY +/-{half_range_mm:g} mm around initial reference | =/-: range +/-5 mm (0..50 mm)"))
        else:
            rows.append(("Reset", "random within configured sample/workspace bounds"))
        self.write("\nSTAGE-2 INFERENCE")
        self.write("-" * 72)
        for label, value in rows:
            self.write(f"  {label:<12} {value}")
        self.write("-" * 72)

    def workspace(self, reference, lower, upper, points):
        self.write("\nREFERENCE & RESET")
        self.write(f"  Initial reference: {self.pose(reference)}")
        for index, axis in enumerate("XYZ"):
            self.write(f"  Workspace {axis} [mm]  {lower[index] * 1000:.1f} .. {upper[index] * 1000:.1f}")
        self.write(f"  Reset Z {SAMPLE_Z_M * 1000:g} mm | Euler XYZ [deg] {SAMPLE_EULER_XYZ_DEG}")
        self.write("  Random XY reset around this reference | =/-: next range | r: resample")

    def prepare_episode(self, episode_id):
        self._last_status_at = -float("inf")
        self.write(f"\nEPISODE {episode_id + 1:02d}/{self.episodes:02d} | preparing reset")

    def moving(self, target, label):
        if label == "sampled workspace reset pose":
            label = "random start"
        elif label == "+Z 1 cm retreat":
            label = "retreat +Z 10 mm"
        self.write(f"  Move    {label} | {self.pose(target)}")

    def arrived(self, measured):
        self.write(f"  Reached {self.pose(measured)}")

    def active(self):
        self.write(f"  RUNNING | maximum {self.duration:g} s | s: success | f: failure | q: exit")
        self.write("  s/f ends the episode after the current chunk.")

    def status(self, episode_id, elapsed, steps, clipped, boundary_ms):
        now = time.monotonic()
        if now - self._last_status_at < 1.0:
            return
        self._last_status_at = now
        line = (f"  RUN {episode_id + 1:02d}/{self.episodes:02d} | {elapsed:.1f}/{self.duration:g} s"
                f" | steps {steps} | clipped {clipped} | boundary {boundary_ms:.1f} ms")
        if self.stream.isatty():
            self.stream.write("\r" + line.ljust(self._line_width))
            self.stream.flush()
            self._line_width = len(line)
        else:
            self.write(line)

    def finish_episode(self, episode_id, execution, elapsed, steps, clipped, boundary_times):
        info = execution.info
        if info.get("workspace_violation"):
            outcome = "WORKSPACE"
        elif execution.stop_requested:
            outcome = "STOPPED"
        elif info.get("success"):
            outcome = "SUCCESS"
        elif execution.terminated:
            outcome = "FAILURE"
        elif execution.truncated:
            outcome = "TRUNCATED"
        else:
            outcome = "PAUSED"
        self.outcomes[outcome] += 1
        boundary = sum(boundary_times) / len(boundary_times) if boundary_times else 0.0
        self.write(
            f"  [{outcome}] Episode {episode_id + 1:02d}/{self.episodes:02d} | {elapsed:.1f} s"
            f" | {steps} steps | clipped {clipped}/{steps} | boundary avg {boundary:.1f} ms"
            + (" | time limit reached" if info.get("timed_out") else "")
        )
        if info.get("workspace_error"):
            self.write(f"  Reason: {info['workspace_error']}")

    def session_summary(self, reason):
        completed = sum(v for k, v in self.outcomes.items() if k not in {"STOPPED", "PAUSED"})
        successes = self.outcomes["SUCCESS"]
        rate = f"{100 * successes / completed:.1f}%" if completed else "n/a"
        counts = " | ".join(f"{key.lower()} {count}" for key, count in self.outcomes.items())
        self.write("\n" + "-" * 72)
        self.write(f"Session {reason} | {completed} completed episodes | success rate {rate}")
        if counts:
            self.write("  " + counts)
