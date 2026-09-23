"""Keyboard events for in-process FR3 Stage-2 takeover.

Only the Stage-2 servo worker sends robot commands. This module owns keyboard
events and intentionally has no robot connection.
"""

from __future__ import annotations

import math
import select
import threading
import time
from collections import deque

import numpy as np

from scripts.fr3_keyboard_teleop import KEY_DIRECTIONS, discover_keyboard


# Stage-2 reserves S/F for episode outcomes, including during human takeover.
# Move -X to X rather than making success depend on the current control mode.
STAGE2_DIRECTIONS = {key: value for key, value in KEY_DIRECTIONS.items() if key != "KEY_S"}
STAGE2_DIRECTIONS["KEY_X"] = KEY_DIRECTIONS["KEY_S"]


class HumanInputMonitor:
    """Thread-safe evdev or X11 keyboard state for one active episode."""

    _OUTCOMES = {
        "KEY_S": "s", "KEY_F": "f", "KEY_F8": "s", "KEY_F9": "f", "KEY_ESC": "q",
    }
    _RELEASE_DEBOUNCE_S = 0.06

    def __init__(self, backend: str, keyboard_path: str | None = None):
        if backend not in {"evdev", "pynput"}:
            raise ValueError(f"unsupported keyboard backend: {backend}")
        if backend == "pynput" and keyboard_path is not None:
            raise ValueError("--teleop-keyboard applies only to evdev")
        self.backend = backend
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._pressed: set[str] = set()
        self._pending_release: dict[str, float] = {}
        self._toggle_down = False
        self._toggle_pending = False
        self._toggle_requested_at: float | None = None
        self.last_consumed_toggle_at: float | None = None
        self._toggle_release_deadline: float | None = None
        self._outcomes: deque[str] = deque()
        self._failure: BaseException | None = None
        self._device = None
        self._listener = None
        self._thread = None
        if backend == "evdev":
            self._device = discover_keyboard(keyboard_path)
            try:
                self._device.set_clockid(time.CLOCK_MONOTONIC)
            except (AttributeError, OSError):
                pass
            try:
                self._device.grab()
            except BaseException:
                self._device.close()
                raise
            self._thread = threading.Thread(target=self._read_evdev, name="stage2-keyboard", daemon=True)
            self._thread.start()
        else:
            try:
                from pynput import keyboard
            except ImportError as error:
                raise RuntimeError("pynput backend requires `pip install pynput` and X11") from error

            def name_for(key):
                special = {
                    keyboard.Key.space: "KEY_SPACE",
                    keyboard.Key.f8: "KEY_F8",
                    keyboard.Key.f9: "KEY_F9",
                    keyboard.Key.esc: "KEY_ESC",
                }
                if key in special:
                    return special[key]
                char = getattr(key, "char", None)
                name = f"KEY_{char.upper()}" if isinstance(char, str) else ""
                return name if name in STAGE2_DIRECTIONS or name in self._OUTCOMES else None

            self._listener = keyboard.Listener(
                on_press=lambda key: self._event(name_for(key), True),
                on_release=lambda key: self._event(name_for(key), False),
                suppress=True,
            )
            self._listener.start()
            self._listener.wait()
            if not self._listener.is_alive():
                raise RuntimeError("Stage-2 pynput listener failed to start")

    def _read_evdev(self):
        try:
            from evdev import ecodes

            names = {
                **{getattr(ecodes, key): key for key in STAGE2_DIRECTIONS},
                ecodes.KEY_S: "KEY_S",
                ecodes.KEY_F: "KEY_F",
                ecodes.KEY_SPACE: "KEY_SPACE",
                ecodes.KEY_F8: "KEY_F8",
                ecodes.KEY_F9: "KEY_F9",
                ecodes.KEY_ESC: "KEY_ESC",
            }
            while not self._stopping.is_set():
                ready, _, _ = select.select([self._device.fd], [], [], 0.05)
                if not ready:
                    continue
                for event in self._device.read():
                    if event.type == ecodes.EV_KEY and event.value != 2:
                        self._event(names.get(event.code), event.value == 1)
        except BaseException as error:
            if not self._stopping.is_set():
                with self._lock:
                    self._failure = error

    def _event(self, key: str | None, down: bool):
        if key is None:
            return
        with self._lock:
            self._expire_releases()
            if key in STAGE2_DIRECTIONS:
                if down:
                    self._pending_release.pop(key, None)
                    self._pressed.add(key)
                elif self.backend == "pynput":
                    self._pending_release[key] = time.monotonic() + self._RELEASE_DEBOUNCE_S
                else:
                    self._pressed.discard(key)
            elif key == "KEY_SPACE":
                if down:
                    self._toggle_release_deadline = None
                    if not self._toggle_down:
                        self._toggle_pending = True
                        self._toggle_requested_at = time.monotonic()
                        print("Stage-2 control switch requested; waiting for chunk boundary", flush=True)
                    self._toggle_down = True
                elif self.backend == "pynput":
                    self._toggle_release_deadline = time.monotonic() + self._RELEASE_DEBOUNCE_S
                else:
                    self._toggle_down = False
            elif key in self._OUTCOMES and down:
                self._outcomes.append(self._OUTCOMES[key])

    def _check(self):
        if self._failure is not None:
            raise RuntimeError("Stage-2 keyboard listener failed") from self._failure
        if self._listener is not None and not self._listener.is_alive() and not self._stopping.is_set():
            raise RuntimeError("Stage-2 pynput listener stopped unexpectedly")

    def _expire_releases(self):
        now = time.monotonic()
        if self._toggle_release_deadline is not None and now >= self._toggle_release_deadline:
            self._toggle_down = False
            self._toggle_release_deadline = None
        for key, deadline in tuple(self._pending_release.items()):
            if now >= deadline:
                self._pressed.discard(key)
                del self._pending_release[key]

    def poll(self) -> str | None:
        with self._lock:
            self._check()
            return self._outcomes.popleft() if self._outcomes else None

    def direction(self) -> np.ndarray:
        with self._lock:
            self._check()
            self._expire_releases()
            direction = sum((STAGE2_DIRECTIONS[key] for key in self._pressed), start=np.zeros(3))
        norm = float(np.linalg.norm(direction))
        return direction if norm == 0 else direction / norm

    def consume_toggle(self, *, human: bool) -> bool:
        with self._lock:
            self._check()
            self._expire_releases()
            if not self._toggle_pending:
                return False
            self._toggle_pending = False
            if human and self._pressed:
                print("Release all direction keys, then press Space again to return to policy", flush=True)
                return False
            self.last_consumed_toggle_at = self._toggle_requested_at
            return True

    def close(self):
        self._stopping.set()
        if self._listener is not None:
            self._listener.stop()
            self._listener.join(timeout=1.0)
            self._listener = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._device is not None:
            try:
                self._device.ungrab()
            finally:
                self._device.close()
                self._device = None


def validate_human_input(backend: str, keyboard_path: str | None):
    """Fail before connecting the robot when the requested input is unavailable."""
    if backend == "evdev":
        device = discover_keyboard(keyboard_path)
        try:
            device.grab()
            device.ungrab()
        finally:
            device.close()
    elif backend == "pynput":
        if keyboard_path is not None:
            raise ValueError("--teleop-keyboard applies only to evdev")
        from pynput import keyboard  # noqa: F401
    else:
        raise ValueError(f"unsupported keyboard backend: {backend}")


def validate_teleop_speed(speed_m_s: float, fps: float, max_step_m: float):
    if not math.isfinite(speed_m_s) or speed_m_s <= 0:
        raise ValueError("--teleop-speed-m-s must be finite and positive")
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(max_step_m) or max_step_m <= 0:
        return  # generic Stage-2 argument validation reports these values
    if speed_m_s / fps > max_step_m:
        raise ValueError("teleop speed per tick exceeds --max-step-m")
