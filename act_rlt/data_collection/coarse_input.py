"""Key presses/releases for coarse collection; this module never moves the arm."""
from __future__ import annotations

from collections import deque
import os
import select
import sys
import termios
import threading
import time
import tty

import numpy as np

from scripts.fr3_keyboard_teleop import KEY_DIRECTIONS, discover_keyboard

XY_KEYS = {name[-1].lower(): value[:2] for name, value in KEY_DIRECTIONS.items()
           if name in {"KEY_W", "KEY_S", "KEY_A", "KEY_D"}}
CONTROL_KEYS = {"p", "x", "q", "enter", "=", "-"}


class CoarseKeyboard:
    """Use real release events on X11/evdev, or discrete steps in an SSH terminal."""

    def __init__(self, backend="auto", keyboard_path=None):
        self.backend = ("pynput" if os.environ.get("DISPLAY") else "terminal") if backend == "auto" else backend
        if keyboard_path and self.backend != "evdev":
            raise ValueError("--keyboard applies only to --input-backend evdev")
        self._lock = threading.Lock()
        self._held = set()
        self._releases = {}
        self._events = deque()
        self._stopping = threading.Event()
        self._failure = None
        self._device = self._thread = self._listener = None
        self._fd = self._terminal = None
        try:
            if self.backend == "terminal":
                if not sys.stdin.isatty():
                    raise RuntimeError("coarse collection requires an interactive terminal")
                self._fd = sys.stdin.fileno()
                self._terminal = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
            elif self.backend == "pynput":
                from pynput import keyboard

                def name_for(key):
                    if key == keyboard.Key.enter:
                        return "enter"
                    if key == keyboard.Key.esc:
                        return "q"
                    char = getattr(key, "char", None)
                    return char.lower() if isinstance(char, str) else None

                self._listener = keyboard.Listener(
                    on_press=lambda key: self.feed(name_for(key), True),
                    on_release=lambda key: self.feed(name_for(key), False),
                    suppress=False,
                )
                self._listener.start()
                self._listener.wait()
                if not self._listener.is_alive():
                    raise RuntimeError("pynput keyboard listener failed to start")
                # Consume no terminal input in X11 mode: only callbacks drive the state.
                if sys.stdin.isatty():
                    self._fd = sys.stdin.fileno()
                    self._terminal = termios.tcgetattr(self._fd)
                    tty.setcbreak(self._fd)
            elif self.backend == "evdev":
                self._device = discover_keyboard(keyboard_path)
                self._device.grab()
                self._thread = threading.Thread(target=self._read_evdev, name="coarse-keyboard", daemon=True)
                self._thread.start()
            else:
                raise ValueError(f"unsupported input backend: {self.backend}")
        except BaseException:
            self.close()
            raise

    def _expire(self):
        now = time.monotonic()
        for key, deadline in tuple(self._releases.items()):
            if now >= deadline:
                self._held.discard(key)
                del self._releases[key]

    def feed(self, key, down):
        if key not in XY_KEYS and key not in CONTROL_KEYS:
            return
        with self._lock:
            self._expire()
            if down:
                self._releases.pop(key, None)
                if key not in self._held:
                    self._events.append(key)
                self._held.add(key)
            elif self.backend == "pynput":
                # X11 auto-repeat synthesizes release/press pairs; match existing teleop debounce.
                self._releases[key] = time.monotonic() + 0.06
            else:
                self._held.discard(key)

    def _read_evdev(self):
        try:
            from evdev import ecodes
            names = {getattr(ecodes, "KEY_" + key.upper()): key for key in (*XY_KEYS, "p", "x", "q")}
            names.update({ecodes.KEY_ENTER: "enter", ecodes.KEY_KPENTER: "enter",
                          ecodes.KEY_ESC: "q", ecodes.KEY_EQUAL: "=", ecodes.KEY_MINUS: "-"})
            ctrl = set()
            while not self._stopping.is_set():
                if not select.select([self._device.fd], [], [], 0.05)[0]:
                    continue
                for event in self._device.read():
                    if event.type != ecodes.EV_KEY or event.value == 2:
                        continue
                    if event.code in {ecodes.KEY_LEFTCTRL, ecodes.KEY_RIGHTCTRL}:
                        (ctrl.add if event.value else ctrl.discard)(event.code)
                    if ctrl and event.code == ecodes.KEY_C and event.value:
                        self.feed("q", True)
                    else:
                        self.feed(names.get(event.code), event.value == 1)
        except BaseException as error:
            if not self._stopping.is_set():
                self._failure = error

    def drain(self):
        if self._failure is not None:
            raise RuntimeError("coarse keyboard listener failed") from self._failure
        if self._listener is not None and not self._listener.is_alive():
            raise RuntimeError("coarse keyboard listener stopped")
        if self.backend == "pynput" and self._fd is not None:
            while select.select([self._fd], [], [], 0)[0]:
                if not os.read(self._fd, 1024):
                    raise EOFError("coarse terminal closed")
        if self.backend == "terminal":
            while select.select([self._fd], [], [], 0)[0]:
                char = os.read(self._fd, 1).decode(errors="ignore").lower()
                if not char:
                    raise EOFError("coarse terminal closed")
                if char == "\x03":
                    raise KeyboardInterrupt
                key = "enter" if char in {"\r", "\n"} else ("q" if char == "\x1b" else char)
                if key in XY_KEYS or key in CONTROL_KEYS:
                    self._events.append(key)
        with self._lock:
            self._expire()
            result = list(self._events)
            self._events.clear()
            return result

    def direction_keys_down(self):
        with self._lock:
            self._expire()
            return bool(self._held.intersection(XY_KEYS))

    def direction(self):
        with self._lock:
            self._expire()
            direction = sum((XY_KEYS[key] for key in self._held if key in XY_KEYS), start=np.zeros(2))
        norm = np.linalg.norm(direction)
        return direction if norm == 0 else direction / norm

    def close(self):
        self._stopping.set()
        if self._listener is not None:
            self._listener.stop()
            self._listener.join(timeout=1)
            self._listener = None
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        if self._device is not None:
            try:
                self._device.ungrab()
            finally:
                self._device.close()
                self._device = None
        if self._terminal is not None:
            termios.tcsetattr(self._fd, termios.TCSAFLUSH, self._terminal)
            self._terminal = None
