"""Receive SpaceMouse input from an isolated ROS Python process."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import signal
import subprocess
import threading
import time

import numpy as np

from act_rlt.human_input import HumanInputMonitor


class SpaceMouseInput:
    def __init__(self, linear_speed=0.0135, angular_speed=0.09, deadzone=0.05):
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._failure = None
        self._velocity = np.zeros(6)
        self._received_at = float('-inf')
        self._logs = deque(maxlen=12)
        self._closed = False
        script = Path(__file__).resolve().parents[1] / 'scripts/fr3_spacemouse_input.sh'
        self._process = subprocess.Popen(
            ['bash', str(script), '--linear-speed', str(linear_speed),
             '--angular-speed', str(angular_speed), '--deadzone', str(deadzone)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )
        self._thread = threading.Thread(target=self._read, name='stage2-spacemouse', daemon=True)
        self._thread.start()
        try:
            if not self._ready.wait(10):
                with self._lock:
                    logs = '\n'.join(self._logs)
                raise RuntimeError('SpaceMouse produced no input within 10 s: ' + logs)
            self.velocity()
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self._process.stdout:
                if not line.startswith('SPACEMOUSE '):
                    with self._lock:
                        self._logs.append(line.rstrip())
                    continue
                timestamp, values = json.loads(line[len('SPACEMOUSE '):])
                values = np.asarray(values, dtype=float)
                if values.shape != (6,) or not np.isfinite(values).all() or not np.isfinite(timestamp):
                    raise ValueError('Invalid SpaceMouse sample')
                with self._lock:
                    self._velocity = values
                    self._received_at = timestamp
                self._ready.set()
            raise RuntimeError('SpaceMouse input process exited: ' + '\n'.join(self._logs))
        except BaseException as error:
            with self._lock:
                self._failure = error
            self._ready.set()

    def velocity(self):
        with self._lock:
            if self._failure is not None:
                raise RuntimeError('SpaceMouse input failed') from self._failure
            # Same monotonic clock in both processes; queued old samples stay stale.
            if time.monotonic() - self._received_at > 0.2:
                return np.zeros(6)
            return self._velocity.copy()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._process.poll() is None:
            try:
                self._process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._thread.join(timeout=2)
        self._process.stdout.close()


class SpaceMouseHumanInputMonitor(HumanInputMonitor):
    """Keep keyboard outcome/switch events, use only SpaceMouse for motion."""
    def __init__(self, backend, keyboard_path, spacemouse):
        self.spacemouse = spacemouse
        super().__init__(backend, keyboard_path, motion_keyboard=False)

    def motion_velocity(self):
        return self.spacemouse.velocity()

    def consume_toggle(self, *, human):
        requested = super().consume_toggle(human=human)
        if requested and human and self.motion_velocity().any():
            print('Release SpaceMouse, then press Space again to return to policy', flush=True)
            return False
        return requested
