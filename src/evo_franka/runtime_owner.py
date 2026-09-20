"""Cooperative, process-scoped ownership of a robot (arm and Hand together).

This prevents accidental use through Evo-RLT, not direct native-library access.
The fixed local lock path must be shared by all cooperating installations.
"""
from __future__ import annotations

import fcntl
import ipaddress
import json
import os
from pathlib import Path
import threading
import time

_LOCK_DIR = Path("/tmp/evo-franka-runtime")
_owners: dict[str, "RuntimeOwner"] = {}
_mutex = threading.RLock()


class OwnershipError(RuntimeError):
    """Ownership is missing or another server already owns this robot."""


def _key(robot_ip: str) -> str:
    # Numeric addresses avoid different hostname aliases creating different locks.
    address = ipaddress.ip_address(robot_ip.strip())
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return str(address)


def require_owner(robot_ip: str) -> None:
    with _mutex:
        owner = _owners.get(_key(robot_ip))
        if owner is None or owner.pid != os.getpid() or owner.fd is None:
            raise OwnershipError(
                f"No runtime ownership for {robot_ip}; use the Evo-RLT control server")


class RuntimeOwner:
    """Acquired only by ControlServer.run; retained across session reconnects."""

    def __init__(self, robot_ip: str):
        self.key = _key(robot_ip)
        self.pid = os.getpid()
        self.fd: int | None = None

    def acquire(self) -> None:
        with _mutex:
            if self.pid != os.getpid() or self.key in _owners:
                raise OwnershipError(f"Runtime ownership already held for {self.key}")
            _LOCK_DIR.mkdir(mode=0o755, exist_ok=True)
            if _LOCK_DIR.is_symlink() or not _LOCK_DIR.is_dir():
                raise OwnershipError(f"Invalid ownership directory: {_LOCK_DIR}")
            path = _LOCK_DIR / f"{self.key}.lock"
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    holder = os.pread(fd, 4096, 0).decode(errors="replace")
                    raise OwnershipError(f"Robot {self.key} is already owned: {holder}") from exc
                metadata = json.dumps({"pid": self.pid, "robot_ip": self.key,
                                       "started_at": time.time()}).encode()
                os.ftruncate(fd, 0)
                os.write(fd, metadata)
            except BaseException:
                os.close(fd)
                raise
            self.fd = fd
            _owners[self.key] = self

    def release(self) -> None:
        with _mutex:
            if self.pid != os.getpid():
                raise OwnershipError("A child process cannot release its parent's ownership")
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
                _owners.pop(self.key, None)
            # Never unlink: waiters must always lock the same inode.


def _after_fork() -> None:
    global _mutex
    for owner in _owners.values():
        if owner.fd is not None:
            os.close(owner.fd)  # close child's duplicate; never LOCK_UN the shared description
            owner.fd = None
    _owners.clear()
    _mutex = threading.RLock()


os.register_at_fork(after_in_child=_after_fork)
