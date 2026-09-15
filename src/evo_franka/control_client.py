#!/usr/bin/env python3
"""ZMQ client that mirrors FrankaArmController's API over the control server.

Drop-in for ``FrankaArmController``: same public methods, but each call is
forwarded to the standalone :mod:`evo_franka.control_server`
that owns the franky 1 kHz FCI loop in its OWN process. So the collector's
recording + JPEG encoding can no longer starve the control loop (the episode-N
"hang").

  * state reads (get_joint_angles / get_tool_pose / get_tool_force_raw / ...)
    are served from a last-value cache fed by the server's PUSH stream -- local,
    never blocked by an in-flight move;
  * blocking moves (move_*) use an accepted + poll-until-idle handshake;
  * servo_*/speed_* are fire-and-forget (the server coalesces to the latest);
  * stop_*/recover/reset_session are interrupts the server processes even mid-move
    (so the collector's watchdog stop_move actually preempts a wedged move).

Mirrors ArrebolBlack/franka-control's robot_client. Selected by
robots/franka/adapter.py when FRANKA_CONTROL_IPC is set.
"""
from __future__ import annotations

import logging
import os
import threading
import time

import msgpack
import zmq

from evo_franka.base_errors import JointSwingExceeded
from evo_franka import constants as C
from evo_franka.errors import (FrankaError, FrankaMotionRefused, FrankaReflex,
                               FrankaSessionLost)
from evo_franka.ipc_codec import jsonable as _jsonable

logger = logging.getLogger("control_client")

# The server's _err() envelope carries error_type (the exception class name).
# Re-raise it TYPED here so callers see ONE error contract whether the driver
# runs in-process or behind the control server -- typed branches (the
# collector's start-resample loop, teleop's session healing) work identically
# on both paths, and no caller ever matches message text. Unknown names (and
# dispatch-level rejections, which carry no error_type) stay RuntimeError.
# Every class here subclasses RuntimeError, so existing `except RuntimeError`
# sites keep catching them unchanged.
_TYPED_ERRORS = {c.__name__: c for c in (
    JointSwingExceeded, FrankaMotionRefused, FrankaReflex, FrankaSessionLost,
    FrankaError)}


def _raise_failure(result: dict, fallback: str) -> None:
    raise _TYPED_ERRORS.get(result.get("error_type"), RuntimeError)(
        result.get("error", fallback))


class FrankaArmControllerClient:
    """Same constructor shape as FrankaArmController(robot_ip, dynamics_factor).

    robot_ip / relative_dynamics_factor are honored by the SERVER (started
    separately, e.g. ``python -m evo_franka.control_server --ip ...``); they are kept
    here only for API parity. The client connects to the server on
    FRANKA_CONTROL_HOST (default 127.0.0.1)."""

    def __init__(self, robot_ip=None,
                 relative_dynamics_factor=C.DEFAULT_DYNAMICS_FACTOR,
                 *args, host=None, cmd_port=None, state_port=None, **kwargs):
        if robot_ip is None:
            robot_ip = C.DEFAULT_ROBOT_IP
        self._robot_ip = robot_ip
        self._rdf = relative_dynamics_factor
        self._host = host or os.environ.get("FRANKA_CONTROL_HOST", C.CONTROL_SERVER_HOST)
        self._cmd_port = cmd_port or C.CONTROL_SERVER_CMD_PORT
        self._state_port = state_port or C.CONTROL_SERVER_STATE_PORT

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.setsockopt(zmq.RCVTIMEO, C.CONTROL_CLIENT_RCV_TIMEOUT_MS)
        self._sock.setsockopt(zmq.SNDTIMEO, C.CONTROL_CLIENT_RCV_TIMEOUT_MS)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(f"tcp://{self._host}:{self._cmd_port}")

        self._pull = self._ctx.socket(zmq.PULL)
        self._pull.setsockopt(zmq.RCVHWM, 2)
        self._pull.setsockopt(zmq.RCVTIMEO, C.CONTROL_CLIENT_STATE_RCV_TIMEOUT_MS)
        self._pull.setsockopt(zmq.LINGER, 0)
        self._pull.connect(f"tcp://{self._host}:{self._state_port}")

        self._lock = threading.Lock()        # serializes the command (DEALER) socket
        self._cache = None
        self._cache_lock = threading.Lock()
        self._stream_on = True
        self.robot = None                     # truthy after connect() -- FrankaAdapter.is_connected probe
        # Streamed-servo faults observed so far. None = no baseline yet; the first
        # servo call ADOPTS the server's current count without raising, so a
        # long-lived server's faults from a PREVIOUS run never surface as this
        # client's. See _raise_pending_servo_fault.
        self._servo_fault_seq = None

        self._rx = threading.Thread(target=self._state_rx, daemon=True, name="state-rx")
        self._rx.start()
        logger.info("control client -> %s (cmd %d / state %d)",
                    self._host, self._cmd_port, self._state_port)

    # -- transport -----------------------------------------------------------
    def _send(self, command, params=None, expect_reply=True):
        msg = {"command": command}
        if params:
            msg["params"] = params
        packed = msgpack.packb(msg, use_bin_type=True)
        with self._lock:
            try:
                self._sock.send_multipart([b"", packed])
                if not expect_reply:
                    return None
                parts = self._sock.recv_multipart()
            except zmq.Again:
                return {"success": False, "error": "server timeout"}
            except Exception as e:
                return {"success": False, "error": str(e)}
        try:
            return msgpack.unpackb(parts[-1], raw=False)
        except Exception as e:
            return {"success": False, "error": f"bad reply: {e}"}

    def _state_rx(self):
        while self._stream_on:
            try:
                raw = self._pull.recv()
                while True:                     # drain to the newest snapshot
                    try:
                        raw = self._pull.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                snap = msgpack.unpackb(raw, raw=False)
                with self._cache_lock:
                    self._cache = snap
            except zmq.Again:
                continue
            except Exception:
                if self._stream_on:
                    continue
                break

    def _cached(self, key: str, default=None):
        """Read ONE field from the PUSH-stream cache, never blocking -- the single
        owner of a hot-path state read on this client.

        The 30 Hz action path may only read state this way. `_snap()` is the
        blocking reader (it falls back to a `get_state` round trip when no
        snapshot has arrived), which is right for setup and preflight and wrong
        for anything called per servo tick. Keeping the never-block rule in ONE
        named place is why this exists: it was open-coded twice and asserted in a
        docstring a third time, and the assertion was the one that was false."""
        with self._cache_lock:
            return (self._cache or {}).get(key, default)

    def _snap(self) -> dict:
        with self._cache_lock:
            if self._cache is not None:
                return self._cache
        resp = self._send("get_state")          # cold start: no stream yet
        if resp and resp.get("success"):
            snap = resp.get("state", {}) or {}
            with self._cache_lock:
                self._cache = snap
            return snap
        raise RuntimeError(f"no robot state: {resp.get('error') if resp else 'no reply'}")

    def _rpc(self, method, *args, **kwargs):
        resp = self._send(method, {"args": _jsonable(list(args)), "kwargs": _jsonable(kwargs)})
        if not resp or not resp.get("success"):
            raise RuntimeError(f"{method} failed: {resp.get('error') if resp else 'no reply'}")
        return resp.get("value")

    def _wait_idle(self, timeout) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            resp = self._send("get_state")
            if not resp or not resp.get("success"):
                return {"success": False, "error": resp.get("error") if resp else "no reply"}
            if (resp.get("state") or {}).get("worker_status") == "idle":
                result = resp.get("result")
                return result if result is not None else {"success": True, "value": None}
            time.sleep(C.CONTROL_CLIENT_IDLE_POLL_S)
        return {"success": False, "error": f"timeout after {timeout:.0f}s"}

    def _blocking(self, method, args, kwargs, timeout):
        # A scripted move ENDS the servo stream: the server stops the servo before
        # issuing one (control_server._start_move / _spawn_move), so anything the
        # stream faulted with belongs to the stream that is now over. Async issue
        # included -- it stops the servo on exactly the same seam.
        self._servo_stream_ended(method)
        if kwargs.get("is_async"):              # non-blocking by contract -- just dispatch
            return self._rpc(method, *args, **kwargs)
        try:
            return self._blocking_once(method, args, kwargs, timeout)
        except RuntimeError as e:
            # The control-server lost its FCI session (e.g. it was restarted or
            # crashed mid-run): the command comes back "not connected; call
            # connect() first" (server's robot is None). Re-establish the session
            # ONCE and retry, so a control-server restart does not kill a live
            # collection -- the collector's generic motion-failure retry only
            # resamples + backs off, it never reconnects. Never recurse on connect
            # itself; only this specific session-lost signature triggers a retry.
            if method in ("connect", "disconnect") or "not connected" not in str(e).lower():
                raise
            logger.warning("control link lost during %s (%s) -> reconnect + retry once", method, e)
            self.connect()                       # re-send connect -> server re-opens the FCI
            return self._blocking_once(method, args, kwargs, timeout)

    def _blocking_once(self, method, args, kwargs, timeout):
        resp = self._send(method, {"args": _jsonable(list(args)), "kwargs": _jsonable(kwargs)})
        if not resp or not resp.get("success"):
            raise RuntimeError(f"{method} rejected: {resp.get('error') if resp else 'no reply'}")
        result = self._wait_idle(timeout)
        if not result.get("success"):
            _raise_failure(result, f"{method} failed")
        return result.get("value")

    @staticmethod
    def _safe(fn):
        try:
            fn()
        except Exception:
            pass

    # -- lifecycle -----------------------------------------------------------
    def connect(self, *args, **kwargs):
        resp = self._send("connect", {"args": _jsonable(list(args)), "kwargs": _jsonable(kwargs)})
        if not resp or not resp.get("success"):
            raise RuntimeError(f"connect rejected: {resp.get('error') if resp else 'no reply'}")
        result = self._wait_idle(C.CONTROL_CLIENT_CONNECT_TIMEOUT_S)
        if not result.get("success"):
            raise RuntimeError(f"connect failed: {result.get('error')}")
        self.robot = True
        return None

    def disconnect(self, *args, **kwargs):
        resp = self._send("disconnect", {"args": [], "kwargs": {}})
        if resp and resp.get("success"):
            self._wait_idle(C.CONTROL_CLIENT_CONNECT_TIMEOUT_S)
        self.robot = None
        return None

    def ping(self) -> bool:
        """Liveness probe over the existing IPC protocol: True iff the server answers.
        Cheap + side-effect-free (no FCI touch) -- the panel's supervised-server health
        check (tools/arm/control_server_run.sh ping) calls this to decide healthy/stale."""
        resp = self._send("ping")
        return bool(resp and resp.get("success"))

    def reset_session(self):
        self._servo_stream_ended("reset_session")   # an interrupt: ends the stream
        return self._rpc("reset_session")

    def recover(self):
        self._servo_stream_ended("recover")         # an interrupt: ends the stream
        return self._rpc("recover")

    def wait_until_ready(self, *a, **k):
        resp = self._send("wait_until_ready", {"args": _jsonable(list(a)), "kwargs": _jsonable(k)})
        if not resp or not resp.get("success"):
            raise RuntimeError(f"wait_until_ready rejected: {resp.get('error') if resp else 'no reply'}")
        result = self._wait_idle(C.CONTROL_CLIENT_CONNECT_TIMEOUT_S)
        if not result.get("success"):
            raise RuntimeError(result.get("error", "wait_until_ready failed"))
        return result.get("value")

    # -- state reads (last-value cache; the hot 30 Hz path) ------------------
    def get_joint_angles(self):
        return list(self._snap().get("joints") or [0.0] * 7)

    def get_joint_speeds(self):
        return list(self._snap().get("joint_speeds") or [0.0] * 7)

    def get_tool_pose(self):
        return list(self._snap().get("pose") or [0.0] * 6)

    def get_tool_force_raw(self):
        return list(self._snap().get("force_raw") or [0.0] * 6)

    def is_running(self):
        return bool(self._snap().get("is_running", False))

    def robot_mode(self):
        return str(self._snap().get("robot_mode", ""))

    def is_user_stopped(self):
        return bool(self._snap().get("is_user_stopped", False))

    def robot_mode_nowait(self) -> str:
        """The arm's mode from the PUSH cache, or "" before the stream is up.

        `robot_mode()` goes through `_snap()`, whose cold path is a BLOCKING
        `get_state` round trip on `self._lock` -- the same lock every streamed
        servo send takes -- capped at CONTROL_CLIENT_RCV_TIMEOUT_MS (10 s). That
        is correct for preflight, which must have a true answer before the arm
        moves and can afford to wait for it. It is wrong inside the 30 Hz servo
        loop, where the adapter's lag guard reads the mode purely to LABEL a
        warning: a diagnostic that can stall the action path for ten seconds
        stops the servo stream it was meant to explain, and franky then replans
        the resumed JointMotion from a target the arm is no longer near.

        So the servo path takes this reader instead, and its contract is the
        honest one for a hot path: never block, and say "" rather than wait when
        the stream has not delivered a snapshot yet."""
        return str(self._cached("robot_mode", "") or "")

    @property
    def state_fallbacks_total(self):
        return int(self._cached("state_fallbacks_total", 0))

    def reset_fallback_counters(self):
        return self._rpc("reset_fallback_counters")

    # -- state reads ---------------------------------------------------------
    def get_tool_force(self):
        return self._rpc("get_tool_force")

    def get_gripper_width(self):
        state = self.get_gripper_motion_state()
        width = state.get("width")
        if width is None:
            # Compatibility with an older server that has no gripper snapshot.
            return self._rpc("get_gripper_width")
        return float(width)

    def is_grasped(self, expected_width_m=None):
        # The control-server's Hand has no measured part width; the argument is
        # accepted (contract held convention) and not sent over the wire.
        state = self.get_gripper_motion_state()
        grasped = state.get("grasped")
        if grasped is None:
            return bool(self._rpc("is_grasped"))
        return bool(grasped)

    def get_gripper_motion_state(self):
        state = self._snap().get("gripper_motion")
        if isinstance(state, dict):
            return dict(state)
        return dict(self._rpc("get_gripper_motion_state"))

    def has_gripper(self):
        # the control-server owns the Hand; ask it (the client is a remote proxy with
        # no local .gripper attribute, so the adapter must query has_gripper via RPC).
        return bool(self._rpc("has_gripper"))

    # -- kinematics / config (RPC) -------------------------------------------
    def fk(self, *a, **k):
        return self._rpc("fk", *a, **k)

    def ik(self, *a, **k):
        return self._rpc("ik", *a, **k)

    def manipulability(self, *a, **k):
        return self._rpc("manipulability", *a, **k)

    def set_ft_zero(self, *a, **k):
        return self._rpc("set_ft_zero", *a, **k)

    def set_collision_behavior(self, *a, **k):
        return self._rpc("set_collision_behavior", *a, **k)

    def set_speed(self, fraction):
        """Push the motion-speed fraction to the SERVER's controller (a fast RPC).

        Without this the operator's collection speed died here: `self._rdf` was
        stored for API parity and never transmitted, and the server -- launched
        with `--ip` only -- ran every remote collection at
        DEFAULT_DYNAMICS_FACTOR. So 0.1 and 0.2 produced identical arm behavior,
        which is exactly why the number was unreadable."""
        self._rdf = self._rpc("set_speed", fraction)
        return self._rdf

    # -- blocking moves (accepted + poll-until-idle) -------------------------
    def move_joint(self, *a, **k):
        return self._blocking("move_joint", a, k, C.CONTROL_CLIENT_MOVE_TIMEOUT_S)

    def move_tool(self, *a, **k):
        return self._blocking("move_tool", a, k, C.CONTROL_CLIENT_MOVE_TIMEOUT_S)

    def move_tool_traj(self, *a, **k):
        return self._blocking("move_tool_traj", a, k, C.CONTROL_CLIENT_MOVE_TIMEOUT_S)

    def move_tool_impedance(self, *a, **k):
        return self._blocking("move_tool_impedance", a, k, C.CONTROL_CLIENT_MOVE_TIMEOUT_S)

    def move_until_force(self, *a, **k):
        v = self._blocking("move_until_force", a, k, C.CONTROL_CLIENT_MOVE_TIMEOUT_S)
        return tuple(v) if isinstance(v, list) else v   # (pose, triggered)

    # -- servo / speed (fire-and-forget; the 30 Hz action path) --------------
    def _raise_pending_servo_fault(self) -> None:
        """Raise the last streamed-servo failure the SERVER reported, once.

        A streamed servo is fire-and-forget -- it must be, because a 30 Hz action
        path cannot pay a round trip per step -- so its exception has no reply to
        travel on. It used to travel nowhere at all: the server logged a warning
        into its own file on the cell and the client returned success, forever.
        The cost was a silent deadlock. One `cartesian_reflex` latches the arm,
        libfranka then rejects EVERY subsequent servo ('command not possible in
        the current mode ("Reflex")'), the measured pose freezes, and because the
        policy conditions on that frozen pose it re-emits the same action forever.
        Observed on franka-right 2026-08-25: ~18 s of rejected servos while the
        adapter's lag guard reported an unchanging 11 mm lead and nothing stopped.

        The in-process FrankaArmController raises that failure straight out of
        send_action. The two controllers are documented drop-in equivalents, so
        this restores the parity: the fault is published on the state snapshot the
        client already receives every tick (no hot-path cost) and is raised from
        the NEXT servo call -- ~33 ms later, which is as synchronous as an
        asynchronous stream can honestly be.
        """
        seen = self._servo_fault_seq
        # _cached, never _snap(): this runs inside the 30 Hz action loop, and
        # _snap()'s cold path is a blocking round trip. No snapshot yet => no
        # fault reported yet either, so a missing count is simply zero.
        seq = int(self._cached("servo_fault_seq") or 0)
        if seen is None:                        # first servo of a stream: adopt, never raise
            self._servo_fault_seq = seq
            return
        if seq <= seen:
            return
        self._servo_fault_seq = seq
        _raise_failure(self._cached("servo_fault") or {},
                       "streamed servo failed on the control server")

    def _servo_stream_ended(self, why: str) -> None:
        """A servo stream is over -- forget its fault tally so the NEXT stream
        cannot inherit it.

        A streamed-servo fault is delivered late by construction: it rides the
        state snapshot and is raised from the caller's NEXT servo call. That is
        honest while the same stream keeps streaming, and a lie the moment the
        stream ends first -- because the "next servo call" is then the first tick
        of an unrelated later stream, which gets an exception describing an arm
        state that is seconds old and no longer true.

        Observed on franka-right, 2026-08-26, twice in one evening. A servo tick
        near the END of the `pick` episode failed on the server; pick's guard had
        already fired, so pick never called servo again and finished SUCCESSFULLY
        (grasp held, 18.2 mm). `lift` ran a scripted move, clearing the latched
        error it left behind. Then `insert` opened, and its very first servo tick
        raised pick's fault verbatim -- `robot_mode=RobotMode.Reflex`,
        `control_command_success_rate: 1` -- before sending anything at all. The
        step that aborted had commanded no motion; the step blamed by the message
        had already been declared a success. The same counter mis-attribution
        aborted `pick` attempt 2 in an earlier run with attempt 1's fault, and
        attempt 3 then ran clean -- the one-round-delayed, one-shot abort that is
        the signature of a monotonic counter consumed one raise at a time.

        The client already refuses to inherit a PREVIOUS RUN's faults (a
        long-lived server outlives its clients). This is the same rule at the
        right granularity: the baseline belongs to a stream, not to a client.
        Resetting to None reuses the existing adopt-on-first-servo path rather
        than adding a second way to set the baseline.

        A fault the server reported but no servo call ever collected is DROPPED,
        so it is said here instead -- the arm's latched error is still surfaced
        (loudly) by the next motion's _warn_and_recover, and the deadlock case
        this channel exists for is untouched: while a stream keeps streaming into
        a latched arm, every tick faults and the very next tick still raises."""
        if self._servo_fault_seq is None:
            return
        undelivered = int(self._cached("servo_fault_seq") or 0) - self._servo_fault_seq
        self._servo_fault_seq = None
        if undelivered > 0:
            logger.warning(
                "servo stream ended (%s) with %d streamed-servo fault(s) never "
                "collected by a servo call -- reporting here rather than raising them "
                "into the next stream: %s", why, undelivered,
                (self._cached("servo_fault") or {}).get("error", "(no detail)"))

    def servo_tool(self, *a, **k):
        self._raise_pending_servo_fault()
        self._send("servo_tool", {"args": _jsonable(list(a)), "kwargs": _jsonable(k)}, expect_reply=False)

    def servo_joint(self, *a, **k):
        self._raise_pending_servo_fault()
        self._send("servo_joint", {"args": _jsonable(list(a)), "kwargs": _jsonable(k)}, expect_reply=False)

    # -- stops / interrupts (best-effort; never raise on a timeout) ----------
    # Every one of these is the END of a servo stream on the server
    # (control_server._handle_interrupt clears _servo_active), so each drops the
    # fault baseline -- see _servo_stream_ended.
    def stop_move(self):
        self._servo_stream_ended("stop_move")
        resp = self._send("stop_move", {"args": [], "kwargs": {}})
        return bool(resp and resp.get("success"))

    def stop_servo(self, *a, **k):
        self._servo_stream_ended("stop_servo")
        resp = self._send("stop_servo", {"args": _jsonable(list(a)), "kwargs": _jsonable(k)})
        return bool(resp and resp.get("success"))

    def stop_joints(self, *a, **k):
        self._servo_stream_ended("stop_joints")
        resp = self._send("stop_joints", {"args": _jsonable(list(a)), "kwargs": _jsonable(k)})
        return bool(resp and resp.get("success"))

    def stop_tool(self, *a, **k):
        self._servo_stream_ended("stop_tool")
        resp = self._send("stop_tool", {"args": _jsonable(list(a)), "kwargs": _jsonable(k)})
        return bool(resp and resp.get("success"))

    # -- gripper (RPC) -------------------------------------------------------
    def open_gripper(self, *a, **k):
        return self._rpc("open_gripper", *a, **k)

    def close_gripper(self, *a, **k):
        return bool(self._rpc("close_gripper", *a, **k))

    def start_open_gripper(self, *a, **k):
        return dict(self._rpc("start_open_gripper", *a, **k))

    def start_close_gripper(self, *a, **k):
        return dict(self._rpc("start_close_gripper", *a, **k))

    def stop_gripper(self):
        resp = self._send("stop_gripper", {"args": [], "kwargs": {}})
        if not resp or not resp.get("success"):
            raise RuntimeError(
                f"stop_gripper failed: {resp.get('error') if resp else 'no reply'}")
        return bool(resp.get("value"))

    def move_gripper(self, *a, **k):
        return self._rpc("move_gripper", *a, **k)

    def home_gripper(self, *a, **k):
        return self._rpc("home_gripper", *a, **k)

    # -- cleanup -------------------------------------------------------------
    def close(self):
        self._stream_on = False
        if self._rx.is_alive():
            self._rx.join(timeout=2.0)
        self._safe(self._pull.close)
        self._safe(self._sock.close)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _main() -> int:
    """CLI health probe: ``python -m evo_franka.control_client --ping`` -> exit 0 iff
    the control server answers. The panel's supervised-server health check runs this on
    the cell (via tools/arm/control_server_run.sh ping) -- it keeps the zmq/msgpack imports
    and the module name on the CELL, so the panel host (which has neither) never imports
    this and its command line carries no backend reference."""
    import argparse

    ap = argparse.ArgumentParser(description="Franka control-server client (health ping).")
    ap.add_argument("--ping", action="store_true", help="ping the server; exit 0 iff it answers")
    ap.add_argument("--host", default=None)
    ap.add_argument("--cmd-port", type=int, default=None)
    ap.add_argument("--state-port", type=int, default=None)
    args = ap.parse_args()
    if not args.ping:
        ap.error("nothing to do (pass --ping)")
    cli = FrankaArmControllerClient(host=args.host, cmd_port=args.cmd_port,
                                    state_port=args.state_port)
    try:
        ok = cli.ping()
    finally:
        cli.close()
    print("PONG" if ok else "DEAD")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_main())
