#!/usr/bin/env python3
"""ZMQ control server -- hosts the franky FrankaArmController in its own process.

Why: the Franka FCI 1 kHz loop must not share a GIL/CPU with the collector's
recording + JPEG encoding + dataset writers; in-process, an encode burst starves
the control management and a move appears to "hang" (the episode-N freeze). This
server runs the SAME robots/franka/driver.py (unchanged) alone in a dedicated
process, so franky's C++ control thread (Ruckig OTG at 1 kHz) keeps its own GIL
and can be pinned to its own cores. The collector drives it as a thin client
(robots/franka/control_client.py).

Architecture (mirrors ArrebolBlack/franka-control):
  * main thread       -- ZMQ ROUTER recv/send only; never touches the controller.
  * controller thread -- owns FrankaArmController; drains the command queue,
                         applies the latest streamed servo target, runs the
                         servo-gap watchdog, and polls + PUSHes a state snapshot
                         every cycle (even during a blocking move).
  * move helper       -- a blocking move runs on a spawned thread so the
                         controller thread stays free to process stop/recover
                         (this is how a watchdog stop_move actually preempts).

Command classes (msgpack {command, params:{args,kwargs}}):
  streaming (servo_*/speed_*)  fire-and-forget, coalesced to the latest -- no reply.
  blocking  (connect/move_*/wait_until_ready)  reply {accepted}; the client polls
            get_state until worker_status=="idle" and reads the result.
  interrupt (stop_*/recover/reset_session)  processed even while busy.
  fast      (everything else: async gripper start/set_*/fk/ik/...) enqueue + reply.

Run (control machine, FCI env):  python -m evo_franka.control_server --ip 172.16.0.2
then run the collector with FRANKA_CONTROL_IPC=1.
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import msgpack
import zmq

from evo_franka.driver import FrankaArmController
from evo_franka.ipc_codec import jsonable as _jsonable
from evo_franka import constants as C

logger = logging.getLogger("control_server")

_STREAMING = {"servo_tool", "servo_joint"}
_MOVES = {"move_joint", "move_tool", "move_tool_traj",
          "move_tool_impedance", "move_until_force"}
_INLINE = {"connect", "disconnect", "wait_until_ready"}
_INTERRUPT = {"stop_move", "stop_servo", "stop_joints", "stop_tool",
              "stop_gripper", "recover", "reset_session"}
_BLOCKING = _MOVES | _INLINE          # set busy + reply {accepted}; client polls
# The TRAJECTORY move runs DURING data recording (collection forward/backward), so
# its state-staleness is what corrupts the dataset -- route it SINGLE-THREADED
# on the controller thread (issue async, poll is_running() inline) so the
# interleaved state reads stay fresh. It is clean (branch-continuous IK
# + _warn_and_recover run BEFORE the motion; no reflex-robust/limit-restore to
# lose). Point-to-point move_joint/move_tool (homing/lift BETWEEN episodes, not
# during recording) and move_until_force/impedance keep the legacy helper path,
# preserving their reflex-robust retry + limit-restore -- their staleness is
# harmless because nothing is recording then.
_ASYNC_MOVES = {"move_tool_traj"}
_MOVE_ENGAGE_GRACE_S = 0.20           # after issuing async, allow is_running() to come True
_HEALTH_LOG_EVERY = 500               # state-loop cycles between health samples (~5 s @ 100 Hz)
_HEALTH_CCSR_OK = 0.99                # actively commanding + ccsr below this => health WARNING
                                      # (a control_command_success_rate drop precedes a comms fault)


@dataclass
class _Cmd:
    name: str
    params: dict
    event: threading.Event = field(default_factory=threading.Event)
    result: Optional[dict] = None


class ControlServer:
    def __init__(self, robot_ip, relative_dynamics_factor,
                 cmd_port=C.CONTROL_SERVER_CMD_PORT,
                 state_port=C.CONTROL_SERVER_STATE_PORT,
                 state_poll_hz=C.CONTROL_SERVER_STATE_POLL_HZ):
        self._ip = robot_ip
        self._rdf = relative_dynamics_factor
        self._cmd_port = cmd_port
        self._state_port = state_port
        self._interval = 1.0 / max(1.0, float(state_poll_hz))

        self._ctrl: Optional[FrankaArmController] = None
        self._connected = False

        self._q: "queue.Queue[_Cmd]" = queue.Queue()
        self._event = threading.Event()

        self._busy = False
        self._worker_result: Optional[dict] = None
        self._helper: Optional[threading.Thread] = None
        # single-threaded async move: the controller thread issues the motion and
        # polls is_running() inline (so the interleaved state reads stay fresh).
        self._move_active: Optional[_Cmd] = None
        self._move_t0 = 0.0

        self._streaming = None
        self._streaming_lock = threading.Lock()
        self._last_servo_t = 0.0
        self._servo_active = False
        # The LAST streamed-servo failure + how many have happened. A streamed
        # servo is fire-and-forget by design (a 30 Hz action path must not pay a
        # round trip), so its exception cannot be returned -- but it MUST still
        # reach the caller, or a latched reflex silently rejects every command
        # while the policy keeps streaming into a dead arm. These ride the state
        # snapshot the client already consumes every tick, at no hot-path cost;
        # the client raises when the count advances (see control_client.servo_*).
        self._servo_fault: Optional[dict] = None
        self._servo_fault_seq = 0

        self._snap: dict = {}
        self._snap_lock = threading.Lock()
        self._seq = 0
        self._sched_logged = False
        # health watchdog: silent while healthy, WARNING each ~5 s while bad,
        # one INFO on recovery (see _log_health)
        self._health_bad = False
        self._health_bad_since = 0.0

        self._running = False
        self._ctx: Optional[zmq.Context] = None
        self._cmd_sock = None

    # -- lifecycle -----------------------------------------------------------
    def _build_controller(self) -> FrankaArmController:
        """Construct the controller and STAMP it as living inside the dedicated
        control server. The driver's fault classifier tailors its starved-
        control-path advice on this: the un-hosted advice is 'isolate the 1 kHz
        loop in the control server (FRANKA_CONTROL_IPC)', and a controller that
        IS the control server must not be told to go run itself -- that stale
        advice sent an operator to re-apply a fix already in place (rate-0.97
        abort under 'cores=20-23 rt=yes', 2026-08-26). The stamp is a fact about
        WHERE the controller runs, which only this constructor site knows."""
        ctrl = FrankaArmController(self._ip, self._rdf)
        ctrl.hosted_by_control_server = True
        return ctrl

    def run(self):
        self._ctx = zmq.Context()
        self._cmd_sock = self._ctx.socket(zmq.ROUTER)
        self._cmd_sock.setsockopt(zmq.RCVTIMEO, 200)
        self._cmd_sock.setsockopt(zmq.LINGER, 0)
        self._cmd_sock.bind(f"tcp://*:{self._cmd_port}")
        self._ctrl = self._build_controller()
        self._running = True
        logger.info("control server up: cmd tcp://*:%d  state tcp://*:%d  (FCI %s)",
                    self._cmd_port, self._state_port, self._ip)

        loop = threading.Thread(target=self._controller_loop, daemon=True, name="controller")
        loop.start()
        try:
            while self._running:
                try:
                    parts = self._cmd_sock.recv_multipart()
                except zmq.Again:
                    continue
                if len(parts) < 3:
                    continue
                identity = parts[0]
                reply = self._dispatch(parts[2])
                if reply is not None:
                    self._cmd_sock.send_multipart(
                        [identity, b"", msgpack.packb(reply, use_bin_type=True)])
        finally:
            self._running = False
            self._event.set()
            loop.join(timeout=5.0)
            self._safe(self._cmd_sock.close)
            self._safe(self._ctx.term)
            logger.info("control server stopped")

    # -- main thread: dispatch (never touches the controller) ----------------
    def _dispatch(self, raw) -> Optional[dict]:
        try:
            msg = msgpack.unpackb(raw, raw=False)
        except Exception:
            return {"success": False, "error": "invalid msgpack"}
        name = msg.get("command", "")
        params = msg.get("params", {}) or {}

        if name == "get_state":
            return self._state_reply()
        if name == "ping":
            return {"success": True, "connected": self._connected, "busy": self._busy}
        if name == "shutdown":
            self._running = False
            return {"success": True}

        if name in _STREAMING:
            with self._streaming_lock:
                self._streaming = (name, params.get("args", []), params.get("kwargs", {}))
            self._event.set()
            return None  # fire-and-forget

        if name in _INTERRUPT:
            return self._enqueue(name, params)

        if name in _BLOCKING:
            if self._busy:
                return {"success": False, "error": "worker busy"}
            self._busy = True
            self._worker_result = None
            return self._enqueue(name, params, blocking=True)

        # fast rpc
        if self._busy:
            return {"success": False, "error": "worker busy"}
        return self._enqueue(name, params)

    def _state_reply(self) -> dict:
        with self._snap_lock:
            snap = dict(self._snap)
        snap["worker_status"] = "busy" if self._busy else "idle"
        return {"success": True, "state": snap, "result": self._worker_result}

    def _enqueue(self, name, params, blocking=False) -> dict:
        cmd = _Cmd(name, params or {})
        self._q.put(cmd)
        self._event.set()
        if blocking:
            return {"success": True, "accepted": True}
        if not cmd.event.wait(timeout=C.CONTROL_SERVER_CMD_WAIT_S):
            return {"success": False, "error": "controller-thread timeout"}
        return cmd.result if cmd.result is not None else {"success": False, "error": "no result"}

    # -- controller thread ---------------------------------------------------
    def _controller_loop(self):
        push = self._ctx.socket(zmq.PUSH)
        push.setsockopt(zmq.SNDHWM, 2)
        push.setsockopt(zmq.LINGER, 0)
        push.bind(f"tcp://*:{self._state_port}")
        try:
            while self._running:
                self._event.wait(timeout=self._interval)
                self._event.clear()

                # 1. latest streamed servo/speed target (coalesced)
                stream = None
                with self._streaming_lock:
                    if self._streaming is not None:
                        stream, self._streaming = self._streaming, None
                if stream is not None and self._connected and not self._busy:
                    method, args, kwargs = stream
                    try:
                        getattr(self._ctrl, method)(*args, **kwargs)
                        self._last_servo_t = time.monotonic()
                        self._servo_active = True
                    except Exception as e:
                        logger.warning("stream %s failed: %s", method, e)
                        # Publish it so the CALLER learns. Without this the fault
                        # died in this log file: the in-process driver raises a
                        # servo failure straight out of send_action, while the IPC
                        # path returned success forever -- the two control paths
                        # are meant to be drop-in equivalents, and this is where
                        # they diverged.
                        self._servo_fault_seq += 1
                        self._servo_fault = {"method": method, "error": str(e),
                                             "error_type": type(e).__name__}

                # 2. drain command queue
                while True:
                    try:
                        cmd = self._q.get_nowait()
                    except queue.Empty:
                        break
                    self._process(cmd)

                # 3. reap a finished move: async single-threaded path first, then
                #    the legacy helper path (move_until_force/impedance)
                self._poll_active_move()
                if self._helper is not None and not self._helper.is_alive():
                    self._helper.join()
                    self._helper = None
                    self._busy = False

                # 4. servo-gap watchdog -> graceful stop (handoff ramp on resume)
                if (self._servo_active and self._connected and not self._busy
                        and time.monotonic() - self._last_servo_t > C.CONTROL_SERVER_SERVO_GAP_S):
                    self._safe(self._ctrl.stop_servo)
                    self._servo_active = False

                # 5. poll + push the latest state snapshot
                if self._connected:
                    snap = self._read_snapshot()
                    with self._snap_lock:
                        self._snap = snap
                    try:
                        push.send(msgpack.packb(snap, use_bin_type=True), zmq.NOBLOCK)
                    except (zmq.Again, zmq.ZMQError):
                        pass
                    if self._seq % _HEALTH_LOG_EVERY == 0:
                        self._log_health()
        finally:
            self._safe(push.close)
            self._teardown()

    def _process(self, cmd: _Cmd):
        name = cmd.name
        if name in _INLINE:
            self._run_inline(cmd)
        elif name in _ASYNC_MOVES:
            self._start_move(cmd)              # single-threaded: issue async, poll inline
        elif name in _MOVES:
            self._spawn_move(cmd)              # legacy helper path: move_until_force/impedance
        elif name in _INTERRUPT:
            self._handle_interrupt(cmd)
        else:
            self._run_fast(cmd)

    def _run_inline(self, cmd: _Cmd):
        """connect / disconnect / wait_until_ready -- on the controller thread."""
        name = cmd.name
        try:
            if name == "connect":
                if not self._connected:
                    self._ctrl.connect(*cmd.params.get("args", []),
                                       **cmd.params.get("kwargs", {}))
                    self._connected = True
                else:
                    # already-open session: a prior run may have crashed and left
                    # the robot in Reflex/error -- heal it so every new client
                    # starts from a READY robot (a fresh connect does this; an
                    # already-open one skipped it -> the next move hit Reflex).
                    self._heal_session(cmd.params)
                res = {"success": True, "value": None}
            elif name == "disconnect":
                if self._connected:
                    self._ctrl.disconnect()
                    self._connected = False
                res = {"success": True, "value": None}
            else:
                res = self._call(name, cmd.params)
        except Exception as e:
            res = self._err(e)
        self._worker_result = res
        self._busy = False
        cmd.result = res
        cmd.event.set()

    def _heal_session(self, params):
        """A new client connected to an already-open session. A prior run may
        have crashed and left the robot in Reflex/error (a fresh connect() clears
        that via recover_from_errors; an already-open session skipped it, so the
        next move hit a Reflex robot -> Net Exception + an 11s GIL-held freeze).
        Heal it through the driver's SINGLE recovery routine (heal_session:
        recover in place, else full reconnect) -- the server no longer
        re-implements 'recover-or-reconnect' (item 3)."""
        self._ctrl.heal_session(*params.get("args", []), **params.get("kwargs", {}))
        self._connected = True

    def _start_move(self, cmd: _Cmd):
        """Issue a single-motion move ASYNC on the controller thread and track it
        (no helper thread). The controller loop then polls _poll_active_move()
        each cycle, so the interleaved _read_snapshot() reads stay on THIS thread
        and remain fresh throughout the move -- the fix for the cross-thread
        state-starvation that froze recorded state during moves. A synchronous
        issue-time failure (IK / JointSwingExceeded raised before any motion) is
        reported immediately."""
        if self._servo_active:
            self._safe(self._ctrl.stop_servo)
            self._servo_active = False
        params = dict(cmd.params or {})
        kwargs = dict(params.get("kwargs", {}))
        kwargs["is_async"] = True                 # force async issue -> returns at once
        params["kwargs"] = kwargs
        try:
            self._call(cmd.name, params)          # issue the franky motion (async)
        except Exception as e:                    # planning/IK error BEFORE any motion
            logger.error("move issue FAILED: %s (%s)", cmd.name, e)
            self._worker_result = self._err(e)
            self._busy = False
            return
        logger.info("move start: %s (async, single-threaded)", cmd.name)
        self._move_active = cmd
        self._move_t0 = time.monotonic()

    def _poll_active_move(self):
        """Controller-thread completion poll for an async move. Same thread as the
        state read, so neither starves the other. Declares the move done ONLY once
        finish_move() confirms join_motion(0)==True -- is_running()==False alone is
        just a hint, so a transient is_in_control glitch can't mis-sequence the
        move. A move exceeding the hang timeout is stopped + surfaced, so a wedged
        motion can never pin _busy forever."""
        cmd = self._move_active
        if cmd is None:
            return
        # Hang cap: the DRIVER owns it (check_move_hang == the async twin of the
        # blocking _safe_move deadline). F4 deleted the server's re-implementation
        # of the MOTION_JOIN_TIMEOUT_S cap so it lives in exactly one place. Checked
        # every cycle regardless of is_running(), so a move wedged while still 'in
        # control' is caught too; it raises FrankaMotionRefused which we surface and
        # clear busy on, so a wedged motion can never pin the worker.
        try:
            self._ctrl.check_move_hang()
        except Exception as e:
            self._safe(self._ctrl.finish_move)    # reap the intentional abort's deferred fault
            logger.error("move HANG: %s -> stopped (%s)", cmd.name, e)
            self._worker_result = self._err(e)
            self._move_active = None
            self._busy = False
            return
        elapsed = time.monotonic() - self._move_t0
        if elapsed < _MOVE_ENGAGE_GRACE_S:
            return                                # let the motion engage
        try:
            running = self._ctrl.is_running()
        except Exception:
            running = True                        # read glitch -> assume running (timeout bounds it)
        if running:
            return
        try:
            done = self._ctrl.finish_move()       # True IFF joined; raises a deferred fault
        except Exception as e:
            logger.error("move FAILED: %s after %.2fs", cmd.name, elapsed)
            self._worker_result = self._err(e)
            self._move_active = None
            self._busy = False
            return
        if not done:
            return                                # is_running glitched False -> keep polling
        self._worker_result = {"success": True, "value": None}
        logger.info("move done:  %s in %.2fs | %s", cmd.name, elapsed, self._fault_detail())
        self._move_active = None
        self._busy = False

    def _spawn_move(self, cmd: _Cmd):
        # ensure a clean control-mode transition out of any active servo
        if self._servo_active:
            self._safe(self._ctrl.stop_servo)
            self._servo_active = False

        def _run():
            t0 = time.monotonic()
            logger.info("move start: %s", cmd.name)
            try:
                self._worker_result = self._call(cmd.name, cmd.params)
                logger.info("move done:  %s in %.2fs | %s",
                            cmd.name, time.monotonic() - t0, self._fault_detail())
            except Exception as e:
                logger.error("move FAILED: %s after %.2fs", cmd.name, time.monotonic() - t0)
                self._worker_result = self._err(e)

        self._helper = threading.Thread(target=_run, daemon=True, name=f"move-{cmd.name}")
        self._helper.start()

    def _handle_interrupt(self, cmd: _Cmd):
        # break a running async move first (watchdog/stop preempts it on this thread)
        if self._move_active is not None:
            self._safe(self._ctrl.stop_move)
            self._safe(self._ctrl.finish_move)   # reap+discard the intentional abort
            self._move_active = None
        # break a running legacy move helper (move_until_force/impedance)
        if self._helper is not None and self._helper.is_alive():
            self._safe(self._ctrl.stop_move)
            self._helper.join(timeout=C.STOP_JOIN_TIMEOUT_S + 5.0)
            if self._helper.is_alive():
                logger.error("move helper did not stop; shutting server down")
                self._running = False
                cmd.result = {"success": False, "error": "helper stuck; server shutting down"}
                cmd.event.set()
                return
            self._helper = None
        self._busy = False
        self._servo_active = False
        try:
            cmd.result = self._call(cmd.name, cmd.params)
        except Exception as e:
            cmd.result = self._err(e)
        cmd.event.set()

    def _run_fast(self, cmd: _Cmd):
        try:
            cmd.result = self._call(cmd.name, cmd.params)
        except Exception as e:
            cmd.result = self._err(e)
        cmd.event.set()

    def _call(self, name, params) -> dict:
        if self._ctrl is None:
            raise RuntimeError("controller not constructed")
        fn = getattr(self._ctrl, name, None)
        if not callable(fn):
            raise RuntimeError(f"unknown method: {name}")
        value = fn(*params.get("args", []), **params.get("kwargs", {}))
        return {"success": True, "value": _jsonable(value)}

    def _read_snapshot(self) -> dict:
        ctrl, prev, snap = self._ctrl, self._snap, {}

        def rd(key, fn, default):
            try:
                snap[key] = _jsonable(fn())
            except Exception:
                snap[key] = prev.get(key, default)

        rd("joints", ctrl.get_joint_angles, [0.0] * 7)
        rd("joint_speeds", ctrl.get_joint_speeds, [0.0] * 7)
        rd("pose", ctrl.get_tool_pose, [0.0] * 6)
        rd("force_raw", ctrl.get_tool_force_raw, [0.0] * 6)
        rd("is_running", ctrl.is_running, False)
        rd("robot_mode", ctrl.robot_mode, "")
        rd("is_user_stopped", ctrl.is_user_stopped, False)
        # Polling a Franky BoolFuture with wait(0) is non-blocking. Width and
        # grasped remain the last reliable values while it runs, explicitly
        # marked measurement_stale, then refresh once on completion.
        rd("gripper_motion", ctrl.get_gripper_motion_state, {
            "command_id": 0, "command": None, "status": "unavailable",
            "result": None, "error": None, "width": None, "grasped": None,
            "measurement_stale": True,
        })
        try:
            snap["state_fallbacks_total"] = int(ctrl.state_fallbacks_total)
        except Exception:
            snap["state_fallbacks_total"] = prev.get("state_fallbacks_total", 0)
        # Streamed-servo faults ride the snapshot: the fire-and-forget servo has no
        # reply channel, and this one already reaches the client every tick.
        snap["servo_fault"] = self._servo_fault
        snap["servo_fault_seq"] = self._servo_fault_seq
        self._seq += 1
        snap["seq"] = self._seq
        snap["ts"] = time.monotonic()
        return snap

    def _teardown(self):
        try:
            if self._ctrl is not None and self._connected:
                self._ctrl.disconnect()
        except Exception:
            pass
        self._connected = False

    def _err(self, e) -> dict:
        logger.error("command fault: %s: %s | %s", type(e).__name__, e, self._fault_detail())
        return {"success": False, "error": str(e), "error_type": type(e).__name__}

    def _fault_detail(self) -> str:
        """franky/libfranka context for a post-mortem: the actual reason a move
        aborted (current_errors / last_motion_errors) + the FCI success rate."""
        r = getattr(self._ctrl, "robot", None)
        if r is None:
            return "robot=None"
        bits = []
        for getter in (
            lambda: "mode=%s" % r.state.robot_mode,
            lambda: "has_errors=%s" % bool(r.has_errors),
            lambda: "ccsr=%.3f" % float(r.state.control_command_success_rate),
            lambda: "current_errors=%s" % r.state.current_errors,
            lambda: "last_motion_errors=%s" % r.state.last_motion_errors,
        ):
            try:
                bits.append(getter())
            except Exception:
                pass
        return " ".join(bits) or "no-detail"

    def _log_health(self) -> None:
        """Health watchdog on the ~5 s cadence of the caller. SILENT while the
        FCI link is healthy; a WARNING each cadence while it is not; a single
        INFO on recovery. "Bad" = a latched robot error, or the command
        success-rate dropping while we are actually commanding (idle sends no
        commands, so a ~0 ccsr there is normal, not a fault). Dumps thread
        scheduling once (the RT check)."""
        r = getattr(self._ctrl, "robot", None)
        if r is None:
            return
        try:
            ccsr = float(r.state.control_command_success_rate)
            mode = r.state.robot_mode
            errors = bool(r.has_errors)
        except Exception:
            return                       # state momentarily unreadable: not a verdict
        busy = bool(self._busy)
        payload = "ccsr=%.3f mode=%s errors=%s busy=%s" % (ccsr, mode, errors, busy)
        bad = errors or ((busy or self._servo_active) and ccsr < _HEALTH_CCSR_OK)
        if bad:
            if not self._health_bad:
                self._health_bad = True
                self._health_bad_since = time.monotonic()
            logger.warning("FCI health: %s", payload)      # repeats each ~5 s while bad
        elif self._health_bad:
            self._health_bad = False
            logger.info("FCI health: %s (recovered after %.1fs)",
                        payload, time.monotonic() - self._health_bad_since)
        # healthy and was healthy -> nothing

        if not self._sched_logged:
            self._sched_logged = True
            self._log_thread_sched("after first state")

    @staticmethod
    def _log_thread_sched(tag: str = "") -> None:
        """Log each thread's scheduling policy/priority -- confirms whether the
        FCI control loop is SCHED_FIFO (real-time)."""
        pol = {0: "OTHER", 1: "FIFO", 2: "RR", 3: "BATCH", 5: "IDLE", 6: "DEADLINE"}
        rows = []
        for d in glob.glob("/proc/self/task/*"):
            try:
                tid = int(os.path.basename(d))
                comm = open(d + "/comm").read().strip()
                rows.append("%s=%s/%d" % (comm, pol.get(os.sched_getscheduler(tid), "?"),
                                          os.sched_getparam(tid).sched_priority))
            except Exception:
                continue
        logger.info("thread sched (%s): %s", tag, "  ".join(rows))

    @staticmethod
    def _safe(fn, *a):
        try:
            fn(*a)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(
        description="Franka control server (ZMQ, hosts the franky FrankaArmController)")
    ap.add_argument("--ip", default=C.DEFAULT_ROBOT_IP, help="Franka FCI IP")
    ap.add_argument("--dyn", type=float, default=C.DEFAULT_DYNAMICS_FACTOR,
                    help="relative dynamics (velocity) factor")
    ap.add_argument("--cmd-port", type=int, default=C.CONTROL_SERVER_CMD_PORT)
    ap.add_argument("--state-port", type=int, default=C.CONTROL_SERVER_STATE_PORT)
    ap.add_argument("--poll-hz", type=float, default=C.CONTROL_SERVER_STATE_POLL_HZ)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ControlServer(args.ip, args.dyn, args.cmd_port, args.state_port, args.poll_hz).run()


if __name__ == "__main__":
    main()
