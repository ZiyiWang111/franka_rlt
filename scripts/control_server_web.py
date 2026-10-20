#!/usr/bin/env python3
"""Minimal teleop launcher with an automatically managed control server."""

import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "website/control_server/index.html"
ROBOT_IP = "172.16.0.2"
CONTROL_PYTHON = "/home/embint/robolab-venv/bin/python"
LOG = Path(os.environ.get("CTRL_LOG", "/tmp/evo-rlt-franka-control-server.log"))
PID_FILE = Path(os.environ.get("CTRL_PIDF", "/tmp/evo-rlt-franka-control-server.pid"))
TELEOP_LOG = Path("/tmp/evo-rlt-franka-teleop.log")
TELEOP = None
TELEOP_CONTROL_PID = None
MESSAGES = deque(maxlen=100)
ACTION_LOCK = threading.Lock()


def control_pid():
    """Only accept a live control server for this robot, never an unrelated PID."""
    try:
        pid = int(PID_FILE.read_text().strip())
        if pid <= 1:
            return None
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if [b"-m", b"evo_franka.control_server"] != args[1:3]:
            return None
        ip_index = args.index(b"--ip") + 1
        if args[ip_index] != ROBOT_IP.encode():
            return None
        return pid
    except (OSError, ValueError, IndexError):
        return None


def start_control():
    pid = control_pid()
    if pid:
        MESSAGES.append(f"control server 已运行，PID {pid}")
        return
    command = ["bash", str(ROOT / "scripts/start_evo_franka_control_server.sh"),
               ROBOT_IP, CONTROL_PYTHON]
    MESSAGES.append(f"$ bash scripts/start_evo_franka_control_server.sh {ROBOT_IP} {CONTROL_PYTHON}")
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=25)
    MESSAGES.append(result.stdout.rstrip() or f"启动脚本退出码：{result.returncode}")
    if result.returncode:
        raise RuntimeError(f"启动失败，退出码 {result.returncode}")


def kill_control(expected_pid=None):
    pid = control_pid()
    if expected_pid is not None and pid != expected_pid:
        return
    if pid is None:
        MESSAGES.append("没有找到运行中的 control server")
        return
    try:
        if control_pid() != pid:
            MESSAGES.append("control server 已退出")
            return
        MESSAGES.append(f"停止 control server，PID {pid}")
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):
            if control_pid() != pid:
                break
            time.sleep(0.1)
        else:
            if control_pid() == pid:
                os.kill(pid, signal.SIGKILL)
            MESSAGES.append("退出超时，已发送 SIGKILL")
        MESSAGES.append("control server 已停止")
    except ProcessLookupError:
        MESSAGES.append("control server 已退出")


def finish_teleop(process):
    global TELEOP, TELEOP_CONTROL_PID
    if TELEOP is not process:
        return
    try:
        kill_control(TELEOP_CONTROL_PID)
    finally:
        TELEOP = None
        TELEOP_CONTROL_PID = None


def watch_teleop(process):
    code = process.wait()
    with ACTION_LOCK:
        if TELEOP is process:
            MESSAGES.append(f"teleop 已退出，退出码 {code}")
            try:
                finish_teleop(process)
            except OSError as exc:
                MESSAGES.append(f"关闭 control server 失败：{exc}")


def start_teleop():
    global TELEOP, TELEOP_CONTROL_PID
    if TELEOP is not None:
        if TELEOP.poll() is None:
            MESSAGES.append("teleop 已运行")
            return
        finish_teleop(TELEOP)
    try:
        start_control()
        TELEOP_CONTROL_PID = control_pid()
        if TELEOP_CONTROL_PID is None:
            raise RuntimeError("control server 未运行，无法启动 teleop")
        command = [CONTROL_PYTHON, "-u", str(ROOT / "scripts/fr3_keyboard_teleop.py"),
                   "--allow-motion", "--input-backend", "pynput", "--fps", "15",
                   "--speed-m-s", "0.02", "--gripper-force-n", "30"]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        with TELEOP_LOG.open("wb") as log:
            TELEOP = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT)
        MESSAGES.append(f"启动 teleop，PID {TELEOP.pid}")
        threading.Thread(target=watch_teleop, args=(TELEOP,), daemon=True).start()
    except (OSError, RuntimeError, subprocess.SubprocessError):
        kill_control(TELEOP_CONTROL_PID)
        TELEOP_CONTROL_PID = None
        raise


def stop_teleop():
    process = TELEOP
    if process is None:
        MESSAGES.append("teleop 未运行")
        return
    if process.poll() is None:
        MESSAGES.append("退出 teleop")
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    finish_teleop(process)


def tail(path):
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 65536))
            return stream.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def output():
    return "\n".join(list(MESSAGES)) + "\n\n[teleop]\n" + tail(TELEOP_LOG) + "\n[control server]\n" + tail(LOG)


class Handler(BaseHTTPRequestHandler):
    def send(self, status, content, content_type):
        body = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send(200, PAGE.read_text(), "text/html; charset=utf-8")
        elif self.path == "/api/output":
            self.send(200, json.dumps({"output": output()}), "application/json")
        else:
            self.send(404, "Not found", "text/plain")

    def do_POST(self):
        action = {"/api/teleop/start": start_teleop,
                  "/api/teleop/stop": stop_teleop}.get(self.path)
        if action is None:
            self.send(404, "Not found", "text/plain")
            return
        if not ACTION_LOCK.acquire(blocking=False):
            self.send(409, json.dumps({"error": "正在执行，请稍候"}), "application/json")
            return
        try:
            action()
            self.send(200, "{}", "application/json")
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            MESSAGES.append(str(exc))
            self.send(500, json.dumps({"error": str(exc)}), "application/json")
        finally:
            ACTION_LOCK.release()

    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    def shutdown_signal(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, shutdown_signal)
    print(f"本机：http://127.0.0.1:{args.port}", flush=True)
    print(f"外部电脑：http://<本机IP>:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with ACTION_LOCK:
            stop_teleop()
        server.server_close()


if __name__ == "__main__":
    main()
