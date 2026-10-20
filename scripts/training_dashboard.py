#!/usr/bin/env python3
"""Read-only local ACT training dashboard. Python standard library only."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import io
import json
import math
from pathlib import Path
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'website/training_monitor'
DEFAULT_RUNS = [
    ('Camera only', '#3975f6', 'act_dataset_928_addon_camera_only_action6_act'),
    ('Camera + TCP Z', '#12a68b', 'act_dataset_928_addon_rgb_z_action6_act'),
]
PROGRESS = re.compile(r'(\d+)/(\d+)\s+\[([^<\]]+)(?:<[^\]]*)?\]')
METRIC = re.compile(r'\b(loss|grdn|lr|updt_s|data_s):([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)', re.I)
TIMESTAMP = re.compile(r'\b\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\b')


def duration_seconds(value):
    parts = value.strip().split(':')
    try:
        return sum(float(part) * 60 ** i for i, part in enumerate(reversed(parts)))
    except ValueError:
        return 0


class LogParser:
    def __init__(self, state=None):
        self.state = state or dict(step=0, total=30000, elapsed=0, points=[], carry='', complete=False, failed=False)

    def feed(self, text):
        lines = re.split(r'[\r\n]', self.state['carry'] + text)
        self.state['carry'] = lines.pop()
        by_step = {point['step']: point for point in self.state['points']}
        for line in lines:
            resume = re.search(r'USER_RESUME_CHECKPOINT step=(\d+) target=(\d+)', line)
            if resume:
                step, total = map(int, resume.groups())
                if self.state.get('resume_start_step') != step:
                    self.state['resume_elapsed_base'] = self.state['elapsed']
                    self.state['resume_start_step'] = step
                self.state.update(failed=False, complete=False, total=total)
            progress = PROGRESS.search(line) if 'Training:' in line else None
            if progress:
                step, total, elapsed = progress.groups()
                self.state.update(step=int(step), total=int(total), elapsed=duration_seconds(elapsed) + self.state.get('resume_elapsed_base', 0))
            values = {key: float(value) for key, value in METRIC.findall(line)}
            if 'loss' in values and all(math.isfinite(value) for value in values.values()):
                # LeRobot rounds its printed `step:2K`. tqdm retains the exact
                # iteration (e.g. 1640/30000), which is the curve's authority.
                exact = re.search(r'\bstep:(\d+)\b(?![.KMBkmb])', line)
                step = int(exact[1]) if exact else self.state['step']
                stamp = TIMESTAMP.search(line)
                by_step[step] = dict(step=step, elapsed=self.state['elapsed'],
                                     timestamp=stamp[0] if stamp else None, **values)
                self.state['step'] = max(step, self.state['step'])
            if 'End of training' in line:
                self.state['complete'] = True
            if 'Traceback (most recent call last)' in line or 'ChildFailedError' in line:
                self.state['failed'] = True
        self.state['points'] = [by_step[step] for step in sorted(by_step)]


REMOTE_PROBE = r'''
import json, os, subprocess
from pathlib import Path
result = {'runs': [], 'gpus': []}
for run in REQUEST:
    path = Path(run['log'])
    row = {'id': run['id'], 'alive': Path('/proc/%s' % run['pid']).exists()}
    control = Path(run['output']) / 'stop_at_checkpoint.json'
    if control.is_file():
        try:
            row['stop_control'] = json.loads(control.read_text())
        except (OSError, ValueError):
            pass
    control_data = row.get('stop_control', {})
    if control_data.get('active_pid'):
        row['alive'] = Path('/proc/%s' % control_data['active_pid']).exists()
    elif control_data.get('status') == 'queued_for_resume':
        row['alive'] = False
    if path.is_file():
        stat = path.stat()
        identity = '%s:%s' % (stat.st_dev, stat.st_ino)
        reset = identity != run.get('identity') or stat.st_size < run.get('offset', 0)
        offset = 0 if reset else run.get('offset', 0)
        with path.open('rb') as stream:
            stream.seek(offset)
            data = stream.read(8 * 1024 * 1024)
            row.update(offset=stream.tell(), identity=identity, reset=reset,
                       text=data.decode('utf-8', errors='replace'), modified=stat.st_mtime)
        checkpoints = Path(run['output']) / 'checkpoints'
        row['checkpoints'] = sorted(p.name for p in checkpoints.iterdir()
                                   if p.name.isdigit() and p.is_dir()) if checkpoints.is_dir() else []
    else:
        row['error'] = '训练日志不存在'
    result['runs'].append(row)
try:
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
    for line in gpu.stdout.splitlines():
        index, used, total, utilization = [int(value.strip()) for value in line.split(',')]
        result['gpus'].append(dict(index=index, used_mb=used, total_mb=total, utilization=utilization))
except Exception:
    pass
print(json.dumps(result))
'''


class Monitor:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.parsers = {}
        self.cursors = {}
        self.runs = []
        for label, color, name in DEFAULT_RUNS:
            manifest = json.loads((ROOT / 'logs' / f'{name}_20261019.launch.json').read_text())
            self.runs.append(dict(id=name, label=label, color=color, **manifest))
            self.parsers[name] = LogParser()
        self.last_success = None
        self.last_attempt = None
        self.error = None
        self.remote = {}
        if args.cache.is_file():
            try:
                saved = json.loads(args.cache.read_text())
                self.last_success = saved.get('last_success')
                for name in self.parsers:
                    if name in saved.get('parsers', {}):
                        self.parsers[name] = LogParser(saved['parsers'][name])
                # v2 understands resume markers. Replay old caches once so a
                # marker already consumed by the previous parser is not lost.
                self.cursors = saved.get('cursors', {}) if saved.get('schema_version') == 2 else {}
                self.remote = saved.get('remote', {})
            except (ValueError, OSError):
                pass

    def poll(self):
        request = [dict(id=run['id'], pid=run['pid'], log=run['log'], output=run['output'],
                        **self.cursors.get(run['id'], {})) for run in self.runs]
        script = 'REQUEST = ' + repr(request) + '\n' + REMOTE_PROBE
        command = ['ssh', '-i', str(self.args.key), '-o', 'IdentitiesOnly=yes',
                   '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=10',
                   '-o', 'ControlMaster=auto', '-o', 'ControlPersist=2h',
                   '-o', f'ControlPath={self.args.control_path}', self.args.host,
                   '/workspace/wangziyi/miniconda3/envs/evo-rlt/bin/python -']
        self.last_attempt = time.time()
        try:
            result = subprocess.run(command, input=script, text=True, capture_output=True, timeout=25, check=True)
            response = json.loads(result.stdout)
            with self.lock:
                for run in response['runs']:
                    name = run['id']
                    if run.get('reset'):
                        self.parsers[name] = LogParser()
                    if 'text' in run:
                        self.parsers[name].feed(run.pop('text'))
                        self.cursors[name] = dict(offset=run['offset'], identity=run['identity'])
                self.remote = response
                self.last_success = time.time()
                self.error = None
                saved = dict(schema_version=2, last_success=self.last_success, cursors=self.cursors,
                             parsers={name: parser.state for name, parser in self.parsers.items()}, remote=self.remote)
                self.args.cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.args.cache.with_suffix('.tmp')
                temporary.write_text(json.dumps(saved))
                temporary.replace(self.args.cache)
        except (subprocess.SubprocessError, ValueError, OSError) as error:
            detail = (getattr(error, 'stderr', None) or str(error)).strip()
            if 'Permission denied' in detail:
                detail = 'SSH 认证已失效，请在终端重新登录服务器；页面保留上次成功读取的数据。'
            with self.lock:
                self.error = detail[-500:]

    def loop(self):
        while not self.stop.is_set():
            self.poll()
            self.stop.wait(self.args.interval)

    def snapshot(self):
        with self.lock:
            runs = []
            remote = {run['id']: run for run in self.remote.get('runs', [])}
            for manifest in self.runs:
                state = copy.deepcopy(self.parsers[manifest['id']].state)
                state.pop('carry', None)
                status = remote.get(manifest['id'], {})
                control = status.get('stop_control', {})
                if control.get('target_step'):
                    state['total'] = control['target_step']
                latest = state['points'][-1] if state['points'] else {}
                points = state['points'][-25:]
                rate = 0
                if len(points) > 1 and points[-1]['elapsed'] > points[0]['elapsed']:
                    rate = (points[-1]['step'] - points[0]['step']) / (points[-1]['elapsed'] - points[0]['elapsed'])
                if not rate and state['elapsed']:
                    rate = state['step'] / state['elapsed']
                remaining = max(0, state['total'] - state['step'])
                outcome = ('queued' if control.get('status') == 'queued_for_resume' else
                           'stopped' if control.get('status') == 'saved_and_stopped' else
                           'failed' if control.get('status') in ('resume_failed', 'process_exited_before_checkpoint', 'checkpoint_validation_failed_paused') else
                           'completed' if state['complete'] or control.get('status') == 'completed' else 'failed' if state['failed'] else
                           'running' if status.get('alive') else 'stopped' if 'alive' in status else 'waiting')
                runs.append(dict(id=manifest['id'], label=manifest['label'], color=manifest['color'],
                    gpus=control.get('gpus', manifest['gpus']), inputs='wrist RGB + TCP Z' if 'rgb_z' in manifest['id'] else 'wrist RGB',
                    frames=manifest['dataset_frames'], episodes=manifest['dataset_episodes'],
                    batch=manifest['global_batch_size'], started_at=manifest['started_at'],
                    output=manifest['output'], log=manifest['log'], status=outcome,
                    latest=latest, rate=rate, eta_seconds=remaining / rate if rate else None,
                    log_modified=status.get('modified'), checkpoints=status.get('checkpoints', []),
                    error=status.get('error'), **state))
                runs[-1]['stop_control'] = control
            return dict(server=self.args.host, poll_seconds=self.args.interval, now=time.time(),
                last_success=self.last_success, last_attempt=self.last_attempt, error=self.error,
                connected=self.last_success is not None and self.error is None and time.time() - self.last_success < self.args.interval + 40,
                runs=runs, gpus=self.remote.get('gpus', []))


def csv_export(snapshot):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=['run', 'step', 'loss', 'lr', 'grdn', 'updt_s', 'data_s', 'elapsed', 'timestamp'])
    writer.writeheader()
    for run in snapshot['runs']:
        for point in run['points']:
            writer.writerow(dict(run=run['id'], **{key: point.get(key) for key in writer.fieldnames if key != 'run'}))
    return stream.getvalue().encode('utf-8-sig')


def make_handler(monitor):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            content_type = 'application/json; charset=utf-8'
            if path == '/api/status':
                body = json.dumps(monitor.snapshot(), allow_nan=False).encode()
            elif path == '/api/history.csv':
                body, content_type = csv_export(monitor.snapshot()), 'text/csv; charset=utf-8'
            elif path in ('/', '/app.js', '/style.css'):
                file = ASSETS / {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}[path]
                body = file.read_bytes()
                content_type = {'/': 'text/html; charset=utf-8', '/app.js': 'text/javascript; charset=utf-8', '/style.css': 'text/css; charset=utf-8'}[path]
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            if path.endswith('.csv'):
                self.send_header('Content-Disposition', 'attachment; filename="act-training-history.csv"')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--host', default='wangziyi@10.0.40.163')
    parser.add_argument('--key', type=Path, default=Path('/home/embint/.ssh/id_ed25519_ziyi'))
    parser.add_argument('--control-path', default='/tmp/evo-act-camera-%C')
    parser.add_argument('--cache', type=Path, default=ROOT / 'logs/training_dashboard_cache.json')
    args = parser.parse_args()
    if args.interval < 5:
        parser.error('interval must be at least 5 seconds')
    monitor = Monitor(args)
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(monitor))
    worker = threading.Thread(target=monitor.loop, daemon=True)
    worker.start()
    print(f'Training dashboard: http://{args.bind}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop.set()
        server.server_close()


if __name__ == '__main__':
    main()
