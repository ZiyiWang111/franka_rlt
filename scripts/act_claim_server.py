"""Private authenticated, persistent, single-winner training claim service."""
import argparse
import fcntl
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading


class ClaimStore:
    def __init__(self, path):
        self.path = Path(path)
        self.mutex = threading.Lock()
        self.record = json.loads(self.path.read_text()) if self.path.exists() else {'status': 'WAITING', 'owner': None}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        with tmp.open('w') as stream:
            stream.write(json.dumps(self.record, indent=2) + '\n')
            stream.flush()
            import os
            os.fsync(stream.fileno())
        tmp.replace(self.path)

    def claim(self, payload):
        with self.mutex:
            if self.record['owner'] is None:
                self.record = dict(status='CLAIMED', owner=payload['owner'], host=payload['host'],
                                   gpu_uuids=payload['gpu_uuids'])
                self.save()
            return dict(self.record)

    def update(self, payload):
        with self.mutex:
            if self.record['owner'] != payload['owner']:
                raise PermissionError('Not claim owner')
            assert payload['status'] in ['RUNNING', 'COMPLETE', 'FAILED', 'START_FAILED']
            self.record.update({k: v for k, v in payload.items() if k in ['status', 'pid', 'exit_code', 'error', 'log', 'output_dir']})
            self.save()
            return dict(self.record)


def handler_for(store, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token):
                self.send_error(403)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 <= size <= 8192:
                    raise ValueError('Request too large')
                data = json.loads(self.rfile.read(size) or b'{}')
                if self.path == '/claim':
                    result = store.claim(data)
                elif self.path == '/update':
                    result = store.update(data)
                elif self.path == '/status':
                    with store.mutex:
                        result = dict(store.record)
                else:
                    self.send_error(404)
                    return
                body = json.dumps(result).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                self.send_error(400)
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings', type=Path, required=True)
    args = p.parse_args()
    settings = json.loads(args.settings.read_text())
    lock = args.settings.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    store = ClaimStore(settings['claim_file'])
    ThreadingHTTPServer((settings['bind'], settings['port']), handler_for(store, settings['token'])).serve_forever()


if __name__ == '__main__':
    main()
