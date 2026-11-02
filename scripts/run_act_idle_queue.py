"""Wait for five idle samples two minutes apart, then run single-GPU ACT jobs.

Example: python scripts/run_act_idle_queue.py --project /path/to/project \
    --python /path/to/env/bin/python --configs configs/fine_1027/*.json
Run under nohup; inspect logs/act_idle_queue/status.json and the job logs.
"""
import argparse
import csv
import datetime
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def now():
    return datetime.datetime.now().astimezone().isoformat()


def snapshot():
    """A failed/unsupported query raises; callers must reset all idle streaks."""
    def query(fields, kind):
        result = subprocess.check_output(
            ['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'],
            text=True, timeout=20)
        return list(csv.reader(io.StringIO(result), skipinitialspace=True))
    gpus = query('index,uuid,memory.used,utilization.gpu', 'gpu')
    apps = query('gpu_uuid,pid', 'compute-apps')
    busy = {row[0].strip() for row in apps if row}
    return [dict(index=int(i), uuid=u.strip(), used_mib=int(m), util=int(v),
                 compute_process=u.strip() in busy) for i, u, m, v in gpus]


def advance_streaks(previous, gpus, reserved=()):
    """Track the same physical GPU; any busy/missing sample breaks the streak."""
    return {g['uuid']: previous.get(g['uuid'], 0) + 1
            if g['uuid'] not in reserved and not g['compute_process']
            and g['used_mib'] < 1024 and g['util'] <= 5 else 0 for g in gpus}


def checkpoint_complete(output, steps):
    folder = output / 'checkpoints/last'
    state = folder / 'training_state/training_step.json'
    return (state.exists() and json.loads(state.read_text()).get('step') == steps
            and (folder / 'pretrained_model/model.safetensors').exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--configs', type=Path, nargs='+', required=True)
    parser.add_argument('--gpus', type=int, nargs='+', help='Restrict to these physical GPU indices')
    parser.add_argument('--idle-samples', type=int, default=5)
    parser.add_argument('--log-dir', type=Path, default=Path('logs/act_idle_queue'))
    parser.add_argument('--check-only', action='store_true', help='Validate inputs; never launch or wait')
    a = parser.parse_args()
    if a.idle_samples < 1:
        parser.error('--idle-samples must be positive')
    root = a.project.resolve()
    # Keep virtualenv's Python symlink: resolving it would bypass the venv.
    python = a.python.absolute()
    logdir = root / a.log_dir
    logdir.mkdir(parents=True, exist_ok=True)
    # One scheduler per project. Cross-project copies also cooperate with UUID locks.
    lock = (logdir / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(PYTHONPATH=f'{root}/src:{root}', OMP_NUM_THREADS='4',
               HF_DATASETS_CACHE=str(root / '.cache/hf_datasets'),
               TORCH_HOME=str(root / '.cache/torch'))
    jobs = []
    for path in a.configs:
        path = (root / path).resolve()
        cfg = json.loads(path.read_text())
        output = Path(cfg['output_dir'])
        if not output.is_absolute() or output.exists():
            raise ValueError(f'Output must be a new absolute path: {output}')
        if cfg['steps'] not in (30000, 50000) or cfg['batch_size'] not in (16, 32):
            raise ValueError('Expected 30000/50000 steps and batch_size=16/32')
        subprocess.run([str(python), str(root / 'scripts/train_act_fine_queue.py'),
                        '--config', str(path), '--validate-only'],
                       env=env, cwd=root, check=True)
        jobs.append(dict(name=cfg['job_name'], config=str(path), output_dir=str(output),
                         steps=cfg['steps'], status='WAITING_GPU'))
    if len({j['name'] for j in jobs}) != len(jobs) or len({j['output_dir'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate job names or outputs')
    print(json.dumps(dict(event='PREFLIGHT_OK', time=now(), jobs=jobs)), flush=True)
    if a.check_only:
        return
    record = dict(pid=os.getpid(), started_at=now(), interval_seconds=120,
                  consecutive_idle_samples=a.idle_samples, allowed_gpu_indices=a.gpus,
                  status='WAITING_GPU', jobs=jobs)
    active = {}
    streaks = {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        # Stop scheduling only; already-running training is left running.
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def save():
        record['updated_at'] = now()
        tmp = logdir / 'status.tmp'
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(logdir / 'status.json')

    save()
    while not stopping:
        sample_started = time.monotonic()
        for uuid, (proc, job, gpu_lock, stream) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            stream.close()
            gpu_lock.close()
            del active[uuid]
            ok = code == 0 and checkpoint_complete(Path(job['output_dir']), job['steps'])
            job.update(status='COMPLETE' if ok else 'FAILED', exit_code=code, finished_at=now())
            streaks[uuid] = 0
            print(json.dumps(dict(event='JOB_FINISHED', **job)), flush=True)
        try:
            gpus = snapshot()
            if a.gpus is not None:
                gpus = [g for g in gpus if g['index'] in a.gpus]
            streaks = advance_streaks(streaks, gpus, active)
            record.update(gpus=gpus, idle_streaks=streaks, query_error=None)
            print(json.dumps(dict(event='GPU_SAMPLE', time=now(), gpus=gpus, streaks=streaks)), flush=True)
            for gpu in gpus:
                uuid = gpu['uuid']
                waiting = next((j for j in jobs if j['status'] == 'WAITING_GPU'), None)
                if waiting is None or streaks[uuid] < a.idle_samples or stopping:
                    continue
                gpu_lock = Path(f'/tmp/act-idle-{os.getuid()}-{uuid}.lock').open('a')
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    gpu_lock.close()
                    streaks[uuid] = 0
                    continue
                # Recheck immediately before launch; this isn't a resource reservation
                # against unrelated launchers on a server without a scheduler.
                try:
                    fresh = {g['uuid']: g for g in snapshot()}
                except Exception:
                    gpu_lock.close()
                    raise
                if advance_streaks({}, fresh.values(), active).get(uuid, 0) != 1:
                    gpu_lock.close()
                    streaks[uuid] = 0
                    continue
                if Path(waiting['output_dir']).exists():
                    gpu_lock.close()
                    waiting.update(status='FAILED', error='Output directory already exists', finished_at=now())
                    continue
                stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                logfile = logdir / f"{waiting['name']}_{stamp}.log"
                stream = logfile.open('ab', buffering=0)
                cmd = [str(python), '-u', str(root / 'scripts/train_act_fine_queue.py'),
                       '--config', waiting['config']]
                try:
                    proc = subprocess.Popen(cmd, cwd=root, env=dict(env, CUDA_VISIBLE_DEVICES=uuid),
                                            stdin=subprocess.DEVNULL, stdout=stream,
                                            stderr=subprocess.STDOUT, start_new_session=True,
                                            pass_fds=(gpu_lock.fileno(),))
                except Exception as error:
                    stream.close()
                    gpu_lock.close()
                    waiting.update(status='FAILED', error=str(error), finished_at=now())
                    continue
                active[uuid] = (proc, waiting, gpu_lock, stream)
                waiting.update(status='RUNNING', pid=proc.pid, gpu=gpu['index'], gpu_uuid=uuid,
                               started_at=now(), log=str(logfile), command=cmd)
                streaks[uuid] = 0
                save()
                print(json.dumps(dict(event='JOB_STARTED', **waiting)), flush=True)
        except Exception as error:
            streaks = {}
            record.update(query_error=str(error), idle_streaks={})
            print(json.dumps(dict(event='CHECK_ERROR', time=now(), error=str(error))), flush=True)
        if all(j['status'] in ('COMPLETE', 'FAILED') for j in jobs):
            record['status'] = 'FAILED' if any(j['status'] == 'FAILED' for j in jobs) else 'COMPLETE'
            record['finished_at'] = now()
            save()
            return
        record['status'] = 'RUNNING' if active else 'WAITING_GPU'
        save()
        # Sleep in short intervals so SIGTERM can promptly stop scheduling.
        while not stopping and time.monotonic() - sample_started < 120:
            time.sleep(max(0, min(1, 120 - (time.monotonic() - sample_started))))
    record['status'] = 'SCHEDULER_STOPPED'
    save()


if __name__ == '__main__':
    main()
