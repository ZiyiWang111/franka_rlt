"""Two-host ACT watcher: qualify local GPUs, acquire one global claim, train once."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import uuid

from run_act_idle_queue import snapshot, advance_streaks, checkpoint_complete


def eligible_gpus(gpus, counts):
    return [g for g in gpus if counts.get(g['uuid'], 0) >= 3][:2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings', type=Path, required=True)
    args = p.parse_args()
    settings = json.loads(args.settings.read_text())
    root = Path(settings['project'])
    config = root / 'configs/vga_1023_z_orientation/train_config.json'
    cfg = json.loads(config.read_text())
    logs = root / 'logs/orientation'
    logs.mkdir(parents=True, exist_ok=True)
    singleton = (logs / 'watcher.lock').open('a')
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = uuid.uuid4().hex
    state = dict(pid=os.getpid(), owner=owner, host=settings['host'], status='PREFLIGHT',
                 config=str(config), training_enabled=True)

    def save(**updates):
        state.update(updates, updated_at=datetime.datetime.now().astimezone().isoformat())
        path = logs / 'status.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(path)
        print(json.dumps(state), flush=True)

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def api(path, data=None):
        req = urllib.request.Request(settings['claim_url'] + path,
                  data=json.dumps(data or {}).encode(),
                  headers={'Authorization': 'Bearer ' + settings['token'], 'Content-Type': 'application/json'})
        with opener.open(req, timeout=15) as response:
            return json.load(response)

    env = dict(os.environ, PYTHONPATH=f'{root}/src:{root}', OMP_NUM_THREADS='4',
               HF_DATASETS_CACHE=str(root / '.cache/hf_datasets'), TORCH_HOME=str(root / '.cache/torch'))
    py = settings['python']
    script = root / 'scripts/train_act_z_orientation.py'
    save()
    # Validate both launch sizes without allocating a CUDA context.
    for world in (1, 2):
        subprocess.run([py, str(script), '--config', str(config), '--validate-only',
                        '--world-size', str(world)], cwd=root, env=env, check=True)
    counts = {}
    save(status='WAITING_GPU', poll_seconds=120, required_idle_samples=3)
    while True:
        locks = []
        won = False
        try:
            global_state = api('/status')
            if global_state['owner'] is not None:
                save(status='OTHER_SERVER_CLAIMED', global_claim=global_state)
                return
            gpus = snapshot()
            counts = advance_streaks(counts, gpus)
            save(gpus=gpus, idle_streaks=counts, query_error=None)
            candidates = eligible_gpus(gpus, counts)
            if candidates:
                for g in candidates:
                    handle = Path(f"/tmp/act-idle-{os.getuid()}-{g['uuid']}.lock").open('a')
                    locks.append(handle)
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Fresh GPU snapshot while holding cooperating local locks.
                fresh = {g['uuid']: g for g in snapshot()}
                if not all(advance_streaks({}, fresh.values()).get(g['uuid']) == 1 for g in candidates):
                    counts = {}
                    continue
                claim = api('/claim', dict(owner=owner, host=settings['host'],
                            gpu_uuids=[g['uuid'] for g in candidates]))
                if claim['owner'] != owner:
                    save(status='OTHER_SERVER_CLAIMED', global_claim=claim)
                    return
                won = True
                # Once claimed, failures never silently release/reassign the job.
                output = Path(cfg['output_dir'])
                assert not output.exists(), f'Output already exists: {output}'
                world = len(candidates)
                train_env = dict(env, CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in candidates))
                cmd = [py, '-u']
                if world == 2:
                    cmd += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
                cmd += [str(script), '--config', str(config)]
                logfile = logs / 'train.log'
                with logfile.open('ab', buffering=0) as stream:
                    proc = subprocess.Popen(cmd, cwd=root, env=train_env, stdin=subprocess.DEVNULL,
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                            pass_fds=tuple(h.fileno() for h in locks))
                    save(status='RUNNING', training_pid=proc.pid, command=cmd,
                         gpu_indices=[g['index'] for g in candidates],
                         gpu_uuids=[g['uuid'] for g in candidates],
                         global_batch=64, per_gpu_batch=64 // world, log=str(logfile))
                    try:
                        api('/update', dict(owner=owner, status='RUNNING', pid=proc.pid,
                            output_dir=str(output), log=str(logfile)))
                    except Exception as error:
                        save(claim_update_error=str(error))
                    code = proc.wait()
                ok = code == 0 and checkpoint_complete(output, 50000)
                status = 'COMPLETE' if ok else 'FAILED'
                save(status=status, exit_code=code)
                try:
                    api('/update', dict(owner=owner, status=status, exit_code=code))
                except Exception as error:
                    save(claim_update_error=str(error))
                return
        except Exception as error:
            counts = {}
            save(status='START_FAILED' if won else 'WAITING_GPU', query_error=str(error), idle_streaks={})
            if won:
                try:
                    api('/update', dict(owner=owner, status='START_FAILED', error=str(error)))
                except Exception:
                    pass
                raise
        finally:
            for handle in locks:
                handle.close()
        time.sleep(120)


if __name__ == '__main__':
    main()
