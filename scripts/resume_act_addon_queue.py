#!/usr/bin/env python3
"""Run the two ACT resumes sequentially on a checked pair of idle GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.stop_act_at_checkpoint import validate_checkpoint, write_json


NAMES = ['act_dataset_928_addon_camera_only_action6_act', 'act_dataset_928_addon_rgb_z_action6_act']


def idle(gpus):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True)
    stats = {int(parts[0]): (int(parts[1]), int(parts[2]))
             for line in output.splitlines() if (parts := line.split(','))}
    return all(stats[gpu][0] < 100 and stats[gpu][1] == 0 for gpu in gpus)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpus', nargs=2, type=int, default=[0, 1])
    args = parser.parse_args()
    root = args.root.resolve()
    old_guard = root / 'logs/act_addon_stop_at_10000_20261019.status.json'
    if old_guard.exists():
        pid = json.loads(old_guard.read_text())['guard_pid']
        if Path(f'/proc/{pid}').exists():
            raise RuntimeError('Old checkpoint-stop guard still exists; refusing to resume')
    records = []
    for name in NAMES:
        output = root / 'outputs' / 'ethernet_act' / name
        validate_checkpoint(output, 10000)
        control = output / 'stop_at_checkpoint.json'
        if control.exists():
            existing = json.loads(control.read_text())
            if existing.get('status') in ('training_to_30000', 'queued_for_resume'):
                raise RuntimeError(f'{name} is already running or queued')
            write_json(output / 'completed_stop_at_10000.json', existing)
        record = dict(name=name, status='queued_for_resume', target_step=30000, resume_step=10000,
                      gpus=args.gpus, queue_pid=os.getpid(), active_pid=None, queued_at=time.time())
        records.append(record)
    status = root / 'logs/act_addon_resume_20261019.status.json'
    for record in records:
        write_json(root / 'outputs' / 'ethernet_act' / record['name'] / 'stop_at_checkpoint.json', record)
    write_json(status, dict(queue_pid=os.getpid(), runs=records))
    for record in records:
        name = record['name']
        while not idle(args.gpus):
            print(f'Waiting for GPUs {args.gpus}: {name}', flush=True)
            time.sleep(10)
        config = root / 'outputs' / 'ethernet_act' / name / 'checkpoints/010000/pretrained_model/train_config.json'
        command = [os.sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
                   '--nnodes=1', '--nproc_per_node=2', '--module', 'scripts.resume_act_addon_tmp',
                   f'--config_path={config}', '--resume=true', '--batch_size=8']
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, args.gpus)), OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONPATH=f'{root}/src:{root}')
        with (root / 'logs' / f'{name}_20261019.log').open('a') as log:
            log.write('\n--- Resume from complete 010000 checkpoint; original scheduler and global batch 16 ---\n')
            log.flush()
            process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        record.update(status='training_to_30000', active_pid=process.pid, command=command, started_at=time.time())
        write_json(root / 'outputs' / 'ethernet_act' / name / 'stop_at_checkpoint.json', record)
        write_json(status, dict(queue_pid=os.getpid(), runs=records))
        print(f'Started {name}, PID {process.pid}, GPUs {args.gpus}', flush=True)
        exit_code = process.wait()
        record.update(exit_code=exit_code, ended_at=time.time())
        try:
            if exit_code != 0:
                raise RuntimeError(f'training exited with code {exit_code}')
            record['validation'] = validate_checkpoint(root / 'outputs' / 'ethernet_act' / name, 30000)
        except Exception as error:
            record.update(status='resume_failed', error=str(error))
        else:
            record.update(status='completed', saved_step=30000)
        write_json(root / 'outputs' / 'ethernet_act' / name / 'stop_at_checkpoint.json', record)
        write_json(status, dict(queue_pid=os.getpid(), runs=records))
        print(f'{name}: {record["status"]}', flush=True)
    print('Resume queue finished.', flush=True)


if __name__ == '__main__':
    main()
