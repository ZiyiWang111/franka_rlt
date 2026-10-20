#!/usr/bin/env python3
"""Resume specified paused ACT jobs; stop each only after a complete checkpoint.

The trainer updates checkpoints/last only AFTER saving all model/processor and
optimizer/scheduler/RNG files. That link is the commit marker for this guard.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def process_identity(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        return fields[19], fields[0]  # start time, process state
    except FileNotFoundError:
        return None, None


def checkpoint_ready(output, step):
    checkpoint = output / 'checkpoints' / f'{step:06d}'
    last = output / 'checkpoints/last'
    if not last.is_symlink() or last.resolve() != checkpoint:
        return False
    files = [
        'pretrained_model/config.json', 'pretrained_model/model.safetensors',
        'pretrained_model/train_config.json', 'pretrained_model/policy_preprocessor.json',
        'pretrained_model/policy_postprocessor.json', 'training_state/training_step.json',
        'training_state/optimizer_state.safetensors', 'training_state/optimizer_param_groups.json',
        'training_state/scheduler_state.json', 'training_state/rng_state.safetensors',
    ]
    if not all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size for name in files):
        return False
    try:
        return json.loads((checkpoint / 'training_state/training_step.json').read_text())['step'] == step
    except (ValueError, KeyError):
        return False


def validate_checkpoint(output, step):
    from safetensors import safe_open

    if not checkpoint_ready(output, step):
        raise RuntimeError('checkpoint is not committed and complete')
    checkpoint = output / 'checkpoints' / f'{step:06d}'
    for file in checkpoint.rglob('*.json'):
        json.loads(file.read_text())
    tensors = []
    for file in checkpoint.rglob('*.safetensors'):
        # Opening validates header + tensor offsets/lengths without allocating
        # CUDA tensors or loading hundreds of MB of optimizer state into RAM.
        with safe_open(file, framework='numpy') as handle:
            if not list(handle.keys()):
                raise RuntimeError(f'empty tensor file: {file}')
        tensors.append(str(file.relative_to(checkpoint)))
    return dict(checkpoint=str(checkpoint), tensor_files=tensors,
                checkpoint_bytes=sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file()))


def verified_signal(run, sig):
    pids = [*run['ranks'], run['parent']] if sig == signal.SIGCONT else [run['parent'], *run['ranks']]
    for pid in pids:
        identity, state = process_identity(pid)
        if identity is None or state == 'Z':
            continue
        if identity != run['identities'][str(pid)]:
            raise RuntimeError(f'PID {pid} identity changed; refusing to signal')
        if pid == run['parent']:
            os.kill(pid, sig)
        else:
            if os.getpgid(pid) != pid:
                raise RuntimeError(f'unexpected process group for rank {pid}')
            os.killpg(pid, sig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pause-record', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--step', type=int, default=10000)
    args = parser.parse_args()
    if args.step < 1:
        parser.error('step must be positive')
    root = args.root.resolve()
    runs = json.loads(args.pause_record.read_text())['runs']
    for run in runs:
        run['output'] = str(root / 'outputs' / run['name'])
        run['identities'] = {}
        for pid in [run['parent'], *run['ranks']]:
            command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            if run['name'] not in command:
                raise RuntimeError(f'PID {pid} is not the specified ACT job')
            identity, state = process_identity(pid)
            if state not in ('T', 't'):
                raise RuntimeError(f'job {pid} is not paused: {state}')
            run['identities'][str(pid)] = identity
        run.update(status='armed', target_step=args.step, guard_pid=os.getpid(), armed_at=time.time())
    status_file = root / 'logs/act_addon_stop_at_10000_20261019.status.json'
    write_json(status_file, dict(guard_pid=os.getpid(), runs=runs))
    for run in runs:
        write_json(Path(run['output']) / 'stop_at_checkpoint.json', run)
    for run in runs:
        verified_signal(run, signal.SIGCONT)
        run.update(status='running_to_checkpoint', paused=False)
        write_json(Path(run['output']) / 'stop_at_checkpoint.json', run)
        print(f"Resumed {run['name']}; will stop after checkpoint {args.step}", flush=True)
    write_json(status_file, dict(guard_pid=os.getpid(), runs=runs))
    while any(run['status'] == 'running_to_checkpoint' for run in runs):
        for run in runs:
            if run['status'] != 'running_to_checkpoint':
                continue
            output = Path(run['output'])
            if checkpoint_ready(output, args.step):
                # Freeze the job before validation. Never terminate if a file
                # is incomplete or checkpoint validation fails.
                verified_signal(run, signal.SIGSTOP)
                try:
                    run['validation'] = validate_checkpoint(output, args.step)
                except Exception as error:
                    run.update(status='checkpoint_validation_failed_paused', paused=True, error=str(error))
                else:
                    verified_signal(run, signal.SIGTERM)
                    verified_signal(run, signal.SIGCONT)  # deliver pending TERM to stopped processes
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        if all(process_identity(pid)[1] in (None, 'Z') for pid in [run['parent'], *run['ranks']]):
                            break
                        time.sleep(.2)
                    else:
                        verified_signal(run, signal.SIGKILL)
                    run.update(status='saved_and_stopped', stopped_at=time.time(),
                               checkpoint_saved=True, saved_step=args.step, paused=False)
                    print(f"Saved and stopped {run['name']} at checkpoint {args.step}", flush=True)
            elif any(process_identity(pid)[1] in (None, 'Z') for pid in [run['parent'], *run['ranks']]):
                run.update(status='process_exited_before_checkpoint', stopped_at=time.time())
                # A failed DDP peer must not leave its remaining rank running.
                verified_signal(run, signal.SIGSTOP)
                print(f"ERROR: {run['name']} exited before saving; remaining ranks paused", flush=True)
            else:
                continue
            write_json(output / 'stop_at_checkpoint.json', run)
            write_json(status_file, dict(guard_pid=os.getpid(), runs=runs))
        time.sleep(.2)
    print('Guard finished; see status JSON for checkpoint validation and exit states.', flush=True)


if __name__ == '__main__':
    main()
