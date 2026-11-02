"""Wait for VGA completion and an idle GPU, audit raw encoder inputs, train USB token."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from run_act_idle_queue import snapshot, checkpoint_complete


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--after-output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=4)
    p.add_argument('--python', required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads(a.config.read_text())
    output = Path(cfg['output_dir'])
    assert not output.exists(), output
    logs = root / 'logs/usb_token'
    logs.mkdir(parents=True, exist_ok=True)
    lock = (logs / 'launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record = dict(pid=os.getpid(), status='WAITING_VGA_AND_GPU', gpu=a.gpu,
                  config=str(a.config), output_dir=str(output))

    def save(**updates):
        record.update(updates, updated_at=datetime.datetime.now().astimezone().isoformat())
        path = logs / 'status.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(path)
        print(json.dumps(record), flush=True)
    save()
    gpu_lock = None
    while True:
        if checkpoint_complete(a.after_output, 30000):
            gpu = next(g for g in snapshot() if g['index'] == a.gpu)
            if not gpu['compute_process'] and gpu['used_mib'] < 1024 and gpu['util'] <= 5:
                gpu_lock = Path(f"/tmp/act-idle-{os.getuid()}-{gpu['uuid']}.lock").open('a')
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    gpu_lock.close()
        time.sleep(120)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], PYTHONPATH=f'{root}/src:{root}',
               TORCH_HOME=str(root / '.cache/torch'), OMP_NUM_THREADS='4',
               HF_DATASETS_CACHE=str(root / '.cache/hf_datasets'))
    report = root / 'configs/usb_1027_token/input_audit.json'
    cmd = [a.python, '-u', str(root / 'scripts/check_act_rl_token_inputs.py'),
           '--checkpoint', cfg['policy']['act_pretrained_path'], '--dataset', cfg['dataset']['root'],
           '--device', 'cuda', '--batch-size', str(cfg['batch_size']),
           '--reserve-adam-state', '--report', str(report)]
    save(status='AUDITING', gpu_uuid=gpu['uuid'])
    with (logs / 'audit.log').open('ab', buffering=0) as stream:
        result = subprocess.run(cmd, env=env, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        save(status='AUDIT_FAILED', exit_code=result.returncode)
        return
    audit = json.loads(report.read_text())
    assert audit['checks_passed'] and not audit['extra_positional_embedding'] and audit['act_frozen']
    entry = ('from act_rlt.dataset_compat import install_scalar_state_dataset_support; '
             'install_scalar_state_dataset_support(); from act_rlt import register; register(); '
             'from lerobot.scripts.lerobot_train import main; main()')
    cmd = [a.python, '-u', '-c', entry, f'--config_path={a.config}']
    with (logs / 'train.log').open('ab', buffering=0) as stream:
        proc = subprocess.Popen(cmd, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT,
                                pass_fds=(gpu_lock.fileno(),))
        save(status='RUNNING', training_pid=proc.pid, command=cmd)
        code = proc.wait()
    ok = code == 0 and checkpoint_complete(output, cfg['steps'])
    save(status='COMPLETE' if ok else 'FAILED', exit_code=code)


if __name__ == '__main__':
    main()
