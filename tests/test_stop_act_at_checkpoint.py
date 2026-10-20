import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from safetensors.numpy import save_file
import numpy as np

from scripts.stop_act_at_checkpoint import checkpoint_ready, validate_checkpoint, write_json


def test_status_can_be_written_before_first_checkpoint_creates_output(tmp_path):
    path = tmp_path / 'new_output/stop_at_checkpoint.json'
    write_json(path, {'target_step': 10000})
    assert json.loads(path.read_text())['target_step'] == 10000


def make_checkpoint(tmp_path):
    output = tmp_path / 'run'
    checkpoint = output / 'checkpoints/010000'
    model = checkpoint / 'pretrained_model'
    state = checkpoint / 'training_state'
    model.mkdir(parents=True)
    state.mkdir()
    for name in ['config.json', 'train_config.json', 'policy_preprocessor.json', 'policy_postprocessor.json']:
        (model / name).write_text('{}')
    save_file({'weight': np.zeros((2, 2), dtype=np.float32)}, model / 'model.safetensors')
    for name in ['optimizer_state.safetensors', 'rng_state.safetensors']:
        save_file({'state': np.zeros(2, dtype=np.float32)}, state / name)
    for name in ['optimizer_param_groups.json', 'scheduler_state.json']:
        (state / name).write_text('{}')
    (state / 'training_step.json').write_text(json.dumps({'step': 10000}))
    return output, checkpoint


def test_no_stop_until_last_link_commits_all_state(tmp_path):
    output, checkpoint = make_checkpoint(tmp_path)
    assert not checkpoint_ready(output, 10000)
    (output / 'checkpoints/last').symlink_to('010000')
    assert checkpoint_ready(output, 10000)
    result = validate_checkpoint(output, 10000)
    assert len(result['tensor_files']) == 3
    assert result['checkpoint_bytes'] > 0
    (checkpoint / 'training_state/optimizer_state.safetensors').unlink()
    assert not checkpoint_ready(output, 10000)


def test_wrong_saved_step_is_not_ready(tmp_path):
    output, checkpoint = make_checkpoint(tmp_path)
    (output / 'checkpoints/last').symlink_to('010000')
    (checkpoint / 'training_state/training_step.json').write_text('{"step": 9999}')
    assert not checkpoint_ready(output, 10000)


def test_guard_resumes_and_terminates_only_after_committed_checkpoint(tmp_path):
    root = tmp_path
    ids = root / 'ids.json'
    parent_code = '''
import json,os,signal,subprocess,sys,time
from pathlib import Path
ranks=[subprocess.Popen([sys.executable,'-c','import time; time.sleep(90)','run'],start_new_session=True) for _ in range(2)]
for rank in ranks:os.killpg(rank.pid,signal.SIGSTOP)
Path(sys.argv[1]).write_text(json.dumps({'name':'run','parent':os.getpid(),'ranks':[p.pid for p in ranks],'gpus':[]}))
os.kill(os.getpid(),signal.SIGSTOP)
time.sleep(90)
'''
    parent = subprocess.Popen([sys.executable, '-c', parent_code, str(ids), 'run'], start_new_session=True)
    guard = None
    run = None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if ids.exists() and Path(f'/proc/{parent.pid}/status').read_text().split('State:', 1)[1].lstrip().startswith('T'):
                break
            time.sleep(.02)
        assert ids.exists()
        run = json.loads(ids.read_text())
        record = root / 'paused.json'
        record.write_text(json.dumps({'runs': [run]}))
        guard = subprocess.Popen([sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts/stop_act_at_checkpoint.py'),
            '--pause-record', str(record), '--root', str(root)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        status_path = root / 'outputs/run/stop_at_checkpoint.json'
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if status_path.exists() and json.loads(status_path.read_text())['status'] == 'running_to_checkpoint':
                break
            time.sleep(.02)
        assert json.loads(status_path.read_text())['status'] == 'running_to_checkpoint'
        assert parent.poll() is None  # no checkpoint yet: job must be alive
        output, checkpoint = make_checkpoint(root / 'outputs')
        (output / 'checkpoints/last').symlink_to('010000')
        stdout, stderr = guard.communicate(timeout=15)
        assert guard.returncode == 0, stderr
        assert 'Saved and stopped run' in stdout
        saved = json.loads(status_path.read_text())
        assert saved['status'] == 'saved_and_stopped'
        assert saved['checkpoint_saved'] is True
        assert saved['saved_step'] == 10000
        parent.wait(timeout=3)
        assert checkpoint_ready(output, 10000)
    finally:
        if guard is not None and guard.poll() is None:
            guard.kill()
            guard.wait()
        for pid in (run or {}).get('ranks', []):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if parent.poll() is None:
            os.kill(parent.pid, signal.SIGKILL)
            parent.wait()
