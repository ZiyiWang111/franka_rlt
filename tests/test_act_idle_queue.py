"""Idle qualification must be per GPU and must fail closed on lost samples."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    'act_idle_queue', Path(__file__).parents[1] / 'scripts/run_act_idle_queue.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def gpu(uuid='GPU-a', used=0, util=0, process=False):
    return dict(index=0, uuid=uuid, used_mib=used, util=util, compute_process=process)


def test_five_consecutive_samples():
    streaks = {}
    for count in range(1, 6):
        streaks = queue.advance_streaks(streaks, [gpu()])
        assert streaks['GPU-a'] == count


@pytest.mark.parametrize('busy', [gpu(process=True), gpu(used=1024), gpu(util=6)])
def test_busy_sample_resets_streak(busy):
    assert queue.advance_streaks({'GPU-a': 4}, [busy]) == {'GPU-a': 0}
    assert queue.advance_streaks({'GPU-a': 0}, [gpu()]) == {'GPU-a': 1}


def test_missing_or_reindexed_gpu_cannot_inherit_streak():
    assert queue.advance_streaks({'GPU-a': 4}, []) == {}
    assert queue.advance_streaks({'GPU-a': 4}, [gpu(uuid='GPU-b')]) == {'GPU-b': 1}


def test_active_job_excluded_even_during_startup():
    assert queue.advance_streaks({'GPU-a': 4}, [gpu()], reserved=['GPU-a']) == {'GPU-a': 0}


def test_snapshot_keeps_busy_process_even_at_zero_util(monkeypatch):
    def output(cmd, **kw):
        if any('--query-gpu=' in arg for arg in cmd):
            return '0, GPU-a, 500, 0\n1, GPU-b, 0, 0\n'
        return 'GPU-a, 42\n'
    monkeypatch.setattr(subprocess, 'check_output', output)
    result = queue.advance_streaks({'GPU-a': 4, 'GPU-b': 4}, queue.snapshot())
    assert result == {'GPU-a': 0, 'GPU-b': 5}


def test_unsupported_gpu_metrics_cannot_qualify(monkeypatch):
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: '0, GPU-a, N/A, 0\n')
    with pytest.raises(ValueError):
        queue.snapshot()


def test_requires_final_weights_and_training_step(tmp_path):
    final = tmp_path / 'checkpoints/last'
    (final / 'training_state').mkdir(parents=True)
    state = final / 'training_state/training_step.json'
    state.write_text(json.dumps({'step': 50000}))
    assert not queue.checkpoint_complete(tmp_path, 50000)
    (final / 'pretrained_model').mkdir()
    (final / 'pretrained_model/model.safetensors').touch()
    assert queue.checkpoint_complete(tmp_path, 50000)
    state.write_text(json.dumps({'step': 49000}))
    assert not queue.checkpoint_complete(tmp_path, 50000)


@pytest.mark.parametrize('busy_sample,launch_second,steps,samples,batch',
                         [(None, 480, 50000, 5, 32), (4, 1080, 50000, 5, 32),
                          (None, 0, 30000, 1, 32), (None, 0, 30000, 1, 16)])
def test_scheduler_launches_only_after_five_idle_polls(tmp_path, monkeypatch, busy_sample, launch_second, steps, samples, batch):
    config = tmp_path / 'job.json'
    config.write_text(json.dumps(dict(job_name='test', output_dir=str(tmp_path / 'output'),
                                     steps=steps, batch_size=batch)))
    clock = [0.0]
    launches = []
    monkeypatch.setattr(sys, 'argv', ['queue', '--project', str(tmp_path), '--python', sys.executable,
                                    '--configs', str(config), '--gpus', '0', '--idle-samples', str(samples)])
    monkeypatch.setattr(queue.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(queue.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(queue.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: None)
    monkeypatch.setattr(queue, 'checkpoint_complete', lambda *a: True)
    monkeypatch.setattr(queue, 'snapshot', lambda: [gpu(uuid='GPU-queue-unit-test',
                        process=busy_sample is not None and int(clock[0] // 120) == busy_sample)])

    class Child:
        pid = 999999

        def poll(self):
            return 0

    def launch(cmd, **kw):
        launches.append((clock[0], kw['env']['CUDA_VISIBLE_DEVICES']))
        return Child()
    monkeypatch.setattr(subprocess, 'Popen', launch)
    queue.main()
    assert launches == [(launch_second, 'GPU-queue-unit-test')]
    status = json.loads((tmp_path / 'logs/act_idle_queue/status.json').read_text())
    assert status['status'] == 'COMPLETE'
