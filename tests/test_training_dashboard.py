import argparse
import csv
import io
import json
from pathlib import Path
import subprocess

import pytest

from scripts.training_dashboard import LogParser, Monitor, csv_export, duration_seconds


def training_line(step, printed='2K', loss='.249'):
    return (f'\rTraining: 5%|▌| {step}/30000 [07:30<1:34:55, 4.98step/s]'
            f'INFO 2026-10-19 11:22:29 ot_train.py:451 step:{printed} loss:{loss} '
            'grdn:7.050 lr:3.0e-05 updt_s:0.119 data_s:0.167\n')


def test_exact_progress_survives_rounded_log_step_and_arbitrary_chunks():
    text = training_line(1640) + training_line(1660)
    parser = LogParser()
    for char in text:
        parser.feed(char)
    assert [point['step'] for point in parser.state['points']] == [1640, 1660]
    assert parser.state['points'][0]['lr'] == 3e-5
    assert parser.state['elapsed'] == 450


def test_parser_cache_resume_and_duplicate_records():
    parser = LogParser()
    text = training_line(1640)
    parser.feed(text[:60])
    restored = LogParser(json.loads(json.dumps(parser.state)))
    restored.feed(text[60:] + text)
    assert len(restored.state['points']) == 1
    assert restored.state['points'][0]['step'] == 1640


def test_completion_and_error_are_explicit_not_inferred_from_final_iteration():
    parser = LogParser()
    parser.feed(training_line(30000, '30K'))
    assert not parser.state['complete']
    parser.feed('INFO End of training\n')
    assert parser.state['complete']
    failed = LogParser()
    failed.feed('Traceback (most recent call last):\n')
    assert failed.state['failed']


def test_plain_metrics_and_unrelated_progress():
    parser = LogParser()
    parser.feed('Downloading: 1/2 [00:01<00:02]\n')
    assert parser.state['step'] == 0
    parser.feed('INFO step:20 loss:0.92 lr:3e-7\n')
    assert parser.state['points'][0]['step'] == 20


def test_resuming_retains_history_and_clears_old_controlled_stop_error():
    parser = LogParser()
    parser.feed(training_line(10000, '10K'))
    parser.feed('Traceback (most recent call last):\n')
    assert parser.state['failed']
    parser.feed('USER_RESUME_CHECKPOINT step=10000 target=30000\n')
    parser.feed('USER_RESUME_CHECKPOINT step=10000 target=30000\n')
    parser.feed('Training: 33%|x| 10020/30000 [00:06<1:20:00, 4step/s]INFO step:10K loss:.14\n')
    assert not parser.state['failed']
    assert parser.state['elapsed'] == 456
    assert [point['step'] for point in parser.state['points']] == [10000, 10020]


@pytest.mark.parametrize('value,seconds', [('07:30',450),('1:02:03',3723),('00:00',0)])
def test_duration(value, seconds):
    assert duration_seconds(value) == seconds


def make_monitor(tmp_path):
    return Monitor(argparse.Namespace(cache=tmp_path/'cache.json', key=Path('/tmp/key'),
        control_path='/tmp/master-%C', host='example', interval=10))


def test_poll_incremental_reset_and_auth_failure_preserve_history(tmp_path, monkeypatch):
    monitor = make_monitor(tmp_path)
    name = monitor.runs[0]['id']
    response = {'gpus': [], 'runs': [dict(id=name, alive=True, reset=True, identity='1:2',
        offset=100, text=training_line(1640), checkpoints=[])]}
    calls = []
    def run(command, **kwargs):
        calls.append(kwargs['input'])
        return subprocess.CompletedProcess(command, 0, json.dumps(response), '')
    monkeypatch.setattr(subprocess, 'run', run)
    monitor.poll()
    assert monitor.snapshot()['connected']
    assert monitor.snapshot()['runs'][0]['points'][0]['step'] == 1640
    response['runs'][0].update(reset=False, offset=200, text=training_line(1660))
    monitor.poll()
    assert "'offset': 100" in calls[-1]
    assert len(monitor.snapshot()['runs'][0]['points']) == 2
    restored = make_monitor(tmp_path)
    assert restored.cursors[name]['offset'] == 200
    def denied(*args, **kwargs):
        raise subprocess.CalledProcessError(255, args[0], stderr='Permission denied')
    monkeypatch.setattr(subprocess, 'run', denied)
    monitor.poll()
    assert not monitor.snapshot()['connected']
    assert '认证' in monitor.snapshot()['error']
    assert len(monitor.snapshot()['runs'][0]['points']) == 2
    monkeypatch.setattr(subprocess, 'run', run)
    response['runs'][0].update(reset=True, identity='1:3', offset=20, text=training_line(20,'20'))
    monitor.poll()
    assert [p['step'] for p in monitor.snapshot()['runs'][0]['points']] == [20]


def test_csv_contains_raw_values_and_exact_steps():
    snapshot = {'runs': [dict(id='camera', points=[dict(step=1640, loss=.249, lr=3e-5)])]}
    rows = list(csv.DictReader(io.StringIO(csv_export(snapshot).decode('utf-8-sig'))))
    assert rows[0]['step'] == '1640'
    assert rows[0]['loss'] == '0.249'
    assert rows[0]['run'] == 'camera'
