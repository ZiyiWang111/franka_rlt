"""One global claim must prevent concurrent/restarted watchers launching twice."""
import importlib.util
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

SCRIPTS = Path(__file__).parents[1] / 'scripts'


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


claim_module = module('act_claim_server')


def payload(owner):
    return dict(owner=owner, host='server6' if int(owner) % 2 else 'server7', gpu_uuids=['GPU-a'])


def test_simultaneous_claims_have_only_one_winner(tmp_path):
    store = claim_module.ClaimStore(tmp_path / 'claim.json')
    with ThreadPoolExecutor(max_workers=12) as workers:
        results = list(workers.map(lambda n: store.claim(payload(str(n))), range(40)))
    assert len({r['owner'] for r in results}) == 1
    winner = results[0]['owner']
    assert claim_module.ClaimStore(tmp_path / 'claim.json').claim(payload('99'))['owner'] == winner


def test_only_owner_can_update_and_failure_does_not_release(tmp_path):
    store = claim_module.ClaimStore(tmp_path / 'claim.json')
    store.claim(payload('1'))
    with pytest.raises(PermissionError):
        store.update(dict(owner='2', status='RUNNING'))
    store.update(dict(owner='1', status='FAILED', exit_code=1))
    result = store.claim(payload('2'))
    assert result['owner'] == '1' and result['status'] == 'FAILED'


def test_eligible_single_or_two_gpus(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    watcher = module('watch_act_z_orientation')
    gpus = [dict(index=i, uuid=f'GPU-{i}') for i in range(3)]
    assert watcher.eligible_gpus(gpus, {'GPU-0': 2}) == []
    assert watcher.eligible_gpus(gpus, {'GPU-1': 3}) == [gpus[1]]
    assert watcher.eligible_gpus(gpus, {g['uuid']: 3 for g in gpus}) == gpus[:2]
