import datasets
import pytest

from lerobot.datasets import dataset_reader
from scripts.train_act_tcp_z_tmp import install_scalar_state_dataset_support


@pytest.mark.parametrize('state_dim', [1, 6, 7])
def test_scalar_state_fix_preserves_vectors_and_scalar_indices(monkeypatch, state_dim):
    original = dataset_reader.get_hf_features_from_features
    monkeypatch.setattr(dataset_reader, 'get_hf_features_from_features', original)
    features = {
        'observation.state': {'shape': (state_dim,), 'dtype': 'float32'},
        'timestamp': {'shape': (1,), 'dtype': 'float32'},
        'index': {'shape': (1,), 'dtype': 'int64'},
        'action': {'shape': (6,), 'dtype': 'float32'},
    }
    install_scalar_state_dataset_support()
    installed = dataset_reader.get_hf_features_from_features
    install_scalar_state_dataset_support()
    assert dataset_reader.get_hf_features_from_features is installed
    actual = installed(features)
    assert isinstance(actual['observation.state'], datasets.Sequence)
    assert actual['observation.state'].length == state_dim
    assert actual['timestamp'] == original(features)['timestamp']
    assert actual['index'] == original(features)['index']
    assert actual['action'] == original(features)['action']
    table = datasets.Dataset.from_dict({
        'observation.state': [[.4] * state_dim, [.39] * state_dim],
        'timestamp': [0., 1. / 30], 'index': [0, 1],
        'action': [[0.] * 6, [0.] * 6],
    }, features=actual)
    assert len(table[0]['observation.state']) == state_dim
