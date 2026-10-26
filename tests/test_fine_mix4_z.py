import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.eval_fine_mix4_z import masked_error_sums
from scripts.train_fine_mix4_z import load_config


def test_validation_error_excludes_padded_future_actions():
    target=torch.zeros(2,3,6)
    predicted=torch.ones_like(target)
    predicted[0,2]=1000
    predicted[1,1:]=1000
    padding=torch.tensor([[False,False,True],[False,True,True]])
    absolute,squared,count=masked_error_sums(predicted,target,padding)
    assert count==3
    np.testing.assert_array_equal(absolute,np.full(6,3))
    np.testing.assert_array_equal(squared,np.full(6,3))


def test_approved_split_excludes_validation_from_training_and_covers_all_sources():
    root=Path(__file__).resolve().parents[1]/'configs/fine_mix4_z'
    config=load_config(root/'train_config.json')
    split=json.loads((root/'split.json').read_text())
    training=set(config.dataset.episodes)
    validation=set(split['validation_episodes'])
    assert not training & validation
    assert training | validation==set(range(1800))
    assert config.batch_size*2==32 and config.steps==100000
    assert config.scheduler.num_warmup_steps==2000 and config.save_freq==config.steps
    assert len(training)==1620 and len(validation)==180
    for source in split['sources'].values():
        assert not set(source['train_episodes']) & set(source['validation_episodes'])
