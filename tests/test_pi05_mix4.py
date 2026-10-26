import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.train_pi05_mix4 import validation


def test_validation_masks_padding_groups_sources_and_preserves_rng(tmp_path):
    split=json.loads(Path('datasets/ethernet_fine/fine_mix4_z/meta/split.json').read_text())
    root=tmp_path/'data';(root/'meta').mkdir(parents=True)
    (root/'meta/split.json').write_text(json.dumps(split))
    (root/'meta/info.json').write_text(json.dumps({'fps':30}))
    out=tmp_path/'run'
    eps=[s['validation_episodes'][0]+s['merged_episode_offset'] for s in split['sources'].values()]

    class Dataset:
        def __init__(self,*a,**kw):
            assert len(kw['episodes'])==40
        def __len__(self):return 4
        def __getitem__(self,i):
            action=torch.ones(16,6);action[8:]=999
            return {'action':action,'action_is_pad':torch.arange(16)>=8,'episode_index':eps[i]}

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.p=torch.nn.Parameter(torch.zeros(()))
        def predict_action_chunk(self,batch):
            return torch.randn_like(batch['action'])*0

    cfg=SimpleNamespace(dataset=SimpleNamespace(root=root),output_dir=out,
                        policy=SimpleNamespace(chunk_size=16,num_inference_steps=10))
    policy=Policy().train()
    torch.manual_seed(12);rng=torch.get_rng_state().clone()
    def synchronous_loader(*a,**kw):
        # Exercise metrics/RNG here without worker IPC in the sandbox.
        kw['num_workers']=0
        return torch.utils.data.DataLoader(*a,**kw)

    with patch('lerobot.datasets.lerobot_dataset.LeRobotDataset',Dataset), \
         patch('scripts.train_pi05_mix4.DataLoader',synchronous_loader):
        validation(policy,lambda x:x,lambda x:x,cfg,5000)
    assert policy.training and torch.equal(rng,torch.get_rng_state())
    metrics=json.loads((out/'validation/step_005000.json').read_text())['metrics']
    assert metrics['overall']['valid_future_actions']==32
    for name,m in metrics.items():
        assert m['mae']==[1.0]*6 and m['rmse']==[1.0]*6
        assert m['frames']==(4 if name=='overall' else 1)
