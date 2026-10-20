#!/usr/bin/env python3
"""Train fresh RGB + measured absolute TCP Z ACT; output six TCP deltas."""
import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil

import draccus
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from scripts.train_act_camera_only_tmp import resolve_reference
from act_rlt.dataset_compat import install_scalar_state_dataset_support


@dataclass
class TCPZTrainConfig(TrainPipelineConfig):
    def validate(self):
        scheduler = self.scheduler
        super().validate()
        if scheduler is not None:
            self.scheduler = scheduler
        if not isinstance(self.policy, ACTConfig):
            raise ValueError('Expected ACT')
        state = self.policy.robot_state_feature
        if state is None or tuple(state.shape) != (1,) or self.policy.env_state_feature:
            raise ValueError('Only measured scalar TCP Z may enter the state branch')
        if len(self.policy.image_features) != 1 or self.policy.use_vae:
            raise ValueError('Expected one camera and use_vae=False')
        if tuple(self.policy.action_feature.shape) != (6,):
            raise ValueError('Expected six TCP delta outputs')


def build_config(args):
    reference = resolve_reference(args.reference_checkpoint.resolve())
    data = deepcopy(json.loads((reference / 'train_config.json').read_text()))
    root = args.root.resolve()
    info = json.loads((root / 'meta/info.json').read_text())
    if info['features']['observation.state'] != {'dtype': 'float32', 'shape': [1], 'names': ['ee_z']}:
        raise ValueError('Dataset state must contain only ee_z')
    if info['features']['action']['names'] != ['dx', 'dy', 'dz', 'drx', 'dry', 'drz']:
        raise ValueError('Dataset must have six TCP delta labels')
    visual = {key: value for key, value in data['policy']['input_features'].items()
              if value['type'] == 'VISUAL'}
    for key, feature in visual.items():
        c, h, w = feature['shape']
        if info['features'][key]['shape'] != [h, w, c]:
            raise ValueError(f'Camera differs from reference: {key}')
    data['policy']['input_features'] = {**visual, 'observation.state': {'type': 'STATE', 'shape': [1]}}
    data['policy']['output_features'] = {'action': {'type': 'ACTION', 'shape': [6]}}
    data['policy'].update(pretrained_path=None, device=args.device, push_to_hub=False)
    data['policy']['normalization_mapping']['STATE'] = 'MEAN_STD'
    data['dataset'].update(root=str(root), repo_id=root.name)
    data.update(output_dir=str(args.output.resolve()), resume=False, checkpoint_path=None,
                job_name='act_rgb_tcp_z_action6')
    data['wandb']['enable'] = False
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size < 1 or args.batch_size <= 0 or args.batch_size % world_size:
        raise ValueError('Global batch must be positive and divisible by WORLD_SIZE')
    data['batch_size'] = args.batch_size // world_size
    cfg = draccus.decode(TCPZTrainConfig, data)
    cfg.validate()
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-checkpoint', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16, help='GLOBAL batch size across all GPUs')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    cfg = build_config(args)
    if int(os.environ.get('RANK', '0')) == 0:
        print(json.dumps(dict(dataset=str(cfg.dataset.root), output=str(cfg.output_dir),
            inputs=draccus.encode(cfg.policy.input_features), state_names=['ee_z'], state_units='m',
            action_names=['dx', 'dy', 'dz', 'drx', 'dry', 'drz'], steps=cfg.steps,
            global_batch_size=args.batch_size, batch_size_per_process=cfg.batch_size,
            processes=int(os.environ.get('WORLD_SIZE', '1')), scheduler=draccus.encode(cfg.scheduler)),
            indent=2), flush=True)
    if args.validate_only:
        return
    install_scalar_state_dataset_support()
    from lerobot.scripts.lerobot_train import train
    train.__wrapped__(cfg)
    if int(os.environ.get('RANK', '0')) == 0:
        shutil.copy2(__file__, Path(cfg.output_dir) / Path(__file__).name)


if __name__ == '__main__':
    main()
