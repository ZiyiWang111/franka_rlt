"""Train confirmed wrist + Z/orientation ACT, global batch 64 on one/two GPUs."""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path

import draccus
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig


@dataclass
class OrientationConfig(TrainPipelineConfig):
    def validate(self):
        scheduler = self.scheduler
        super().validate()
        if scheduler is not None:
            self.scheduler = scheduler
        assert isinstance(self.policy, ACTConfig) and not self.policy.use_vae
        assert set(self.policy.input_features) == {'observation.images.wrist', 'observation.state'}
        assert tuple(self.policy.robot_state_feature.shape) == (4,)
        assert tuple(self.policy.action_feature.shape) == (6,)
        assert self.policy.chunk_size == self.policy.n_action_steps == 4
        assert self.steps == 50000 and self.save_freq == 10000 and self.eval_freq == 0
        assert self.dataset.episodes is None
        info = json.loads((Path(self.dataset.root) / 'meta/info.json').read_text())
        assert info['features']['observation.state']['names'] == ['ee_z', 'ee_rx', 'ee_ry', 'ee_rz']
        assert info['features']['action']['names'] == ['dx', 'dy', 'dz', 'drx', 'dry', 'drz']
        assert self.dataset.image_transforms.enable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--world-size', type=int, choices=[1, 2], help='Only used for validation')
    args = parser.parse_args()
    data = json.loads(args.config.read_text())
    assert data['batch_size'] == 64, 'Config batch_size is GLOBAL'
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if args.validate_only and args.world_size:
        world = args.world_size
    assert world in (1, 2)
    data['batch_size'] = 64 // world
    cfg = draccus.decode(OrientationConfig, data)
    cfg.validate()
    print('ACT_ORIENTATION_CONFIG', json.dumps(dict(global_batch=64, world_size=world,
          per_gpu_batch=cfg.batch_size, steps=cfg.steps, use_cvae=cfg.policy.use_vae,
          state=['ee_z', 'ee_rx', 'ee_ry', 'ee_rz'])), flush=True)
    if not args.validate_only:
        from lerobot.scripts.lerobot_train import train
        train.__wrapped__(cfg)


if __name__ == '__main__':
    main()
