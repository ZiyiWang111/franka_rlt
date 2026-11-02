"""Train wrist RGB + measured ee_z -> TCP delta ACT, from a validated config."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import draccus
from act_rlt.dataset_compat import install_scalar_state_dataset_support
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig


@dataclass
class FineTrainConfig(TrainPipelineConfig):
    def validate(self):
        scheduler = self.scheduler
        super().validate()
        if scheduler is not None:
            self.scheduler = scheduler
        assert isinstance(self.policy, ACTConfig)
        assert set(self.policy.input_features) == {'observation.images.wrist', 'observation.state'}
        assert tuple(self.policy.robot_state_feature.shape) == (1,)
        assert tuple(self.policy.action_feature.shape) == (6,)
        assert self.policy.chunk_size == self.policy.n_action_steps == 4
        assert self.steps == self.save_freq and self.steps in (30000, 50000)
        assert self.batch_size in (16, 32)
        assert self.dataset.episodes is None and self.eval_freq == 0
        assert self.policy.use_vae and self.policy.latent_dim == 32
        assert self.policy.n_vae_encoder_layers == 4 and self.policy.kl_weight == 10
        meta = Path(self.dataset.root) / 'meta'
        info = json.loads((meta / 'info.json').read_text())
        assert info['features']['observation.state']['names'] == ['ee_z']
        assert info['features']['action']['names'] == ['dx', 'dy', 'dz', 'drx', 'dry', 'drz']
        h, w, channels = info['features']['observation.images.wrist']['shape']
        assert tuple(self.policy.image_features['observation.images.wrist'].shape) == (channels, h, w)
        assert (meta / 'stats.json').is_file()
        assert list((Path(self.dataset.root) / 'data').rglob('*.parquet'))
        assert list((Path(self.dataset.root) / 'videos/observation.images.wrist').rglob('*.mp4'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    cfg = draccus.decode(FineTrainConfig, json.loads(args.config.read_text()))
    cfg.validate()
    print('ACT_FINE_CONFIG', json.dumps(dict(job=cfg.job_name, steps=cfg.steps,
          batch=cfg.batch_size, chunk=cfg.policy.chunk_size, cvae=cfg.policy.use_vae,
          validation=False, image_augmentation=cfg.dataset.image_transforms.enable)), flush=True)
    if args.validate_only:
        return
    install_scalar_state_dataset_support()
    from lerobot.scripts import lerobot_train
    original = lerobot_train.update_policy
    step = 0

    def update(*a, **kw):
        nonlocal step
        tracker, parts = original(*a, **kw)
        step += 1
        if step == 1:
            print('ACT_FIRST_BATCH', json.dumps({k: list(v.shape) for k, v in a[2].items()
                  if k in ('observation.state', 'action', 'action_is_pad')}), flush=True)
        if step == 1 or step % cfg.log_freq == 0:
            print('CVAE_METRICS', json.dumps(dict(step=step,
                  reconstruction_l1=parts['l1_loss'], kl=parts['kld_loss'],
                  total_loss=parts['l1_loss'] + cfg.policy.kl_weight * parts['kld_loss'])), flush=True)
        return tracker, parts

    lerobot_train.update_policy = update
    lerobot_train.train.__wrapped__(cfg)


if __name__ == '__main__':
    main()
