"""Train the approved RGB/Z ACT; evaluate held-out episodes at completion."""
import argparse
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path

import draccus
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from act_rlt.dataset_compat import install_scalar_state_dataset_support


@dataclass
class FineTrainConfig(TrainPipelineConfig):
    def validate(self):
        scheduler = self.scheduler
        super().validate()
        if scheduler is not None:
            self.scheduler = scheduler
        if not isinstance(self.policy, ACTConfig):
            raise ValueError('Expected ACT')
        if set(self.policy.input_features) != {'observation.images.wrist','observation.state'}:
            raise ValueError('Only wrist RGB and measured Z may enter the model')
        if tuple(self.policy.robot_state_feature.shape) != (1,) or tuple(self.policy.action_feature.shape) != (6,):
            raise ValueError('Expected scalar Z and six TCP deltas')
        if tuple(self.policy.input_features['observation.images.wrist'].shape) != (3,720,1280):
            raise ValueError('Expected native wrist RGB 1280x720')
        if self.policy.chunk_size != 16:
            raise ValueError('Expected chunk_size=16')
        if self.policy.use_vae and (self.policy.latent_dim != 32 or self.policy.n_vae_encoder_layers != 4
                                    or self.policy.kl_weight != 10):
            raise ValueError('Approved VAE configuration: latent_dim=32, four encoder layers, KL weight=10')


def load_config(path):
    cfg = draccus.decode(FineTrainConfig,json.loads(path.read_text()))
    cfg.validate()
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--validate-only',action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    world = int(os.environ.get('WORLD_SIZE','1'))
    if int(os.environ.get('RANK','0')) == 0:
        print(json.dumps(dict(config=str(args.config),per_gpu_batch=cfg.batch_size,
            world_size=world,global_batch=cfg.batch_size*world,steps=cfg.steps,
            save_freq=cfg.save_freq,scheduler=draccus.encode(cfg.scheduler)),indent=2),flush=True)
    if not args.validate_only:
        if world != 2 and not (world == 1 and cfg.policy.use_vae):
            raise ValueError('Expected two ranks, or one rank for the VAE experiment')
        if cfg.batch_size * world != 32:
            raise ValueError('Expected global batch size 32')
        install_scalar_state_dataset_support()
        from lerobot.scripts import lerobot_train
        if cfg.policy.use_vae:
            original_update = lerobot_train.update_policy
            completed_updates = 0

            def update_with_vae_metrics(*args, **kwargs):
                global completed_updates
                tracker, components = original_update(*args, **kwargs)
                completed_updates += 1
                if int(os.environ.get('RANK', '0')) == 0 and completed_updates % cfg.log_freq == 0:
                    print('VAE_METRICS',json.dumps(dict(step=completed_updates,
                        reconstruction_l1=components['l1_loss'],kl=components['kld_loss'],
                        kl_weight=cfg.policy.kl_weight,
                        total_loss=components['l1_loss']+cfg.policy.kl_weight*components['kld_loss'])),flush=True)
                return tracker, components

            lerobot_train.update_policy = update_with_vae_metrics
        lerobot_train.train.__wrapped__(cfg)
        if int(os.environ.get('RANK','0')) == 0:
            import torch
            from scripts.eval_fine_mix4_z import evaluate
            gc.collect()
            torch.cuda.empty_cache()
            evaluate(Path(cfg.output_dir),Path(cfg.dataset.root),Path(cfg.dataset.root)/'meta/split.json',
                     device='cuda:0',batch_size=cfg.batch_size,workers=cfg.num_workers)
