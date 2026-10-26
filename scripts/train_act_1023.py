"""Train approved ACT/CVAE pairs on all 1023 episodes, without validation."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import draccus
from act_rlt.dataset_compat import install_scalar_state_dataset_support
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig


@dataclass
class TrainConfig(TrainPipelineConfig):
    def validate(self):
        scheduler=self.scheduler
        super().validate()
        if scheduler is not None:self.scheduler=scheduler
        assert isinstance(self.policy,ACTConfig)
        assert set(self.policy.input_features)=={'observation.images.wrist','observation.state'}
        assert tuple(self.policy.robot_state_feature.shape)==(1,)
        assert tuple(self.policy.action_feature.shape)==(6,)
        assert tuple(self.policy.image_features['observation.images.wrist'].shape)==(3,720,1280)
        assert self.policy.chunk_size==self.policy.n_action_steps==4
        assert self.steps==40000 and self.batch_size==16 and self.save_freq==40000
        assert self.dataset.episodes is None and self.eval_freq==0
        if self.policy.use_vae:
            assert self.policy.latent_dim==32 and self.policy.n_vae_encoder_layers==4
            assert self.policy.kl_weight==10


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--validate-only',action='store_true')
    args=p.parse_args()
    cfg=draccus.decode(TrainConfig,json.loads(args.config.read_text()))
    cfg.validate()
    print('ACT_1023_CONFIG',json.dumps(dict(steps=cfg.steps,batch=cfg.batch_size,
        chunk=cfg.policy.chunk_size,use_cvae=cfg.policy.use_vae,validation=False)),flush=True)
    if not args.validate_only:
        install_scalar_state_dataset_support()
        from lerobot.scripts import lerobot_train
        original=lerobot_train.update_policy
        step=0

        def update(*a,**kw):
            global step
            tracker,components=original(*a,**kw)
            step+=1
            if step==1:
                print('ACT_FIRST_BATCH',json.dumps({k:list(v.shape) for k,v in a[2].items()
                    if k in ['observation.state','action','action_is_pad']}),flush=True)
            if cfg.policy.use_vae and step%cfg.log_freq==0:
                print('CVAE_METRICS',json.dumps(dict(step=step,
                    reconstruction_l1=components['l1_loss'],kl=components['kld_loss'],
                    total_loss=components['l1_loss']+cfg.policy.kl_weight*components['kld_loss'])),flush=True)
            return tracker,components

        lerobot_train.update_policy=update
        lerobot_train.train.__wrapped__(cfg)
