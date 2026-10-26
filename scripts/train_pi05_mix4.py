"""Train RGB/Z PI05 with masked flow loss and held-out action validation."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import draccus
import numpy as np
import torch
from torch.utils.data import DataLoader

from act_rlt.dataset_compat import install_scalar_state_dataset_support
from evo_rlt.adapters.lerobot.policies.pi05_masked_loss import enable_masked_pi05_training
from lerobot.configs.train import TrainPipelineConfig
from scripts.eval_fine_mix4_z import masked_error_sums


def validation(policy,pre,post,cfg,step,full=False):
    """Use fixed data and RNG; preserve training RNG and module mode."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root=Path(cfg.dataset.root)
    split=json.loads((root/'meta/split.json').read_text())
    episodes=split['validation_episodes'] if full else [
        ep+s['merged_episode_offset'] for s in split['sources'].values()
        for ep in s['validation_episodes'][:10]]
    fps=json.loads((root/'meta/info.json').read_text())['fps']
    dataset=LeRobotDataset(root.name,root=root,episodes=episodes,
        delta_timestamps={'action':[i/fps for i in range(cfg.policy.chunk_size)]},
        image_transforms=None,video_backend='pyav')
    # A separate generator keeps DataLoader initialization out of training RNG.
    loader=DataLoader(dataset,batch_size=4,shuffle=False,num_workers=2,
                      generator=torch.Generator().manual_seed(1000))
    names=list(split['sources'])
    ends=np.array([s['merged_episode_offset']+len(s['train_episodes'])+len(s['validation_episodes'])
                   for s in split['sources'].values()])
    sums={n:dict(abs=np.zeros(6),square=np.zeros(6),valid=0,frames=0) for n in ['overall',*names]}
    mode=policy.training
    device=next(policy.parameters()).device
    try:
        with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []),torch.inference_mode():
            torch.manual_seed(1000)
            policy.eval()
            for batch in loader:
                target=batch['action'].clone().cpu()
                padding=batch['action_is_pad'].clone().cpu()
                sources=np.searchsorted(ends,batch['episode_index'].cpu().numpy(),side='right')
                predicted=post(policy.predict_action_chunk(pre(batch))).float().cpu()
                for name,m in sums.items():
                    selected=np.ones(len(target),dtype=bool) if name=='overall' else sources==names.index(name)
                    if not selected.any():continue
                    a,s,n=masked_error_sums(predicted[selected],target[selected],padding[selected])
                    m['abs']+=a;m['square']+=s;m['valid']+=n;m['frames']+=int(selected.sum())
    finally:
        policy.train(mode)
    metrics={n:dict(frames=m['frames'],valid_future_actions=m['valid'],
        mae=(m['abs']/m['valid']).tolist(),rmse=np.sqrt(m['square']/m['valid']).tolist())
        for n,m in sums.items()}
    if full:assert metrics['overall']['frames']==split['validation_frames']
    report=dict(step=step,full=full,seed=1000,episodes=len(episodes),padding_excluded=True,
        action_names=['dx','dy','dz','drx','dry','drz'],units=['m']*3+['rad']*3,
        num_inference_steps=cfg.policy.num_inference_steps,metrics=metrics)
    out=Path(cfg.output_dir);(out/'validation').mkdir(parents=True,exist_ok=True)
    (out/'validation'/f'step_{step:06d}.json').write_text(json.dumps(report,indent=2)+'\n')
    if full:(out/'validation_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PI05_VALIDATION',json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    # Import registers PI05 before polymorphic config decoding.
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    cfg=draccus.decode(TrainPipelineConfig,json.loads(args.config.read_text()))
    cfg.validate()
    assert isinstance(cfg.policy,PI05Config)
    assert cfg.policy.robot_state_feature.shape==(1,) and cfg.policy.action_feature.shape==(6,)
    assert set(cfg.policy.input_features)=={'observation.images.wrist','observation.state'}
    assert cfg.policy.chunk_size==16 and cfg.batch_size==16
    assert cfg.steps==30000 and cfg.save_freq==30000
    assert cfg.policy.dtype=='bfloat16' and cfg.policy.gradient_checkpointing
    assert not cfg.policy.train_expert_only and not cfg.policy.freeze_vision_encoder
    assert not cfg.policy.use_relative_actions
    print('PI05_CONFIG',json.dumps(dict(chunk=cfg.policy.chunk_size,per_gpu_batch=cfg.batch_size,
        world_size=os.environ.get('WORLD_SIZE','1'),steps=cfg.steps,lr=cfg.optimizer.lr)),flush=True)
    if args.validate_only:return
    assert int(os.environ.get('WORLD_SIZE','1'))==2
    install_scalar_state_dataset_support()
    enable_masked_pi05_training()
    from lerobot.scripts import lerobot_train
    original_update=lerobot_train.update_policy
    original_processors=lerobot_train.make_pre_post_processors
    original_save=lerobot_train.save_checkpoint
    processors={}

    def capture_processors(*a,**kw):
        pre,post=original_processors(*a,**kw)
        processors.update(pre=pre,post=post)
        return pre,post

    completed=0
    def update(*a,**kw):
        nonlocal completed
        tracker,components=original_update(*a,**kw)
        completed+=1
        if completed==1:
            print('PI05_FIRST_BATCH',json.dumps({k:list(v.shape) for k,v in a[2].items()
                if k in ['observation.state','action','action_is_pad']}),flush=True)
        if completed%5000==0 and completed<cfg.steps:
            accelerator=kw['accelerator']
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                validation(accelerator.unwrap_model(a[1]),processors['pre'],processors['post'],
                           cfg,completed)
            accelerator.wait_for_everyone()
        return tracker,components

    lerobot_train.make_pre_post_processors=capture_processors
    lerobot_train.update_policy=update

    def save_and_validate(*a,**kw):
        original_save(*a,**kw)
        if kw['step']==cfg.steps:
            validation(kw['policy'],processors['pre'],processors['post'],cfg,cfg.steps,full=True)

    lerobot_train.save_checkpoint=save_and_validate
    # Rank 1 waits while rank 0 evaluates or writes the full checkpoint.
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
    accelerator=Accelerator(step_scheduler_with_optimizer=False,kwargs_handlers=[
        DistributedDataParallelKwargs(find_unused_parameters=True),
        InitProcessGroupKwargs(timeout=timedelta(hours=2))])
    lerobot_train.train.__wrapped__(cfg,accelerator=accelerator)


if __name__=='__main__':main()
