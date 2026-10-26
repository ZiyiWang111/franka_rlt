"""Final offline action prediction metrics on held-out episodes, including per-source metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def masked_error_sums(predicted, target, padding):
    """Sum errors only for real future actions, excluding episode-end padding."""
    mask = ~padding.bool()
    error = predicted - target
    if predicted.shape != target.shape or mask.shape != predicted.shape[:2]:
        raise ValueError('Unexpected chunk or padding shape')
    valid = mask.unsqueeze(-1)
    return (error.abs()*valid).sum((0,1)).double().cpu().numpy(), (error.square()*valid).sum((0,1)).double().cpu().numpy(), int(mask.sum())


def evaluate(run,root,split_path,device='cuda:0',batch_size=16,workers=4):
    from act_rlt.dataset_compat import install_scalar_state_dataset_support
    from act_rlt.infer import resolve_checkpoint
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    install_scalar_state_dataset_support()
    checkpoint=resolve_checkpoint(run)
    config=PreTrainedConfig.from_pretrained(str(checkpoint),local_files_only=True)
    config.device=device
    policy=ACTPolicy.from_pretrained(str(checkpoint),config=config,local_files_only=True).to(device).eval()
    pre,post=make_pre_post_processors(config,pretrained_path=str(checkpoint),
        preprocessor_overrides={'device_processor':{'device':device}})
    split=json.loads(split_path.read_text())
    dataset=LeRobotDataset(root.name,root=root,episodes=split['validation_episodes'],
        delta_timestamps={'action':[i/dataset_fps(root) for i in range(config.chunk_size)]},
        image_transforms=None,video_backend='pyav')
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=device.startswith('cuda'))
    metrics={name:{'absolute':np.zeros(6),'squared':np.zeros(6),'normalized':np.zeros(6),'valid':0,'frames':0}
             for name in ['overall',*split['sources']]}
    source_names=list(split['sources'])
    ends=np.array([entry['merged_episode_offset']+len(entry['train_episodes'])+len(entry['validation_episodes'])
                   for entry in split['sources'].values()])
    with torch.inference_mode():
        for batch in loader:
            target=batch['action'].clone().to(device)
            padding=batch['action_is_pad'].clone().to(device)
            sources=np.searchsorted(ends,batch['episode_index'].cpu().numpy(),side='right')
            prepared=pre(batch)
            normalized_target=prepared['action'].clone()
            predicted=policy.predict_action_chunk(prepared)
            # Saved postprocessors normally return actions to CPU for robots.
            # Keep predictions on the metric tensors' device during GPU eval.
            raw=post(predicted.clone()).to(device)
            for name in metrics:
                selected=np.ones(len(target),dtype=bool) if name=='overall' else sources==source_names.index(name)
                if not selected.any():
                    continue
                selection=torch.as_tensor(selected,device=device)
                absolute,squared,count=masked_error_sums(raw[selection],target[selection],padding[selection])
                normalized,_,_=masked_error_sums(predicted[selection],normalized_target[selection],padding[selection])
                m=metrics[name];m['absolute']+=absolute;m['squared']+=squared;m['normalized']+=normalized
                m['valid']+=count;m['frames']+=int(selected.sum())
    result={}
    for name,m in metrics.items():
        count=m['valid']
        if count==0:
            raise ValueError(f'No validation samples for {name}')
        result[name]=dict(frames=m['frames'],valid_future_actions=count,
            mae=(m['absolute']/count).tolist(),rmse=np.sqrt(m['squared']/count).tolist(),
            normalized_l1=float((m['normalized']/count).mean()))
    assert result['overall']['frames']==split['validation_frames']
    report=dict(checkpoint=str(checkpoint),action_names=['dx','dy','dz','drx','dry','drz'],
                units=['m','m','m','rad','rad','rad'],padding_excluded=True,image_augmentation=False,metrics=result)
    path=run/'validation_metrics.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print('FINAL_VALIDATION',json.dumps(report),flush=True)
    return report


def dataset_fps(root):
    return json.loads((root/'meta/info.json').read_text())['fps']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--split',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    evaluate(args.run,args.root,args.split,args.device)
