"""Audit frozen CVAE extraction and raw encoder-out RL Token inputs, without optimizer steps."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch


def check(checkpoint,root,tokens=1,device='cpu',batch_size=1,reserve_adam_state=False):
    from act_rlt.dataset_compat import install_scalar_state_dataset_support
    from act_rlt.configuration_act_rlt_token import ACTRLTokenConfig
    from act_rlt.modeling_act_rlt_token import ACTRLTokenPolicy
    from act_rlt.processor_act_rlt_token import make_act_rlt_token_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    torch.set_num_threads(2)
    torch.manual_seed(1000)
    if device.startswith('cuda'):
        assert torch.cuda.is_available()
        torch.cuda.reset_peak_memory_stats()
    install_scalar_state_dataset_support()
    act=PreTrainedConfig.from_pretrained(checkpoint,local_files_only=True)
    cfg=ACTRLTokenConfig(act_pretrained_path=str(checkpoint),device=device,
        input_features=act.input_features,output_features=act.output_features,
        chunk_size=act.chunk_size,rl_token_num_rl_tokens=tokens,push_to_hub=False)
    policy=ACTRLTokenPolicy(cfg).to(device).train()
    pre,_=make_act_rlt_token_pre_post_processors(cfg)
    dataset=LeRobotDataset(root.name,root=root,episodes=[0],
        delta_timestamps={'action':[i/30 for i in range(act.chunk_size)]},video_backend='pyav')
    sample=dataset[0]
    batch=pre({k:v.unsqueeze(0).repeat(batch_size,*([1]*v.ndim))
               if isinstance(v,torch.Tensor) else [v]*batch_size for k,v in sample.items()})
    hidden,pos=policy.extract_act_hidden_and_pos(batch)
    assert torch.count_nonzero(pos)>0
    posterior_calls=[];latent_inputs=[];encoder_inputs=[]
    handles=[
        policy._act.model.encoder_latent_input_proj.register_forward_pre_hook(
            lambda m,a:latent_inputs.append(a[0].detach().clone())),
        policy.rl_token.encoder.register_forward_pre_hook(
            lambda m,a:encoder_inputs.append(a[0].detach().clone())),
    ]
    if act.use_vae:
        handles.append(policy._act.model.vae_encoder.register_forward_hook(
            lambda *a:posterior_calls.append(True)))
    try:
        with patch.object(policy.rl_token,'decode',wraps=policy.rl_token.decode) as decode:
            loss,_=policy(batch)
            memory,teacher=decode.call_args.args
            assert memory.shape==(batch_size,tokens,act.dim_model)
            assert torch.equal(teacher,hidden)
        torch.testing.assert_close(encoder_inputs[-1][:,:-tokens],hidden,rtol=0,atol=0)
        assert not torch.equal(encoder_inputs[-1][:,:-tokens],hidden+pos)
        with torch.no_grad():
            policy.encode_multi(batch)
            policy.encode(batch)
        assert all(torch.equal(x[:,:-tokens],hidden) for x in encoder_inputs)
        assert not posterior_calls and latent_inputs
        assert all(torch.count_nonzero(x)==0 for x in latent_inputs)
        assert not policy._act.training
        assert all(not p.requires_grad for p in policy._act.parameters())
        assert torch.isfinite(loss)
        loss.backward()
        grad=policy.rl_token.rl_token_embed.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0
        assert all(p.grad is None for p in policy._act.parameters())
        # Allocate the Adam first/second moments without applying updates.
        states=[torch.zeros_like(p) for p in policy.rl_token.parameters() for _ in range(2)] \
            if reserve_adam_state else []
        if device.startswith('cuda'):torch.cuda.synchronize()
    finally:
        for h in handles:h.remove()
    return dict(checks_passed=True,checkpoint=str(checkpoint),act_use_cvae=act.use_vae,
        encoder_input='encoder_out only',extra_positional_embedding=False,
        reconstruction_target='encoder_out',hidden_shape=list(hidden.shape),
        rl_memory_shape=list(memory.shape),posterior_calls=len(posterior_calls),
        latent='all zeros',act_frozen=True,encoder_equivalence_verified=policy._encoder_equivalence_verified,
        rl_token_gradient_norm=float(grad.norm()),optimizer_steps=0,batch_size=batch_size,
        adam_state_reserved=reserve_adam_state,device=device,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.startswith('cuda') else None,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if device.startswith('cuda') else None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=1)
    p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--reserve-adam-state',action='store_true')
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();report=check(a.checkpoint.resolve(),a.dataset.resolve(),a.tokens,
                                a.device,a.batch_size,a.reserve_adam_state)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
