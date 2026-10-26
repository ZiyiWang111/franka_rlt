"""Verify the installed ACT variational objective and inference on a real batch."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def verify(policy, batch):
    """Exercise the actual model: objective, posterior masks, gradients and zero-latent inference."""
    policy.train()
    captures = {}
    def posterior_hook(module, inputs, output):
        captures['posterior'] = output
    def action_hook(module, inputs, output):
        captures['predicted'] = output
    posterior_handle=policy.model.vae_encoder_latent_output_proj.register_forward_hook(posterior_hook)
    action_handle=policy.model.action_head.register_forward_hook(action_hook)
    try:
        torch.manual_seed(1000)
        loss, components=policy(batch)
        latent=captures['posterior']
        mu, logvar=latent.chunk(2,dim=-1)
        q=torch.distributions.Normal(mu,torch.exp(.5*logvar))
        p=torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(mu))
        kl=torch.distributions.kl_divergence(q,p).sum(-1).mean()
        reconstruction=(F.l1_loss(captures['predicted'],batch['action'],reduction='none')*
                        (~batch['action_is_pad']).unsqueeze(-1)).mean()
        torch.testing.assert_close(loss,reconstruction+policy.config.kl_weight*kl,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(kl.detach(),torch.tensor(components['kld_loss'],device=kl.device),rtol=1e-5,atol=1e-5)
        gradient=torch.autograd.grad(reconstruction,latent,retain_graph=True)[0]
        grad_mu, grad_logvar=gradient.chunk(2,dim=-1)
        assert torch.isfinite(gradient).all() and grad_mu.norm()>0 and grad_logvar.norm()>0
        loss.backward()
        norms={}
        for name in ('vae_encoder_action_input_proj','vae_encoder_robot_state_input_proj',
                     'vae_encoder_latent_output_proj','encoder_latent_input_proj'):
            grad=getattr(policy.model,name).weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0,name
            norms[name]=float(grad.norm())
        loss_value=float(loss.detach())
        policy.zero_grad(set_to_none=True)
        changed=dict(batch)
        changed['action']=batch['action'].clone()
        changed['action'][batch['action_is_pad']]+=100
        assert batch['action_is_pad'].any(), 'Verification batch must include padded future actions'
        with torch.no_grad():
            torch.manual_seed(44); policy(batch); posterior_a=captures['posterior'].clone()
            torch.manual_seed(44); policy(changed); posterior_b=captures['posterior'].clone()
        torch.testing.assert_close(posterior_a,posterior_b,rtol=1e-5,atol=1e-5)
    finally:
        posterior_handle.remove();action_handle.remove()
    prior=[]
    posterior_calls=[]
    def prior_hook(module,inputs): prior.append(inputs[0].detach().clone())
    def encoder_hook(*args): posterior_calls.append(True)
    prior_handle=policy.model.encoder_latent_input_proj.register_forward_pre_hook(prior_hook)
    encoder_handle=policy.model.vae_encoder.register_forward_hook(encoder_hook)
    try:
        observation={key:value for key,value in batch.items() if key not in ('action','action_is_pad')}
        with torch.inference_mode():
            first=policy.predict_action_chunk(observation)
            second=policy.predict_action_chunk(changed)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        assert not posterior_calls and all(torch.count_nonzero(value)==0 for value in prior)
        assert first.shape[-2:]==(policy.config.chunk_size,6) and torch.isfinite(first).all()
    finally:
        prior_handle.remove();encoder_handle.remove()
    return dict(loss=loss_value,reconstruction_l1=components['l1_loss'],kl=components['kld_loss'],
                latent_shape=list(mu.shape),reparameterization_grad_mu=float(grad_mu.norm()),
                reparameterization_grad_logvar=float(grad_logvar.norm()),gradient_norms=norms,
                padded_actions_ignored_by_posterior=True,inference_ignores_actions=True,
                inference_posterior_encoder_calls=0,inference_latent='all zeros',checks_passed=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    from act_rlt.dataset_compat import install_scalar_state_dataset_support
    from lerobot.configs.policies import PreTrainedConfig
    import lerobot.policies.act.modeling_act as modeling
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    torch.set_num_threads(2)
    install_scalar_state_dataset_support()
    config_data=json.loads(args.config.read_text())
    config_data['policy']['device']='cpu'
    config_data['policy']['pretrained_backbone_weights']=None
    import draccus
    config=draccus.decode(PreTrainedConfig,config_data['policy'])
    episodes=config_data['dataset'].get('episodes')
    episode=episodes[0] if episodes else 0
    dataset=LeRobotDataset(args.dataset.name,root=args.dataset,episodes=[episode],
                          delta_timestamps={'action':[i/30 for i in range(config.chunk_size)]},video_backend='pyav')
    sample=dataset[len(dataset)-2]
    stats=dataset.meta.stats
    stats['observation.images.wrist'].update(mean=[[[.485]],[[.456]],[[.406]]],std=[[[.229]],[[.224]],[[.225]]])
    pre,_=make_pre_post_processors(config,dataset_stats=stats)
    batch={key:value.unsqueeze(0) if isinstance(value,torch.Tensor) else [value] for key,value in sample.items()}
    batch=pre(batch)
    policy=modeling.ACTPolicy(config)
    report=verify(policy,batch)
    report.update(implementation_sha256=hashlib.sha256(Path(modeling.__file__).read_bytes()).hexdigest(),
                  input_shape=list(batch['observation.images.wrist'].shape),
                  state_shape=list(batch['observation.state'].shape),
                  valid_future_actions=int((~batch['action_is_pad']).sum()))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
