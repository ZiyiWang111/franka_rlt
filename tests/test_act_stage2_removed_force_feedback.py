from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from act_rlt import infer_stage2, train_stage2
from act_rlt.stage2 import ACTStage2Config, Stage2Learner, OnlineStage2Metrics, strip_removed_force_feedback_config
from act_rlt.stage2_networks import Stage2ChunkActor
from evo_rlt.core.replay_buffer import ReplayBuffer


LEGACY_FORCE_CONFIG = dict(use_force_feedback=False, force_scale_n=10., torque_scale_nm=1.,
                           wrench_offset=[0.] * 6, wrench_clip=5.)


def actor(config):
    return Stage2ChunkActor(config.state_dim, config.proprio_dim, config.chunk_dim,
                            input_fusion=config.input_fusion)


@pytest.mark.parametrize('layout', ['balanced', 'token_identity'])
def test_legacy_disabled_force_metadata_resumes_and_infers(monkeypatch, tmp_path, layout):
    config = ACTStage2Config(stage1_checkpoint='/tmp/stage1', act_checkpoint='/tmp/act',
                            device='cpu', proprio_dim=1, input_fusion=layout)
    learner = Stage2Learner(SimpleNamespace(actor=actor(config)), config)
    learner.initialized = True
    train_stage2.save_checkpoint(tmp_path, config, learner, ReplayBuffer(), OnlineStage2Metrics())
    checkpoint = tmp_path / 'checkpoints/latest.pt'
    saved = torch.load(checkpoint, weights_only=False)
    saved['config'].update(LEGACY_FORCE_CONFIG)
    torch.save(saved, checkpoint)
    clean = strip_removed_force_feedback_config(saved['config'])
    assert ACTStage2Config(**clean).state_dim == 769
    assert saved['config']['use_force_feedback'] is False  # Cleanup never mutates loaded metadata.
    loaded = Stage2Learner(SimpleNamespace(actor=actor(config)), config)
    train_stage2.load_checkpoint(checkpoint, config, loaded, ReplayBuffer())
    for k, v in learner.policy.actor.state_dict().items():
        torch.testing.assert_close(loaded.policy.actor.state_dict()[k], v, atol=0, rtol=0)

    class FakePolicy(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.actor = actor(config)
    monkeypatch.setattr(infer_stage2, 'ACTStage2Policy', FakePolicy)
    monkeypatch.setattr(infer_stage2, 'resolve_pretrained_model', Path)
    policy, inferred_config = infer_stage2.load_frozen_policy(checkpoint, device='cpu')
    assert inferred_config.state_dim == 769
    assert policy.actor.fusion.output_dim == (384 if layout == 'balanced' else 928)
    assert not hasattr(policy.actor.fusion, 'wrench')
    assert all(not parameter.requires_grad for parameter in policy.parameters())
    for k, v in learner.policy.actor.state_dict().items():
        torch.testing.assert_close(policy.actor.state_dict()[k], v, atol=0, rtol=0)


def test_removed_force_architecture_cannot_be_silently_loaded():
    with pytest.raises(ValueError, match='removed Stage-2 force-feedback branch'):
        strip_removed_force_feedback_config(dict(LEGACY_FORCE_CONFIG, use_force_feedback=True))


@pytest.mark.parametrize('flag,value', [('--use-force-feedback', []), ('--force-scale-n', ['10']),
                                      ('--torque-scale-nm', ['1']), ('--wrench-clip', ['5']),
                                      ('--wrench-offset', ['0'] * 6)])
def test_removed_force_cli_is_rejected(flag, value):
    with pytest.raises(SystemExit):
        train_stage2.build_parser().parse_args(['--stage1-checkpoint', 'a', '--act-checkpoint', 'b', flag, *value])
