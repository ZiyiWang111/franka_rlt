from __future__ import annotations

from pathlib import Path

import pytest
import torch

from act_rlt import register
from act_rlt.act_encoder import (
    capture_act_forward_encoder_hidden,
    extract_act_encoder_hidden_and_pos,
    prepare_act_batch,
    predict_act_chunk_with_encoder_hidden_and_pos,
)
from act_rlt.configuration_act_rlt_token import ACTRLTokenConfig
from evo_rlt.core.rl_token import RLTokenModule
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import get_policy_class
from lerobot.utils.constants import OBS_STATE


ACT_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "outputs/act_rlt_001_state7_act/checkpoints/last/pretrained_model"
)


def test_policy_registers_with_lerobot_dynamic_factory() -> None:
    register()
    assert PreTrainedConfig.get_choice_class("act_rlt_token") is ACTRLTokenConfig
    assert get_policy_class("act_rlt_token").__name__ == "ACTRLTokenPolicy"


def test_stage1_and_stage2_rl_token_shapes_are_distinct() -> None:
    module = RLTokenModule(
        token_dim=16,
        nhead=2,
        num_enc_layers=1,
        num_dec_layers=1,
        ff_dim=32,
        num_rl_tokens=1,
    ).eval()
    act_hidden = torch.randn(3, 11, 16)

    assert module.encode_multi(act_hidden).shape == (3, 1, 16)
    assert module.encode(act_hidden).shape == (3, 16)
    assert module.reconstruction_loss(act_hidden).ndim == 0


def test_position_correspondence_changes_rl_token_output() -> None:
    torch.manual_seed(11)
    module = RLTokenModule(
        token_dim=16, nhead=2, num_enc_layers=1, num_dec_layers=1, ff_dim=32
    ).eval()
    hidden = torch.randn(1, 11, 16)
    pos = torch.randn(1, 11, 16)
    permutation = torch.randperm(hidden.shape[1])
    with torch.no_grad():
        original = module.encode(hidden + pos)
        permuted_hidden = module.encode(hidden[:, permutation] + pos)
        permuted_pairs = module.encode((hidden + pos)[:, permutation])
    assert (original - permuted_hidden).abs().max().item() > 1e-5
    torch.testing.assert_close(original, permuted_pairs, atol=1e-6, rtol=1e-6)


def test_config_defaults_to_confirmed_stage1_architecture() -> None:
    config = ACTRLTokenConfig(
        act_pretrained_path="/tmp/act",
        device="cpu",
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.wrist": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 480, 640)
            ),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    assert config.rl_token_dim == 768
    assert config.rl_token_enc_layers == 3
    assert config.rl_token_dec_layers == 3
    assert config.rl_token_ff_dim == 4096
    assert config.rl_token_num_rl_tokens == 1
    assert config.get_optimizer_preset().lr == pytest.approx(2e-4)
    assert config.get_optimizer_preset().grad_clip_norm == pytest.approx(1.0)


@pytest.mark.skipif(not ACT_CHECKPOINT.is_dir(), reason="local ACT checkpoint is unavailable")
def test_real_act_encoder_only_helper_matches_forward_hook() -> None:
    config = PreTrainedConfig.from_pretrained(ACT_CHECKPOINT, local_files_only=True)
    config.device = "cpu"
    policy = ACTPolicy.from_pretrained(
        ACT_CHECKPOINT,
        config=config,
        local_files_only=True,
        strict=True,
    ).eval()
    torch.manual_seed(7)
    batch = {OBS_STATE: torch.randn(1, 7)}
    for key, feature in config.image_features.items():
        batch[key] = torch.randn(1, *feature.shape)
    prepared = prepare_act_batch(policy, batch)
    decoder_positions = []

    def capture_decoder_pos(_module, _args, kwargs):
        decoder_positions.append(kwargs["encoder_pos_embed"].detach())

    handle = policy.model.decoder.register_forward_pre_hook(capture_decoder_pos, with_kwargs=True)

    try:
        with torch.inference_mode():
            baseline_action = policy.predict_action_chunk(batch)
            extracted, pos = extract_act_encoder_hidden_and_pos(policy.model, prepared)
            captured = capture_act_forward_encoder_hidden(policy.model, prepared)
            hooked_action, hooked_hidden, hooked_pos = predict_act_chunk_with_encoder_hidden_and_pos(
                policy, batch
            )
    finally:
        handle.remove()

    assert extracted.shape == (1, 602, 768)  # empirical for this checkpoint, not an API constant
    assert torch.equal(extracted, captured)
    assert pos.shape == (1, 602, 768)
    assert torch.equal(pos.transpose(0, 1), decoder_positions[0])
    assert torch.equal(hooked_pos, pos)
    assert torch.equal(hooked_hidden, extracted)
    assert torch.equal(hooked_action, baseline_action)
