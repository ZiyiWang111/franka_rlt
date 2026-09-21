from __future__ import annotations

import einops
import torch
from torch import Tensor

from lerobot.policies.act.modeling_act import ACT, ACTPolicy
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


def prepare_act_batch(policy: ACTPolicy, batch: dict[str, Tensor]) -> dict[str, Tensor | list[Tensor]]:
    """Arrange named camera tensors in the exact order used by ACTPolicy.forward."""
    prepared: dict[str, Tensor | list[Tensor]] = dict(batch)
    if policy.config.image_features:
        prepared[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]
    return prepared


def extract_act_encoder_hidden_and_pos(
    model: ACT, batch: dict[str, Tensor | list[Tensor]]
) -> tuple[Tensor, Tensor]:
    """Return ACT encoder output [B,S,D] and its decoder memory position [1,S,D].

    This mirrors the encoder-input construction in LeRobot 0.5.1 ACT.forward,
    stopping before the ACT action decoder. Sequence length S is derived from
    the tensors and is deliberately not a configuration constant.
    """
    if model.training:
        raise RuntimeError("ACT must be in eval mode for deterministic encoder-only extraction")

    images = batch.get(OBS_IMAGES)
    if images is not None:
        if not isinstance(images, list) or not images:
            raise ValueError(f"{OBS_IMAGES} must be a non-empty list of camera tensors")
        batch_size = images[0].shape[0]
        device = images[0].device
    else:
        env_state = batch.get(OBS_ENV_STATE)
        if not isinstance(env_state, Tensor):
            raise ValueError(f"batch must contain {OBS_IMAGES} or {OBS_ENV_STATE}")
        batch_size = env_state.shape[0]
        device = env_state.device

    # A frozen ACT is always evaluated with the same zero latent used by
    # ACT.forward outside VAE training, including when use_vae=True.
    latent_sample = torch.zeros(
        (batch_size, model.config.latent_dim), dtype=torch.float32, device=device
    )
    encoder_in_tokens = [model.encoder_latent_input_proj(latent_sample)]
    encoder_in_pos_embed = list(model.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

    if model.config.robot_state_feature:
        state = batch.get(OBS_STATE)
        if not isinstance(state, Tensor):
            raise ValueError(f"batch is missing {OBS_STATE}")
        encoder_in_tokens.append(model.encoder_robot_state_input_proj(state))
    if model.config.env_state_feature:
        env_state = batch.get(OBS_ENV_STATE)
        if not isinstance(env_state, Tensor):
            raise ValueError(f"batch is missing {OBS_ENV_STATE}")
        encoder_in_tokens.append(model.encoder_env_state_input_proj(env_state))

    if model.config.image_features:
        if not isinstance(images, list):
            raise ValueError(f"batch is missing {OBS_IMAGES}")
        if len(images) != len(model.config.image_features):
            raise ValueError(
                f"expected {len(model.config.image_features)} cameras, got {len(images)}"
            )
        for image in images:
            cam_features = model.backbone(image)["feature_map"]
            cam_pos_embed = model.encoder_cam_feat_pos_embed(cam_features).to(
                dtype=cam_features.dtype
            )
            cam_features = model.encoder_img_feat_input_proj(cam_features)
            cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
            cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
            encoder_in_tokens.extend(list(cam_features))
            encoder_in_pos_embed.extend(list(cam_pos_embed))

    tokens = torch.stack(encoder_in_tokens, dim=0)
    positions = torch.stack(encoder_in_pos_embed, dim=0)
    hidden_sbd = model.encoder(tokens, pos_embed=positions)
    return hidden_sbd.transpose(0, 1).contiguous(), positions.transpose(0, 1).contiguous()


def extract_act_encoder_hidden(model: ACT, batch: dict[str, Tensor | list[Tensor]]) -> Tensor:
    """Return the frozen ACT encoder output [B,S,D]."""
    return extract_act_encoder_hidden_and_pos(model, batch)[0]


def capture_act_forward_encoder_hidden(
    model: ACT, batch: dict[str, Tensor | list[Tensor]]
) -> Tensor:
    """Capture ACT.forward's encoder_out for one numerical-equivalence check."""
    captured: list[Tensor] = []

    def hook(_module, _args, output: Tensor) -> None:
        captured.append(output.detach())

    handle = model.encoder.register_forward_hook(hook)
    try:
        model(batch)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one ACT encoder output, captured {len(captured)}")
    return captured[0].transpose(0, 1).contiguous()


def predict_act_chunk_with_encoder_hidden(
    policy: ACTPolicy, batch: dict[str, Tensor]
) -> tuple[Tensor, Tensor]:
    """Run ACT once and return ``(action_chunk, encoder_hidden)``.

    Stage 2 needs both ACT's action proposal and the exact encoder output used
    to produce that proposal. Capturing the encoder during
    :meth:`ACTPolicy.predict_action_chunk` avoids a second ResNet/transformer
    forward and guarantees that both tensors come from the same observation.

    Returns:
        action_chunk: ``[B,H,A]`` in ACT's normalized action space.
        encoder_hidden: ``[B,S,D]`` in the same token order as Stage 1.
    """
    if policy.training or policy.model.training:
        raise RuntimeError("ACT must be in eval mode for Stage-2 extraction")

    captured: list[Tensor] = []

    def hook(_module, _args, output: Tensor) -> None:
        captured.append(output.detach())

    handle = policy.model.encoder.register_forward_hook(hook)
    try:
        action_chunk = policy.predict_action_chunk(batch)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one ACT encoder output, captured {len(captured)}")
    hidden = captured[0].transpose(0, 1).contiguous()
    return action_chunk, hidden


def predict_act_chunk_with_encoder_hidden_and_pos(
    policy: ACTPolicy, batch: dict[str, Tensor]
) -> tuple[Tensor, Tensor, Tensor]:
    """Capture ACT's action, encoder output and corresponding decoder memory position."""
    if policy.training or policy.model.training:
        raise RuntimeError("ACT must be in eval mode for Stage-2 extraction")

    hidden_outputs: list[Tensor] = []
    positions: list[Tensor] = []

    def encoder_hook(_module, _args, output: Tensor) -> None:
        hidden_outputs.append(output.detach())

    def decoder_hook(_module, _args, kwargs) -> None:
        positions.append(kwargs["encoder_pos_embed"].detach())

    encoder_handle = policy.model.encoder.register_forward_hook(encoder_hook)
    decoder_handle = policy.model.decoder.register_forward_pre_hook(decoder_hook, with_kwargs=True)
    try:
        action_chunk = policy.predict_action_chunk(batch)
    finally:
        encoder_handle.remove()
        decoder_handle.remove()
    if len(hidden_outputs) != 1 or len(positions) != 1:
        raise RuntimeError(
            f"expected one ACT encoder output and position, got {len(hidden_outputs)} and {len(positions)}"
        )
    hidden = hidden_outputs[0].transpose(0, 1).contiguous()
    pos = positions[0].transpose(0, 1).contiguous()
    if pos.shape[1:] != hidden.shape[1:] or pos.shape[0] not in (1, hidden.shape[0]):
        raise RuntimeError(f"ACT position {tuple(pos.shape)} does not match hidden {tuple(hidden.shape)}")
    return action_chunk, hidden, pos
