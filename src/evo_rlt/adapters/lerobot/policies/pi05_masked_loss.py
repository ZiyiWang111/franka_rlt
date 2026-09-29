"""Training-only PI05 loss reduction that excludes episode-tail padding."""

import torch


def reduce_action_losses(losses, action_is_pad=None, reduction="mean"):
    if losses.ndim != 3 or reduction not in ("mean", "none"):
        raise ValueError("Expected [batch, horizon, action_dim] losses and mean/none reduction")
    if action_is_pad is None:
        valid = torch.ones(losses.shape[:2], dtype=torch.bool, device=losses.device)
    else:
        if action_is_pad.shape != losses.shape[:2] or action_is_pad.dtype != torch.bool:
            raise ValueError("action_is_pad must be a boolean [batch, horizon] tensor")
        valid = ~action_is_pad.to(losses.device)
    masked = losses.masked_fill(~valid.unsqueeze(-1), 0)
    counts = valid.sum(dim=1).clamp_min(1).to(losses.dtype)
    per_sample_dim = masked.sum(dim=1) / counts.unsqueeze(-1)
    per_sample = per_sample_dim.mean(dim=-1)
    metrics = {
        "loss": per_sample.mean().detach().item(),
        "loss_per_dim": per_sample_dim.mean(dim=0).detach().float().cpu().tolist(),
    }
    return (per_sample if reduction == "none" else per_sample.mean()), metrics


def pi05_masked_forward(self, batch, reduction="mean"):
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, image_masks = self._preprocess_images(batch)
    actions = self.prepare_action(batch)
    losses = self.model.forward(
        images, image_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], actions
    )
    losses = losses[:, :, :self.config.output_features[ACTION].shape[0]]
    return reduce_action_losses(losses, batch.get("action_is_pad"), reduction)


def enable_masked_pi05_training():
    # Scoped to this Python process; pretrained weight/config formats stay standard.
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    PI05Policy.forward = pi05_masked_forward
