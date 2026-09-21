from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor
from typing_extensions import Unpack

from evo_rlt.core.rl_token import RLTokenModule
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy

from act_rlt.act_encoder import (
    capture_act_forward_encoder_hidden,
    extract_act_encoder_hidden_and_pos,
    prepare_act_batch,
)
from act_rlt.configuration_act_rlt_token import ACTRLTokenConfig


log = logging.getLogger(__name__)


class ACTRLTokenPolicy(PreTrainedPolicy):
    """Train an RL-token autoencoder over a frozen ACT encoder's hidden states."""

    config_class = ACTRLTokenConfig
    name = "act_rlt_token"

    def __init__(
        self,
        config: ACTRLTokenConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        *args,
        **kwargs,
    ) -> None:
        del dataset_stats
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.config.validate_features()

        self.rl_token = RLTokenModule(
            token_dim=config.rl_token_dim,
            nhead=config.rl_token_nhead,
            num_enc_layers=config.rl_token_enc_layers,
            num_dec_layers=config.rl_token_dec_layers,
            ff_dim=config.rl_token_ff_dim,
            num_rl_tokens=config.rl_token_num_rl_tokens,
            inference_only=False,
        )

        act = self._load_act_policy()
        self._validate_act_compatibility(act.config)
        for parameter in act.parameters():
            parameter.requires_grad = False
        act.eval()
        # Keep the frozen reference policy out of state_dict and optimizer params.
        object.__setattr__(self, "_act", act)
        self._encoder_equivalence_verified = not config.verify_encoder_equivalence
        self._logged_hidden_shape = False

    def _load_act_policy(self) -> ACTPolicy:
        path = Path(self.config.act_pretrained_path).expanduser().resolve()
        if not (path / "config.json").is_file() or not (path / "model.safetensors").is_file():
            raise FileNotFoundError(f"incomplete ACT pretrained_model directory: {path}")
        act_config = PreTrainedConfig.from_pretrained(path, local_files_only=True)
        if not isinstance(act_config, ACTConfig):
            raise TypeError(f"expected ACT checkpoint at {path}, got {act_config.type!r}")
        act_config.device = self.config.device
        return ACTPolicy.from_pretrained(
            path,
            config=act_config,
            local_files_only=True,
            strict=True,
        )

    def _validate_act_compatibility(self, act_config: ACTConfig) -> None:
        if act_config.use_vae:
            raise ValueError("this Stage-1 implementation requires an ACT checkpoint with use_vae=false")
        if act_config.dim_model != self.config.rl_token_dim:
            raise ValueError(
                f"ACT dim_model={act_config.dim_model} does not match "
                f"rl_token_dim={self.config.rl_token_dim}"
            )
        if act_config.chunk_size != self.config.chunk_size:
            raise ValueError(
                f"ACT chunk_size={act_config.chunk_size} does not match "
                f"Stage-1 chunk_size={self.config.chunk_size}"
            )
        expected_inputs = {key: tuple(value.shape) for key, value in act_config.input_features.items()}
        actual_inputs = {key: tuple(value.shape) for key, value in self.config.input_features.items()}
        if actual_inputs != expected_inputs:
            raise ValueError(
                "dataset input features must exactly match the ACT checkpoint: "
                f"expected={expected_inputs}, actual={actual_inputs}"
            )
        expected_action = tuple(act_config.action_feature.shape)
        actual_action = tuple(self.config.action_feature.shape)
        if actual_action != expected_action:
            raise ValueError(
                f"dataset action shape {actual_action} does not match ACT checkpoint {expected_action}"
            )

    def get_optim_params(self) -> list[dict]:
        return [{"params": list(self.rl_token.parameters()), "lr": self.config.rl_token_lr}]

    def reset(self) -> None:
        pass

    def extract_act_hidden_and_pos(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Return frozen ACT hidden states and ACT's matching memory positions."""
        prepared = prepare_act_batch(self._act, batch)
        with torch.no_grad():
            hidden, pos = extract_act_encoder_hidden_and_pos(self._act.model, prepared)
            if not self._encoder_equivalence_verified:
                reference = capture_act_forward_encoder_hidden(self._act.model, prepared)
                if not torch.equal(hidden, reference):
                    max_error = (hidden - reference).abs().max().item()
                    raise RuntimeError(
                        "ACT encoder-only extraction differs from ACT.forward hook "
                        f"(max_abs_error={max_error:.3e})"
                    )
                self._encoder_equivalence_verified = True
                log.info("verified ACT encoder-only extraction against ACT.forward hook")
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.rl_token_dim:
            raise RuntimeError(
                f"expected ACT hidden [B,S,{self.config.rl_token_dim}], got {tuple(hidden.shape)}"
            )
        if pos.shape[1:] != hidden.shape[1:] or pos.shape[0] not in (1, hidden.shape[0]):
            raise RuntimeError(f"ACT position {tuple(pos.shape)} does not match hidden {tuple(hidden.shape)}")
        if not self._logged_hidden_shape:
            log.info("ACT hidden shape entering RL Token: %s", tuple(hidden.shape))
            self._logged_hidden_shape = True
        return hidden.float(), pos.float()

    def extract_act_hidden(self, batch: dict[str, Tensor]) -> Tensor:
        """Return frozen ACT encoder hidden states as dynamic [B,S,D]."""
        return self.extract_act_hidden_and_pos(batch)[0]

    def encode_multi(self, batch: dict[str, Tensor]) -> Tensor:
        """Stage-1 bottleneck interface: return [B,N,D]."""
        hidden, pos = self.extract_act_hidden_and_pos(batch)
        return self.rl_token.encode_multi(hidden + pos)

    def encode(self, batch: dict[str, Tensor]) -> Tensor:
        """Stage-2 Actor/Critic interface: mean-pool RL tokens to [B,D]."""
        hidden, pos = self.extract_act_hidden_and_pos(batch)
        return self.rl_token.encode(hidden + pos)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        hidden, pos = self.extract_act_hidden_and_pos(batch)
        rl_memory = hidden + pos
        loss_recon = self.rl_token.reconstruction_loss(hidden, encoder_tokens=rl_memory)
        loss = self.config.recon_weight * loss_recon
        return loss, {
            "loss": loss.detach().item(),
            "loss_recon": loss_recon.detach().item(),
            "act_sequence_length": hidden.shape[1],
            "act_hidden_dim": hidden.shape[2],
        }

    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        raise NotImplementedError("ACTRLTokenPolicy is Stage-1 training-only")

    def select_action(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        raise NotImplementedError("ACTRLTokenPolicy is Stage-1 training-only")

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if hasattr(self, "_act"):
            self._act.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        super().cuda(device)
        self._act.cuda(device)
        return self

    def cpu(self):
        super().cpu()
        self._act.cpu()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self._act.eval()
        return self
