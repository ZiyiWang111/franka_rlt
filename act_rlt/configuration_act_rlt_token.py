from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig, LRSchedulerConfig


@PreTrainedConfig.register_subclass("act_rlt_token")
@dataclass
class ACTRLTokenConfig(PreTrainedConfig):
    """Stage-1 reconstruction training with a frozen ACT encoder."""

    act_pretrained_path: str = ""

    rl_token_dim: int = 768
    rl_token_nhead: int = 8
    rl_token_enc_layers: int = 3
    rl_token_dec_layers: int = 3
    rl_token_ff_dim: int = 4096
    rl_token_num_rl_tokens: int = 1

    recon_weight: float = 1.0
    verify_encoder_equivalence: bool = True

    # Used only to ask LeRobot for the same action window as the ACT checkpoint.
    # Actions remain unused by ACT encoder-only reconstruction training.
    chunk_size: int = 16

    rl_token_lr: float = 2e-4
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.act_pretrained_path:
            raise ValueError("act_pretrained_path must point to a local ACT pretrained_model directory")
        if self.rl_token_dim <= 0 or self.rl_token_ff_dim <= 0:
            raise ValueError("RL-token dimensions must be positive")
        if self.rl_token_nhead <= 0 or self.rl_token_dim % self.rl_token_nhead:
            raise ValueError("rl_token_dim must be divisible by rl_token_nhead")
        if self.rl_token_enc_layers <= 0 or self.rl_token_dec_layers <= 0:
            raise ValueError("RL-token encoder and decoder must each have at least one layer")
        if self.rl_token_num_rl_tokens <= 0:
            raise ValueError("rl_token_num_rl_tokens must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.recon_weight <= 0:
            raise ValueError("recon_weight must be positive")

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("ACT RL-token training requires at least one image feature")
        if self.robot_state_feature is None:
            raise ValueError("ACT RL-token training requires observation.state")
        if self.action_feature is None:
            raise ValueError("ACT RL-token training requires the dataset action feature")

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(lr=self.rl_token_lr, weight_decay=0.0, grad_clip_norm=1.0)

    def get_scheduler_preset(self) -> LRSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.rl_token_lr,
            decay_lr=5e-6,
            num_warmup_steps=200,
            num_decay_steps=5000,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
