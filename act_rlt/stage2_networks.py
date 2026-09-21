"""Balanced input fusion for the ACT-backed Stage-2 actor and critics."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from evo_rlt.core.utils import build_mlp


class ChunkInputFusion(nn.Module):
    """Give token, joint state, and action equal-width normalized branches."""

    def __init__(self, state_dim: int, proprio_dim: int, chunk_dim: int, fusion_dim: int) -> None:
        super().__init__()
        token_dim = state_dim - proprio_dim
        if token_dim <= 0 or proprio_dim <= 0 or chunk_dim <= 0 or fusion_dim <= 0:
            raise ValueError("fusion input and output dimensions must be positive")
        self.token_dim = token_dim
        self.token = self._branch(token_dim, fusion_dim)
        self.proprio = self._branch(proprio_dim, fusion_dim)
        self.action = self._branch(chunk_dim, fusion_dim)

    @staticmethod
    def _branch(input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim), nn.Tanh())

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        token = state[..., : self.token_dim]
        proprio = state[..., self.token_dim :]
        return torch.cat(
            (self.token(token), self.proprio(proprio), self.action(action)), dim=-1
        )


class Stage2ChunkActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        proprio_dim: int,
        chunk_dim: int,
        fusion_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        fixed_std: float = 0.1,
        ref_dropout_p: float = 0.5,
    ) -> None:
        super().__init__()
        self.fusion = ChunkInputFusion(state_dim, proprio_dim, chunk_dim, fusion_dim)
        self.net = build_mlp(
            3 * fusion_dim, hidden_dim, chunk_dim, num_layers,
            activation="gelu", layer_norm=True,
        )
        self.fixed_std = fixed_std
        self.ref_dropout_p = ref_dropout_p

    def forward(
        self, state_vec: Tensor, ref_chunk_flat: Tensor, training: bool = False
    ) -> tuple[Tensor, Tensor]:
        if training:
            keep = (
                torch.rand(state_vec.shape[0], 1, device=state_vec.device)
                > self.ref_dropout_p
            )
            actor_ref = ref_chunk_flat * keep.to(ref_chunk_flat.dtype)
        else:
            actor_ref = ref_chunk_flat
        mean = self.net(self.fusion(state_vec, actor_ref))
        return mean, torch.full_like(mean, self.fixed_std)

    def sample(
        self, state_vec: Tensor, ref_chunk_flat: Tensor, training: bool = False
    ) -> tuple[Tensor, Tensor]:
        mean, std = self.forward(state_vec, ref_chunk_flat, training)
        return mean + std * torch.randn_like(std), mean


class Stage2ChunkCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        proprio_dim: int,
        chunk_dim: int,
        fusion_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.fusion = ChunkInputFusion(state_dim, proprio_dim, chunk_dim, fusion_dim)
        self.net = build_mlp(
            3 * fusion_dim, hidden_dim, 1, num_layers,
            activation="gelu", layer_norm=True,
        )

    def forward(self, state_vec: Tensor, action_flat: Tensor) -> Tensor:
        return self.net(self.fusion(state_vec, action_flat))


class Stage2TwinCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        proprio_dim: int,
        chunk_dim: int,
        fusion_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        kwargs = dict(
            state_dim=state_dim, proprio_dim=proprio_dim,
            chunk_dim=chunk_dim, fusion_dim=fusion_dim,
            hidden_dim=hidden_dim, num_layers=num_layers,
        )
        self.q1 = Stage2ChunkCritic(**kwargs)
        self.q2 = Stage2ChunkCritic(**kwargs)

    def forward(self, state_vec: Tensor, action_flat: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(state_vec, action_flat), self.q2(state_vec, action_flat)

    def min_q(self, state_vec: Tensor, action_flat: Tensor) -> Tensor:
        q1, q2 = self.forward(state_vec, action_flat)
        return torch.minimum(q1, q2)
