"""ACT-backed RL Token Stage-2 policy, learner, and online loop.

The online unit is one action chunk: collect one transition, then run ``G``
replay updates. Warmup only fills replay with frozen-ACT actions; it never runs
Actor/Critic optimization.
"""

from __future__ import annotations

import copy
import logging
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import torch
from torch import Tensor, nn

from act_rlt.act_encoder import predict_act_chunk_with_encoder_hidden
from act_rlt.configuration_act_rlt_token import ACTRLTokenConfig
from act_rlt.modeling_act_rlt_token import ACTRLTokenPolicy
from act_rlt.stage2_networks import Stage2ChunkActor, Stage2TwinCritic
from evo_rlt.core.interfaces import (
    TRANSITION_SOURCE_HUMAN_OVERRIDE,
    TRANSITION_SOURCE_RL_AUTONOMOUS,
    TRANSITION_SOURCE_WARMUP_VLA,
    ChunkTransition,
)
from evo_rlt.core.losses import discounted_chunk_return
from evo_rlt.core.replay_buffer import ReplayBuffer
from evo_rlt.core.utils import flatten_chunk, soft_update, unflatten_chunk
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.constants import OBS_STATE


log = logging.getLogger(__name__)


def resolve_pretrained_model(path: str | Path) -> Path:
    """Resolve a LeRobot run, checkpoint, or pretrained-model directory."""
    from evo_rlt.artifact_paths import resolve_artifact_path

    path = resolve_artifact_path(path)
    candidates = (path, path / "pretrained_model", path / "checkpoints/last/pretrained_model")
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
            return candidate
    raise FileNotFoundError(f"no complete pretrained_model found under {path}")


def strip_removed_force_feedback_config(config: dict) -> dict:
    """Load legacy no-force checkpoints while rejecting the removed architecture."""
    config = dict(config)
    if config.pop("use_force_feedback", False):
        raise ValueError(
            "This checkpoint used the removed Stage-2 force-feedback branch; "
            "its Actor/Critic and replay are incompatible. Start a new run without force feedback."
        )
    for key in ("force_scale_n", "torque_scale_nm", "wrench_offset", "wrench_clip"):
        config.pop(key, None)
    return config


@dataclass
class ACTStage2Config:
    """Confirmed first-version Stage-2 settings for the FR3 insertion task."""

    stage1_checkpoint: str
    act_checkpoint: str
    device: str = "cuda"
    chunk_length: int = 4
    action_dim: int = 6
    proprio_dim: int = 7
    temporal_ensemble_coeff: float = 0.01

    actor_hidden_dim: int = 256
    actor_num_layers: int = 2
    fusion_dim: int = 128
    input_fusion: str = "balanced"
    actor_fixed_std: float = 0.3
    actor_ref_dropout_p: float = 0.5
    actor_lr: float = 3e-4

    critic_hidden_dim: int = 256
    critic_num_layers: int = 2
    critic_lr: float = 3e-4

    gamma: float = 0.99
    td_chunks: int = 1
    beta: float = 2.0
    tau: float = 0.005
    batch_size: int = 256
    utd_ratio: int = 5
    actor_update_interval: int = 2
    grad_clip_norm: float = 1.0
    target_q_clip: float = 1.0
    target_q_clipping: bool = True
    learner_version: int = 3
    bc_init_steps: int = 1000
    critic_init_steps: int = 1000
    temporal_collection: bool = False
    max_step_m: float = 0.002
    max_step_rad: float = 0.02

    replay_capacity: int = 200_000
    # None preserves non-overlapping replay for any configured chunk length.
    replay_stride: int | None = None
    warmup_steps: int = 4_000
    total_env_steps: int = 10_000
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.td_chunks, int) or self.td_chunks < 1:
            raise ValueError("td_chunks must be a positive integer")
        if self.replay_stride is None:
            self.replay_stride = self.chunk_length
        if (not isinstance(self.replay_stride, int) or self.replay_stride <= 0
                or self.chunk_length % self.replay_stride):
            raise ValueError("replay_stride must be a positive divisor of chunk_length")
        if self.target_q_clip != 1.0 or self.learner_version != 3:
            raise ValueError("Stage-2 v3 requires target_q_clip=1; use target_q_clipping=False to disable clipping")
        if not isinstance(self.target_q_clipping, bool):
            raise ValueError("target_q_clipping must be boolean")
        if self.input_fusion not in {"token_identity", "balanced"}:
            raise ValueError("input_fusion must be token_identity or balanced")
        if self.bc_init_steps < 0 or self.critic_init_steps < 0:
            raise ValueError("initialization steps must be non-negative")
        if self.temporal_collection:
            raise ValueError("chunk TD learning requires temporal_collection=False")
        if not all(torch.isfinite(torch.tensor(v)) and v > 0 for v in (self.max_step_m, self.max_step_rad)):
            raise ValueError("physical action limits must be finite and positive")
        positive_ints = {
            "chunk_length": self.chunk_length,
            "action_dim": self.action_dim,
            "actor_hidden_dim": self.actor_hidden_dim,
            "actor_num_layers": self.actor_num_layers,
            "fusion_dim": self.fusion_dim,
            "critic_hidden_dim": self.critic_hidden_dim,
            "critic_num_layers": self.critic_num_layers,
            "batch_size": self.batch_size,
            "utd_ratio": self.utd_ratio,
            "actor_update_interval": self.actor_update_interval,
            "replay_capacity": self.replay_capacity,
            "warmup_steps": self.warmup_steps,
            "total_env_steps": self.total_env_steps,
        }
        invalid = [name for name, value in positive_ints.items() if value <= 0]
        if invalid:
            raise ValueError(f"Stage-2 integer settings must be positive: {invalid}")
        if self.proprio_dim not in (0, 1, 6, 7):
            raise ValueError("proprio_dim must be 0 (RGB only), 1 (TCP Z), 6 (wrench), or 7 (joints)")
        for name, value in {
            "actor_fixed_std": self.actor_fixed_std,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "gamma": self.gamma,
            "beta": self.beta,
            "tau": self.tau,
            "grad_clip_norm": self.grad_clip_norm,
            "temporal_ensemble_coeff": self.temporal_ensemble_coeff,
        }.items():
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"{name} must be finite")
        if self.actor_fixed_std <= 0 or self.actor_lr <= 0 or self.critic_lr <= 0:
            raise ValueError("standard deviation and learning rates must be positive")
        if self.beta < 0 or self.grad_clip_norm <= 0:
            raise ValueError("beta must be non-negative and grad_clip_norm must be positive")
        if self.temporal_ensemble_coeff < 0:
            raise ValueError("temporal_ensemble_coeff must be non-negative")
        if not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("gamma and tau must be in (0, 1]")
        if not 0 <= self.actor_ref_dropout_p < 1:
            raise ValueError("actor_ref_dropout_p must be in [0, 1)")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")

    @property
    def state_dim(self) -> int:
        return 768 + self.proprio_dim

    @property
    def chunk_dim(self) -> int:
        return self.chunk_length * self.action_dim

    def to_dict(self) -> dict:
        return asdict(self)


class ACTStage2Policy(nn.Module):
    """Frozen ACT + frozen RL Token encoder + trainable chunk Actor."""

    def __init__(
        self,
        config: ACTStage2Config,
        *,
        stage1_policy: ACTRLTokenPolicy | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        stage1 = stage1_policy or self._load_stage1()
        self._validate_backbone(stage1)
        for parameter in stage1.parameters():
            parameter.requires_grad = False
        stage1.eval()
        # The frozen model is intentionally excluded from Stage-2 state_dict and
        # optimizer parameters; the Stage-1 and ACT paths are checkpoint metadata.
        object.__setattr__(self, "_stage1", stage1)

        self.actor = Stage2ChunkActor(
            state_dim=config.state_dim,
            proprio_dim=config.proprio_dim,
            chunk_dim=config.chunk_dim,
            fusion_dim=config.fusion_dim,
            hidden_dim=config.actor_hidden_dim,
            num_layers=config.actor_num_layers,
            fixed_std=config.actor_fixed_std,
            ref_dropout_p=config.actor_ref_dropout_p,
            input_fusion=config.input_fusion,
        )

    def _load_stage1(self) -> ACTRLTokenPolicy:
        stage1_path = resolve_pretrained_model(self.config.stage1_checkpoint)
        act_path = resolve_pretrained_model(self.config.act_checkpoint)
        loaded = PreTrainedConfig.from_pretrained(stage1_path, local_files_only=True)
        if not isinstance(loaded, ACTRLTokenConfig):
            raise TypeError(f"expected act_rlt_token checkpoint, got {loaded.type!r}")
        # The ACT run may have been copied or renamed after RL Token training.
        loaded.act_pretrained_path = str(act_path)
        loaded.device = self.config.device
        return ACTRLTokenPolicy.from_pretrained(
            stage1_path,
            config=loaded,
            local_files_only=True,
            strict=True,
        )

    def _validate_backbone(self, stage1: ACTRLTokenPolicy) -> None:
        act_cfg = stage1._act.config
        if stage1.config.rl_token_dim != 768:
            raise ValueError(f"Stage-2 requires RL token dim 768, got {stage1.config.rl_token_dim}")
        if self.config.chunk_length > act_cfg.chunk_size:
            raise ValueError(
                f"chunk_length={self.config.chunk_length} exceeds ACT horizon={act_cfg.chunk_size}"
            )
        act_action_dim = int(act_cfg.action_feature.shape[0])
        if act_action_dim not in (self.config.action_dim, self.config.action_dim + 1):
            raise ValueError(
                f"FR3 Stage-2 expects six arm deltas with an optional gripper, got {act_action_dim}"
            )
        act_state_dim = int(act_cfg.robot_state_feature.shape[0]) if act_cfg.robot_state_feature else 0
        if act_state_dim != self.config.proprio_dim:
            raise ValueError(
                f"Stage-2 expects {self.config.proprio_dim}D auxiliary state, got {act_state_dim}"
            )

    @property
    def stage1(self) -> ACTRLTokenPolicy:
        return self._stage1

    @property
    def act_action_dim(self) -> int:
        return int(self._stage1._act.config.action_feature.shape[0])

    @torch.no_grad()
    def encode_and_reference(
        self, batch: dict[str, Tensor], *, timing=None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``state_vec``, Stage-2 reference, and full ACT reference.

        All outputs use the already-normalized tensors produced by ACT's saved
        preprocessor. ACT and the RL Token encoder stay frozen and in eval mode.
        """
        self._stage1.eval()
        device = next(value.device for value in batch.values() if isinstance(value, Tensor))
        with timing.measure("act", device) if timing else nullcontext():
            full_ref, hidden = predict_act_chunk_with_encoder_hidden(self._stage1._act, batch)
        with timing.measure("rl_token", hidden.device) if timing else nullcontext():
            z_rl = self._stage1.rl_token.encode(hidden.float())
        state_vec = z_rl
        if self.config.proprio_dim:
            proprio = batch[OBS_STATE].float()
            state_vec = torch.cat([z_rl, proprio], dim=-1)
        ref = full_ref[:, : self.config.chunk_length, : self.config.action_dim].float()
        if state_vec.shape[-1] != self.config.state_dim:
            raise RuntimeError(f"unexpected Stage-2 state shape: {tuple(state_vec.shape)}")
        return state_vec, ref, full_ref.float()

    def actor_chunk(
        self,
        state_vec: Tensor,
        ref_chunk: Tensor,
        *,
        deterministic: bool,
    ) -> Tensor:
        ref_flat = flatten_chunk(ref_chunk)
        sampled, mean = self.actor.sample(state_vec, ref_flat, training=False)
        return unflatten_chunk(mean if deterministic else sampled, self.config.chunk_length)

    def train(self, mode: bool = True):
        super().train(mode)
        self._stage1.eval()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._stage1.to(*args, **kwargs)
        self._stage1.eval()
        return self


@dataclass
class LearnerStepMetrics:
    critic_loss: float
    actor_loss: float | None
    critic_grad_norm: float
    actor_grad_norm: float | None
    critic_updates: int
    actor_updates: int
    diagnostics: dict[str, float] = field(default_factory=dict)


class Stage2Learner:
    """Twin-Q learner with delayed Actor and target-critic updates."""

    def __init__(self, policy: ACTStage2Policy, config: ACTStage2Config) -> None:
        self.policy = policy
        self.config = config
        self.critic = Stage2TwinCritic(
            state_dim=config.state_dim,
            proprio_dim=config.proprio_dim,
            chunk_dim=config.chunk_dim,
            fusion_dim=config.fusion_dim,
            hidden_dim=config.critic_hidden_dim,
            num_layers=config.critic_num_layers,
            input_fusion=config.input_fusion,
        ).to(config.device)
        self.target_critic = copy.deepcopy(self.critic).eval()
        self.target_actor = copy.deepcopy(policy.actor).eval()
        self.target_actor.requires_grad_(False)
        self.initialized = False
        self.action_projection = lambda action: action
        for parameter in self.target_critic.parameters():
            parameter.requires_grad = False
        self.actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.critic_updates = 0
        self.actor_updates = 0

    def _batch_to_device(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {key: value.to(self.config.device) for key, value in batch.items()}

    def _sample_batch(self, replay: ReplayBuffer, *, td: bool = False) -> dict[str, Tensor]:
        if isinstance(replay, ReplayBuffer):
            return self._batch_to_device(
                replay.sample(self.config.batch_size, include_bc_target=True,
                              td_chunks=self.config.td_chunks if td else 1)
            )
        return self._batch_to_device(replay.sample(self.config.batch_size))

    def configure_action_projection(self, postprocessor) -> None:
        """Differentiate through the same physical norm bounds as the robot.

        ACT uses mean/std normalization. Recover its affine inverse from the
        saved processor; reject non-affine processors rather than guessing.
        """
        with torch.no_grad():
            action_dim = self.policy.act_action_dim
            probe = torch.stack([torch.zeros(action_dim), torch.ones(action_dim), torch.full((action_dim,), 2.)]).to(self.config.device)
            physical = postprocessor(probe).to(self.config.device)
            offset, scale = physical[0, :6], physical[1, :6] - physical[0, :6]
            if not torch.isfinite(physical).all() or (scale <= 0).any() or not torch.allclose(
                physical[2, :6], offset + 2 * scale, atol=1e-6):
                raise ValueError("Stage-2 requires affine ACT mean/std action normalization")

        def project(action):
            shaped = action.reshape(-1, self.config.chunk_length, 6)
            physical = shaped * scale + offset
            parts = []
            for start, limit in ((0, self.config.max_step_m), (3, self.config.max_step_rad)):
                part = physical[..., start:start + 3]
                parts.append(part * (limit / part.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1))
            return ((torch.cat(parts, dim=-1) - offset) / scale).reshape_as(action)
        self.action_projection = project

    def initialize(self, replay: ReplayBuffer) -> None:
        """Offline initialization at the warmup boundary, before Actor control."""
        if self.initialized:
            return
        if len(replay) < self.config.batch_size:
            raise ValueError("not enough replay for Actor/Critic initialization")
        for step in range(self.config.bc_init_steps):
            batch = self._sample_batch(replay)
            mu, _ = self.policy.actor(batch["state_vec"], batch["ref_chunk_flat"])
            bc_target = batch.get("bc_target_flat", batch["ref_chunk_flat"])
            loss = self.config.chunk_dim * _bc_loss(
                mu, bc_target, batch.get("actual_steps"), chunk_length=self.config.chunk_length
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.config.grad_clip_norm,
                                          error_if_nonfinite=True)
            self.actor_optimizer.step()
            if step % 100 == 0:
                print(f"BC initialization {step}/{self.config.bc_init_steps}: loss={loss.item():.5f}", flush=True)
        self.target_actor.load_state_dict(self.policy.actor.state_dict())
        for step in range(self.config.critic_init_steps):
            metrics = self.update_once(replay, update_actor=False)
            if step % 100 == 0:
                print(f"Critic initialization {step}/{self.config.critic_init_steps}: {metrics.diagnostics}", flush=True)
        self.initialized = True

    def update_once(self, replay: ReplayBuffer, *, update_actor: bool = True) -> LearnerStepMetrics:
        batch = self._sample_batch(replay, td=True)
        self.critic.train()
        self.policy.actor.train()

        x, action = batch["state_vec"], batch["exec_chunk_flat"]
        td_next_state = batch.get("td_next_state_vec", batch["next_state_vec"])
        td_next_ref = batch.get("td_next_ref_flat", batch["next_ref_flat"])
        td_steps = batch.get("td_steps", batch["actual_steps"])
        td_done = batch.get("td_done", batch["done"])
        with torch.no_grad():
            next_action, _ = self.target_actor(td_next_state, td_next_ref)
            next_action = self.action_projection(next_action)
            raw_next_q = self.target_critic.min_q(td_next_state, next_action)
            next_q = raw_next_q.clamp(0.0, 1.0) if self.config.target_q_clipping else raw_next_q
            reward = discounted_chunk_return(batch.get("td_reward_seq", batch["reward_seq"]), self.config.gamma, td_steps)
            target = reward + self.config.gamma ** td_steps.float().unsqueeze(-1) * (
                1 - td_done.unsqueeze(-1)) * next_q
            if not torch.isfinite(target).all():
                raise ValueError("non-finite TD target; check Critic output and replay")
            if self.config.target_q_clipping and ((target < 0).any() or (target > 1.00001).any()):
                raise ValueError("invalid terminal-success TD target; check reward/done in replay")
        q1, q2 = self.critic(x, action)
        c_loss = (q1 - target).square().mean() + (q2 - target).square().mean()
        # Evaluate the frozen ACT reference on exactly the same replay states.
        # This is diagnostic-only: reference actions are not substituted into
        # either the TD loss or the Actor objective.
        with torch.no_grad():
            reference_q = self.critic.min_q(
                x, self.action_projection(batch["ref_chunk_flat"])
            )
        diagnostics = {
            "q1_mean": q1.detach().mean().item(), "q2_mean": q2.detach().mean().item(),
            "q_replay_min": torch.minimum(q1, q2).detach().min().item(),
            "q_replay_max": torch.maximum(q1, q2).detach().max().item(),
            "q_reference_mean": reference_q.mean().item(),
            "q_reference_max": reference_q.max().item(),
            "target_q_raw_mean": raw_next_q.mean().item(),
            "target_q_raw_max": raw_next_q.max().item(),
            "target_q_clipping_enabled": self.config.target_q_clipping,
            "target_q_clip_fraction": ((raw_next_q < 0) | (raw_next_q > 1)).float().mean().item()
            if self.config.target_q_clipping else 0.0,
            "target_q_clip_low_fraction": (raw_next_q < 0).float().mean().item()
            if self.config.target_q_clipping else 0.0,
            "target_q_clip_high_fraction": (raw_next_q > 1).float().mean().item()
            if self.config.target_q_clipping else 0.0,
            "td_target_mean": target.mean().item(), "td_target_max": target.max().item(),
            "td_steps_mean": td_steps.float().mean().item(),
            "reward_mean": reward.mean().item(), "done_fraction": batch["done"].mean().item(),
            "td_error_abs_mean": (q1.detach() - target).abs().mean().item(),
        }
        self.critic_optimizer.zero_grad(set_to_none=True)
        c_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.grad_clip_norm, error_if_nonfinite=True
        )
        self.critic_optimizer.step()
        self.critic_updates += 1

        actor_loss_value: float | None = None
        actor_grad_value: float | None = None
        if update_actor and self.critic_updates % self.config.actor_update_interval == 0:
            critic_params = list(self.critic.parameters())
            for parameter in critic_params:
                parameter.requires_grad_(False)
            try:
                mu, _ = self.policy.actor(x, batch["ref_chunk_flat"], training=True)
                actor_q = self.critic.min_q(x, self.action_projection(mu))
                bc_target = batch.get("bc_target_flat", batch["ref_chunk_flat"])
                bc = _bc_loss(mu, bc_target, batch.get("actual_steps"), chunk_length=self.config.chunk_length)
                a_loss = -actor_q.mean() + self.config.beta * bc
                diagnostics.update(q_actor_mean=actor_q.detach().mean().item(),
                                   q_actor_max=actor_q.detach().max().item(),
                                   bc_loss=bc.detach().item(),
                                   actor_reference_rmse=(mu.detach()-batch["ref_chunk_flat"]).square().mean().sqrt().item())
                with torch.no_grad():
                    deployed_mu, _ = self.policy.actor(x, batch["ref_chunk_flat"])
                    deployed_q = self.critic.min_q(x, self.action_projection(deployed_mu))
                    diagnostics.update(q_actor_deploy_mean=deployed_q.mean().item(),
                                       q_actor_deploy_max=deployed_q.max().item())
                self.actor_optimizer.zero_grad(set_to_none=True)
                a_loss.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(), self.config.grad_clip_norm, error_if_nonfinite=True
                )
                self.actor_optimizer.step()
            finally:
                for parameter in critic_params:
                    parameter.requires_grad_(True)
            self.actor_updates += 1
            actor_loss_value = float(a_loss.detach().item())
            actor_grad_value = float(actor_grad)
            soft_update(self.target_actor, self.policy.actor, self.config.tau)

        # Update targets after the online critic optimizer step.
        soft_update(self.target_critic, self.critic, self.config.tau)
        self.target_critic.eval()
        return LearnerStepMetrics(
            critic_loss=float(c_loss.detach().item()),
            actor_loss=actor_loss_value,
            critic_grad_norm=float(critic_grad),
            actor_grad_norm=actor_grad_value,
            critic_updates=self.critic_updates,
            actor_updates=self.actor_updates,
            diagnostics=diagnostics,
        )

    def state_dict(self) -> dict:
        return {
            "learner_version": 3,
            "input_fusion": self.config.input_fusion,
            "initialized": self.initialized,
            "target_actor": self.target_actor.state_dict(),
            "actor": self.policy.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "critic_updates": self.critic_updates,
            "actor_updates": self.actor_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("learner_version") != 3:
            raise ValueError("old Stage-2 network cannot resume as v3; use --replay-from for a fresh learner")
        if state.get("input_fusion", "balanced") != self.config.input_fusion:
            raise ValueError("Stage-2 input fusion changed; use --replay-from with a new output directory")
        self.target_actor.load_state_dict(state["target_actor"])
        self.initialized = bool(state["initialized"])
        self.policy.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.critic_updates = int(state["critic_updates"])
        self.actor_updates = int(state["actor_updates"])


@dataclass
class ChunkExecution:
    """Result of executing one normalized Stage-2 action chunk."""

    next_batch: dict[str, Tensor]
    exec_chunk: Tensor
    reward_seq: Tensor
    actual_steps: int
    done: bool
    terminated: bool
    truncated: bool
    intervention: bool = False
    stop_requested: bool = False
    info: dict = field(default_factory=dict)
    # The real-time Stage-2 collector performs policy inference inside its
    # inference worker. These fields carry the exact state/reference paired with
    # the fixed executed chunk back to the generic learner loop.
    state_vec: Tensor | None = None
    ref_chunk: Tensor | None = None
    next_state_vec: Tensor | None = None
    next_ref_chunk: Tensor | None = None
    bc_target_chunk: Tensor | None = None
    # CPU-only intermediate observations; excluded from info/JSON/checkpoints.
    replay_observations: dict = field(default_factory=dict)


class Stage2Environment(Protocol):
    def reset(self, *, episode_id: int, warmup: bool) -> dict[str, Tensor]: ...

    def execute_chunk(self, action_chunk: Tensor, full_reference: Tensor) -> ChunkExecution: ...


@dataclass
class OnlineStage2Metrics:
    env_steps: int = 0
    warmup_env_steps: int = 0
    online_env_steps: int = 0
    chunks: int = 0
    episodes: int = 0
    successes: int = 0
    human_chunks: int = 0
    human_env_steps: int = 0
    critic_updates: int = 0
    actor_updates: int = 0
    replay_overlap_chunks: int = 0
    last_learner: LearnerStepMetrics | None = None


def _unbatch_cpu(tensor: Tensor) -> Tensor:
    tensor = tensor.detach().float().cpu()
    if tensor.ndim > 0 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


def _bc_loss(mean: Tensor, target: Tensor, actual_steps: Tensor | None, *, chunk_length: int) -> Tensor:
    """Mean squared error over executed steps, including short terminal chunks."""
    dimensions = mean.shape[-1] // chunk_length
    error = (mean - target).reshape(-1, chunk_length, dimensions).square()
    if actual_steps is None:
        return error.mean()
    mask = (
        torch.arange(chunk_length, device=mean.device)[None, :]
        < actual_steps.reshape(-1, 1)
    ).unsqueeze(-1)
    return (error * mask).sum() / (mask.sum().clamp_min(1) * dimensions)


def run_online_stage2(
    policy: ACTStage2Policy,
    learner: Stage2Learner,
    env: Stage2Environment,
    replay: ReplayBuffer,
    config: ACTStage2Config,
    *,
    on_chunk: Callable[[OnlineStage2Metrics, ChunkExecution], None] | None = None,
    initial_metrics: OnlineStage2Metrics | None = None,
) -> OnlineStage2Metrics:
    """Run warmup then online Stage 2 with chunk-level UTD semantics.

    Warmup ends at the first episode boundary at or above warmup_steps.
    Warmup transitions never trigger updates. After warmup, each collected
    kept chunk triggers ``utd_ratio`` critic updates, independent of C.
    With episode replay review, chunks are staged and updates run only after
    acceptance at the episode boundary; rejected chunks never train the model.
    """
    metrics = initial_metrics or OnlineStage2Metrics()

    if metrics.online_env_steps >= config.total_env_steps:
        return metrics
    overlap = None
    if config.replay_stride < config.chunk_length:
        from act_rlt.replay_overlap import EpisodeReplayOverlap

        if not callable(getattr(env, "encode_replay_observation", None)):
            raise ValueError("overlapping replay requires intermediate-observation encoding")
        overlap = EpisodeReplayOverlap(config.chunk_length, config.replay_stride)

    review_replay = bool(getattr(env, "review_episode_replay", False))
    pending_replay = []
    pending_steps = 0
    pending_chunks = 0

    def finish_overlap(execution):
        nonlocal pending_steps, pending_chunks
        discard = review_replay and (
            execution.info.get("discard_episode_replay", False) or execution.stop_requested
        )
        if overlap is not None:
            if discard:
                overlap.clear()
                execution.info["replay_overlap_added"] = 0
            else:
                started = time.monotonic()
                extra, stats = overlap.finish(
                    env.encode_replay_observation,
                    done=execution.done, terminated=execution.terminated, truncated=execution.truncated,
                )
                for transition in extra:
                    if review_replay:
                        pending_replay.append(transition)
                    else:
                        replay.add(transition)
                metrics.replay_overlap_chunks += len(extra)
                execution.info.update(stats)
                execution.info["replay_overlap_encode_wall_ms"] = (time.monotonic() - started) * 1000
        if not review_replay:
            return 1
        updates_for_chunks = 0
        execution.info["episode_replay_discarded"] = bool(discard)
        execution.info["episode_replay_steps"] = pending_steps
        execution.info["episode_replay_chunks"] = len(pending_replay)
        if not discard:
            for transition in pending_replay:
                replay.add(transition)
            if warmup:
                metrics.warmup_env_steps += pending_steps
            else:
                metrics.online_env_steps += pending_steps
            updates_for_chunks = pending_chunks
        pending_replay.clear()
        pending_steps = pending_chunks = 0
        return updates_for_chunks

    def update_committed_replay(execution, update_chunks):
        # Confirmed timing: no updates for a chunk that started in warmup.
        # Confirmed UTD: G updates per collected chunk, never G*C.
        if not warmup and update_chunks and len(replay) >= config.batch_size:
            updates = []
            learner_started = time.monotonic()
            for _ in range(config.utd_ratio * update_chunks):
                updates.append(learner.update_once(replay))
            execution.info["learner_updates_wall_ms"] = (time.monotonic() - learner_started) * 1000
            metrics.last_learner = updates[-1]
            for name in ("critic_loss", "actor_loss", "critic_grad_norm", "actor_grad_norm"):
                values = [getattr(u, name) for u in updates if getattr(u, name) is not None]
                setattr(metrics.last_learner, name, sum(values) / len(values) if values else None)
            keys = set().union(*(u.diagnostics for u in updates))
            metrics.last_learner.diagnostics = {
                key: (max if key.endswith("_max") else min if key.endswith("_min") else
                      lambda values: sum(values) / len(values))(
                          [u.diagnostics[key] for u in updates if key in u.diagnostics])
                for key in keys
            }
            metrics.critic_updates = learner.critic_updates
            metrics.actor_updates = learner.actor_updates


    # A mid-episode checkpoint resumes from a fresh physical reset, so allocate
    # a new replay episode id rather than reusing the interrupted one.
    episode_id = metrics.episodes
    episode_step = 0
    if replay.buffer:
        episode_id = max(int(transition.episode_id) for transition in replay.buffer) + 1
    warmup = metrics.warmup_env_steps < config.warmup_steps
    batch = env.reset(
        episode_id=episode_id,
        warmup=warmup,
    )

    # Match Evo-RLT's two-loop semantics: total_env_steps is the online budget
    # after warmup, not a combined warmup + online limit.
    while metrics.online_env_steps < config.total_env_steps:
        metrics.last_learner = None
        if not warmup and not learner.initialized:
            initialization_start = time.monotonic()
            learner.initialize(replay)
            exclude_time = getattr(env, "exclude_learning_time", None)
            if exclude_time is not None:
                exclude_time(time.monotonic() - initialization_start)
        policy.eval()
        execute_policy_chunk = getattr(env, "execute_policy_chunk", None)
        if callable(execute_policy_chunk):
            # Warmup has a minimum length, but must collect the terminal
            # transition before initializing or handing control to the Actor.
            remaining_steps = (
                None if warmup
                else config.total_env_steps - metrics.online_env_steps - pending_steps
            )
            execution = execute_policy_chunk(
                policy, batch, warmup=warmup, remaining_steps=remaining_steps
            )
            required = {
                "state_vec": execution.state_vec,
                "ref_chunk": execution.ref_chunk,
                "next_state_vec": execution.next_state_vec,
                "next_ref_chunk": execution.next_ref_chunk,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise RuntimeError(f"real-time Stage-2 execution omitted metadata: {missing}")
            state_vec = execution.state_vec
            ref_chunk = execution.ref_chunk
            next_state = execution.next_state_vec
            next_ref = execution.next_ref_chunk
        else:
            with torch.inference_mode():
                state_vec, ref_chunk, full_ref = policy.encode_and_reference(batch)
                action_chunk = (
                    ref_chunk
                    if warmup
                    else policy.actor_chunk(state_vec, ref_chunk, deterministic=False)
                )
            execution = env.execute_chunk(action_chunk, full_ref)
            with torch.inference_mode():
                next_state, next_ref, _ = policy.encode_and_reference(execution.next_batch)
        if not 0 <= execution.actual_steps <= config.chunk_length:
            raise RuntimeError(
                f"actual_steps must be in [0,{config.chunk_length}], got {execution.actual_steps}"
            )
        if execution.exec_chunk.shape != (config.chunk_length, config.action_dim):
            raise RuntimeError(f"bad executed chunk shape: {tuple(execution.exec_chunk.shape)}")
        if execution.reward_seq.shape != (config.chunk_length,):
            raise RuntimeError(f"bad reward sequence shape: {tuple(execution.reward_seq.shape)}")

        # A workspace refusal can happen before the first action is sent. It is
        # a real episode boundary but not a transition: no environment step or
        # reward occurred, so do not fabricate replay data or run updates.
        if execution.actual_steps == 0:
            if not execution.done or not execution.truncated:
                raise RuntimeError("zero-step execution must be a truncated episode")
            committed_chunks = finish_overlap(execution)
            if review_replay:
                update_committed_replay(execution, committed_chunks)
            execution.info.setdefault("warmup", warmup)
            metrics.episodes += 1
            if on_chunk is not None:
                on_chunk(metrics, execution)
            if execution.stop_requested:
                break
            episode_id += 1
            episode_step = 0
            warmup = metrics.warmup_env_steps < config.warmup_steps
            batch = env.reset(
                episode_id=episode_id,
                warmup=warmup,
            )
            continue

        transition = ChunkTransition(
            start_step=episode_step,
            state_vec=_unbatch_cpu(state_vec),
            exec_chunk=_unbatch_cpu(execution.exec_chunk),
            ref_chunk=_unbatch_cpu(ref_chunk),
            reward_seq=_unbatch_cpu(execution.reward_seq),
            next_state_vec=_unbatch_cpu(next_state),
            next_ref_chunk=_unbatch_cpu(next_ref),
            done=torch.tensor(float(execution.done)),
            intervention=torch.tensor(float(execution.intervention)),
            actual_steps=torch.tensor(execution.actual_steps, dtype=torch.int64),
            source=torch.tensor(
                TRANSITION_SOURCE_HUMAN_OVERRIDE if execution.intervention else
                TRANSITION_SOURCE_WARMUP_VLA if warmup else TRANSITION_SOURCE_RL_AUTONOMOUS
            ),
            episode_id=torch.tensor(episode_id),
            is_critical=torch.tensor(1.0),
            terminated=torch.tensor(float(execution.terminated)),
            truncated=torch.tensor(float(execution.truncated)),
            bc_target_chunk=(
                _unbatch_cpu(execution.bc_target_chunk)
                if execution.bc_target_chunk is not None else None
            ),
        )
        if review_replay:
            pending_replay.append(transition)
            pending_steps += execution.actual_steps
            pending_chunks += 1
        else:
            replay.add(transition)
        episode_step += execution.actual_steps
        if overlap is not None:
            overlap.add(transition, execution.replay_observations)
        execution.replay_observations.clear()
        metrics.env_steps += execution.actual_steps
        if not review_replay:
            if warmup:
                metrics.warmup_env_steps += execution.actual_steps
            else:
                metrics.online_env_steps += execution.actual_steps
        metrics.chunks += 1
        if execution.intervention:
            metrics.human_chunks += 1
            metrics.human_env_steps += execution.actual_steps

        update_chunks = 0 if review_replay else 1
        if (execution.done or execution.info.get("collector_paused", False)
                or (not warmup and metrics.online_env_steps + pending_steps >= config.total_env_steps)):
            update_chunks = finish_overlap(execution)

        update_committed_replay(execution, update_chunks)

        execution.info.setdefault("warmup", warmup)
        if execution.done:
            metrics.episodes += 1
            metrics.successes += int(bool(execution.info.get("success", False)))

        if on_chunk is not None:
            on_chunk(metrics, execution)
        if execution.stop_requested:
            break
        if execution.done:
            episode_id += 1
            episode_step = 0
            warmup = metrics.warmup_env_steps < config.warmup_steps
            batch = env.reset(
                episode_id=episode_id,
                warmup=warmup,
            )
        else:
            batch = execution.next_batch

    return metrics


__all__ = [
    "ACTStage2Config",
    "ACTStage2Policy",
    "ChunkExecution",
    "LearnerStepMetrics",
    "OnlineStage2Metrics",
    "Stage2Environment",
    "Stage2Learner",
    "resolve_pretrained_model",
    "run_online_stage2",
]
