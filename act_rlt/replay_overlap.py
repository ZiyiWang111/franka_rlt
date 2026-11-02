"""Build extra replay windows from one stopped Stage-2 trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from evo_rlt.core.interfaces import ChunkTransition


@dataclass
class ReplayObservation:
    # Owned CPU sensor data, never normalized or encoded by the servo worker.
    observation: dict
    sequence: int
    selected_at: float
    fresh_after: float


class EpisodeReplayOverlap:
    def __init__(self, chunk_length: int, stride: int):
        if stride <= 0 or chunk_length % stride:
            raise ValueError("replay stride must be positive and divide chunk length")
        self.chunk_length = chunk_length
        self.stride = stride
        self.clear()

    def clear(self):
        self.steps = 0
        self.chunks: list[tuple[int, ChunkTransition]] = []
        self.anchors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.observations: dict[int, ReplayObservation] = {}

    def add(self, transition: ChunkTransition, observations: dict[int, ReplayObservation]):
        count = int(transition.actual_steps)
        if not 0 < count <= self.chunk_length:
            raise ValueError("overlap requires a nonempty executed chunk")
        if self.chunks and int(transition.episode_id) != int(self.chunks[0][1].episode_id):
            raise ValueError("cannot mix episodes in replay overlap")
        start = self.steps
        self.chunks.append((start, transition))
        self.anchors[start] = (transition.state_vec, transition.ref_chunk)
        self.anchors[start + count] = (transition.next_state_vec, transition.next_ref_chunk)
        self.observations.update({start + offset: obs for offset, obs in observations.items()
                                  if 0 < offset < count})
        self.steps += count

    def finish(
        self,
        encode: Callable[[dict], tuple[torch.Tensor, torch.Tensor]],
        *,
        done: bool,
        terminated: bool,
        truncated: bool,
    ) -> tuple[list[ChunkTransition], dict]:
        """Encode missing anchors once, preserve t -> t+C, then release images."""
        stats = {"replay_overlap_added": 0, "replay_overlap_skipped_missing": 0,
                 "replay_overlap_skipped_source": 0, "replay_overlap_encoded_anchors": 0,
                 "replay_overlap_cached_observations": len(self.observations)}
        try:
            return self._build(encode, done, terminated, truncated, stats), stats
        finally:
            self.clear()

    def _build(self, encode, done, terminated, truncated, stats):
        if not self.chunks or self.stride == self.chunk_length:
            return []
        actions, rewards, sources, human, critical, targets = [], [], [], [], [], []
        original_starts = set()
        for start, transition in self.chunks:
            count = int(transition.actual_steps)
            original_starts.add(start)
            actions.append(transition.exec_chunk[:count])
            rewards.append(transition.reward_seq[:count])
            sources.extend([int(transition.source)] * count)
            human.extend([bool(transition.intervention)] * count)
            critical.extend([float(transition.is_critical)] * count)
            target = transition.bc_target_chunk
            targets.append((transition.exec_chunk if target is None else target)[:count])
        actions, rewards, targets = torch.cat(actions), torch.cat(rewards), torch.cat(targets)
        episode_id = self.chunks[0][1].episode_id
        result = []

        def anchor(step):
            if step not in self.anchors:
                state, reference = encode(self.observations.pop(step).observation)
                self.anchors[step] = (state.detach().float().cpu(), reference.detach().float().cpu())
                stats["replay_overlap_encoded_anchors"] += 1
            return self.anchors[step]

        for start in range(0, self.steps - self.chunk_length + 1, self.stride):
            if start in original_starts:
                continue
            end = start + self.chunk_length
            if len(set(zip(sources[start:end], human[start:end]))) != 1:
                stats["replay_overlap_skipped_source"] += 1
                continue
            if any(step not in self.anchors and step not in self.observations for step in (start, end)):
                stats["replay_overlap_skipped_missing"] += 1
                continue
            state, reference = anchor(start)
            next_state, next_reference = anchor(end)
            terminal = done and end == self.steps
            result.append(ChunkTransition(
                start_step=start,
                state_vec=state, exec_chunk=actions[start:end].clone(), ref_chunk=reference,
                reward_seq=rewards[start:end].clone(), next_state_vec=next_state,
                next_ref_chunk=next_reference, done=torch.tensor(float(terminal)),
                intervention=torch.tensor(float(human[start])),
                actual_steps=torch.tensor(self.chunk_length, dtype=torch.int64),
                source=torch.tensor(sources[start]), episode_id=episode_id.clone(),
                is_critical=torch.tensor(max(critical[start:end])),
                terminated=torch.tensor(float(terminal and terminated)),
                truncated=torch.tensor(float(terminal and truncated)),
                bc_target_chunk=targets[start:end].clone() if human[start] else None,
            ))
        stats["replay_overlap_added"] = len(result)
        return result
