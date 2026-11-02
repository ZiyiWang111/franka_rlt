"""Incremental index for temporal chunk TD, including overlapping windows."""
from __future__ import annotations

import random
import torch


class ChunkTDIndex:
    def __init__(self, chunks, horizon):
        self.horizon = horizon
        self.lookup = {}
        self.positions = {}
        self.ready = {}
        self.legacy = {}
        # Without time indices, a nonterminal snapshot tail cannot be
        # distinguished reliably from appended legacy overlap windows.
        self.legacy_complete = {int(item.episode_id) for item in chunks if bool(item.done)}
        self._choices = None
        for chunk in chunks:
            self.add(chunk)

    def _chain(self, key):
        chain = []
        for _ in range(self.horizon):
            item = self.lookup.get(key)
            if item is None:
                return None
            if chain and (not torch.equal(chain[-1].next_state_vec, item.state_vec)
                          or not torch.equal(chain[-1].next_ref_chunk, item.ref_chunk)):
                return None
            chain.append(item)
            if bool(item.done):
                return chain
            key = (key[0], key[1] + int(item.actual_steps))
        return chain

    def add(self, item):
        episode = int(item.episode_id)
        start = getattr(item, "start_step", None)
        count = int(item.actual_steps)
        if count <= 0 or episode < 0:
            return
        if start is None:
            if episode not in self.legacy_complete:
                return
            # Legacy checkpoints store original chunks in temporal order, then
            # append overlap windows. Accept only a provably contiguous prefix.
            previous = self.legacy.get(episode)
            if previous is None:
                start = 0
            else:
                end, prev = previous
                if (bool(prev.done) or not torch.equal(prev.next_state_vec, item.state_vec)
                        or not torch.equal(prev.next_ref_chunk, item.ref_chunk)):
                    return
                start = end
            self.legacy[episode] = (start + count, item)
        key = (episode, int(start))
        if key in self.lookup:
            return
        self.lookup[key] = item
        self.positions[id(item)] = key
        length = item.exec_chunk.shape[0]
        for offset in range(self.horizon):
            ancestor = (episode, int(start) - offset * length)
            chain = self._chain(ancestor)
            if chain:
                self.ready[ancestor] = chain
        self._choices = None

    def remove(self, item):
        key = self.positions.pop(id(item), None)
        if key is None:
            return
        self.lookup.pop(key, None)
        for offset in range(self.horizon):
            self.ready.pop((key[0], key[1] - offset * item.exec_chunk.shape[0]), None)
        self._choices = None

    def sample(self, batch_size):
        if self._choices is None:
            self._choices = list(self.ready)
        if not self._choices:
            raise ValueError("no complete TD horizon or episode-terminal prefix in replay")
        chains = [self.ready[key] for key in random.sample(self._choices, min(batch_size, len(self._choices)))]
        rewards, steps = [], []
        for chain in chains:
            seq = torch.cat([item.reward_seq[:int(item.actual_steps)] for item in chain])
            steps.append(len(seq))
            padded = seq.new_zeros(self.horizon * chain[0].exec_chunk.shape[0])
            padded[:len(seq)] = seq
            rewards.append(padded)
        return [chain[0] for chain in chains], {
            "td_reward_seq": torch.stack(rewards),
            "td_steps": torch.tensor(steps, dtype=torch.int64),
            "td_done": torch.stack([chain[-1].done for chain in chains]),
            "td_next_state_vec": torch.stack([chain[-1].next_state_vec for chain in chains]),
            "td_next_ref_flat": torch.stack([chain[-1].next_ref_chunk for chain in chains]).flatten(1),
        }
