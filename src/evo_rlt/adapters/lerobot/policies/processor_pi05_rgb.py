"""Persistent OpenPI-style task-only tokenization for PI05 ablations."""

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from lerobot.processor import ProcessorStep
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK


@dataclass
class Pi05RGBTokenizerStep(ProcessorStep):
    # Serialized by full import path, so inference requires this installed module.
    tokenizer_name: str = "google/paligemma-3b-pt-224"
    max_length: int = 200
    task_key: str = "task"
    discrete_state_input: bool = False

    def __post_init__(self):
        if self.discrete_state_input:
            raise ValueError("RGB tokenizer requires discrete_state_input=False")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def get_config(self):
        return {k: getattr(self, k) for k in (
            "tokenizer_name", "max_length", "task_key", "discrete_state_input"
        )}

    def __call__(self, transition):
        result = transition.copy()
        obs = dict(transition.get(TransitionKey.OBSERVATION) or {})
        obs.pop(OBS_STATE, None)
        tasks = transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key]
        if isinstance(tasks, str):
            tasks = [tasks]
        rows, masks = [], []
        for task in tasks:
            text = task.strip().replace("_", " ").replace("\n", " ")
            ids = [self.tokenizer.bos_token_id]
            ids += self.tokenizer.encode(text, add_special_tokens=False)
            ids += self.tokenizer.encode("\n", add_special_tokens=False)
            ids = ids[:self.max_length]
            masks.append([True] * len(ids) + [False] * (self.max_length - len(ids)))
            rows.append(ids + [0] * (self.max_length - len(ids)))
        obs[OBS_LANGUAGE_TOKENS] = torch.tensor(rows, dtype=torch.long)
        obs[OBS_LANGUAGE_ATTENTION_MASK] = torch.tensor(masks, dtype=torch.bool)
        result[TransitionKey.OBSERVATION] = obs
        return result

    def transform_features(self, features):
        return features


def enable_rgb_training():
    """Apply after either loading saved processors or building new ones."""
    import lerobot.policies.factory as factory
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.processor import TokenizerProcessorStep

    original = factory.make_pre_post_processors

    def make_processors(policy_cfg, *args, **kwargs):
        pre, post = original(policy_cfg, *args, **kwargs)
        if policy_cfg.type != "pi05" or policy_cfg.use_relative_actions:
            raise ValueError("RGB training requires PI05 with use_relative_actions=False")
        if any(isinstance(step, Pi05RGBTokenizerStep) for step in pre.steps):
            return pre, post
        state_steps = [s for s in pre.steps if isinstance(s, Pi05PrepareStateTokenizerProcessorStep)]
        token_steps = [s for s in pre.steps if isinstance(s, TokenizerProcessorStep)]
        if len(state_steps) != 1 or len(token_steps) != 1:
            raise ValueError("Unexpected PI05 pipeline; refusing an incomplete state ablation")
        token_step = token_steps[0]
        pre.steps = [
            Pi05RGBTokenizerStep(tokenizer_name=s.tokenizer_name,
                                 max_length=s.max_length, task_key=s.task_key)
            if s is token_step else s
            for s in pre.steps if s is not state_steps[0]
        ]
        return pre, post

    factory.make_pre_post_processors = make_processors
