"""Run approved wrist-only ACT configurations, preserving the explicit scheduler."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import draccus
from lerobot.configs.train import TrainPipelineConfig
from act_rlt.dataset_compat import install_scalar_state_dataset_support


@dataclass
class CoarseTrainConfig(TrainPipelineConfig):
    def validate(self):
        scheduler = self.scheduler
        super().validate()
        if scheduler is not None:
            self.scheduler = scheduler
        assert list(self.policy.image_features) == ['observation.images.wrist']
        assert self.policy.chunk_size == 10
        assert self.policy.robot_state_feature.shape in ((7,), (1,), (13,), [7], [1], [13])
        assert self.policy.action_feature.shape in ((7,), (6,), [7], [6])
        if tuple(self.policy.robot_state_feature.shape) == (13,):
            assert tuple(self.policy.action_feature.shape) == (6,)
        assert self.policy.use_vae is False


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    import lerobot.policies.act.configuration_act  # register ACT config
    cfg = draccus.decode(CoarseTrainConfig, json.loads(args.config.read_text()))
    cfg.validate()
    print(json.dumps(draccus.encode(cfg), indent=2), flush=True)
    if not args.validate_only:
        install_scalar_state_dataset_support()
        from lerobot.scripts.lerobot_train import train
        train.__wrapped__(cfg)
