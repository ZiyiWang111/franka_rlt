"""Use an explicit LR scheduler while retaining ACT's optimizer parameter groups."""

from dataclasses import dataclass

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig


@dataclass
class ScheduledTrainPipelineConfig(TrainPipelineConfig):
    def validate(self) -> None:
        scheduler = self.scheduler
        super().validate()
        if not isinstance(self.policy, ACTConfig):
            raise ValueError("This entry point supports ACT policies only")
        # LeRobot 0.5.1 resets scheduler when applying policy optimizer presets.
        # Keep those presets (including the lower backbone LR) and the schedule.
        if scheduler is not None:
            self.scheduler = scheduler


@parser.wrap()
def main(cfg: ScheduledTrainPipelineConfig) -> None:
    from lerobot.scripts.lerobot_train import train

    # This config is already parsed. LeRobot's wrapper checks exact types and
    # would reparse it as the base config, losing our validate override.
    train.__wrapped__(cfg)


if __name__ == "__main__":
    main()
