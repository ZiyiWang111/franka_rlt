#!/usr/bin/env python3
"""Resume ACT model/optimizer/scheduler/RNG using LeRobot's saved train config."""
import json
from pathlib import Path

from lerobot.configs import parser
from scripts.train_act_camera_only_tmp import install_camera_only_forward
from scripts.train_act_tcp_z_tmp import install_scalar_state_dataset_support


def main():
    config_path = parser.parse_arg('config_path')
    if not config_path or parser.parse_arg('resume') != 'true':
        raise ValueError('Pass --config_path=.../pretrained_model/train_config.json --resume=true')
    config_path = Path(config_path).resolve()
    step_file = config_path.parent.parent / 'training_state/training_step.json'
    saved_step = json.loads(step_file.read_text())['step']
    config = json.loads(config_path.read_text())
    if config['policy']['type'] != 'act':
        raise ValueError('Expected an ACT checkpoint')
    install_camera_only_forward()
    install_scalar_state_dataset_support()
    from lerobot.scripts import lerobot_train

    original_tqdm = lerobot_train.tqdm

    def absolute_progress(*args, **kwargs):
        if kwargs.get('desc') == 'Training':
            kwargs['initial'] = saved_step
            kwargs['total'] += saved_step
        return original_tqdm(*args, **kwargs)

    lerobot_train.tqdm = absolute_progress
    print(f'USER_RESUME_CHECKPOINT step={saved_step} target={config["steps"]}', flush=True)
    lerobot_train.main()


if __name__ == '__main__':
    main()
