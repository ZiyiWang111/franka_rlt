import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


def dry_run(tmp_path, *extra):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"total_episodes": 1, "total_frames": 20}))
    script = Path(__file__).resolve().parents[1] / "act_rlt/train.py"
    text = subprocess.check_output(
        [sys.executable, str(script), "--root", str(root),
         "--output", str(tmp_path / "output"), "--dry-run", *extra], text=True
    )
    return shlex.split(text.splitlines()[-1])


def test_explicit_scheduler_survives_act_preset_and_scales_both_groups(tmp_path):
    draccus = pytest.importorskip("draccus")
    torch = pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    from act_rlt.train_scheduled import ScheduledTrainPipelineConfig

    command = dry_run(tmp_path, "--steps", "30000", "--batch-size", "16")
    cfg = draccus.parse(ScheduledTrainPipelineConfig, args=command[3:])
    cfg.validate()
    assert (cfg.steps, cfg.batch_size, cfg.save_freq) == (30000, 16, 10000)
    assert cfg.use_policy_training_preset
    assert cfg.scheduler.num_warmup_steps == 1000
    optimizer = cfg.optimizer.build([
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": cfg.policy.optimizer_lr},
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": cfg.policy.optimizer_lr_backbone},
    ])
    scheduler = cfg.scheduler.build(optimizer, cfg.steps)
    for _ in range(cfg.steps):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([3e-6, 1e-6])
    assert cfg.to_dict()["scheduler"]["type"] == "cosine_decay_with_warmup"


def test_smoke_warmup_fits_run(tmp_path):
    command = dry_run(tmp_path, "--preset", "smoke")
    assert "--scheduler.num_warmup_steps=10" in command
    assert "--save_freq=100" in command


def test_scheduler_can_be_disabled(tmp_path):
    command = dry_run(tmp_path, "--scheduler", "none")
    assert command[2:4] == ["-m", "lerobot.scripts.lerobot_train"]
    assert not any(arg.startswith("--scheduler.") for arg in command)
