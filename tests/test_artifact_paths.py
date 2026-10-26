from pathlib import Path

import pytest

from evo_rlt import artifact_paths


@pytest.mark.parametrize("category", artifact_paths.OUTPUT_CATEGORIES)
@pytest.mark.parametrize("prefix", ["", "/old/training/checkout/"])
def test_resolve_checkpoint_paths_after_move(tmp_path, monkeypatch, category, prefix):
    monkeypatch.setattr(artifact_paths, "PROJECT_ROOT", tmp_path)
    target = tmp_path / "outputs" / category / "run/checkpoints/last/pretrained_model"
    target.mkdir(parents=True)
    for layout in ("outputs/run", f"outputs/{category}/run"):
        old = prefix + layout + "/checkpoints/last/pretrained_model"
        assert artifact_paths.resolve_artifact_path(old) == target.resolve()


def test_resolve_dataset_and_preserve_explicit_existing_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_paths, "PROJECT_ROOT", tmp_path)
    target = tmp_path / "datasets/ethernet_fine/demo"
    target.mkdir(parents=True)
    assert artifact_paths.resolve_artifact_path("/old/checkout/datasets/demo") == target
    coarse = tmp_path / "datasets/ethernet_coarse/demo"
    coarse.mkdir(parents=True)
    assert artifact_paths.resolve_artifact_path("/old/checkout/datasets/ethernet_coarse/demo") == coarse
    explicit = tmp_path / "other/outputs/demo"
    explicit.mkdir(parents=True)
    assert artifact_paths.resolve_artifact_path(explicit) == explicit
    missing = tmp_path / "unrelated/missing"
    assert artifact_paths.resolve_artifact_path(missing) == missing


def test_duplicate_run_names_require_explicit_category(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_paths, "PROJECT_ROOT", tmp_path)
    for category in artifact_paths.OUTPUT_CATEGORIES[:2]:
        (tmp_path / "outputs" / category / "same_run").mkdir(parents=True)
    with pytest.raises(ValueError, match="Ambiguous"):
        artifact_paths.resolve_artifact_path("/old/outputs/same_run")


def test_inference_resolves_old_run_and_checkpoint_paths(tmp_path, monkeypatch):
    from act_rlt.infer import resolve_checkpoint

    monkeypatch.setattr(artifact_paths, "PROJECT_ROOT", tmp_path)
    run = tmp_path / "outputs/ethernet_act/model"
    model = run / "checkpoints/000100/pretrained_model"
    model.mkdir(parents=True)
    (run / "checkpoints/last").symlink_to("000100")
    for name in ("config.json", "model.safetensors", "policy_preprocessor.json",
                 "policy_postprocessor.json"):
        (model / name).touch()
    assert resolve_checkpoint(Path("/old/outputs/model")).resolve() == model
    assert resolve_checkpoint(Path("/old/outputs/model/checkpoints/last")).resolve() == model


def test_stage2_resume_accepts_relocated_model_paths(tmp_path, monkeypatch):
    import random
    from dataclasses import asdict
    from types import SimpleNamespace

    import numpy as np
    import torch

    from act_rlt.stage2 import ACTStage2Config
    from act_rlt.train_stage2 import OnlineStage2Metrics, load_checkpoint

    monkeypatch.setattr(artifact_paths, "PROJECT_ROOT", tmp_path)
    paths = {}
    for key, category in (("act_checkpoint", "ethernet_act"),
                          ("stage1_checkpoint", "ethernet_token")):
        target = tmp_path / "outputs" / category / key / "pretrained_model"
        target.mkdir(parents=True)
        paths[key] = str(target)
    config = ACTStage2Config(**paths, device="cpu")
    saved_config = config.to_dict()
    for key in paths:
        saved_config[key] = f"/old/training/checkout/outputs/{key}/pretrained_model"
    saved = {
        "config": saved_config,
        "learner": {"learner_version": 3},
        "metrics": asdict(OnlineStage2Metrics()),
        "replay": [],
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    checkpoint = tmp_path / "latest.pt"
    torch.save(saved, checkpoint)
    restored = []
    learner = SimpleNamespace(load_state_dict=restored.append)
    replay = SimpleNamespace(add=lambda _: pytest.fail("Replay should be empty"))
    assert load_checkpoint(checkpoint, config, learner, replay).env_steps == 0
    assert restored == [saved["learner"]]
    saved["config"]["beta"] += 1
    saved["config"]["act_checkpoint"] = "/old/outputs/different_model/pretrained_model"
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="resume config differs"):
        load_checkpoint(checkpoint, config, learner, replay)
