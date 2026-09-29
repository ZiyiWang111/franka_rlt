import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from evo_rlt.adapters.lerobot.policies.pi05_masked_loss import pi05_masked_forward, reduce_action_losses


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_pi05_franka", PROJECT / "scripts/prepare_pi05_franka.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    for name in ["meta/episodes/chunk-000", "data/chunk-000", "videos/observation.images.wrist/chunk-000"]:
        (root / name).mkdir(parents=True)
    state = np.arange(42, dtype=np.float32).reshape(6, 7)
    action = state / 100
    action[:, -1] = .0128
    ids = np.array([0, 0, 0, 1, 1, 1])
    vectors = {"observation.state": state, "action": action, "task_index": np.full((6, 1), 3)}
    table = pa.table({
        "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 7)),
        "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 7)),
        "episode_index": ids, "task_index": [3] * 6,
        "frame_index": [0, 1, 2, 0, 1, 2],
        "complementary_info.policy_action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 7)),
    })
    for index, part in enumerate([table.slice(0, 4), table.slice(4)]):
        pq.write_table(part, root / f"data/chunk-000/file-{index:03d}.parquet")
    episodes = []
    for ep in range(2):
        row = {"episode_index": ep, "length": 3, "tasks": ["old task"]}
        for key, values in vectors.items():
            row.update({f"stats/{key}/{k}": v for k, v in prepare.exact_stats(values[ids == ep]).items()})
        episodes.append(row)
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.table({"task_index": [3], "task": ["old task"]}), root / "meta/tasks.parquet")
    info = {"total_frames": 6, "total_episodes": 2, "total_tasks": 1, "features": {
        "observation.state": {"dtype": "float32", "shape": [7], "names": prepare.STATE_NAMES},
        "action": {"dtype": "float32", "shape": [7], "names": prepare.ACTION_NAMES + ["gripper_target_width"]},
        "observation.images.wrist": {"dtype": "video", "shape": [720, 1280, 3]},
    }}
    (root / "meta/info.json").write_text(json.dumps(info))
    # Simulate the old approximation: average episode quantiles.
    stats = {k: prepare.exact_stats(v) for k, v in vectors.items()}
    for key in vectors:
        for quantile in prepare.QUANTILES:
            stats[key][quantile] = np.mean([r[f"stats/{key}/{quantile}"] for r in episodes], axis=0).tolist()
    (root / "meta/stats.json").write_text(json.dumps(stats))
    (root / "videos/observation.images.wrist/chunk-000/file-000.mp4").write_bytes(b"unchanged-video-fixture")
    return root


def test_dataset_conversion_relabels_and_computes_global_quantiles(source, tmp_path):
    original_hashes = hashes(source)
    dst = tmp_path / "prepared"
    prepare.prepare_dataset(source, dst)
    assert hashes(source) == original_hashes
    info = json.loads((dst / "meta/info.json").read_text())
    assert info["features"]["action"]["names"] == prepare.ACTION_NAMES
    tables = [pq.read_table(p) for p in sorted((dst / "data").rglob("*.parquet"))]
    table = pa.concat_tables(tables)
    assert table["action"].type.list_size == 6
    assert table["task_index"].to_pylist() == [0] * 6
    expected = np.arange(42, dtype=np.float32).reshape(6, 7)[:, :6] / 100
    np.testing.assert_array_equal(table["action"].to_pylist(), expected)
    stats = json.loads((dst / "meta/stats.json").read_text())
    np.testing.assert_allclose(stats["action"]["q01"], np.quantile(expected, .01, axis=0))
    assert stats["action"]["q01"] != json.loads((source / "meta/stats.json").read_text())["action"]["q01"][:6]
    assert pq.read_table(dst / "meta/tasks.parquet").to_pylist() == [{"task_index": 0, "task": prepare.TASK}]
    eps = pq.read_table(dst / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    for ep in eps:
        assert ep["tasks"] == [prepare.TASK]
        np.testing.assert_allclose(ep["stats/action/q99"], np.quantile(expected[ep["episode_index"] * 3:][:3], .99, axis=0))
        assert ep["stats/task_index/max"] == [0]
    assert (dst / "videos/observation.images.wrist/chunk-000/file-000.mp4").read_bytes() == b"unchanged-video-fixture"
    assert table["complementary_info.policy_action"].type.list_size == 7


def test_conversion_refuses_existing_destination(source):
    before = hashes(source)
    with pytest.raises(FileExistsError):
        prepare.prepare_dataset(source, source)
    assert hashes(source) == before


def test_masked_forward_excludes_time_padding_and_unused_action_dims():
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    values = torch.ones(2, 3, 32)
    values[1] = 4
    values[0, 1:] = 10000
    values[:, :, 6:] = 10000
    values.requires_grad_()
    policy = SimpleNamespace(
        config=SimpleNamespace(output_features={"action": SimpleNamespace(shape=(6,))}),
        _preprocess_images=lambda batch: ([], []), prepare_action=lambda batch: batch["action"],
        model=SimpleNamespace(forward=lambda *args: values),
    )
    batch = {"action": torch.zeros(2, 3, 6), "action_is_pad": torch.tensor([[False, True, True], [False, False, False]]),
             OBS_LANGUAGE_TOKENS: torch.zeros(2, 1), OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1)}
    loss, metrics = pi05_masked_forward(policy, batch)
    assert loss.item() == pytest.approx(2.5)
    assert metrics["loss_per_dim"] == pytest.approx([2.5] * 6)
    loss.backward()
    assert torch.count_nonzero(values.grad[0, 1:]) == 0
    assert torch.count_nonzero(values.grad[:, :, 6:]) == 0
    per_sample, _ = pi05_masked_forward(policy, batch, reduction="none")
    torch.testing.assert_close(per_sample, torch.tensor([1., 4.]))


def test_no_padding_preserves_loss_and_all_padding_is_zero():
    values = torch.rand(2, 5, 6, requires_grad=True)
    loss, _ = reduce_action_losses(values)
    torch.testing.assert_close(loss, values.mean())
    zero, _ = reduce_action_losses(values, torch.ones(2, 5, dtype=torch.bool))
    zero.backward()
    assert zero.item() == 0 and torch.count_nonzero(values.grad) == 0


def test_training_recipe_keeps_state_and_disables_relative_actions(source, tmp_path):
    import draccus
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    dst = tmp_path / "prepared"
    prepare.prepare_dataset(source, dst)
    result = subprocess.run([
        sys.executable, str(PROJECT / "scripts/train_pi05_franka.py"), "--root", str(dst),
        "--base-model", str(tmp_path / "base"), "--output", str(tmp_path / "run"), "--dry-run",
    ], text=True, capture_output=True, check=True)
    command = shlex.split(result.stdout.splitlines()[-1])
    assert command[3] == "evo_rlt.cli.train_pi05_masked"
    assert "--save_freq=10000" in command and "--policy.input_features=null" in command
    policy_args = [x.replace("--policy.", "--", 1) for x in command if x.startswith("--policy.") and not x.startswith("--policy.path=")]
    cfg = draccus.parse(PI05Config, args=policy_args)
    assert not cfg.use_relative_actions and not cfg.train_expert_only
    assert cfg.gradient_checkpointing and cfg.chunk_size == 10
    assert cfg.scheduler_decay_steps == 30000
    assert cfg.get_scheduler_preset().decay_lr == pytest.approx(5e-6)
