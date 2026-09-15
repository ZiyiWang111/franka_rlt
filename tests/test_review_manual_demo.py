from __future__ import annotations

import numpy as np
import pandas as pd

import evo_rlt.cli.review_manual_demo as review_module
from evo_rlt.cli.review_manual_demo import (
    DEFAULT_DATASET_ROOT,
    _parser,
    _review_from_tail,
    analyze_dataframe,
)


STATE_NAMES = ["ee_x", "ee_y", "ee_z", "gripper_grasped"]


def _episode(index: int, xyz: np.ndarray, grasped: np.ndarray) -> pd.DataFrame:
    delta = np.diff(xyz, axis=0, append=xyz[-1:])
    action = np.concatenate([delta, np.zeros_like(delta)], axis=1)
    state = np.concatenate([xyz, grasped[:, None]], axis=1)
    return pd.DataFrame(
        {
            "episode_index": index,
            "frame_index": np.arange(len(xyz)),
            "timestamp": np.arange(len(xyz)) / 30,
            "action": list(action),
            "observation.state": list(state),
        }
    )


def test_analyzer_flags_frozen_episode_but_not_complete_pick() -> None:
    n = 100
    good_xyz = np.column_stack(
        [np.linspace(0, 0.6, n), np.zeros(n), np.zeros(n)]
    )
    good_grasp = np.r_[np.zeros(40), np.ones(60)]
    frozen_xyz = np.tile([0.1, 0.3, 0.2], (n, 1))
    frozen_grasp = np.zeros(n)
    df = pd.concat(
        [_episode(0, good_xyz, good_grasp), _episode(1, frozen_xyz, frozen_grasp)],
        ignore_index=True,
    )

    candidates = analyze_dataframe(
        df,
        state_names=STATE_NAMES,
        fps=30,
        require_pick_sequence=True,
    )

    assert [candidate.episode_index for candidate in candidates] == [1]
    assert "path<0.1m" in candidates[0].reasons
    assert "static_tail>=2s" in candidates[0].reasons
    assert "incomplete_pick/post_grasp<0.25m" in candidates[0].reasons


def test_parser_defaults_to_act_rlt_001_latest_episode() -> None:
    args = _parser().parse_args([])

    assert args.root == DEFAULT_DATASET_ROOT
    assert args.repo_id is None
    assert args.episode_index is None
    assert not args.suspicious_only


def test_tail_review_descends_and_deletes_only_yes(monkeypatch, tmp_path) -> None:
    state = {"total": 3}
    visualized = []
    deleted = []
    answers = iter(["y", "n", "q"])

    def fake_load_dataset(root):
        total = state["total"]
        frame = pd.DataFrame(
            {
                "episode_index": np.arange(total),
                "timestamp": np.zeros(total),
            }
        )
        return {"total_episodes": total, "fps": 15}, frame, b"snapshot"

    def fake_delete(repo_id, root, rejected, output_root, snapshot, *, preserve_backup):
        deleted.extend(rejected)
        state["total"] -= 1
        assert not preserve_backup
        return root

    monkeypatch.setattr(review_module, "_load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        review_module,
        "_visualize_episode",
        lambda episode_index, **kwargs: visualized.append(episode_index),
    )
    monkeypatch.setattr(review_module, "_apply_deletions", fake_delete)
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    _review_from_tail(repo_id="act_rlt_001", root=tmp_path, viz_command="viz")

    assert visualized == [2, 1, 0]
    assert deleted == [2]
