from __future__ import annotations

import json
import subprocess

import pytest

import evo_rlt.cli.review_manual_demo as review


def _dataset(root, total=5):
    (root / "meta").mkdir(parents=True)
    snapshot = json.dumps({"total_episodes": total}).encode()
    (root / "meta" / "info.json").write_bytes(snapshot)
    (root / "original.txt").write_text("original")
    return snapshot


def _answers(monkeypatch, answers):
    answers = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))


def test_select_dataset_navigates_back_and_retries_invalid_input(monkeypatch, tmp_path):
    root = tmp_path / "datasets"
    (root / "a_empty").mkdir(parents=True)
    target = root / "b_fine" / "sample"
    _dataset(target)
    _answers(monkeypatch, ["invalid", "9", "1", "b", "2", "1"])
    assert review._select_dataset(root) == target


def test_select_dataset_quit_stays_within_base(monkeypatch, tmp_path):
    _answers(monkeypatch, ["b", "q"])
    assert review._select_dataset(tmp_path) is None


def test_count_rejects_invalid_and_all_episode_deletion(monkeypatch):
    _answers(monkeypatch, ["0", "-1", "2.5", "5", "6", "2"])
    assert review._read_count(5) == 2


def test_count_quit_and_single_episode(monkeypatch):
    _answers(monkeypatch, ["q"])
    assert review._read_count(5) is None
    assert review._read_count(1) is None
    assert review._read_count(0) is None


def _fake_editor(monkeypatch, *, total=3, fail=False):
    calls = []
    monkeypatch.setattr(review, "_resolve_editor", lambda: "editor")

    def run(command, *, check):
        calls.append(command)
        assert check
        if fail:
            raise subprocess.CalledProcessError(1, command)
        staging = review.Path(command[command.index("--new_root") + 1])
        _dataset(staging, total)
        (staging / "original.txt").unlink()
        (staging / "filtered.txt").write_text("filtered")

    monkeypatch.setattr(review.subprocess, "run", run)
    return calls


def test_delete_tail_removes_correct_indices_and_replaces_only_after_validation(monkeypatch, tmp_path):
    root = tmp_path / "sample"
    snapshot = _dataset(root)
    calls = _fake_editor(monkeypatch)
    review._delete_tail(root, 2, snapshot)
    assert json.loads(calls[0][-1]) == [3, 4]
    assert (root / "filtered.txt").is_file()
    assert not (root / "original.txt").exists()
    assert list(tmp_path.iterdir()) == [root]


@pytest.mark.parametrize("invalid_count", [0, -1, 5, 6])
def test_delete_tail_validates_count_before_running_editor(tmp_path, invalid_count):
    root = tmp_path / "sample"
    snapshot = _dataset(root)
    with pytest.raises(ValueError):
        review._delete_tail(root, invalid_count, snapshot)
    assert (root / "original.txt").is_file()


@pytest.mark.parametrize("fail,output_total", [(True, 3), (False, 4)])
def test_failed_rebuild_preserves_original(monkeypatch, tmp_path, fail, output_total):
    root = tmp_path / "sample"
    snapshot = _dataset(root)
    _fake_editor(monkeypatch, total=output_total, fail=fail)
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        review._delete_tail(root, 2, snapshot)
    assert (root / "original.txt").read_text() == "original"
    assert (root / "meta" / "info.json").read_bytes() == snapshot


def test_source_change_aborts_before_editing(monkeypatch, tmp_path):
    root = tmp_path / "sample"
    snapshot = _dataset(root)
    calls = _fake_editor(monkeypatch)
    (root / "meta" / "info.json").write_text('{"total_episodes": 6}')
    with pytest.raises(RuntimeError, match="已发生变化"):
        review._delete_tail(root, 2, snapshot)
    assert not calls


def test_main_selects_dataset_and_deletes_count(monkeypatch, tmp_path):
    root = tmp_path / "fine" / "sample"
    _dataset(root)
    monkeypatch.setattr(review, "DATASETS_ROOT", tmp_path)
    calls = _fake_editor(monkeypatch)
    _answers(monkeypatch, ["1", "1", "2"])
    review.main()
    assert json.loads(calls[0][-1]) == [3, 4]


def test_source_change_during_rebuild_preserves_original(monkeypatch, tmp_path):
    root = tmp_path / "sample"
    snapshot = _dataset(root)
    _fake_editor(monkeypatch)
    rebuild = review.subprocess.run

    def run(command, *, check):
        rebuild(command, check=check)
        (root / "meta" / "info.json").write_text('{"total_episodes": 6}')

    monkeypatch.setattr(review.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="重建期间"):
        review._delete_tail(root, 2, snapshot)
    assert (root / "original.txt").is_file()


def test_failed_replacement_rolls_back_original(monkeypatch, tmp_path):
    root = tmp_path / "sample"
    snapshot = _dataset(root)
    _fake_editor(monkeypatch)
    rename = review.Path.rename

    def fail_staging_rename(path, target):
        if ".filtering-" in path.name:
            raise OSError("simulated rename failure")
        return rename(path, target)

    monkeypatch.setattr(review.Path, "rename", fail_staging_rename)
    with pytest.raises(OSError, match="simulated"):
        review._delete_tail(root, 2, snapshot)
    assert (root / "original.txt").is_file()
    assert (root / "meta" / "info.json").read_bytes() == snapshot
