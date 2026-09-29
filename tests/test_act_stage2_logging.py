"""Metrics retain every chunk without file or terminal I/O in the learner."""

import json
import threading
import time

import pytest

from act_rlt.stage2_logging import Stage2MetricsWriter


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_close_drains_all_records_and_freezes_mutable_diagnostics(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"previous_run": true}\n')
    writer = Stage2MetricsWriter(path, format_terminal=str, interval_s=60)
    record = {"chunks": 1, "diagnostics": {"delta": [0.1, 0.2]}}
    writer.append(record)
    record["chunks"] = 999
    record["diagnostics"]["delta"][0] = 999
    for chunk in range(2, 21):
        writer.append({"chunks": chunk})
    # Simulate an exception in training: cleanup still drains pending records.
    try:
        raise RuntimeError("training stopped")
    except RuntimeError:
        writer.close()
    rows = read_records(path)
    assert rows[0] == {"previous_run": True}
    assert rows[1] == {"chunks": 1, "diagnostics": {"delta": [0.1, 0.2]}}
    assert [row["chunks"] for row in rows[1:]] == list(range(1, 21))
    assert not writer._thread.is_alive()
    writer.close()  # Cleanup is idempotent.
    with pytest.raises(RuntimeError, match="closed"):
        writer.append({"chunks": 21})


def test_periodic_flush_keeps_all_chunks_but_prints_only_latest(tmp_path, capsys):
    path = tmp_path / "metrics.jsonl"
    formatted = []
    ready = threading.Event()

    def format_terminal(record):
        formatted.append((record["chunks"], time.monotonic(), threading.get_ident()))
        ready.set()
        return f"chunk={record['chunks']}"

    writer = Stage2MetricsWriter(path, format_terminal=format_terminal, interval_s=0.05)
    try:
        for chunk in (1, 2, 3):
            writer.append({"chunks": chunk})
        assert ready.wait(2)
        assert [row["chunks"] for row in read_records(path)] == [1, 2, 3]
        assert formatted[0][0] == 3
        ready.clear()
        writer.append({"chunks": 4})
        assert ready.wait(2)
        assert [row["chunks"] for row in read_records(path)] == [1, 2, 3, 4]
    finally:
        writer.close()
    assert [item[0] for item in formatted] == [3, 4]
    assert formatted[1][1] - formatted[0][1] >= 0.045
    assert all(item[2] != threading.get_ident() for item in formatted)
    assert capsys.readouterr().out.splitlines() == ["chunk=3", "chunk=4"]


def test_slow_disk_does_not_block_appending_and_serialization_is_off_thread(tmp_path, monkeypatch):
    path = tmp_path / "metrics.jsonl"
    blocked = threading.Event()
    release = threading.Event()
    appended = threading.Event()
    producer_errors = []
    serialization_threads = []
    original_write = Stage2MetricsWriter._write_batch
    original_dumps = json.dumps

    def slow_write(stream, pending):
        if pending:
            blocked.set()
            if not release.wait(2):
                raise RuntimeError("test did not release disk writer")
        original_write(stream, pending)

    def track_serialization(*args, **kwargs):
        serialization_threads.append(threading.get_ident())
        return original_dumps(*args, **kwargs)

    monkeypatch.setattr(Stage2MetricsWriter, "_write_batch", staticmethod(slow_write))
    monkeypatch.setattr(json, "dumps", track_serialization)
    writer = Stage2MetricsWriter(path, format_terminal=str, interval_s=0.02)

    def producer():
        try:
            for chunk in range(2, 12):
                writer.append({"chunks": chunk})
        except BaseException as exc:
            producer_errors.append(exc)
        finally:
            appended.set()

    thread = threading.Thread(target=producer)
    try:
        writer.append({"chunks": 1})
        assert blocked.wait(2)
        thread.start()
        assert appended.wait(0.5), "learner blocked behind disk I/O"
        assert not producer_errors
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(timeout=2)
        writer.close()
    assert [row["chunks"] for row in read_records(path)] == list(range(1, 12))
    assert serialization_threads
    assert set(serialization_threads) == {writer._thread.ident}


def test_open_failure_is_reported_before_startup_returns(tmp_path):
    with pytest.raises(RuntimeError, match="metrics writer failed"):
        Stage2MetricsWriter(tmp_path / "missing" / "metrics.jsonl", format_terminal=str)


def test_background_write_failure_is_reported_to_training_and_cleanup(tmp_path, monkeypatch):
    def fail_write(stream, pending):
        if pending:
            raise OSError("disk full")

    monkeypatch.setattr(Stage2MetricsWriter, "_write_batch", staticmethod(fail_write))
    writer = Stage2MetricsWriter(
        tmp_path / "metrics.jsonl", format_terminal=str, interval_s=0.01,
    )
    writer.append({"chunks": 1})
    writer._thread.join(timeout=2)
    assert not writer._thread.is_alive()
    with pytest.raises(RuntimeError, match="disk full"):
        writer.append({"chunks": 2})
    with pytest.raises(RuntimeError, match="disk full"):
        writer.close()


def test_full_queue_fails_promptly_instead_of_silently_dropping_records(tmp_path, monkeypatch):
    path = tmp_path / "metrics.jsonl"
    blocked = threading.Event()
    release = threading.Event()
    original_write = Stage2MetricsWriter._write_batch

    def slow_write(stream, pending):
        if pending:
            blocked.set()
            if not release.wait(2):
                raise RuntimeError("test did not release disk writer")
        original_write(stream, pending)

    monkeypatch.setattr(Stage2MetricsWriter, "_write_batch", staticmethod(slow_write))
    writer = Stage2MetricsWriter(path, format_terminal=str, interval_s=0.01, queue_capacity=2)
    try:
        writer.append({"chunks": 1})
        assert blocked.wait(2)
        writer.append({"chunks": 2})
        writer.append({"chunks": 3})
        with pytest.raises(RuntimeError, match="queue is full"):
            writer.append({"chunks": 4})
    finally:
        release.set()
        writer.close()
    assert [row["chunks"] for row in read_records(path)] == [1, 2, 3]


@pytest.mark.parametrize("interval_s", [0, -1, float("nan"), float("inf")])
def test_invalid_interval_is_rejected(tmp_path, interval_s):
    with pytest.raises(ValueError, match="interval"):
        Stage2MetricsWriter(tmp_path / "metrics.jsonl", format_terminal=str, interval_s=interval_s)


@pytest.mark.parametrize("queue_capacity", [0, -1, 1.5])
def test_unbounded_or_invalid_queue_is_rejected(tmp_path, queue_capacity):
    with pytest.raises(ValueError, match="capacity"):
        Stage2MetricsWriter(tmp_path / "metrics.jsonl", format_terminal=str, queue_capacity=queue_capacity)
