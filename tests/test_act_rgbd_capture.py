import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from act_rlt.data_collection import record_soft_approach as collector
from act_rlt.data_collection import rgbd_capture as depth


def samples(value=1000, cameras=("wrist", "front")):
    array = np.array([[0, 1, 255, 256], [1000, value, 60000, 65535]], dtype=np.uint16)
    return {name: {"array": array.copy(), "depth_scale_m": 0.001,
                   "color_timestamp_ms": 123.0, "depth_timestamp_ms": 124.0,
                   "color_frame_number": 10, "depth_frame_number": 11} for name in cameras}


def writer(tmp_path, cameras=("wrist", "front")):
    return depth.DepthEpisodeWriter(tmp_path, 0, 15, cameras, tmp_path / "cameras.json")


def test_lossless_16bit_depth_and_rgb_frame_mapping(tmp_path):
    output = writer(tmp_path)
    try:
        first, second = samples(), samples(2000)
        output.submit(first)
        output.submit(second)
        output.finish(2)
        output.publish()
        rows = [json.loads(line) for line in (output.destination / "frames.jsonl").read_text().splitlines()]
        assert [row["frame_index"] for row in rows] == [0, 1]
        assert rows[1]["timestamp_s"] == pytest.approx(1 / 15)
        for index, expected in enumerate((first, second)):
            for camera in expected:
                actual = cv2.imread(str(output.destination / rows[index]["cameras"][camera]["file"]),
                                    cv2.IMREAD_UNCHANGED)
                assert actual.dtype == np.uint16
                np.testing.assert_array_equal(actual, expected[camera]["array"])
        assert json.loads((output.destination / "episode.json").read_text())["dataset_saved"] is False
        output.mark_saved()
        assert json.loads((output.destination / "episode.json").read_text())["dataset_saved"] is True
    finally:
        output.discard()


def test_aborted_depth_staging_is_removed_and_writer_exits(tmp_path):
    output = writer(tmp_path)
    output.submit(samples())
    output.discard()
    assert not output.thread.is_alive()
    assert not output.staging.exists()
    assert not output.destination.exists()


def test_png_failure_and_frame_count_mismatch_prevent_publish(tmp_path, monkeypatch):
    output = writer(tmp_path)
    monkeypatch.setattr(depth.cv2, "imwrite", lambda *args: False)
    try:
        output.submit(samples())
        with pytest.raises(RuntimeError, match="depth saving failed"):
            output.finish(1)
    finally:
        output.discard()
    assert not output.thread.is_alive() and not output.destination.exists()


def test_frame_count_mismatch_is_reported(tmp_path):
    output = writer(tmp_path)
    try:
        output.submit(samples())
        with pytest.raises(RuntimeError, match="frame mismatch"):
            output.finish(2)
    finally:
        output.discard()
    assert not output.destination.exists()


def test_resume_rejects_changed_depth_mode_and_incomplete_dataset_save(tmp_path):
    args = collector.build_parser().parse_args([
        "--dataset", "test/rgbd", "--root", str(tmp_path), "--resume",
    ])
    collector.write_mode_marker(args)
    output = writer(tmp_path)
    try:
        output.submit(samples())
        output.finish(1)
        output.publish()
        with pytest.raises(ValueError, match="unfinished RGB/depth"):
            collector.verify_dataset_mode(args)
        output.mark_saved()
        collector.verify_dataset_mode(args)
        marker = args.root / "meta" / "soft_approach_collection.json"
        metadata = json.loads(marker.read_text())
        metadata["depth_enabled"] = False
        marker.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="RGB-only datasets"):
            collector.verify_dataset_mode(args)
    finally:
        output.discard()


def test_rgb_depth_are_read_once_from_same_frameset_and_copied():
    intrinsics = SimpleNamespace(width=4, height=2, fx=3.0, fy=3.0, ppx=2.0, ppy=1.0,
                                 coeffs=[0]*5, model="none")
    profile = SimpleNamespace(as_video_stream_profile=lambda: SimpleNamespace(get_intrinsics=lambda: intrinsics))

    class Frame:
        def __init__(self, array, number): self.array, self.number, self.profile = array, number, profile
        def get_data(self): return self.array
        def get_frame_number(self): return self.number
        def get_timestamp(self): return self.number * 10.0
        def get_frame_timestamp_domain(self): return "hardware_clock"

    rgb = np.zeros((2, 4, 3), dtype=np.uint8)
    z16 = samples(cameras=("wrist",))["wrist"]["array"]
    color, z = Frame(rgb, 10), Frame(z16, 11)
    frameset = SimpleNamespace(get_color_frame=lambda: color, get_depth_frame=lambda: z)
    calls = []
    robot = depth.RGBDFrankaRobot.__new__(depth.RGBDFrankaRobot)
    robot.depth_enabled = True
    robot._cam_streams = {"wrist": object()}
    robot.depth_calibration = {"wrist": {"depth_scale_m": 0.001}}
    robot._depth_align = {"wrist": SimpleNamespace(process=lambda frames: frames)}
    robot._read_camera_frame = lambda name: calls.append(name) or frameset
    result = robot._image_observation()
    assert calls == ["wrist"]
    assert robot.latest_depth["wrist"]["depth_frame_number"] == 11
    np.testing.assert_array_equal(robot.latest_depth["wrist"]["array"], z16)
    rgb[:] = 255
    z16[:] = 0
    assert not result["wrist"].any()
    assert robot.latest_depth["wrist"]["array"].max() == 65535


def test_depth_recorder_matches_original_one_frame_delayed_actions(tmp_path, monkeypatch):
    rows = []
    dataset = SimpleNamespace(root=tmp_path, num_episodes=0, features={}, add_frame=rows.append)
    frame_number = 0
    robot = SimpleNamespace(_cam_streams={"wrist": object()})
    robot.robot = SimpleNamespace(get_force_feedback=lambda: {
        name: [0.0] * len(names) for name, names in collector.base.FORCE_FEEDBACK_NAMES.items()
    })

    def observe():
        nonlocal frame_number
        robot.latest_depth = samples(1000 + frame_number, ("wrist",))
        pose = [frame_number * 0.001, 0, 0, 0, 0, 0]
        frame_number += 1
        return {**dict(zip(["ee_x", "ee_y", "ee_z", "ee_rx", "ee_ry", "ee_rz"], pose)),
                "gripper_width": 0.01}

    robot.get_observation = observe
    monkeypatch.setattr(collector.base, "build_dataset_frame", lambda *args, **kwargs: {})
    monkeypatch.setattr(collector.base, "manual_demo_health_tick", lambda **kwargs: None)
    recorder = depth.DepthEpisodeRecorder(
        robot=robot, dataset=dataset, observation_processor=lambda obs: obs,
        action_names=["dx", "dy", "dz", "drx", "dry", "drz", "gripper_target_width"],
        task="test", fps=30, episode_time=10, depth_enabled=True,
        depth_calibration_path=tmp_path / "cameras.json")
    try:
        recorder.start()
        recorder.capture()
        assert len(rows) == 0
        recorder.capture()
        assert len(rows) == 1
        assert recorder.finish() == 2
        recorder.prepare_depth_save()
        manifest = recorder.depth_writer.destination / "frames.jsonl"
        assert len(manifest.read_text().splitlines()) == len(rows) == 2
        recorder.mark_depth_saved()
    finally:
        recorder.discard_depth()
