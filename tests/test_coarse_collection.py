from collections import deque
from types import SimpleNamespace
import json
import threading

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from act_rlt.data_collection import record_coarse as coarse
from act_rlt.data_collection import record_sample_space as base
from act_rlt.data_collection.coarse_input import CoarseKeyboard
from act_rlt.data_collection.sampling import SAMPLE_Z_M, SAMPLE_ROTATION_VECTOR


def options(tmp_path, *extra):
    args = coarse.build_parser().parse_args(["--root", str(tmp_path / "dataset"), *extra])
    coarse.validate_args(args)
    return args


class Keyboard:
    backend = "terminal"

    def __init__(self, batches=()):
        self.batches = deque(batches)

    def drain(self):
        return self.batches.popleft() if self.batches else []

    def direction_keys_down(self):
        return False


class Robot:
    def __init__(self, pose):
        self.pose = list(pose)
        self.events = []
        self.config = SimpleNamespace(enable_fci_keepalive=True)
        self.is_connected = True
        self.robot = self
        self.running = False

    def get_tool_pose(self):
        return self.pose.copy()

    def _stop_keepalive(self):
        self.events.append("keepalive_off")

    def _start_keepalive(self):
        self.events.append("keepalive_on")

    def resync_command_pose(self):
        self.events.append("resync")

    def servo_tool(self, target):
        self.events.append(("servo", target.copy()))
        self.pose = target.copy()

    def stop_servo(self):
        self.events.append("stop_servo")
        return True

    def stop_move(self):
        self.events.append("stop_move")
        return True

    def move_tool_traj(self, path, *, is_async):
        assert is_async
        self.events.append(("move", path))
        self.pose = path[-1][:6]

    def is_running(self):
        return self.running

    def _wait_idle(self, timeout):
        self.events.append("idle")
        return {"success": True}


def test_geometry_uses_distinct_fixed_start_and_sampling_goal(tmp_path):
    args = options(tmp_path)
    assert args.move_speed == 0.06 and args.fps == 15
    goal = coarse.point2_from_xy([.49, -.07, .6, 0, 0, 0])
    np.testing.assert_allclose(goal, [.49, -.07, SAMPLE_Z_M, *SAMPLE_ROTATION_VECTOR])
    fixed = coarse.sample_point1(np.random.default_rng(42), 0)
    np.testing.assert_allclose(fixed[:3], [.5534, .0548, .5594])
    expected = Rotation.from_euler("XYZ", [174, 3.9, 92], degrees=True)
    assert (Rotation.from_rotvec(fixed[3:]).inv() * expected).magnitude() < 1e-12
    rng = np.random.default_rng(1)
    samples = np.array([coarse.sample_point1(rng, .05) for _ in range(1000)])
    assert np.all(np.abs(samples[:, :2] - fixed[:2]) <= .05)
    assert np.ptp(samples[:, 0]) > .09 and np.ptp(samples[:, 1]) > .09
    np.testing.assert_allclose(samples[:, 2:], np.tile(fixed[2:], (1000, 1)))


@pytest.mark.parametrize("value", ["-0.5", "5.5", "0.25", "nan"])
def test_rejects_invalid_ranges(tmp_path, value):
    with pytest.raises(ValueError, match="xy-range"):
        options(tmp_path, "--xy-range-cm", value)


def test_pause_is_latched_but_never_aborts_and_range_is_next_only():
    keys = Keyboard([["p", "="], ["p"], ["-"]])
    controls = coarse.Controls(keys, 0)
    assert controls.pressed() is False
    assert controls.pause_requested and controls.half_range == .005
    assert controls.pressed() is False
    controls.poll()
    assert controls.half_range == 0 and controls.pause_requested
    keys.batches.extend([["="] * 30, ["-"] * 30])
    controls.poll()
    assert controls.half_range == .05
    controls.poll()
    assert controls.half_range == 0


def test_x_aborts_immediately_and_q_exits():
    controls = coarse.Controls(Keyboard([["x"], ["q"]]), 0)
    with pytest.raises(base.EpisodeAborted):
        controls.pressed()
    with pytest.raises(KeyboardInterrupt):
        controls.poll()


def test_no_force_schema_and_old_default_unchanged(tmp_path, monkeypatch):
    args = options(tmp_path)
    created = []

    def create(*args, **kwargs):
        created.append(kwargs["features"])
        return SimpleNamespace(features=kwargs["features"])

    monkeypatch.setattr(base.LeRobotDataset, "create", create)
    monkeypatch.setattr(base, "_write_schema_metadata", lambda *a, **kw: None)

    def factory(config):
        robot = base.FrankaRobot.__new__(base.FrankaRobot)
        robot.config = config
        return robot

    no_force, *_ = base.build_dataset_and_robot(args, robot_factory=factory, record_force_feedback=False)
    assert all(not any(word in key for word in ("force", "torque", "wrench")) for key in no_force.features)
    assert no_force.features["observation.state"]["shape"] == (22,)
    assert no_force.features["action"]["shape"] == (7,)
    assert no_force.features["observation.images.wrist"]["shape"] == (480, 640, 3)
    assert no_force.features["observation.images.front"]["shape"] == (480, 640, 3)
    original, *_ = base.build_dataset_and_robot(args, robot_factory=factory)
    assert all(f"observation.{name}" in original.features for name in base.FORCE_FEEDBACK_NAMES)


def test_no_force_recorder_does_not_read_or_write_any_force(tmp_path, monkeypatch):
    frames = []
    observations = [{**{f"ee_{k}": v for k, v in zip(("x", "y", "z", "rx", "ry", "rz"), pose)},
                     "gripper_width": .04}
                    for pose in ([.5, .1, .5, 0, 0, 0], [.51, .1, .5, 0, 0, 0])]
    robot = SimpleNamespace(robot=SimpleNamespace(get_force_feedback=lambda: pytest.fail("Force read!")),
                            get_observation=lambda: observations.pop(0))
    dataset = SimpleNamespace(features={}, add_frame=frames.append)
    names = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper_target_width"]
    recorder = base.EpisodeRecorder(robot=robot, dataset=dataset, observation_processor=lambda x: x,
                                    action_names=names, task="coarse", fps=30, episode_time=10,
                                    record_force_feedback=False)
    monkeypatch.setattr(base, "manual_demo_health_tick", lambda **kw: None)
    monkeypatch.setattr(base, "precise_sleep", lambda _: None)
    monkeypatch.setattr(base, "build_dataset_frame", lambda features, values, prefix: {prefix: dict(values)})
    recorder.start()
    recorder.capture()
    recorder.capture()
    recorder.finish()
    assert len(frames) == 2
    assert frames[0]["action"]["dx"] == pytest.approx(.01)
    assert frames[-1]["action"]["dx"] == 0
    assert all(not any(word in k for word in ("force", "torque", "wrench"))
               for f in frames for k in f["observation"])


def test_paused_wasd_preserves_z_rotation_and_stops_before_saving_xy(tmp_path, monkeypatch):
    args = options(tmp_path)
    target = coarse.point2_from_xy([.50, .05])
    robot = Robot(target)
    controls = coarse.Controls(Keyboard([["d"], ["s"], ["enter"]]), 0)
    controls.pause_requested = True
    monkeypatch.setattr(coarse.time, "sleep", lambda _: None)
    updated = coarse.edit_point2(robot, target, target[:2], controls, args)
    assert updated[0] == pytest.approx(target[0] + .001)
    assert updated[1] == pytest.approx(target[1] + .001)
    np.testing.assert_array_equal(updated[2:], target[2:])
    for event in robot.events:
        if isinstance(event, tuple) and event[0] == "servo":
            np.testing.assert_array_equal(event[1][2:], target[2:])
    assert "stop_servo" in robot.events and robot.events.index("stop_servo") < robot.events.index("idle")
    assert controls.pause_requested is False


def test_teleop_caps_lag_and_rejects_changed_z(tmp_path):
    args = options(tmp_path)
    goal = coarse.point2_from_xy([.5, .05])
    robot = Robot(goal)
    teleop = coarse.XYTeleop(robot, goal, goal[:2], args)
    teleop.step(np.array([.1, 0]))
    assert robot.pose[0] - goal[0] == pytest.approx(.003)
    teleop.stop()
    robot.pose[2] += .01
    with pytest.raises(RuntimeError, match="Z changed"):
        teleop.step(np.array([.001, 0]))


@pytest.mark.parametrize("randomize_yaw", [False, True])
def test_recording_pauses_only_after_save_and_uses_updated_goal_next_episode(tmp_path, monkeypatch, randomize_yaw):
    flags = ["--randomize-point1-yaw"] if randomize_yaw else []
    args = options(tmp_path, "--episodes", "2", *flags)
    events = []
    robot = Robot(coarse.POINT1_POSE)
    controls = coarse.Controls(Keyboard(), 0)
    dataset = SimpleNamespace(meta=SimpleNamespace(total_episodes=5))

    def save():
        events.append("save")
        dataset.meta.total_episodes += 1

    dataset.save_episode = save
    goal = coarse.point2_from_xy([.5, .05])
    updated = coarse.point2_from_xy([.51, .06])

    def collect(recorder, target, keys, settings):
        events.append(("record", target.copy()))
        if dataset.meta.total_episodes == 5:
            keys.pause_requested = True
        return 20

    def edit(*a):
        assert events[-1] == "save"
        events.append("pause")
        controls.half_range = .005
        controls.pause_requested = False
        return updated

    monkeypatch.setattr(coarse, "move_outside_episode", lambda robot, target, *a: events.append(("return", target.copy())))
    monkeypatch.setattr(coarse, "collect_episode", collect)
    monkeypatch.setattr(coarse, "edit_point2", edit)
    monkeypatch.setattr(base, "normalize_lerobot_scalar_buffer", lambda _: None)
    assert coarse.run_collection(dataset, robot, None, controls, args, goal) == 2
    assert [e if isinstance(e, str) else e[0] for e in events] == [
        "return", "record", "save", "pause", "return", "record", "save", "return"]
    assert events[1][1] == goal and events[5][1] == updated
    first = json.loads((args.root / "meta/coarse_episodes/episode_000005.json").read_text())
    second = json.loads((args.root / "meta/coarse_episodes/episode_000006.json").read_text())
    assert first["point1_xy_half_range_m"] == 0
    assert second["point1_xy_half_range_m"] == .005
    assert second["point2_pose"] == updated and second["dataset_saved"]


def test_abort_discards_and_does_not_finish_or_save(tmp_path, monkeypatch):
    args = options(tmp_path)
    cleared = []
    recorder = SimpleNamespace(start=lambda: None, capture=lambda: None, abort_monitor=None,
                               dataset=SimpleNamespace(clear_episode_buffer=lambda: cleared.append(True)))

    def abort(*a, **kw):
        raise base.EpisodeAborted()

    monkeypatch.setattr(base, "run_recorded_move", abort)
    with pytest.raises(base.EpisodeAborted):
        coarse.collect_episode(recorder, coarse.point2_from_xy([.5, .1]), coarse.Controls(Keyboard(), 0), args)
    assert cleared == [True] and recorder.abort_monitor is None


@pytest.mark.parametrize("tracking_error", [0.0, .01])
def test_real_recording_path_uses_point2_and_006_speed_and_checks_arrival(tmp_path, monkeypatch, tracking_error):
    args = options(tmp_path)
    goal = coarse.point2_from_xy([.51, .06])
    robot = Robot(coarse.POINT1_POSE)
    controls = coarse.Controls(Keyboard([["p"]]), 0)
    frames, cleared, finished = [], [], []

    def capture():
        controls.pressed()
        frames.append(robot.pose.copy())

    def finish():
        finished.append(True)
        return len(frames)

    recorder = SimpleNamespace(robot=robot, fps=15, start=lambda: None, capture=capture,
                               set_command=lambda *a: None, clear_command=lambda: None,
                               finish=finish, dataset=SimpleNamespace(clear_episode_buffer=lambda: cleared.append(True)),
                               abort_monitor=None)
    original_move = robot.move_tool_traj

    def move(path, **kwargs):
        original_move(path, **kwargs)
        robot.pose[0] += tracking_error

    robot.move_tool_traj = move
    if tracking_error:
        with pytest.raises(RuntimeError, match="arrival tolerances"):
            coarse.collect_episode(recorder, goal, controls, args)
        assert cleared == [True] and not finished
    else:
        assert coarse.collect_episode(recorder, goal, controls, args) >= 3
        assert finished == [True] and not cleared
    assert controls.pause_requested
    commands = [event[1] for event in robot.events if isinstance(event, tuple) and event[0] == "move"]
    assert commands == [[[ *goal, .06, 0.0, 0.0 ]]]


def test_resume_rejects_ordinary_force_dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    with pytest.raises(ValueError, match="no-force coarse"):
        options(tmp_path, "--resume")
    (root / "meta/coarse_collection.json").write_text(json.dumps({"mode": coarse.MODE}))
    (root / "meta/info.json").write_text(json.dumps({"features": {"observation.external_wrench": {}}}))
    with pytest.raises(ValueError, match="must not contain"):
        options(tmp_path, "--resume")


def test_real_key_release_and_diagonal_normalization(monkeypatch):
    keyboard = CoarseKeyboard.__new__(CoarseKeyboard)
    keyboard.backend = "evdev"
    keyboard._lock = threading.Lock()
    keyboard._held = set()
    keyboard._releases = {}
    keyboard._events = deque()
    keyboard.feed("w", True)
    keyboard.feed("d", True)
    assert np.linalg.norm(keyboard.direction()) == pytest.approx(1)
    keyboard.feed("w", False)
    np.testing.assert_array_equal(keyboard.direction(), [0, 1])
    keyboard.feed("d", False)
    assert not keyboard.direction_keys_down()
    keyboard.feed("p", True)
    keyboard.feed("p", True)
    assert list(keyboard._events).count("p") == 1


@pytest.mark.parametrize("randomize_yaw", [False, True])
def test_return_hold_and_automatic_pause_every_five_successful_new_episodes(tmp_path, monkeypatch, randomize_yaw):
    flags = ["--randomize-point1-yaw"] if randomize_yaw else []
    args = options(tmp_path, "--episodes", "11", *flags)
    assert args.return_speed == .05
    clock = {"now": 0.0}
    monkeypatch.setattr(coarse.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(coarse.time, "sleep", lambda seconds: clock.__setitem__("now", clock["now"] + seconds))
    controls = coarse.Controls(Keyboard(), 0)
    robot = Robot(coarse.POINT1_POSE)
    dataset = SimpleNamespace(meta=SimpleNamespace(total_episodes=7))
    returns, ended, paused = [], [], []

    def move(*a):
        returns.append(clock["now"])

    def collect(*a):
        clock["now"] += 1
        ended.append(clock["now"])
        return 20

    def save():
        dataset.meta.total_episodes += 1
        clock["now"] += .2  # Saving overlaps the requested 0.5 s hold.

    def edit(robot, point2, origin_xy, keys, settings):
        paused.append(dataset.meta.total_episodes - 7)
        keys.pause_requested = False
        clock["now"] += 2  # Operator takes longer than the minimum hold.
        return point2

    dataset.save_episode = save
    monkeypatch.setattr(coarse, "move_outside_episode", move)
    monkeypatch.setattr(coarse, "collect_episode", collect)
    monkeypatch.setattr(coarse, "edit_point2", edit)
    monkeypatch.setattr(base, "normalize_lerobot_scalar_buffer", lambda _: None)
    assert coarse.run_collection(dataset, robot, None, controls, args, coarse.point2_from_xy([.5, .05])) == 11
    assert paused == ([] if randomize_yaw else [5, 10])
    gaps = np.array(returns[1:]) - np.array(ended)
    assert np.all(gaps >= .5 - 1e-9)
    assert gaps[0] == pytest.approx(.5)
    if not randomize_yaw:
        assert gaps[4] >= 2 and gaps[9] >= 2
    else:
        np.testing.assert_allclose(gaps, .5)
    metadata = json.loads((args.root / "meta/coarse_episodes/episode_000007.json").read_text())
    assert metadata["point1_randomize_yaw"] is randomize_yaw


def test_randomized_point1_yaw_preserves_position_and_rx_ry_and_is_seeded():
    rng = np.random.default_rng(42)
    samples = np.array([coarse.sample_point1(rng, 0, randomize_yaw=True) for _ in range(500)])
    np.testing.assert_allclose(samples[:, :3], np.tile(coarse.POINT1_POSITION_M, (500, 1)))
    angles = Rotation.from_rotvec(samples[:, 3:]).as_euler("XYZ", degrees=True)
    np.testing.assert_allclose(angles[:, :2], np.tile([174.0, 3.9], (500, 1)), atol=1e-10)
    assert np.all((angles[:, 2] >= 67) & (angles[:, 2] <= 117))
    assert angles[:, 2].min() < 72 and angles[:, 2].max() > 112
    first = coarse.sample_point1(np.random.default_rng(17), .02, randomize_yaw=True)
    second = coarse.sample_point1(np.random.default_rng(17), .02, randomize_yaw=True)
    np.testing.assert_array_equal(first, second)


def test_p_during_post_recording_hold_still_pauses_before_return(tmp_path, monkeypatch):
    args = options(tmp_path)
    clock = {"now": 0.0}
    monkeypatch.setattr(coarse.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(coarse.time, "sleep", lambda seconds: clock.__setitem__("now", clock["now"] + seconds))
    controls = coarse.Controls(Keyboard([[], ["p"]]), 0)
    goal = coarse.point2_from_xy([.5, .05])
    updated = coarse.point2_from_xy([.51, .06])
    pauses = []

    def edit(*a):
        pauses.append(clock["now"])
        controls.pause_requested = False
        return updated

    monkeypatch.setattr(coarse, "edit_point2", edit)
    assert coarse.prepare_return(None, goal, goal[:2], controls, args, 0) == updated
    assert len(pauses) == 1 and pauses[0] < .5
    assert clock["now"] >= .5
