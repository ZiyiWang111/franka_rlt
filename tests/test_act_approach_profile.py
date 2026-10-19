import copy

import numpy as np
import pytest

from act_rlt.data_collection import record_soft_approach as collector


def simulate(start, target, parameters):
    planner = collector.soft.OnlineApproach(start, target, parameters)
    rows = []
    for index in range(600):
        previous = planner.command.copy()
        acceleration = planner.acceleration.copy()
        diagnostics = planner.step(previous, 1 / parameters.fps)
        assert np.linalg.norm(planner.velocity) <= parameters.speed + 1e-10
        assert np.linalg.norm(planner.acceleration) <= parameters.acceleration + 1e-10
        assert np.linalg.norm(planner.acceleration - acceleration) <= parameters.jerk / parameters.fps + 1e-9
        assert np.linalg.norm(planner.command[:3] - previous[:3]) <= parameters.speed / parameters.fps + 1e-10
        assert np.all((planner.command[:3] - previous[:3]) * (target[:3] - previous[:3]) >= -1e-12)
        rows.append({"time": (index + 1) / parameters.fps,
                     "error": np.linalg.norm(target[:3] - planner.command[:3]),
                     "speed": np.linalg.norm(planner.velocity),
                     "rotation_done": planner.rotation_progress == planner.rotation_total,
                     **diagnostics})
        if planner.at_goal():
            break
    assert planner.at_goal()
    np.testing.assert_array_equal(planner.command, target)
    return planner, rows


def parameters(start, target):
    args = collector.build_parser().parse_args(["--dataset", "test", "--root", "/tmp/offline-unused"])
    return collector.motion_args(args, start_pose=start, target_pose=target)


@pytest.mark.parametrize("x,y,z", [(0, .02, .021), (.02, .025, .021),
                                   (-.02, .015, .01), (0, 0, .021),
                                   (.002, .002, .002), (.02, .02, -.02),
                                   (0, .02, .031), (.02, .025, .031)])
def test_finite_approach_stops_smoothly_with_varied_starts(x, y, z):
    start = np.array([x, y, z, 0, 0, 0])
    target = np.array([0, 0, 0, 0, 0, .14])
    planner, rows = simulate(start, target, parameters(start, target))
    assert planner.terminal_stop is not None
    assert rows[-1]["time"] < 4
    assert rows[-1]["speed"] == 0
    assert next(row["time"] for row in rows if row["rotation_done"]) < 1.4


def test_middle_cruises_and_final_millimetres_have_no_exponential_tail():
    start = np.array([0, .02, .021, 0, 0, 0])
    target = np.array([0, 0, 0, 0, 0, .14])
    config = parameters(start, target)
    planner, rows = simulate(start, target, config)
    cruise = [row["speed"] for row in rows if not row["finite_braking"]
              and row["speed"] > planner.cruise_speed * 0.95]
    assert len(cruise) >= 10
    assert max(cruise) / min(cruise) < 1.15
    tail_start = next(row["time"] for row in rows if row["error"] < .001)
    assert rows[-1]["time"] - tail_start < .5
    # Legacy proportional slowing retains its own behavior outside this stage.
    legacy = copy.copy(config)
    legacy.finite_approach = False
    legacy.slowdown_distance = .03
    _, previous = simulate(start, target, legacy)
    assert rows[-1]["time"] < 0.65 * previous[-1]["time"]


def test_braking_matches_entry_state_and_respects_dense_derivative_checks():
    start = np.array([.01, .02, .021, 0, 0, 0])
    target = np.zeros(6)
    config = parameters(start, target)
    planner = collector.soft.OnlineApproach(start, target, config)
    for _ in range(200):
        initial = planner.command[:3].copy(), planner.velocity.copy(), planner.acceleration.copy()
        planner.step(planner.command.copy(), 1 / config.fps)
        if planner.terminal_stop is not None:
            break
    segment = planner.terminal_stop
    assert segment is not None
    for actual, expected in zip(segment.sample(0), initial):
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    coefficients = segment.coefficients.copy()
    t = np.linspace(0, 1, 2001)
    for order, limit in enumerate((config.speed, config.acceleration, config.jerk), start=1):
        coefficients = coefficients[1:] * np.arange(1, len(coefficients))[:, None]
        values = np.polynomial.polynomial.polyval(t, coefficients).T / segment.duration**order
        assert np.max(np.linalg.norm(values, axis=1)) <= limit
    final = segment.sample(segment.duration)
    np.testing.assert_array_equal(final[0], target[:3])
    np.testing.assert_array_equal(final[1], np.zeros(3))
    np.testing.assert_array_equal(final[2], np.zeros(3))


def test_finite_braking_holds_reference_on_tracking_lag_and_resumes():
    start, target = np.array([0, .02, .021, 0, 0, 0]), np.zeros(6)
    planner = collector.soft.OnlineApproach(start, target, parameters(start, target))
    for _ in range(200):
        planner.step(planner.command.copy(), 1 / 30)
        if planner.terminal_stop is not None:
            break
    held, elapsed = planner.command.copy(), planner.terminal_elapsed
    measured = held.copy()
    measured[0] -= .02
    diagnostics = planner.step(measured, 1 / 30)
    assert diagnostics["position_paused"]
    np.testing.assert_array_equal(planner.command[:3], held[:3])
    assert planner.terminal_elapsed == elapsed
    for _ in range(200):
        planner.step(planner.command.copy(), 1 / 30)
        if planner.at_goal():
            break
    assert planner.at_goal()
