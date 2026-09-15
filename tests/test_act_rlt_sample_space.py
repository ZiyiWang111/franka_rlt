import numpy as np
import pytest

from act_rlt.data_collection.sampling import (
    insertion_pose,
    sample_pose,
)


def test_sample_pose_stays_in_requested_bounds_and_preserves_orientation():
    reference = [0.5, -0.2, 0.3, 1.0, 2.0, 3.0]
    rng = np.random.default_rng(1234)

    samples = np.asarray([sample_pose(reference, rng) for _ in range(10_000)])

    assert np.all(samples[:, 0] >= reference[0] - 0.01)
    assert np.all(samples[:, 0] <= reference[0] + 0.01)
    assert np.all(samples[:, 1] >= reference[1])
    assert np.all(samples[:, 1] <= reference[1] + 0.04)
    assert np.all(samples[:, 2] >= reference[2] - 0.01)
    assert np.all(samples[:, 2] <= reference[2] + 0.01)
    np.testing.assert_array_equal(samples[:, 3:], np.tile(reference[3:], (len(samples), 1)))


def test_sample_pose_does_not_mutate_reference():
    reference = [0.5, -0.2, 0.3, 1.0, 2.0, 3.0]
    original = reference.copy()

    sample_pose(reference, np.random.default_rng(0))

    assert reference == original


def test_insertion_pose_only_offsets_base_y():
    reference = [0.5, -0.2, 0.3, 1.0, 2.0, 3.0]

    final = insertion_pose(reference, 0.01)

    assert final == pytest.approx([0.5, -0.21, 0.3, 1.0, 2.0, 3.0])
    assert reference == [0.5, -0.2, 0.3, 1.0, 2.0, 3.0]
