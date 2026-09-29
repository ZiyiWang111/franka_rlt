import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from act_rlt.data_collection.sampling import (
    SAMPLE_EULER_XYZ_DEG,
    insertion_pose_minus_z,
    sample_z_insertion_pose,
)


def test_z_insertion_sample_uses_the_fixed_collection_box_and_orientation():
    reference = [0.5, -0.2, 0.3, 1.0, 2.0, 3.0]
    rng = np.random.default_rng(1234)

    samples = np.asarray([sample_z_insertion_pose(reference, rng) for _ in range(10_000)])

    assert np.all(samples[:, 0] >= reference[0] - 0.02)
    assert np.all(samples[:, 0] <= reference[0] + 0.02)
    assert np.all(samples[:, 1] >= reference[1] + 0.015)
    assert np.all(samples[:, 1] <= reference[1] + 0.025)
    np.testing.assert_array_equal(samples[:, 2], 0.402)
    expected = Rotation.from_euler("XYZ", SAMPLE_EULER_XYZ_DEG, degrees=True).as_rotvec()
    np.testing.assert_allclose(samples[:, 3:], np.tile(expected, (len(samples), 1)))


def test_z_insertion_sample_does_not_mutate_reference():
    reference = [0.5, -0.2, 0.3, 1.0, 2.0, 3.0]
    original = reference.copy()

    sample_z_insertion_pose(reference, np.random.default_rng(0))

    assert reference == original


def test_fixed_sample_orientation_is_close_to_logged_taught_orientation():
    reference = [0.561755, -0.225792, 0.387321, -2.199291, 2.224642, -0.030117]
    sample = sample_z_insertion_pose(reference, np.random.default_rng(0))
    taught_rotation = Rotation.from_rotvec(reference[3:])
    sample_rotation = Rotation.from_rotvec(sample[3:])
    difference_deg = np.rad2deg((taught_rotation.inv() * sample_rotation).magnitude())

    assert difference_deg < 0.2


def test_z_insertion_final_retreat_geometry():
    reference = [0.5, -0.2, 0.3, 1.0, 2.0, 3.0]

    final = insertion_pose_minus_z(reference, 0.01)
    assert final == pytest.approx([0.5, -0.2, 0.29, 1.0, 2.0, 3.0])
    assert reference == [0.5, -0.2, 0.3, 1.0, 2.0, 3.0]
