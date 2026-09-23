import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from act_rlt.data_collection.sampling import (
    insertion_pose,
    insertion_pose_minus_z,
    sample_pose,
    sample_z_insertion_pose,
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


@pytest.mark.parametrize("sampler", [sample_pose, sample_z_insertion_pose])
def test_rotation_option_perturbs_only_rz_within_five_degrees(sampler):
    reference = [0.5, -0.2, 0.3, 1.0, 2.0, 3.0]
    samples = np.asarray([
        sampler(reference, np.random.default_rng(seed), z_rotation_degrees=5.0)
        for seed in range(100)
    ])

    reference_rotation = Rotation.from_rotvec(reference[3:])
    relative = np.asarray([
        (Rotation.from_rotvec(sample[3:]) * reference_rotation.inv()).as_rotvec()
        for sample in samples
    ])
    offsets_deg = np.rad2deg(relative[:, 2])
    assert np.all(np.abs(offsets_deg) <= 5.0)
    assert np.any(np.abs(offsets_deg) > 0.01)
    np.testing.assert_allclose(relative[:, :2], 0.0, atol=1e-12)


def test_insertion_pose_only_offsets_base_y():
    reference = [0.5, -0.2, 0.3, 1.0, 2.0, 3.0]

    final = insertion_pose(reference, 0.01)

    assert final == pytest.approx([0.5, -0.21, 0.3, 1.0, 2.0, 3.0])
    assert reference == [0.5, -0.2, 0.3, 1.0, 2.0, 3.0]


def test_z_insertion_mode_sample_bounds_and_final_retreat_geometry():
    reference = [0.5, -0.2, 0.3, 1.0, 2.0, 3.0]
    rng = np.random.default_rng(44)
    samples = np.asarray([sample_z_insertion_pose(reference, rng) for _ in range(10_000)])

    assert np.all(samples[:, 0] >= reference[0] - 0.01)
    assert np.all(samples[:, 0] <= reference[0] + 0.01)
    assert np.all(samples[:, 1] >= reference[1] - 0.01)
    assert np.all(samples[:, 1] <= reference[1] + 0.01)
    assert np.all(samples[:, 2] >= reference[2])
    assert np.all(samples[:, 2] <= reference[2] + 0.02)
    np.testing.assert_array_equal(samples[:, 3:], np.tile(reference[3:], (len(samples), 1)))

    final = insertion_pose_minus_z(reference, 0.01)
    assert final == pytest.approx([0.5, -0.2, 0.29, 1.0, 2.0, 3.0])
    assert reference == [0.5, -0.2, 0.3, 1.0, 2.0, 3.0]
