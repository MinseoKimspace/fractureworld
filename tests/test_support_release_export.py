import numpy as np
import pytest

from fracture_world.dynamics_data import load_dynamics_trajectory
from test_dynamics_data import recorded_trajectory
from recorded_cases import update_archive


def test_explicit_release_labels_match_current_native_material_corner_flags(tmp_path):
    case, path = recorded_trajectory(tmp_path, supports=True)
    original = load_dynamics_trajectory(case, path)
    support = original.fixed.reshape(len(original.times), -1, 4)
    labels = support[:-1] & ~support[1:]
    update_archive(path, lambda arrays: arrays.update(corner_kinematic=support, support_release_labels=labels))
    actual = load_dynamics_trajectory(case, path)
    np.testing.assert_array_equal(actual.fixed, original.fixed)
    assert actual.metadata["observed_support_releases"] > 0


@pytest.mark.parametrize("field", ["corner_kinematic", "support_release_labels"])
def test_fabricated_export_labels_are_rejected(tmp_path, field):
    case, path = recorded_trajectory(tmp_path, supports=True)
    original = load_dynamics_trajectory(case, path)
    support = original.fixed.reshape(len(original.times), -1, 4).copy()
    labels = support[:-1] & ~support[1:]
    values = {"corner_kinematic": support, "support_release_labels": labels}
    values[field].flat[0] = ~values[field].flat[0]
    update_archive(path, lambda arrays: arrays.update(values))
    with pytest.raises(ValueError, match="support history|support release labels"):
        load_dynamics_trajectory(case, path)
