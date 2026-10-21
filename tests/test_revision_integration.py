import json
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import pytest
import torch

from fracture_world.dynamics_runtime import RolloutConfig, _correct, fracture_candidates, rollout_dynamics
from fracture_world.mesh import VolumeMesh
from test_dynamics_runtime import portable_rollout_inputs


@pytest.mark.parametrize("method", ["central", "consensus"])
def test_default_correction_follows_the_checkpoint_constant_acceleration_prior(tmp_path, method):
    checkpoint, initial, topology, _ = portable_rollout_inputs(tmp_path)
    saved = torch.load(checkpoint, weights_only=True)
    saved["model_config"]["motion_prior"] = "constant_acceleration"
    torch.save(saved, checkpoint)
    output = tmp_path / method
    report = rollout_dynamics(checkpoint, initial, output, 1, method,
                              RolloutConfig(fracture_samples=0, fracture_policy="marginal_search"), "cpu")
    assert report["status"] == "completed"
    assert report["configuration"]["kinematic_rule"] == "constant_acceleration"
    assert report["all_committed_energy_valid"]
    assert report["all_committed_swept_geometry_valid"]
    with np.load(output / "trajectory.npz", allow_pickle=False) as archive:
        assert_allclose(archive["positions"][-1], topology.rest_positions + [0, 0, -0.00005], atol=1e-8)
        assert_allclose(archive["velocities"][-1], np.tile([0, 0, -0.01], (8, 1)), atol=1e-8)


def test_damped_teacher_is_rejected_by_the_undamped_constrained_model(tmp_path):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path)
    with np.load(initial, allow_pickle=False) as source:
        values = {key: np.array(source[key], copy=True) for key in source.files}
    metadata = json.loads(str(values["metadata_json"].item()))
    metadata["teacher_physics"] = {"rayleigh_mass_damping": 0.15}
    values["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(initial, **values)
    with pytest.raises(ValueError, match="zero damping"):
        rollout_dynamics(checkpoint, initial, tmp_path / "invalid", 1, "consensus", device="cpu")
    assert not (tmp_path / "invalid").exists()


def test_fm_search_contains_only_generated_samples_without_classifier_or_extra_cracks():
    fields = [torch.tensor([0.2, -0.2, -0.4]), torch.tensor([-0.2, 0.2, -0.4])]

    class SampleModel:
        def sample_fracture(self, batch, steps, noise):
            return fields.pop(0)

    batch = {"positions": torch.zeros(4, 3), "broken": torch.zeros(3, dtype=torch.bool)}
    proposal = {"fracture_probability": torch.ones(3)}
    config = RolloutConfig(fracture_policy="fm_search", fracture_samples=2)
    candidates = fracture_candidates(SampleModel(), batch, proposal, config, 0)
    assert {tuple(mask) for mask in candidates} == {(True, False, False), (False, True, False)}


@pytest.mark.parametrize("method", ["central", "consensus"])
def test_physical_correction_can_prefer_a_sampled_fracture_without_reference_labels(method):
    mesh = VolumeMesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                      [[0, 1, 2, 3], [4, 1, 2, 3]])
    mass = np.repeat(mesh.volumes / 4, 4)
    rest = mesh.nodes[mesh.tets].reshape(-1, 3)
    center = np.average(rest, axis=0, weights=mass)
    positions = center + 1.05 * (rest - center)
    velocities = np.zeros_like(positions)
    normal = mesh.faces.normals[mesh.internal_faces[0]]
    displacement = np.vstack([np.tile(-2e-3 * normal, (4, 1)), np.tile(1e-3 * normal, (4, 1))])
    initial = SimpleNamespace(topology=SimpleNamespace(mesh=mesh, masses=mass), dt=0.01,
                              material=dict(rest_density=1.0, youngs_modulus=1000.0, poissons_ratio=0.25),
                              gravity=np.zeros(3), metadata={})
    model = SimpleNamespace(config={"velocity_scale": 1.0, "motion_prior": "backward_euler"})
    proposal = dict(positions=torch.tensor(positions + displacement), velocities=torch.tensor(displacement / initial.dt))
    objectives = {}
    for broken in (False, True):
        result, diagnostic = _correct(initial, positions, velocities, proposal, np.array([broken]),
                                      np.zeros(8, dtype=bool), model, RolloutConfig(), method)
        assert result is not None, diagnostic
        assert result.diagnostics["energy"]["energy_satisfied"]
        objectives[broken] = result.objective
    assert objectives[True] < 0.01 * objectives[False]
