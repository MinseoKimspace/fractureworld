import importlib

import pytest
import torch

from fracture_world.dynamics_model import (
    DYNAMICS_EDGE_TYPES,
    DYNAMICS_INPUT_DIMS,
    NeuralDynamicsCorrector,
    TemporalDynamicsModel,
    build_dynamics_graph,
    build_fracture_adjacency,
)


def dynamics_batch(broken=False):
    rest = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                         [1., 1., 1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    return dict(rest_positions=rest, positions=rest.clone(), velocities=torch.zeros_like(rest),
                masses=torch.ones(8), face_pairs=torch.tensor([[[1, 5], [2, 6], [3, 7]]]),
                broken=torch.tensor([broken]), fixed=torch.tensor([True, False, False, False, False, False, False, False]),
                forces=torch.tensor([[0., 0., -2.]]).expand(8, -1).clone(),
                material=torch.tensor([1000., 1.e6, 0.3, 1.e4]), dt=torch.tensor(0.01))


def small_model(model_type=TemporalDynamicsModel):
    return model_type(hidden=12, heads=3, graph_layers=2, transformer_layers=1,
                      contact_radius=0.05, material_scale=(1000, 1.e6, 1, 1.e4))


def adjacent_face_batch():
    batch = dynamics_batch()
    extra = torch.tensor([[-1., 1., 1.], [1., 1., 1.], [0., 1., 0.], [0., 0., 1.]])
    for key in ("rest_positions", "positions"):
        batch[key] = torch.cat((batch[key], extra.clone()))
    for key in ("velocities", "forces"):
        batch[key] = torch.cat((batch[key], torch.zeros_like(extra)))
    batch["masses"] = torch.ones(12)
    batch["fixed"] = torch.cat((batch["fixed"], torch.zeros(4, dtype=torch.bool)))
    batch["face_pairs"] = torch.cat((batch["face_pairs"], torch.tensor([[[4, 9], [6, 10], [7, 11]]])))
    batch["broken"] = torch.zeros(2, dtype=torch.bool)
    return batch


def test_graph_contains_current_scene_topology_loading_and_new_contact_relations():
    batch = dynamics_batch()
    intact = build_dynamics_graph(batch, contact_radius=0.05)
    assert set(intact.edge_types) == set(DYNAMICS_EDGE_TYPES)
    for name, width in DYNAMICS_INPUT_DIMS.items():
        assert intact[name].x.shape[1] == width
        assert torch.isfinite(intact[name].x).all()
    assert intact["interaction"].contact_pairs.shape == (0, 2)
    assert intact["corner", "bounds", "face"].edge_index.shape == (2, 6)
    batch["broken"][:] = True
    broken = build_dynamics_graph(batch, contact_radius=0.05)
    assert len(broken["interaction"].contact_pairs) == 3
    assert broken["face"].x.shape[0] == 1
    assert broken["corner", "bounds", "face"].edge_index.shape == (2, 0)
    assert broken["face", "borders", "corner"].edge_index.shape == (2, 0)
    assert broken["face"].x[0, -2].item() == 1
    batch["forces"][1, 0] = 10
    loaded = build_dynamics_graph(batch, contact_radius=0.05)
    assert not torch.equal(loaded["corner"].x, broken["corner"].x)
    assert not torch.equal(loaded["interaction"].x, broken["interaction"].x)


def test_temporal_predictor_ballistic_initialization_supports_and_irreversible_breaks():
    batch = dynamics_batch(broken=True)
    batch["velocities"][:, 0] = 0.2
    model = small_model()
    prediction = model(batch)
    expected_velocity = batch["velocities"] + batch["forces"] * batch["dt"] / batch["masses"][:, None]
    expected_position = batch["positions"] + batch["dt"] * expected_velocity
    expected_velocity[batch["fixed"]] = 0
    expected_position[batch["fixed"]] = batch["positions"][batch["fixed"]]
    torch.testing.assert_close(prediction["positions"], expected_position)
    torch.testing.assert_close(prediction["velocities"], expected_velocity)
    assert prediction["fracture_probability"].item() == 1
    assert prediction["broken"].item()


def test_predictor_motion_and_new_fracture_are_trainable_from_current_state():
    torch.manual_seed(5)
    batch = dynamics_batch()
    model = small_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    target_velocity = torch.zeros_like(batch["velocities"])
    target_velocity[1:, 0] = 0.4
    losses = []
    for _ in range(6):
        prediction = model(batch)
        loss = (prediction["velocities"] - target_velocity).square().mean()
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(prediction["fracture_logits"], torch.ones(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert model.motion_head.weight.grad.abs().sum().item() > 0
    assert model.fracture_head.weight.grad.abs().sum().item() > 0


def test_neural_corrector_initially_preserves_the_shared_proposal_and_can_learn_residuals():
    batch = dynamics_batch()
    proposal = {key: value.detach() for key, value in small_model()(batch).items()}
    model = small_model(NeuralDynamicsCorrector)
    corrected = model(batch, proposal)
    for key in ("positions", "velocities", "fracture_logits"):
        torch.testing.assert_close(corrected[key], proposal[key])
    with torch.no_grad():
        model.motion_head.bias[3] = 0.25
        model.fracture_head.bias.fill_(0.5)
    corrected = model(batch, proposal)
    torch.testing.assert_close(corrected["velocities"][1:, 0], proposal["velocities"][1:, 0] + 0.25)
    torch.testing.assert_close(corrected["fracture_logits"], proposal["fracture_logits"] + 0.5)
    torch.testing.assert_close(corrected["positions"][batch["fixed"]], batch["positions"][batch["fixed"]])


def test_fracture_flow_is_conditional_trainable_and_pins_already_broken_faces():
    model = small_model()
    batch = dynamics_batch()
    field = torch.tensor([-0.7])
    velocity = model.fracture_velocity(batch, field, torch.tensor(0.4))
    velocity.square().mean().backward()
    assert model.flow_head[-1].weight.grad.abs().sum().item() > 0
    sampled = model.sample_fracture(batch, steps=2, noise=field)
    assert sampled.shape == (1,) and torch.isfinite(sampled).all()
    batch["broken"][:] = True
    assert model.fracture_velocity(batch, field, 0.4).item() == 0
    assert model.sample_fracture(batch, steps=2, noise=field).item() == 1


def test_flow_recomputes_joint_pattern_from_the_current_neighboring_face_values():
    torch.manual_seed(29)
    model = small_model().eval()
    batch = adjacent_face_batch()
    cache = model.prepare_fracture_flow(batch)
    torch.testing.assert_close(cache.edge_index, torch.tensor([[0, 1], [1, 0]]))
    field = torch.tensor([-0.7, 0.4], requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(lambda values: model.fracture_velocity(batch, values, 0.3, cache), field)
    assert jacobian[0, 1].abs() > 1.e-7
    assert jacobian[1, 0].abs() > 1.e-7
    model.fracture_velocity(batch, field, 0.3, cache).square().sum().backward()
    for parameter in (model.flow_encoder.weight, model.flow_layers[0].local_attention.lin_l.weight,
                      model.flow_layers[0].global_attention.in_proj_weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_independent_legacy_flow_keeps_diagonal_field_jacobian_and_parameter_layout():
    torch.manual_seed(29)
    model = TemporalDynamicsModel(**{**small_model().config, "flow_architecture": "independent"}).eval()
    batch = adjacent_face_batch()
    context = model.fracture_context(batch)
    field = torch.tensor([-0.7, 0.4], requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(lambda values: model.fracture_velocity(batch, values, 0.3, context), field)
    assert jacobian[0, 1] == 0 and jacobian[1, 0] == 0
    assert not any(key.startswith(("flow_encoder.", "flow_layers.")) for key in model.state_dict())
    legacy_features = torch.cat((context, field[:, None], torch.full((2, 1), 0.3),
                                 torch.full((2, 1), torch.sin(torch.tensor(0.3 * torch.pi))),
                                 torch.full((2, 1), torch.cos(torch.tensor(0.3 * torch.pi)))), dim=-1)
    torch.testing.assert_close(model.fracture_velocity(batch, field, 0.3, context), model.flow_head(legacy_features).squeeze(-1))


def test_coupled_flow_and_samples_are_equivariant_to_face_permutations():
    torch.manual_seed(41)
    model = small_model().eval()
    batch = adjacent_face_batch()
    field = torch.tensor([-0.4, 0.7])
    order = torch.tensor([1, 0])
    permuted = {**batch, "face_pairs": batch["face_pairs"][order], "broken": batch["broken"][order]}
    torch.testing.assert_close(model.fracture_velocity(permuted, field[order], 0.4),
                               model.fracture_velocity(batch, field, 0.4)[order], atol=1.e-6, rtol=1.e-5)
    torch.testing.assert_close(model.sample_fracture(permuted, steps=2, noise=field[order]),
                               model.sample_fracture(batch, steps=2, noise=field)[order], atol=1.e-6, rtol=1.e-5)


def test_old_breaks_have_fixed_field_values_and_cannot_inject_arbitrary_noise_into_other_faces():
    torch.manual_seed(43)
    model = small_model().eval()
    batch = adjacent_face_batch()
    batch["broken"][0] = True
    cache = model.prepare_fracture_flow(batch)
    first = model.fracture_velocity(batch, torch.tensor([-10., 0.2]), 0.4, cache)
    second = model.fracture_velocity(batch, torch.tensor([10., 0.2]), 0.4, cache)
    torch.testing.assert_close(first, second)
    assert first[0] == 0
    sampled = model.sample_fracture(batch, steps=2, noise=torch.tensor([-0.7, 0.2]))
    assert sampled[0] == 1
    assert torch.isfinite(sampled).all()


def test_flow_sampling_caches_static_context_and_adjacency_once(monkeypatch):
    from fracture_world import dynamics_model

    model = small_model().eval()
    batch = adjacent_face_batch()
    calls = {"context": 0, "adjacency": 0}
    original_context = model.fracture_context
    original_adjacency = dynamics_model.build_fracture_adjacency

    def context(values):
        calls["context"] += 1
        return original_context(values)

    def adjacency(values):
        calls["adjacency"] += 1
        return original_adjacency(values)

    monkeypatch.setattr(model, "fracture_context", context)
    monkeypatch.setattr(dynamics_model, "build_fracture_adjacency", adjacency)
    model.sample_fracture(batch, steps=3, noise=torch.tensor([-0.1, 0.2]))
    assert calls == {"context": 1, "adjacency": 1}


def test_coupled_sampling_is_reproducible_for_a_fixed_noise_seed():
    torch.manual_seed(47)
    model = small_model().eval()
    batch = adjacent_face_batch()
    torch.manual_seed(53)
    first = model.sample_fracture(batch, steps=2)
    torch.manual_seed(53)
    second = model.sample_fracture(batch, steps=2)
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_face_adjacency_uses_material_cell_incidence_without_self_edges():
    batch = adjacent_face_batch()
    torch.testing.assert_close(build_fracture_adjacency(batch["face_pairs"]), torch.tensor([[0, 1], [1, 0]]))
    disconnected = batch["face_pairs"].clone()
    disconnected[1] += 12
    assert build_fracture_adjacency(disconnected).shape == (2, 0)
    assert build_fracture_adjacency(torch.empty((0, 3, 2), dtype=torch.long)).shape == (2, 0)


def test_single_tetrahedron_without_internal_faces_is_supported():
    batch = dynamics_batch()
    for key in ("rest_positions", "positions", "velocities", "masses", "fixed", "forces"):
        batch[key] = batch[key][:4]
    batch["face_pairs"] = torch.empty((0, 3, 2), dtype=torch.long)
    batch["broken"] = torch.empty(0, dtype=torch.bool)
    model = small_model()
    prediction = model(batch)
    assert prediction["positions"].shape == (4, 3)
    assert prediction["fracture_logits"].shape == (0,)
    assert model.sample_fracture(batch).shape == (0,)
    assert model.fracture_velocity(batch, torch.empty(0), 0.4).shape == (0,)


@pytest.mark.parametrize("key,value,match", [
    ("dt", -0.1, "positive"),
    ("masses", torch.zeros(8), "positive"),
    ("material", torch.tensor([1000., 1.e6, 0.7, 1.e4]), "admissible"),
])
def test_invalid_input_is_rejected_before_prediction(key, value, match):
    batch = dynamics_batch()
    batch[key] = value
    with pytest.raises(ValueError, match=match):
        small_model()(batch)


@pytest.mark.parametrize("config", [{"flow_architecture": "unknown"}, {"flow_layers": 0}, {"flow_layers": True}])
def test_invalid_flow_architecture_is_rejected(config):
    with pytest.raises(ValueError, match="flow_"):
        TemporalDynamicsModel(**config)


def test_model_module_does_not_depend_on_native_or_reference_engine():
    module = importlib.import_module("fracture_world.dynamics_model")
    names = {getattr(value, "__name__", "") for value in vars(module).values()}
    assert not any("femfx" in name or "pybullet" in name for name in names)


def test_future_labels_do_not_change_the_predictor_or_flow_context():
    batch = dynamics_batch()
    model = small_model().eval()
    before = model(batch)
    context = model.fracture_context(batch)
    batch["target_positions"] = torch.full_like(batch["positions"], float("nan"))
    batch["target_velocities"] = torch.full_like(batch["velocities"], 1.e20)
    batch["target_broken"] = ~batch["broken"]
    after = model(batch)
    for key in before:
        torch.testing.assert_close(after[key], before[key])
    torch.testing.assert_close(model.fracture_context(batch), context)


def test_intact_tet_neighbors_are_not_contact_pairs_even_with_a_large_search_radius():
    intact = build_dynamics_graph(dynamics_batch(), contact_radius=4)
    assert len(intact["interaction"].contact_pairs) == 0
    broken = build_dynamics_graph(dynamics_batch(broken=True), contact_radius=4)
    assert len(broken["interaction"].contact_pairs) == 16


def test_optional_support_release_head_learns_without_repinning_free_corners():
    batch = dynamics_batch()
    model = TemporalDynamicsModel(**{**small_model().config, "learn_support_release": True})
    model.train()
    with torch.no_grad():
        model.support_head.bias.fill_(-0.4)
    held = model(batch)
    torch.testing.assert_close(held["next_fixed"], batch["fixed"])
    torch.testing.assert_close(held["velocities"][batch["fixed"]], torch.zeros((1, 3)))
    logits = held["support_release_logits"][batch["fixed"]]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    loss.backward()
    assert model.support_head.weight.grad.abs().sum() > 0
    assert model.support_head.bias.grad.item() < 0
    with torch.no_grad():
        model.support_head.bias.fill_(0.4)
    released = model(batch)
    assert not released["next_fixed"].any()
    assert (released["support_release_probability"][~batch["fixed"]] == 0).all()
    torch.testing.assert_close(released["velocities"][batch["fixed"]], torch.tensor([[0., 0., -0.02]]))
    assert not torch.equal(released["positions"][batch["fixed"]], batch["positions"][batch["fixed"]])


def test_support_release_is_disabled_by_default_and_neural_corrector_starts_with_shared_logits():
    batch = dynamics_batch()
    predictor = small_model()
    with torch.no_grad():
        predictor.support_head.bias.fill_(10)
    proposal = predictor(batch)
    torch.testing.assert_close(proposal["next_fixed"], batch["fixed"])
    corrector = small_model(NeuralDynamicsCorrector)
    corrected = corrector(batch, proposal)
    torch.testing.assert_close(corrected["support_release_logits"], proposal["support_release_logits"])
    torch.testing.assert_close(corrected["next_fixed"], batch["fixed"])
    with torch.no_grad():
        corrector.support_head.bias.fill_(-2)
    updated = corrector(batch, proposal)
    torch.testing.assert_close(updated["support_release_logits"], proposal["support_release_logits"] - 2)


def test_future_support_release_labels_do_not_change_prediction():
    batch = dynamics_batch()
    model = TemporalDynamicsModel(**{**small_model().config, "learn_support_release": True}).eval()
    before = model(batch)
    batch["target_fixed"] = ~batch["fixed"]
    after = model(batch)
    torch.testing.assert_close(after["support_release_logits"], before["support_release_logits"])
    torch.testing.assert_close(after["next_fixed"], before["next_fixed"])


def test_class_weight_offsets_calibrate_decisions_without_changing_training_logits():
    batch = dynamics_batch()
    model = TemporalDynamicsModel(**{**small_model().config, "learn_support_release": True}).eval()
    with torch.no_grad():
        model.fracture_head.weight.zero_()
        model.fracture_head.bias.fill_(1)
        model.support_head.bias.fill_(1)
    uncalibrated = model(batch)
    assert uncalibrated["broken"].item()
    assert not uncalibrated["next_fixed"].any()
    model.fracture_logit_offset = 2
    model.support_logit_offset = 2
    calibrated = model(batch)
    for key in ("fracture_logits", "support_release_logits"):
        torch.testing.assert_close(calibrated[key], uncalibrated[key])
    torch.testing.assert_close(calibrated["fracture_probability"], torch.tensor([-1.]).sigmoid())
    torch.testing.assert_close(calibrated["support_release_probability"][batch["fixed"]], torch.tensor([-1.]).sigmoid())
    torch.testing.assert_close(calibrated["next_fixed"], batch["fixed"])
    assert not calibrated["broken"].item()


def test_neural_corrector_applies_logit_offsets_once_after_residual_addition():
    batch = dynamics_batch()
    predictor = small_model()
    predictor.fracture_logit_offset = 2
    predictor.support_logit_offset = 3
    proposal = predictor(batch)
    corrector = small_model(NeuralDynamicsCorrector)
    corrector.fracture_logit_offset = 2
    corrector.support_logit_offset = 3
    corrected = corrector(batch, proposal)
    for key in ("fracture_logits", "fracture_probability", "support_release_logits", "support_release_probability"):
        torch.testing.assert_close(corrected[key], proposal[key])


def test_proposal_graph_uses_calibrated_probabilities_and_rejects_invalid_probabilities():
    batch = dynamics_batch()
    model = small_model()
    model.fracture_logit_offset = 2
    model.support_logit_offset = 3
    proposal = model(batch)
    graph = build_dynamics_graph(batch, proposal=proposal)
    torch.testing.assert_close(graph["face"].x[:, -1], proposal["fracture_probability"])
    torch.testing.assert_close(graph["corner"].x[:, -1], proposal["support_release_probability"])
    for key in ("fracture_probability", "support_release_probability"):
        invalid = {**proposal, key: torch.full_like(proposal[key], 1.1)}
        with pytest.raises(ValueError, match=key):
            build_dynamics_graph(batch, proposal=invalid)
