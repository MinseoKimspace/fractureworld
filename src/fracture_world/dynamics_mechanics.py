from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from .mesh import VolumeMesh
from .partition import component_labels


FEMFX_REVISION = "95cf656731a2465ae1d400138170af2d6575b5bb"
FEMFX_SOURCE = f"https://github.com/GPUOpen-Effects/FEMFX/tree/{FEMFX_REVISION}/amd_femfx/src"


def _material(material):
    density, young, poisson = (float(np.float32(material[key])) for key in ("rest_density", "youngs_modulus", "poissons_ratio"))
    if not np.isfinite([density, young, poisson]).all() or min(density, young) <= 0 or not -1 < poisson < .5:
        raise ValueError("mechanics requires positive density and Young modulus and -1 < Poisson ratio < .5")
    return density, young / (2 * (1 + poisson)), young * poisson / ((1 + poisson) * (1 - 2 * poisson))


def _corners(mesh, values, name):
    values = np.asarray(values, dtype=float)
    if values.shape not in ((4 * len(mesh.tets), 3), (len(mesh.tets), 4, 3)) or not np.isfinite(values).all():
        raise ValueError(f"{name} requires finite material-corner three-vectors")
    return np.array(values.reshape(-1, 3), copy=True)


def _mask(values, shape, name):
    values = np.asarray(values)
    if values.shape != shape or values.dtype != np.dtype(bool):
        raise ValueError(f"{name} requires a Boolean array with shape {shape}")
    return values.copy()


def _tolerances(absolute, relative):
    if not np.isfinite([absolute, relative]).all() or absolute <= 0 or relative < 0:
        raise ValueError("mechanics tolerances require absolute > 0 and relative >= 0")


def _elastic_state(mesh, material, positions):
    if not isinstance(mesh, VolumeMesh):
        raise TypeError("mechanics mesh must be a VolumeMesh")
    _, shear, lame = _material(material)
    rest = mesh.nodes.astype(np.float32).astype(float)[mesh.tets]
    current = _corners(mesh, positions, "elastic positions").reshape(rest.shape)
    inverse = np.linalg.inv((rest[:, 1:] - rest[:, :1]).transpose(0, 2, 1))
    volumes = np.linalg.det(rest[:, 1:] - rest[:, :1]) / 6
    if (volumes <= 0).any():
        raise ValueError("native float32 rest geometry must preserve positive tetrahedron orientation")
    gradients = np.concatenate((-inverse.sum(axis=1, keepdims=True), inverse), axis=1)
    deformation = (current[:, 1:] - current[:, :1]).transpose(0, 2, 1) @ inverse
    if (np.linalg.det(deformation) <= 0).any():
        raise ValueError("elastic mechanics requires positive tetrahedron orientation")
    left, _, right = np.linalg.svd(deformation)
    signs = np.ones((len(mesh.tets), 3))
    signs[:, -1] = np.where(np.linalg.det(left @ right) < 0, -1., 1.)
    rotations = (left * signs[:, None, :]) @ right
    stretch = rotations.transpose(0, 2, 1) @ deformation
    strain = .5 * (stretch + stretch.transpose(0, 2, 1)) - np.eye(3)
    stress = 2 * shear * strain + lame * np.trace(strain, axis1=1, axis2=2)[:, None, None] * np.eye(3)
    gradient = volumes[:, None, None] * np.einsum("tab,tib->tia", rotations @ stress, gradients)
    return gradients, rotations, stress, gradient, volumes


def _fracture_tensor(stress):
    result = np.array(stress, copy=True)
    result[:, 0, 1] = result[:, 1, 0] = stress[:, 0, 2]
    result[:, 0, 2] = result[:, 2, 0] = stress[:, 1, 2]
    result[:, 1, 2] = result[:, 2, 1] = stress[:, 0, 1]
    return result


def elastic_kernel_arguments(mesh, broken, state):
    if state is None:
        return {}
    groups = mesh.corner_groups(broken)
    _, representatives = np.unique(groups, return_index=True)
    vertex_ids = np.asarray(state["corner_vertex_ids"]).reshape(-1)[representatives]
    return dict(vertex_order=np.argsort(vertex_ids, kind="stable"),
                tetrahedron_order=np.lexsort((np.asarray(state["tetrahedron_local_ids"]), np.asarray(state["piece_ids"]))),
                current_rotations=np.asarray(state["rotations"]),
                current_centers_of_mass=np.asarray(state["centers_of_mass"]),
                current_nodal_masses=np.asarray(state["nodal_masses"]))


@dataclass
class FrozenElasticBalance:
    mesh: VolumeMesh
    material: dict
    positions: np.ndarray
    velocities: np.ndarray
    dt: float
    gravity: np.ndarray
    broken: np.ndarray
    fixed: np.ndarray
    contacts: tuple = ()
    velocity_scale: float = 1.
    absolute_tolerance: float = 1e-8
    relative_tolerance: float = 1e-7
    coefficient_model: str = "analytical"
    current_elastic_state: dict | None = None
    groups: np.ndarray = field(init=False)
    representatives: np.ndarray = field(init=False)
    matrix: sparse.csr_matrix = field(init=False, repr=False)
    rhs: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.mesh, VolumeMesh):
            raise TypeError("mechanics mesh must be a VolumeMesh")
        self.positions = _corners(self.mesh, self.positions, "initial positions")
        self.velocities = _corners(self.mesh, self.velocities, "initial velocities")
        self.broken = _mask(self.broken, (len(self.mesh.internal_faces),), "initial fracture")
        self.fixed = _mask(self.fixed, (len(self.positions),), "initial support")
        self.gravity = np.asarray(self.gravity, dtype=np.float32).astype(float)
        if self.gravity.shape != (3,) or not np.isfinite(self.gravity).all():
            raise ValueError("mechanics gravity must be a finite three-vector")
        if not np.isfinite([self.dt, self.velocity_scale]).all() or min(self.dt, self.velocity_scale) <= 0:
            raise ValueError("mechanics timestep and velocity scale must be finite and positive")
        _tolerances(self.absolute_tolerance, self.relative_tolerance)
        density, shear, lame = _material(self.material)
        gradients, self.rotations, self.standard_stress, corner_gradient, volumes = _elastic_state(self.mesh, self.material, self.positions)
        self.groups = self.mesh.corner_groups(self.broken)
        _, self.representatives = np.unique(self.groups, return_index=True)
        count = len(self.representatives)
        corner_mass = np.repeat(volumes * density / 4, 4)
        self.corner_masses = corner_mass.copy()
        self.node_masses = np.bincount(self.groups, weights=corner_mass, minlength=count)
        coordinates = (3 * self.groups[:, None] + np.arange(3)).ravel()
        self.elastic_gradient = np.bincount(coordinates, weights=corner_gradient.ravel(), minlength=3 * count).reshape(-1, 3)
        rotated_gradients = np.einsum("tab,tib->tia", self.rotations, gradients)
        dot = np.einsum("tia,tja->tij", gradients, gradients)
        blocks = volumes[:, None, None, None, None] * (
            shear * dot[..., None, None] * np.eye(3)
            + shear * np.einsum("tja,tib->tijab", rotated_gradients, rotated_gradients)
            + lame * np.einsum("tia,tjb->tijab", rotated_gradients, rotated_gradients))
        nodes = self.groups.reshape(-1, 4)
        rows = 3 * nodes[:, :, None, None, None] + np.arange(3)[None, None, None, :, None]
        columns = 3 * nodes[:, None, :, None, None] + np.arange(3)[None, None, None, None, :]
        self.stiffness = sparse.coo_matrix((blocks.ravel(), (np.broadcast_to(rows, blocks.shape).ravel(),
                                                          np.broadcast_to(columns, blocks.shape).ravel())),
                                           shape=(3 * count, 3 * count)).tocsr()
        self.stiffness.eliminate_zeros()
        self.matrix = sparse.diags(np.repeat(self.node_masses, 3), format="csr") + self.dt**2 * self.stiffness
        self.rhs = (self.node_masses[:, None] * (self.velocities[self.representatives] + self.dt * self.gravity)
                    - self.dt * self.elastic_gradient).ravel()
        if self.coefficient_model not in {"analytical", "native_float32"}:
            raise ValueError("mechanics coefficient model must select analytical or native_float32")
        self.operator_ordering = None
        if self.coefficient_model == "native_float32":
            from .dynamics_femfx_math import native_frozen_operator

            operator = native_frozen_operator(self.mesh, self.material, self.positions, self.velocities, self.dt,
                                              self.gravity, self.broken, self.fixed,
                                              **elastic_kernel_arguments(self.mesh, self.broken, self.current_elastic_state))
            for name in ("matrix", "rhs", "stiffness", "elastic_gradient", "corner_masses", "node_masses", "groups", "representatives", "rotations"):
                setattr(self, name, getattr(operator, name))
            self.operator_ordering = operator.ordering
        self.support_nodes = np.unique(self.groups[self.fixed])
        touched = []
        for contact in self.contacts:
            ids = np.asarray(contact.corner_ids)
            weights = np.asarray(contact.weights)
            if ids.ndim != 1 or weights.shape != ids.shape or not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any() or (ids >= len(self.positions)).any():
                raise ValueError("mechanics contacts must reference valid material corners")
            touched.extend(self.groups[ids[weights != 0]].tolist())
        self.contact_nodes = np.unique(touched).astype(np.int64)
        self.unknown_nodes = np.union1d(self.support_nodes, self.contact_nodes)
        self.free_nodes = np.setdiff1d(np.arange(count), self.unknown_nodes)

    def factors(self, problem):
        if problem.corner_count != len(self.groups) or not np.isclose(problem.dt, self.dt, rtol=0, atol=0):
            raise ValueError("elastic balance and corrector must use the same corners and timestep")
        factors = []
        for node in self.free_nodes:
            matrix = self.matrix[3 * node:3 * node + 3].tocoo()
            columns = 3 * problem.corner_count + 3 * self.representatives[matrix.col // 3] + matrix.col % 3
            scale = self.node_masses[node] * self.velocity_scale
            factors.append(problem._factor(f"elastic_balance:{node}", matrix.row, columns,
                                           matrix.data * problem.velocity_scale / scale,
                                           self.rhs[3 * node:3 * node + 3] / scale))
        return tuple(factors)

    def diagnostics(self, next_positions, next_velocities):
        positions = _corners(self.mesh, next_positions, "next positions")
        velocities = _corners(self.mesh, next_velocities, "next velocities")
        nodal_velocity = velocities[self.representatives]
        residual = (self.matrix @ nodal_velocity.ravel() - self.rhs).reshape(-1, 3)
        inferred_reaction = residual.copy()
        if self.coefficient_model == "native_float32" and len(self.support_nodes):
            physical = (self.node_masses[:, None] * (nodal_velocity - self.velocities[self.representatives] - self.dt * self.gravity)
                        + self.dt * self.elastic_gradient
                        + self.dt**2 * (self.stiffness @ nodal_velocity.ravel()).reshape(-1, 3))
            inferred_reaction[self.support_nodes] = physical[self.support_nodes]
        normalized = residual / (self.node_masses[:, None] * self.velocity_scale)
        tolerance = self.absolute_tolerance + self.relative_tolerance * max(float(np.max(np.abs(nodal_velocity))) / self.velocity_scale, 1.)
        free_residual = float(np.max(np.abs(normalized[self.free_nodes]))) if len(self.free_nodes) else None
        free_status = "unknown" if free_residual is None else "pass" if free_residual <= tolerance else "fail"
        position_mismatch = float(np.max(np.abs(positions - positions[self.representatives][self.groups]))) / (self.dt * self.velocity_scale)
        velocity_mismatch = float(np.max(np.abs(velocities - nodal_velocity[self.groups]))) / self.velocity_scale
        initial_mismatch = max(float(np.max(np.abs(self.positions - self.positions[self.representatives][self.groups]))) / (self.dt * self.velocity_scale),
                               float(np.max(np.abs(self.velocities - self.velocities[self.representatives][self.groups]))) / self.velocity_scale)
        continuity = max(position_mismatch, velocity_mismatch, initial_mismatch)
        status = "fail" if free_status == "fail" or continuity > tolerance else "unknown" if len(self.unknown_nodes) or free_status == "unknown" else "pass"
        return dict(status=status, free_status=free_status, law="FEMFX frozen-rotation linear implicit Euler velocity balance",
                    corrector_model_status=status, native_replay_status="unknown",
                    native_replay_reason="canonical/current-state coefficient reconstruction does not certify native finite iterations or pending postfracture topology packing",
                    source=f"{FEMFX_SOURCE}/SparseSolvers/FEMFXMpcgSolverSetup.cpp", native_revision=FEMFX_REVISION,
                    start_connectivity_used=True, damping_assumption="zero Rayleigh damping", engine_calls=0,
                    coefficient_precision="source-ordered float32 constitutive and assembly math" if self.coefficient_model == "native_float32" else "native float32 rest geometry/material inputs; reconstructed float64 operators",
                    coefficient_model=self.coefficient_model, operator_ordering=self.operator_ordering,
                    normalized_tolerance=float(tolerance), maximum_free_normalized_residual=free_residual,
                    free_normalized_residuals=normalized[self.free_nodes].tolist(),
                    free_node_ids=self.free_nodes.tolist(), verified_free_nodes=int(len(self.free_nodes)),
                    unknown_node_ids=self.unknown_nodes.tolist(), unknown_nodes=int(len(self.unknown_nodes)),
                    support_reaction_node_ids=self.support_nodes.tolist(), contact_reaction_node_ids=self.contact_nodes.tolist(),
                    unknown_reasons=[reason for values, reason in ((self.support_nodes, "unrecorded support reactions"),
                                                                  (self.contact_nodes, "unrecorded contact impulses and stabilization")) if len(values)],
                    inferred_reaction_impulses_kg_m_per_s=inferred_reaction[self.unknown_nodes].tolist(),
                    support_reaction_scope="reconstructed physical M*delta_v + h*q + h^2*K*v - h*M*g; filtered native identity rows are never used as forces",
                    maximum_old_node_endpoint_mismatch=float(continuity),
                    old_node_endpoint_continuity_satisfied=bool(continuity <= tolerance),
                    residual_scope="free rows of the supplied contact/support partition; excluded reaction rows are not certified")


def fracture_initiation_diagnostics(mesh, material, next_positions, previous_broken, broken,
                                    absolute_tolerance=1e-8, relative_tolerance=1e-7, *,
                                    native_stress_ratios=None, native_stress_observed=None,
                                    coefficient_model="analytical"):
    _tolerances(absolute_tolerance, relative_tolerance)
    previous = _mask(previous_broken, (len(mesh.internal_faces),), "previous fracture")
    broken = _mask(broken, previous.shape, "candidate fracture")
    if np.any(previous & ~broken):
        raise ValueError("fracture initiation cannot heal previously broken faces")
    threshold = float(np.float32(material["fracture_stress_threshold"]))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("fracture initiation requires a finite positive stress threshold")
    _, _, stress, _, _ = _elastic_state(mesh, material, next_positions)
    magnitude = np.max(np.abs(np.linalg.eigvalsh(_fracture_tensor(stress))), axis=1)
    ratio = magnitude / threshold
    if coefficient_model == "native_float32":
        from .dynamics_femfx_math import native_fracture_stress

        ratio = native_fracture_stress(mesh, material, next_positions, previous)["stress_ratios"]
    elif coefficient_model != "analytical":
        raise ValueError("fracture coefficient model must select analytical or native_float32")
    native = None
    if native_stress_ratios is not None:
        native = _mask(native_stress_observed, (len(mesh.tets),), "native stress availability")
        observed_ratios = np.asarray(native_stress_ratios, dtype=float)
        if observed_ratios.shape != native.shape or not np.isfinite(observed_ratios[native]).all() or (observed_ratios[native] < 0).any():
            raise ValueError("native stress ratios must be finite and nonnegative where observed")
        ratio = np.where(native, observed_ratios, ratio)
    groups = mesh.corner_groups(previous)
    star = [set() for _ in range(int(groups.max()) + 1)]
    for corner, group in enumerate(groups):
        star[group].add(corner // 4)
    labels = component_labels(len(mesh.tets), mesh.dual_edges, previous)
    sizes = np.bincount(labels)
    tolerance = absolute_tolerance + relative_tolerance
    threshold_margin = 0. if coefficient_model == "native_float32" or native is not None else tolerance
    faces = []
    for face in np.flatnonzero(broken & ~previous):
        cells = mesh.dual_edges[face]
        vertices = groups[mesh.corner_face_pairs[face, :, 0]]
        witnesses = sorted(set().union(*(star[group] for group in vertices)))
        maximum = float(np.max(ratio[witnesses]))
        large_enough = int(sizes[labels[cells[0]]]) >= 6
        supported = maximum >= 1. - threshold_margin
        availability = native is None or bool(native[witnesses].all()) or bool(np.any(native[witnesses] & (ratio[witnesses] >= 1. - threshold_margin)))
        status = "fail" if not large_enough else "unknown" if not availability else "pass" if supported else "fail"
        faces.append(dict(internal_face=int(face), material_face=int(mesh.internal_faces[face]),
                          status=status,
                          maximum_vertex_star_stress_ratio=maximum, stress_threshold_pa=threshold,
                          overstressed_witness_tetrahedra=[cell for cell in witnesses if ratio[cell] >= 1. - threshold_margin],
                          previous_piece_tetrahedra=int(sizes[labels[cells[0]]]),
                          minimum_piece_size_satisfied=bool(large_enough), stress_support_satisfied=bool(supported)))
    failed = [item["internal_face"] for item in faces if item["status"] == "fail"]
    unknown = [item["internal_face"] for item in faces if item["status"] == "unknown"]
    return dict(status="fail" if failed else "unknown" if unknown else "pass", certificate_scope="necessary initiation conditions only",
                full_topology_status="unknown", native_revision=FEMFX_REVISION,
                native_replay_status="unknown", numerical_law_scope="canonical corrector constitutive threshold; native random splitting and topology preparation are not certified",
                source=f"{FEMFX_SOURCE}/Simulation/FEMFXUpdateTetState.cpp", engine_calls=0,
                stress_measure="spectral radius of pinned FEMFX fracture-component tensor; compression included",
                shear_component_order="native stress1=(xy,yz,zx), fracture assembly interprets it as (yz,zx,xy)",
                maximum_cell_stress_ratio=float(np.max(ratio)), normalized_tolerance=float(tolerance),
                stress_threshold_margin=float(threshold_margin),
                newly_broken_faces=len(faces), unsupported_new_face_ids=failed, faces=faces,
                stress_source="native observed threshold ratios with explicit availability" if native is not None else
                "source-ordered float32 stress and native approximate spectral radius" if coefficient_model == "native_float32" else
                "reconstructed float64 stress from native float32 inputs",
                unknown_new_face_ids=unknown,
                unavailable_checks=["randomized split direction and vertex selection", "crack-tip preference and disabled-face flags",
                                    "native tentative-split local minimum-three-tetrahedra rule"])


def mechanics_diagnostics(initial, transition, contacts, config, *, velocity_scale=None):
    from .dynamics_native_audit import native_solver_diagnostics

    setting = config.get if isinstance(config, dict) else lambda key, default: getattr(config, key, default)
    absolute, relative = setting("absolute_tolerance", 1e-8), setting("relative_tolerance", 1e-7)
    velocity_scale = setting("velocity_scale", None) if velocity_scale is None else velocity_scale
    if velocity_scale is None:
        raise ValueError("teacher mechanics diagnostics require an explicit velocity scale")
    coefficient_model = "native_float32" if setting("mechanics_mode", "disabled") != "disabled" else "analytical"
    balance = FrozenElasticBalance(initial.topology.mesh, initial.material, transition.positions, transition.velocities,
                                   transition.dt, initial.gravity, transition.broken, transition.fixed, tuple(contacts),
                                   velocity_scale, absolute, relative, coefficient_model=coefficient_model,
                                   current_elastic_state=getattr(transition, "elastic_state", None))
    dynamics = balance.diagnostics(transition.target_positions, transition.target_velocities)
    dynamics["certificate_scope"] = "reconstructed corrector-model compatibility; not the native CG stopping criterion"
    settings = initial.metadata.get("native_settings", initial.metadata.get("teacher_physics", {}))
    keys = ("rayleigh_mass_damping", "rayleigh_stiffness_damping", "friction_coefficient")
    compatible = all(key in settings and settings[key] == 0 for key in keys)
    dynamics["teacher_settings_status"] = "undamped_frictionless_recorded" if compatible else "unknown_or_unsupported"
    if not compatible and dynamics["status"] == "pass":
        dynamics["status"] = "unknown"
        dynamics["unknown_reasons"].append("teacher damping/friction not fully recorded as zero")
    positions = _corners(initial.topology.mesh, transition.target_positions, "next positions")
    velocities = _corners(initial.topology.mesh, transition.target_velocities, "next velocities")
    observations = getattr(transition, "observations", None)
    observed = None
    delta_position = np.zeros_like(positions)
    native_kwargs = {}
    if observations is not None and observations.get("single_step_coverage") is True:
        observed = np.asarray(observations["contact_observed"], dtype=bool).reshape(-1)
        delta_position = np.asarray(observations["stabilization_delta_positions"]).reshape(-1, 3)
        active = np.asarray(observations["constraint_solve_active"], dtype=bool).reshape(-1)
        before = np.asarray(observations["pre_contact_velocities"]).reshape(-1, 3)
        delta_velocity = np.asarray(observations["applied_delta_velocities"]).reshape(-1, 3)
        expected_velocity = (before.astype(np.float32) + delta_velocity.astype(np.float32)).astype(float)
        velocity_error = float(np.max(np.abs((velocities - expected_velocity)[observed]))) / velocity_scale if observed.any() else None
        update_status = "fail" if velocity_error is not None and velocity_error != 0 else "pass" if observed.all() else "unknown"
        dynamics["native_velocity_update"] = dict(status=update_status,
                                                 maximum_normalized_residual=velocity_error, arithmetic="native float32 addition")
        prebalance = FrozenElasticBalance(initial.topology.mesh, initial.material, transition.positions, transition.velocities,
                                          transition.dt, initial.gravity, transition.broken, transition.fixed, (), velocity_scale, absolute, relative,
                                          coefficient_model=coefficient_model, current_elastic_state=getattr(transition, "elastic_state", None))
        pre_report = prebalance.diagnostics(positions, before)
        pre_report["certificate_scope"] = "reconstructed current-state coefficients applied to the captured pre-constraint native velocity; source arithmetic and replay scope are explicit"
        pre_report["status"] = "unknown" if not observed.all() else pre_report["status"]
        dynamics["native_preconstraint_elastic_balance"] = pre_report
        dynamics["native_constraint_deltas_recorded"] = bool(observed.all())
        if update_status == "fail" or (observed.all() and pre_report["free_status"] == "fail"):
            dynamics["status"] = "fail"
        elif not observed.all() or active[observed].any():
            dynamics["status"] = "unknown"
            dynamics["unknown_reasons"].append("constraint deltas are observed but per-contact impulse law is not independently certified")
        stress_observed = np.asarray(observations["fracture_stress_observed"], dtype=bool)
        stress_thresholds = np.asarray(observations["fracture_stress_thresholds"], dtype=float)
        ratios = np.divide(observations["fracture_stress_eigenvalues"], stress_thresholds,
                           out=np.zeros_like(stress_thresholds), where=stress_observed)
        native_kwargs = dict(native_stress_ratios=ratios, native_stress_observed=stress_observed)
    native_precision = all(np.array_equal(value, value.astype(np.float32).astype(float)) for value in (positions, velocities, balance.positions))
    if native_precision:
        expected_position = ((balance.positions.astype(np.float32) + delta_position.astype(np.float32)) +
                             np.float32(transition.dt) * velocities.astype(np.float32)).astype(float)
    else:
        expected_position = balance.positions + delta_position + transition.dt * velocities
    residual = (positions - expected_position) / (transition.dt * velocity_scale)
    all_corner_maximum = float(np.max(np.abs(residual)))
    maximum = all_corner_maximum if observed is None else float(np.max(np.abs(residual[observed]))) if observed.any() else None
    tolerance = absolute + relative * max(float(np.max(np.abs(velocities))) / velocity_scale, 1.)
    complete_observation = observed is not None and bool(observed.all())
    if observed is not None:
        acceptance = 0. if native_precision else tolerance
        kinematic_status = "fail" if maximum is not None and maximum > acceptance else "pass" if complete_observation else "unknown"
    else:
        kinematic_status = "unknown" if contacts else "pass" if maximum <= tolerance else "fail"
    kinematics = dict(status=kinematic_status,
                      law="backward Euler endpoint position update", maximum_normalized_residual=maximum,
                      normalized_tolerance=float(tolerance),
                      all_corner_unverified_maximum_normalized_residual=all_corner_maximum,
                      arithmetic="native float32 multiply/add" if native_precision else "float64 analytical state",
                      acceptance_scope="exact captured float32 update identity" if complete_observation and native_precision else "analytical residual with configured tolerance",
                      acceptance_tolerance=0. if complete_observation and native_precision else float(tolerance),
                      stabilization_recorded=complete_observation,
                      unrounded_backward_euler_maximum_normalized_residual=float(np.max(np.abs(positions - balance.positions - transition.dt * velocities))) / (transition.dt * velocity_scale),
                      unknown_reason="native contact stabilization position deltas are not fully recorded" if contacts and not complete_observation else None)
    fracture = fracture_initiation_diagnostics(initial.topology.mesh, initial.material, transition.target_positions,
                                               transition.broken, transition.target_broken, absolute, relative,
                                               coefficient_model=coefficient_model, **native_kwargs)
    native_solver = native_solver_diagnostics(observations, balance, velocity_scale)
    dynamics["native_equation_status"] = native_solver["equation_status"]
    if observations is not None and observations.get("single_step_coverage") is not True:
        for report in (dynamics, kinematics, fracture):
            report["status"] = "unknown"
            report["availability_reason"] = "observation does not cover this complete single native transition"
    return dict(elastic_dynamics=dynamics, fracture_initiation=fracture, kinematics=kinematics,
                native_elastic_solver=native_solver)
