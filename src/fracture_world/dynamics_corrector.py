from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.linalg import null_space, pinvh
from scipy.optimize import linprog

from .mesh import VolumeMesh
from .partition import component_labels


def _finite_array(value, shape, name):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


@dataclass(frozen=True)
class LinearizedContact:
    corner_ids: np.ndarray
    weights: np.ndarray
    normal: np.ndarray
    gap: float
    environment_velocity: float = 0.0
    environment: bool = False

    def __post_init__(self):
        indices = np.asarray(self.corner_ids)
        if indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer) or (indices < 0).any():
            raise ValueError("contact corner_ids must be a nonempty nonnegative integer vector")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("contact corner_ids must be unique")
        weights = _finite_array(self.weights, indices.shape, "contact weights")
        normal = _finite_array(self.normal, (3,), "contact normal")
        if not np.isclose(np.linalg.norm(normal), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("contact normal must have unit length")
        if not np.isfinite([self.gap, self.environment_velocity]).all():
            raise ValueError("contact gap and environment velocity must be finite")
        expected = 1.0 if self.environment else 0.0
        if not np.isclose(weights.sum(), expected, rtol=1e-10, atol=1e-12):
            raise ValueError("internal contact weights must sum to zero; environment weights must sum to one")
        if not self.environment and self.environment_velocity != 0:
            raise ValueError("environment velocity requires an environment contact")
        object.__setattr__(self, "corner_ids", indices.astype(np.int64, copy=True))
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "normal", normal)

    @classmethod
    def pair(cls, corner_a, corner_b, normal, gap, environment_velocity=0.0):
        if corner_b == -1:
            return cls(np.array([corner_a]), np.array([1.0]), normal, gap, environment_velocity, True)
        return cls(np.array([corner_a, corner_b]), np.array([1.0, -1.0]), normal, gap, environment_velocity, False)


@dataclass(frozen=True)
class CorrectorTolerances:
    absolute: float = 1e-8
    relative: float = 1e-7

    def __post_init__(self):
        if not np.isfinite([self.absolute, self.relative]).all() or self.absolute <= 0 or self.relative < 0:
            raise ValueError("corrector tolerances must be finite, with absolute > 0 and relative >= 0")


@dataclass
class _Factor:
    name: str
    indices: np.ndarray
    matrix: sparse.csr_matrix
    bound: np.ndarray
    inequality: bool = False
    penalty: float | None = None
    inverse_gram: np.ndarray = field(init=False, repr=False)
    _soft_inverse: tuple | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.penalty is not None and (not np.isfinite(self.penalty) or self.penalty <= 0 or self.inequality):
            raise ValueError("soft residual factors require a positive finite penalty and equality residuals")
        if len(self.bound) == 1:
            squared_norm = float(self.matrix.data @ self.matrix.data)
            self.inverse_gram = np.array([[1.0 / squared_norm if squared_norm else 0.0]])
        else:
            self.inverse_gram = pinvh((self.matrix @ self.matrix.T).toarray())

    def projection_inverse(self, rho):
        if self.penalty is None:
            return self.inverse_gram
        if self._soft_inverse is None or self._soft_inverse[0] != rho:
            gram = (self.matrix @ self.matrix.T).toarray()
            inverse = np.linalg.solve(gram + (rho / self.penalty) * np.eye(len(self.bound)), np.eye(len(self.bound)))
            self._soft_inverse = (rho, inverse)
        return self._soft_inverse[1]

    def cost(self, value):
        if self.penalty is None:
            return 0.0
        residual = np.asarray(self.matrix @ value).ravel() - self.bound
        return float(0.5 * self.penalty * (residual @ residual))

    def project(self, value, rho=1.0):
        residual = self.bound - self.matrix @ value
        if self.inequality:
            residual = np.maximum(residual, 0)
        return value + np.asarray(self.matrix.T @ (self.projection_inverse(rho) @ residual)).ravel()

    def violation(self, value):
        residual = np.asarray(self.matrix @ value).ravel() - self.bound
        if self.penalty is not None:
            return np.zeros_like(residual)
        return np.maximum(-residual, 0) if self.inequality else np.abs(residual)


@dataclass
class DynamicsProblem:
    mesh: VolumeMesh
    positions: np.ndarray
    velocities: np.ndarray
    predicted_positions: np.ndarray
    predicted_velocities: np.ndarray
    dt: float
    density: float = 1.0
    broken_faces: np.ndarray | None = None
    fixed: np.ndarray | None = None
    external_impulse: np.ndarray | None = None
    contacts: tuple[LinearizedContact, ...] = ()
    position_scale: float | None = None
    velocity_scale: float | None = None
    position_weight: float = 1.0
    velocity_weight: float = 1.0
    kinematic_rule: str = "backward_euler"
    previous_broken_faces: np.ndarray | None = None
    previous_fixed: np.ndarray | None = None
    elastic_balance: object | None = None
    mechanics_mode: str = "disabled"
    mechanics_weight: float = 1.0
    stabilization_offset: np.ndarray | None = None
    position_roundoff: np.ndarray | None = None
    kinematic_velocity_envelope: np.ndarray | None = None
    contact_tolerance: float = 0.0
    native_contact_phase: bool = False
    corner_masses: np.ndarray = field(init=False)
    prior: np.ndarray = field(init=False, repr=False)
    prior_weights: np.ndarray = field(init=False, repr=False)
    factors: tuple[_Factor, ...] = field(init=False, repr=False)
    metadata: dict = field(init=False)
    momentum_islands: tuple[np.ndarray, ...] = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.mesh, VolumeMesh):
            raise TypeError("mesh must be a VolumeMesh")
        if not np.isfinite([self.dt, self.density, self.position_weight, self.velocity_weight]).all():
            raise ValueError("time step, density, and prior weights must be finite")
        if min(self.dt, self.density, self.position_weight, self.velocity_weight) <= 0:
            raise ValueError("time step, density, and prior weights must be positive")
        if self.kinematic_rule not in ("backward_euler", "trapezoidal", "constant_acceleration"):
            raise ValueError("unsupported kinematic integration rule")
        if self.mechanics_mode not in {"disabled", "proximal", "exact"}:
            raise ValueError("mechanics_mode must select disabled, proximal or exact")
        if self.mechanics_mode != "disabled" and (self.elastic_balance is None or self.kinematic_rule != "backward_euler"):
            raise ValueError("FEMFX mechanical balance requires its frozen elastic model and backward_euler kinematics")
        if not np.isfinite(self.mechanics_weight) or self.mechanics_weight <= 0:
            raise ValueError("mechanics_weight must be finite and positive")
        shape = (len(self.mesh.tets), 4, 3)
        for name in ("positions", "velocities", "predicted_positions", "predicted_velocities"):
            setattr(self, name, _finite_array(getattr(self, name), shape, name))
        for name in ("stabilization_offset", "position_roundoff", "kinematic_velocity_envelope"):
            value = getattr(self, name)
            if value is not None:
                value = _finite_array(value, shape, name)
                if name != "stabilization_offset" and (value < 0).any():
                    raise ValueError(f"{name} must be nonnegative")
                setattr(self, name, value)
        if not np.isfinite(self.contact_tolerance) or self.contact_tolerance < 0:
            raise ValueError("contact tolerance must be finite and nonnegative")
        if not isinstance(self.native_contact_phase, bool):
            raise ValueError("native contact phase must be Boolean")
        for name, length in (("broken_faces", len(self.mesh.faces.vertices)), ("fixed", 4 * len(self.mesh.tets))):
            raw = getattr(self, name)
            target_shape = (length,) if name == "broken_faces" else shape[:2]
            if raw is None:
                value = np.zeros(target_shape, dtype=bool)
            else:
                value = np.asarray(raw)
                if value.shape != target_shape or value.dtype != np.dtype(bool):
                    raise ValueError(f"{name} must be a boolean array with shape {target_shape}")
                value = value.copy()
            setattr(self, name, value)
        if self.broken_faces[self.mesh.faces.cells[:, 1] < 0].any():
            raise ValueError("only internal material faces can be broken")
        previous = self.broken_faces if self.previous_broken_faces is None else np.asarray(self.previous_broken_faces)
        if previous.shape != self.broken_faces.shape or previous.dtype != np.dtype(bool):
            raise ValueError("previous_broken_faces must match the Boolean material-face array")
        if np.any(previous & ~self.broken_faces):
            raise ValueError("previously broken material faces cannot become intact")
        self.previous_broken_faces = previous.copy()
        previous_fixed = self.fixed if self.previous_fixed is None else np.asarray(self.previous_fixed)
        if previous_fixed.shape != shape[:2] or previous_fixed.dtype != np.dtype(bool):
            raise ValueError("previous_fixed must match the Boolean material-corner support array")
        self.previous_fixed = previous_fixed.copy()
        self.external_impulse = np.zeros(shape) if self.external_impulse is None else _finite_array(self.external_impulse, shape, "external_impulse")
        self.contacts = tuple(self.contacts)
        count = 4 * len(self.mesh.tets)
        if not all(isinstance(contact, LinearizedContact) for contact in self.contacts):
            raise TypeError("contacts must contain LinearizedContact objects")
        if any((contact.corner_ids >= count).any() for contact in self.contacts):
            raise ValueError("contact corner index is outside the material-corner array")
        extent = float(np.linalg.norm(np.ptp(self.mesh.nodes, axis=0)))
        self.position_scale = extent if self.position_scale is None else float(self.position_scale)
        self.velocity_scale = self.position_scale / self.dt if self.velocity_scale is None else float(self.velocity_scale)
        if not np.isfinite([self.position_scale, self.velocity_scale]).all() or min(self.position_scale, self.velocity_scale) <= 0:
            raise ValueError("position and velocity scales must be finite and positive")
        self.corner_masses = np.broadcast_to(self.mesh.volumes[:, None] * self.density / 4, shape[:2]).copy()
        if self.mechanics_mode != "disabled":
            self.corner_masses = _finite_array(self.elastic_balance.corner_masses.reshape(shape[:2]), shape[:2], "native corner masses")
        fractions = np.repeat((self.corner_masses / self.corner_masses.sum()).ravel(), 3)
        self.prior_weights = np.r_[self.position_weight * fractions, self.velocity_weight * fractions]
        self.prior = np.r_[((self.predicted_positions - self.positions) / self.position_scale).ravel(), (self.predicted_velocities / self.velocity_scale).ravel()]
        self.momentum_islands = self._momentum_islands()
        self.factors = self._build_factors()
        supported = bool(self.fixed.any() or self.previous_fixed.any())
        closed = not supported and not any(contact.environment for contact in self.contacts)
        self.metadata = {
            "engine_calls": 0,
            "state_representation": "one Lagrangian corner per material tetrahedron vertex",
            "topology": "fixed during each convex corrector solve",
            "kinematic_rule": self.kinematic_rule,
            "contact_geometry": "supplied frozen linearized normal constraints",
            "contact_law": "nonpenetration only; no verified impulse or restitution law",
            "momentum_constraint": "linear and angular momentum per closed interaction island; external reaction inferred on supported or environment-contact islands"
            if self.mechanics_mode == "disabled" else "frozen discrete island balance: residual penalty in proximal mode, equality in exact reference; physical linear and angular momentum recorded independently",
            "momentum_partition": "start-of-step intact material connectivity plus current contact hyperedges",
            "momentum_island_count": len(self.momentum_islands),
            "newly_broken_faces": int(np.count_nonzero(self.broken_faces & ~self.previous_broken_faces)),
            "released_support_corners": int(np.count_nonzero(self.previous_fixed & ~self.fixed)),
            "support_reaction_interval": "supports present at either endpoint may contribute an unmeasured reaction during the step",
            "boundary_momentum_verified": closed,
            "closed_boundary": closed,
            "physical_momentum_equality_enforced": closed and self.mechanics_mode == "disabled",
            "discrete_balance_equality_enforced": self.mechanics_mode == "exact",
            "angular_momentum_constraint": closed and self.mechanics_mode == "disabled",
            "mechanics_mode": self.mechanics_mode,
            "mechanics_weight": self.mechanics_weight,
            "fracture_update_order": "endpoint candidate topology" if self.mechanics_mode == "disabled" else "integrate old topology, then copy state into the generated endpoint topology",
            "angular_momentum_scope": "explicit invariant" if self.mechanics_mode == "disabled" else "diagnostic only; pinned frozen-rotation implicit teacher step",
            "frictionless_boundary_projection": not supported and any(contact.environment for contact in self.contacts),
            "external_torque_quadrature": "supplied point impulses act at current material-corner positions",
            "feasibility_check": "shared HiGHS linear feasibility presolve; solver failure is unknown",
            "position_scale": self.position_scale,
            "velocity_scale": self.velocity_scale,
            "prior_objective": "0.5 * sum(corner_mass / total_mass * (position_weight * normalized_position_correction^2 + velocity_weight * normalized_velocity_correction^2))",
            "intact_faces": int(np.count_nonzero(~self.broken_faces[self.mesh.internal_faces])),
            "broken_faces": int(self.broken_faces.sum()),
            "factor_names": [factor.name for factor in self.factors],
            "contact_tolerance_metres": float(self.contact_tolerance),
            "position_arithmetic": "IEEE float32 publication error envelope" if self.position_roundoff is not None else "exact analytical integration",
            "position_stabilization": "frozen minimum-A-response contact correction" if self.stabilization_offset is not None else "none",
        }
        if self.kinematic_rule == "constant_acceleration":
            self.metadata["external_torque_quadrature"] = "internal impulse at the initial state followed by continuous constant external acceleration"

    @property
    def corner_count(self):
        return 4 * len(self.mesh.tets)

    def _factor(self, name, rows, columns, values, bound, inequality=False):
        columns = np.asarray(columns, dtype=np.int64)
        indices, local = np.unique(columns, return_inverse=True)
        matrix = sparse.csr_matrix((values, (rows, local)), shape=(len(bound), len(indices)))
        return _Factor(name, indices, matrix, np.asarray(bound, dtype=float), inequality)

    def _momentum_islands(self):
        edges = list(self.mesh.dual_edges[~self.previous_broken_faces[self.mesh.internal_faces]])
        for contact in self.contacts:
            cells = np.unique(contact.corner_ids[contact.weights != 0] // 4)
            edges.extend((cells[0], cell) for cell in cells[1:])
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        labels = component_labels(len(self.mesh.tets), edges, np.zeros(len(edges), dtype=bool))
        return tuple(np.flatnonzero(np.repeat(labels, 4) == label) for label in np.unique(labels))

    def _island_environment(self, corners):
        return [contact for contact in self.contacts if contact.environment
                and np.isin(contact.corner_ids[contact.weights != 0], corners).any()]

    def _build_factors(self):
        factors = []
        count = self.corner_count
        beta = self.dt * self.velocity_scale / self.position_scale
        if self.kinematic_rule == "trapezoidal":
            beta /= 2
        groups = self.mesh.corner_groups(self.broken_faces[self.mesh.internal_faces])
        constrained = np.isin(groups, groups[self.fixed.ravel()])
        for corner in range(count):
            if constrained[corner]:
                continue
            columns = np.r_[3 * corner + np.arange(3), 3 * count + 3 * corner + np.arange(3)]
            bound = np.zeros(3) if self.kinematic_rule == "backward_euler" else self.dt * self.velocities.reshape(-1, 3)[corner] / (2 * self.position_scale)
            if self.kinematic_rule == "constant_acceleration":
                impulse = self.external_impulse.reshape(-1, 3)[corner]
                bound = -0.5 * self.dt * impulse / (self.corner_masses.ravel()[corner] * self.position_scale)
            if self.stabilization_offset is not None:
                bound = bound + self.stabilization_offset.reshape(-1, 3)[corner] / self.position_scale
            if self.position_roundoff is None:
                factors.append(self._factor(f"kinematic:{corner}", np.tile(np.arange(3), 2), columns, np.r_[np.ones(3), -beta * np.ones(3)], bound))
            else:
                radius = self.position_roundoff.reshape(-1, 3)[corner] / self.position_scale
                for axis in range(3):
                    indices = columns[[axis, 3 + axis]]
                    for sign, side in ((1., "lower"), (-1., "upper")):
                        factors.append(self._factor(f"kinematic:{corner}:{axis}:{side}", [0, 0], indices,
                                                    [sign, -sign * beta], [sign * bound[axis] - radius[axis]], True))
        flat_positions = self.positions.reshape(-1, 3)
        for face in self.mesh.internal_faces:
            if self.broken_faces[face]:
                continue
            cell_a, cell_b = self.mesh.faces.cells[face]
            columns, rows, values, bound = [], [], [], []
            for vertex in self.mesh.faces.vertices[face]:
                corner_a = 4 * cell_a + int(np.flatnonzero(self.mesh.tets[cell_a] == vertex)[0])
                corner_b = 4 * cell_b + int(np.flatnonzero(self.mesh.tets[cell_b] == vertex)[0])
                for offset in (0, 3 * count):
                    for axis in range(3):
                        row = len(bound)
                        rows.extend([row, row])
                        columns.extend([offset + 3 * corner_a + axis, offset + 3 * corner_b + axis])
                        values.extend([1.0, -1.0])
                        bound.append((flat_positions[corner_b, axis] - flat_positions[corner_a, axis]) / self.position_scale if offset == 0 else 0.0)
            factors.append(self._factor(f"intact_face:{face}", rows, columns, values, bound))
        for corner in np.flatnonzero(self.fixed.ravel()):
            columns = np.r_[3 * corner + np.arange(3), 3 * count + 3 * corner + np.arange(3)]
            factors.append(self._factor(f"fixed:{corner}", np.arange(6), columns, np.ones(6), np.zeros(6)))
        for index, contact in enumerate(self.contacts):
            columns = (3 * contact.corner_ids[:, None] + np.arange(3)).ravel()
            values = (contact.weights[:, None] * contact.normal).ravel()
            bound = [(-contact.gap - self.contact_tolerance + self.dt * contact.environment_velocity) / self.position_scale]
            factors.append(self._factor(f"contact:{index}", np.zeros(len(columns), dtype=int), columns, values, bound, True))
            if self.native_contact_phase:
                velocity_columns = columns + 3 * count
                target = contact.environment_velocity - max(contact.gap + self.contact_tolerance, 0.) / self.dt
                factors.append(self._factor(f"contact_velocity:{index}", np.zeros(len(columns), dtype=int), velocity_columns,
                                            values, [target / self.velocity_scale], True))
        for island, corners in enumerate(self.momentum_islands):
            factors.extend(self._momentum_factors(corners, "" if len(self.momentum_islands) == 1 else f":{island}"))
        if self.mechanics_mode != "disabled":
            for factor in self.elastic_balance.factors(self):
                if self.mechanics_mode == "proximal":
                    group = int(factor.name.rsplit(":", 1)[-1])
                    factor.penalty = self.mechanics_weight * self.elastic_balance.node_masses[group] / self.elastic_balance.node_masses.sum()
                factors.append(factor)
        return tuple(factors)

    def _momentum_factors(self, corners, suffix):
        if (self.fixed | self.previous_fixed).ravel()[corners].any():
            return []
        if self.mechanics_mode != "disabled":
            return self._discrete_momentum_factors(corners, suffix)
        factors = []
        mass = self.corner_masses.ravel()[corners]
        positions = self.positions.reshape(-1, 3)[corners]
        velocities = self.velocities.reshape(-1, 3)[corners]
        impulse = self.external_impulse.reshape(-1, 3)[corners]
        columns = 3 * self.corner_count + (3 * corners[:, None] + np.arange(3)).ravel()
        target_momentum = np.sum(mass[:, None] * velocities + impulse, axis=0)
        origin = np.average(positions, axis=0, weights=mass)
        lever = positions - origin
        length = float(np.linalg.norm(np.ptp(positions, axis=0)))
        if length <= 0:
            raise ValueError("momentum islands require nonzero spatial extent")
        moment_arm = lever + self.dt * velocities / 2 if self.kinematic_rule == "trapezoidal" else lever
        old_angular = np.sum(np.cross(lever, mass[:, None] * velocities), axis=0)
        torque = np.sum(np.cross(lever, impulse), axis=0)
        environment = self._island_environment(corners)
        if not environment:
            factors.append(self._factor(f"linear_momentum{suffix}", np.tile(np.arange(3), len(corners)), columns,
                                        np.repeat(mass / mass.sum(), 3), target_momentum / (mass.sum() * self.velocity_scale)))
            if self.mechanics_mode != "disabled":
                return factors
            cross = np.zeros((len(corners), 3, 3))
            cross[:, 0, 1], cross[:, 0, 2] = -moment_arm[:, 2], moment_arm[:, 1]
            cross[:, 1, 0], cross[:, 1, 2] = moment_arm[:, 2], -moment_arm[:, 0]
            cross[:, 2, 0], cross[:, 2, 1] = -moment_arm[:, 1], moment_arm[:, 0]
            factors.append(self._factor(f"angular_momentum{suffix}", np.tile(np.repeat(np.arange(3), 3), len(corners)),
                                        (3 * self.corner_count + 3 * corners[:, None] + np.tile(np.arange(3), 3)).ravel(),
                                        (mass[:, None, None] * cross / (mass.sum() * length)).ravel(),
                                        (old_angular + torque) / (mass.sum() * length * self.velocity_scale)))
        else:
            normals = np.stack([contact.normal for contact in environment])
            tangents = null_space(normals.T @ normals).T
            if len(tangents):
                coefficients = tangents[:, None, :] * (mass / mass.sum())[None, :, None]
                factors.append(self._factor(f"boundary_tangential_momentum{suffix}", np.repeat(np.arange(len(tangents)), 3 * len(corners)),
                                             np.tile(columns, len(tangents)), coefficients.ravel(),
                                             tangents @ target_momentum / (mass.sum() * self.velocity_scale)))
            if np.allclose(normals, normals[0], rtol=1e-10, atol=1e-12):
                normal = normals[0]
                coefficients = mass[:, None] * normal / mass.sum()
                factors.append(self._factor(f"nonnegative_boundary_normal_impulse{suffix}", np.zeros(3 * len(corners), dtype=int),
                                             columns, coefficients.ravel(),
                                             [float(normal @ target_momentum / (mass.sum() * self.velocity_scale))], True))
                if self.mechanics_mode != "disabled":
                    return factors
                coefficients = mass[:, None] * np.cross(normal, moment_arm) / (mass.sum() * length)
                factors.append(self._factor(f"boundary_normal_axis_angular_momentum{suffix}", np.zeros(3 * len(corners), dtype=int), columns,
                                             coefficients.ravel(), [float(normal @ (old_angular + torque) / (mass.sum() * length * self.velocity_scale))]))
        return factors

    def _discrete_momentum_factors(self, corners, suffix):
        balance = self.elastic_balance
        nodes = np.unique(balance.groups[corners])
        rows = (3 * nodes[:, None] + np.arange(3)).ravel()
        aggregate = sparse.csr_matrix((np.ones(len(rows)), (np.tile(np.arange(3), len(nodes)), rows)),
                                      shape=(3, len(balance.rhs)))
        mass = float(self.corner_masses.ravel()[corners].sum())
        matrix = (aggregate @ balance.matrix) / mass
        bound = np.asarray(aggregate @ balance.rhs).ravel() / (mass * self.velocity_scale)
        environment = self._island_environment(corners)
        projections = [(f"discrete_linear_balance{suffix}", np.eye(3), False)]
        if environment:
            normals = np.stack([contact.normal for contact in environment])
            projections = [(f"discrete_boundary_tangential_balance{suffix}", null_space(normals.T @ normals).T, False)]
            if np.allclose(normals, normals[0], rtol=1e-10, atol=1e-12):
                projections.append((f"discrete_nonnegative_boundary_impulse{suffix}", normals[:1], True))
        factors = []
        for name, projection, inequality in projections:
            if not len(projection):
                continue
            coo = (sparse.csr_matrix(projection) @ matrix).tocoo()
            columns = 3 * self.corner_count + 3 * balance.representatives[coo.col // 3] + coo.col % 3
            factor = self._factor(name, coo.row, columns, coo.data, projection @ bound, inequality)
            if self.mechanics_mode == "proximal" and not inequality:
                factor.penalty = self.mechanics_weight * mass / self.corner_masses.sum()
            factors.append(factor)
        return factors

    def unpack(self, state):
        state = _finite_array(state, self.prior.shape, "corrector state")
        cut = 3 * self.corner_count
        return self.positions + self.position_scale * state[:cut].reshape(self.positions.shape), self.velocity_scale * state[cut:].reshape(self.velocities.shape)

    def objective(self, state):
        difference = np.asarray(state) - self.prior
        return float(0.5 * np.dot(self.prior_weights, difference**2) +
                     sum(factor.cost(np.asarray(state)[factor.indices]) for factor in self.factors))

    def encode_state(self, positions, velocities):
        return np.r_[((np.asarray(positions) - self.positions) / self.position_scale).ravel(),
                     (np.asarray(velocities) / self.velocity_scale).ravel()]

    def direction_to_corners(self, direction):
        direction = _finite_array(direction, self.prior.shape, "state direction")
        cut = 3 * self.corner_count
        return (self.position_scale * direction[:cut].reshape(self.positions.shape),
                self.velocity_scale * direction[cut:].reshape(self.velocities.shape))

    def gradient_from_corners(self, position_gradient, velocity_gradient):
        return np.r_[self.position_scale * np.asarray(position_gradient).ravel(),
                     self.velocity_scale * np.asarray(velocity_gradient).ravel()]

    def zero_energy_state(self, positions):
        rest = self.mesh.nodes[self.mesh.tets]
        projected = _rigid_projection(rest, np.asarray(positions).reshape(rest.shape))
        return self.encode_state(projected, np.zeros_like(projected))


def _rigid_projection(rest, positions):
    source = rest - rest.mean(axis=-2, keepdims=True)
    center = positions.mean(axis=-2, keepdims=True)
    left, _, right = np.linalg.svd(np.swapaxes(source, -2, -1) @ (positions - center))
    signs = np.ones(left.shape[:-1])
    signs[..., -1] = np.where(np.linalg.det(left @ right) < 0, -1.0, 1.0)
    return source @ ((left * signs[..., None, :]) @ right) + center


class _SharedDynamicsProblem:
    def __init__(self, source):
        self.source_problem = source
        self.groups = source.mesh.corner_groups(source.broken_faces[source.mesh.internal_faces])
        self.group_count = int(self.groups.max()) + 1
        _, self.representatives = np.unique(self.groups, return_index=True)
        self.reference_positions = source.positions.reshape(-1, 3)[self.representatives]
        position_indices = (3 * self.groups[:, None] + np.arange(3)).ravel()
        self.state_indices = np.r_[position_indices, 3 * self.group_count + position_indices]
        self.offset = np.r_[((self.reference_positions[self.groups] - source.positions.reshape(-1, 3))
                             / source.position_scale).ravel(), np.zeros(3 * source.corner_count)]
        self.prior_weights = np.bincount(self.state_indices, weights=source.prior_weights,
                                        minlength=6 * self.group_count)
        self.prior = np.bincount(self.state_indices, weights=source.prior_weights * (source.prior - self.offset),
                                minlength=6 * self.group_count) / self.prior_weights
        difference = self.prior[self.state_indices] + self.offset - source.prior
        self.objective_offset = float(0.5 * np.dot(source.prior_weights, difference**2))
        self.factors, aliases = self._reduced_factors(source.factors)
        self.metadata = {**source.metadata, "factor_names": [factor.name for factor in self.factors],
                         "state_parameterization": "exact shared next positions and velocities on intact material corners",
                         "full_state_variables": len(source.prior), "solver_state_variables": len(self.prior),
                         "shared_corner_groups": self.group_count, "original_factor_count": len(source.factors),
                         "solver_factor_count": len(self.factors), "factor_aliases": aliases,
                         "irreducible_prediction_cost": self.objective_offset}

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "source_problem"), name)

    def _reduced_factors(self, factors):
        reduced, signatures, aliases = [], {}, {}
        for factor in factors:
            coo = factor.matrix.tocoo()
            columns = self.state_indices[factor.indices[coo.col]]
            indices, local = np.unique(columns, return_inverse=True)
            matrix = sparse.csr_matrix((coo.data, (coo.row, local)), shape=(len(factor.bound), len(indices)))
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
            bound = factor.bound - factor.matrix @ self.offset[factor.indices]
            active = np.diff(matrix.indptr) > 0
            active |= bound > 0 if factor.inequality else bound != 0
            if not active.any():
                aliases[factor.name] = "exactly eliminated"
                continue
            matrix, bound = matrix[active], np.asarray(bound)[active]
            coo = matrix.tocoo()
            used, local = np.unique(coo.col, return_inverse=True)
            indices = indices[used]
            matrix = sparse.csr_matrix((coo.data, (coo.row, local)), shape=(len(bound), len(indices)))
            signature = (factor.inequality, indices.tobytes(), matrix.indptr.tobytes(),
                         matrix.indices.tobytes(), matrix.data.tobytes(), bound.tobytes())
            if factor.penalty is None and signature in signatures:
                aliases[factor.name] = signatures[signature]
                continue
            if factor.penalty is None:
                signatures[signature] = factor.name
            reduced.append(_Factor(factor.name, indices, matrix, bound, factor.inequality, factor.penalty))
        return tuple(reduced), aliases

    def unpack(self, state):
        state = _finite_array(state, self.prior.shape, "shared corrector state")
        cut = 3 * self.group_count
        positions = self.reference_positions + self.position_scale * state[:cut].reshape(-1, 3)
        velocities = self.velocity_scale * state[cut:].reshape(-1, 3)
        return positions[self.groups].reshape(self.positions.shape), velocities[self.groups].reshape(self.velocities.shape)

    def encode_state(self, positions, velocities):
        positions = np.asarray(positions).reshape(-1, 3)[self.representatives]
        velocities = np.asarray(velocities).reshape(-1, 3)[self.representatives]
        return np.r_[((positions - self.reference_positions) / self.position_scale).ravel(),
                     (velocities / self.velocity_scale).ravel()]

    def direction_to_corners(self, direction):
        direction = _finite_array(direction, self.prior.shape, "shared state direction")
        cut = 3 * self.group_count
        positions = self.position_scale * direction[:cut].reshape(-1, 3)
        velocities = self.velocity_scale * direction[cut:].reshape(-1, 3)
        return positions[self.groups].reshape(self.positions.shape), velocities[self.groups].reshape(self.velocities.shape)

    def gradient_from_corners(self, position_gradient, velocity_gradient):
        full = self.source_problem.gradient_from_corners(position_gradient, velocity_gradient)
        return np.bincount(self.state_indices, weights=full, minlength=len(self.prior))

    def objective(self, state):
        difference = np.asarray(state) - self.prior
        return float(self.objective_offset + 0.5 * np.dot(self.prior_weights, difference**2) +
                     sum(factor.cost(np.asarray(state)[factor.indices]) for factor in self.factors))

    def zero_energy_state(self, positions):
        rest = self.mesh.nodes[self.mesh.tets].reshape(-1, 3)[self.representatives]
        target = np.asarray(positions).reshape(-1, 3)[self.representatives]
        projected = np.empty_like(target)
        labels = component_labels(len(self.mesh.tets), self.mesh.dual_edges,
                                  self.broken_faces[self.mesh.internal_faces])
        for label in np.unique(labels):
            corners = np.flatnonzero(np.repeat(labels, 4) == label)
            groups = np.unique(self.groups[corners])
            projected[groups] = _rigid_projection(rest[groups], target[groups])
        shape = self.positions.shape
        return self.encode_state(projected[self.groups].reshape(shape), np.zeros(shape))


def _shared_problem(problem):
    if isinstance(problem, _SharedDynamicsProblem):
        return problem
    groups = problem.mesh.corner_groups(problem.broken_faces[problem.mesh.internal_faces])
    return _SharedDynamicsProblem(problem) if len(np.unique(groups)) < problem.corner_count else problem


@dataclass(frozen=True)
class CorrectorIteration:
    iteration: int
    objective: float
    primal_residual: float
    dual_residual: float
    primal_tolerance: float
    dual_tolerance: float
    constraint_violation: float
    rho: float


@dataclass(frozen=True)
class CorrectorResult:
    method: str
    positions: np.ndarray
    velocities: np.ndarray
    objective: float
    converged: bool
    status: str
    iterations: int
    diagnostics: dict
    history: list[CorrectorIteration]
    runtime_seconds: float


def _limit(value):
    if value is not None and (not isinstance(value, (int, np.integer)) or value < 0):
        raise ValueError("max_iterations must be a nonnegative integer or None")


def _island_momentum_diagnostics(problem, positions, velocities, bound):
    reports = []
    for island, corners in enumerate(problem.momentum_islands):
        mass = problem.corner_masses.ravel()[corners]
        previous_position = problem.positions.reshape(-1, 3)[corners]
        previous_velocity = problem.velocities.reshape(-1, 3)[corners]
        next_position = positions.reshape(-1, 3)[corners]
        next_velocity = velocities.reshape(-1, 3)[corners]
        impulse = problem.external_impulse.reshape(-1, 3)[corners]
        origin = np.average(previous_position, axis=0, weights=mass)
        lever = previous_position - origin
        residual = np.sum(mass[:, None] * (next_velocity - previous_velocity) - impulse, axis=0)
        old_angular = np.sum(np.cross(lever, mass[:, None] * previous_velocity), axis=0)
        new_angular = np.sum(np.cross(next_position - origin, mass[:, None] * next_velocity), axis=0)
        angular_impulse = np.sum(np.cross(lever, impulse), axis=0)
        if problem.kinematic_rule == "constant_acceleration":
            angular_impulse += 0.5 * problem.dt * np.sum(np.cross(next_velocity, impulse), axis=0)
        angular_residual = new_angular - old_angular - angular_impulse
        length = float(np.linalg.norm(np.ptp(previous_position, axis=0)))
        supported = bool((problem.fixed | problem.previous_fixed).ravel()[corners].any())
        environment = bool(problem._island_environment(corners))
        closed = not supported and not environment
        scale = mass.sum() * problem.velocity_scale
        reports.append(dict(island=island, tetrahedra=(corners[::4] // 4).tolist(),
                            boundary="support" if supported else "environment_contact" if environment else "closed",
                            linear_residual_or_inferred_reaction=residual.tolist(),
                            angular_residual_or_inferred_reaction=angular_residual.tolist(),
                            linear_momentum_verified=bool(closed and np.max(np.abs(residual)) / scale <= bound),
                            angular_momentum_verified=bool(closed and np.max(np.abs(angular_residual)) / (scale * length) <= bound)
                            if problem.mechanics_mode == "disabled" else None))
    return reports


def _diagnostics(problem, state, tolerance):
    if isinstance(problem, _SharedDynamicsProblem):
        positions, velocities = problem.unpack(state)
        original = problem.source_problem
        report = _diagnostics(original, original.encode_state(positions, velocities), tolerance)
        for factor in problem.factors:
            violation = float(np.max(factor.violation(np.asarray(state)[factor.indices]), initial=0))
            report["factor_violations"][factor.name] = max(report["factor_violations"].get(factor.name, 0.0), violation)
        report["constraint_violation"] = max(report["factor_violations"].values(), default=0.0)
        report["constraints_satisfied"] = bool(report["constraint_violation"] <= report["constraint_tolerance"])
        report["shared_coordinates"] = {key: problem.metadata[key] for key in (
            "state_parameterization", "full_state_variables", "solver_state_variables", "shared_corner_groups",
            "original_factor_count", "solver_factor_count", "irreducible_prediction_cost")}
        return report
    positions, velocities = problem.unpack(state)
    violations = {factor.name: float(np.max(factor.violation(state[factor.indices]), initial=0)) for factor in problem.factors}
    actual_impulse = np.sum(problem.corner_masses[..., None] * (velocities - problem.velocities), axis=(0, 1))
    supplied_impulse = problem.external_impulse.sum(axis=(0, 1))
    residual = actual_impulse - supplied_impulse
    mass = problem.corner_masses.ravel()
    origin = np.average(problem.positions.reshape(-1, 3), axis=0, weights=mass)
    old_lever = problem.positions.reshape(-1, 3) - origin
    old_angular = np.sum(np.cross(old_lever, mass[:, None] * problem.velocities.reshape(-1, 3)), axis=0)
    new_angular = np.sum(np.cross(positions.reshape(-1, 3) - origin, mass[:, None] * velocities.reshape(-1, 3)), axis=0)
    supplied_angular = np.sum(np.cross(old_lever, problem.external_impulse.reshape(-1, 3)), axis=0)
    if problem.kinematic_rule == "constant_acceleration":
        supplied_angular += 0.5 * problem.dt * np.sum(np.cross(velocities.reshape(-1, 3), problem.external_impulse.reshape(-1, 3)), axis=0)
    angular_residual = new_angular - old_angular - supplied_angular
    angular_length = float(np.linalg.norm(np.ptp(problem.positions.reshape(-1, 3), axis=0)))
    bound = tolerance.absolute + tolerance.relative * max(float(np.max(np.abs(state), initial=0)), 1.0)
    violation = max(violations.values(), default=0.0)
    island_reports = _island_momentum_diagnostics(problem, positions, velocities, bound)
    free_islands = [item for item in island_reports if item["boundary"] == "closed"]
    contacts = [float(contact.gap + np.dot(contact.normal, np.sum(contact.weights[:, None] * (positions - problem.positions).reshape(-1, 3)[contact.corner_ids], axis=0)) - problem.dt * contact.environment_velocity) for contact in problem.contacts]
    return {
        "constraint_violation": violation,
        "constraint_tolerance": float(bound),
        "constraints_satisfied": bool(violation <= bound),
        "factor_violations": violations,
        "objective_components": {
            "prediction_fidelity": float(0.5 * np.dot(problem.prior_weights, (np.asarray(state) - problem.prior)**2)),
            "elastic_residual_penalty": sum(factor.cost(np.asarray(state)[factor.indices]) for factor in problem.factors),
        },
        "linearized_contact_gaps": contacts,
        "linear_momentum_change": actual_impulse.tolist(),
        "supplied_external_impulse": supplied_impulse.tolist(),
        "momentum_residual_or_inferred_boundary_impulse": residual.tolist(),
        "angular_momentum_change": (new_angular - old_angular).tolist(),
        "supplied_external_angular_impulse": supplied_angular.tolist(),
        "angular_residual_or_inferred_boundary_angular_impulse": angular_residual.tolist(),
        "angular_momentum_origin": origin.tolist(),
        "external_torque_quadrature": problem.metadata["external_torque_quadrature"],
        "closed_system_momentum_constraint": problem.metadata["physical_momentum_equality_enforced"],
        "discrete_balance_equality_enforced": problem.metadata["discrete_balance_equality_enforced"],
        "boundary_momentum_verified": bool(problem.metadata["boundary_momentum_verified"] and np.max(np.abs(residual)) / (problem.corner_masses.sum() * problem.velocity_scale) <= bound),
        "angular_momentum_verified": bool(problem.metadata["angular_momentum_constraint"] and np.max(np.abs(angular_residual)) / (problem.corner_masses.sum() * angular_length * problem.velocity_scale) <= bound)
        if problem.mechanics_mode == "disabled" else None,
        "contact_impulse_law_verified": False,
        "momentum_partition": problem.metadata["momentum_partition"],
        "momentum_islands": island_reports,
        "free_momentum_island_count": len(free_islands),
        "free_momentum_islands_verified": all(item["linear_momentum_verified"] and item["angular_momentum_verified"]
                                               for item in free_islands) if free_islands and problem.mechanics_mode == "disabled" else None,
        "free_linear_momentum_islands_verified": all(item["linear_momentum_verified"] for item in free_islands) if free_islands else None,
        "engine_calls": 0,
    }


def _constraint_system(problem):
    systems = []
    for inequality in (False, True):
        rows, columns, values, bounds = [], [], [], []
        offset = 0
        sign = -1.0 if inequality else 1.0
        for factor in problem.factors:
            if factor.inequality != inequality or factor.penalty is not None:
                continue
            coo = factor.matrix.tocoo()
            rows.append(coo.row + offset)
            columns.append(factor.indices[coo.col])
            values.append(sign * coo.data)
            bounds.append(sign * factor.bound)
            offset += len(factor.bound)
        matrix = sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                                   shape=(offset, len(problem.prior))).tocsc() if values else sparse.csc_matrix((0, len(problem.prior)))
        systems.append((matrix, np.concatenate(bounds) if bounds else np.empty(0)))
    return tuple(systems)


def _quadratic_objective(problem):
    matrix = sparse.diags(problem.prior_weights, format="csc")
    linear = -problem.prior_weights * problem.prior
    for factor in problem.factors:
        if factor.penalty is None:
            continue
        coo = factor.matrix.tocoo()
        full = sparse.csc_matrix((coo.data, (coo.row, factor.indices[coo.col])), shape=(len(factor.bound), len(problem.prior)))
        matrix = matrix + factor.penalty * (full.T @ full)
        linear = linear - factor.penalty * np.asarray(full.T @ factor.bound).ravel()
    return sparse.triu(matrix, format="csc"), linear


def _prior_is_minimum(problem):
    return all(factor.cost(problem.prior[factor.indices]) == 0.0 for factor in problem.factors)


def _factor_inverse_gram(problem, rho):
    return sparse.block_diag([factor.projection_inverse(rho) for factor in problem.factors], format="csr")


def _feasibility(problem, tolerances):
    start = perf_counter()
    report = {"method": "scipy.optimize.linprog / HiGHS", "scope": "supplied linear equality and inequality factors", "status": "unknown", "iterations": 0, "engine_calls": 0}
    prior_diagnostics = _diagnostics(problem, problem.prior, tolerances)
    if prior_diagnostics["constraints_satisfied"]:
        report.update(method="direct feasible prior certificate", status="feasible", maximum_violation=prior_diagnostics["constraint_violation"])
    else:
        try:
            (equalities, equality_bounds), (inequalities, inequality_bounds) = _constraint_system(problem)
            result = linprog(np.zeros(len(problem.prior)), A_ub=inequalities if len(inequality_bounds) else None,
                             b_ub=inequality_bounds if len(inequality_bounds) else None,
                             A_eq=equalities if len(equality_bounds) else None, b_eq=equality_bounds if len(equality_bounds) else None,
                             bounds=(None, None), method="highs",
                             options={"primal_feasibility_tolerance": max(1e-10, min(tolerances.absolute, 1e-7))})
            report.update(solver_status=int(result.status), solver_message=str(result.message), iterations=int(getattr(result, "nit", 0) or 0))
            if result.status == 2:
                report["status"] = "infeasible"
            elif result.status == 0 and result.x is not None and np.isfinite(result.x).all():
                certificate = _diagnostics(problem, result.x, tolerances)
                report.update(status="feasible" if certificate["constraints_satisfied"] else "unknown", maximum_violation=certificate["constraint_violation"])
        except Exception as error:
            report["solver_message"] = f"{type(error).__name__}: {error}"
    report["runtime_seconds"] = perf_counter() - start
    return report


def _result(problem, method, state, tolerance, converged, status, iterations, history, start, feasibility=None):
    positions, velocities = problem.unpack(state)
    diagnostics = _diagnostics(problem, state, tolerance)
    if converged and not diagnostics["constraints_satisfied"]:
        converged, status = False, "original_constraint_validation_failed"
    diagnostics["feasibility"] = feasibility or {"method": "not run", "status": "unknown", "iterations": 0, "runtime_seconds": 0.0, "engine_calls": 0}
    return CorrectorResult(method, positions, velocities, problem.objective(state), converged, status, iterations, diagnostics, history, perf_counter() - start)


def solve_central(problem, tolerances=None, max_iterations=None, cancel=None):
    import clarabel

    start = perf_counter()
    problem = _shared_problem(problem)
    tolerances = tolerances or CorrectorTolerances()
    _limit(max_iterations)
    if cancel is not None and cancel():
        return _result(problem, "central", problem.prior, tolerances, False, "interrupted", 0, [], start)
    if max_iterations == 0:
        return _result(problem, "central", problem.prior, tolerances, False, "maximum_iterations", 0, [], start)
    feasibility = _feasibility(problem, tolerances)
    if feasibility["status"] != "feasible":
        status = "PrimalInfeasible" if feasibility["status"] == "infeasible" else "feasibility_unknown"
        return _result(problem, "central", problem.prior, tolerances, False, status, 0, [], start, feasibility)
    if feasibility["method"] == "direct feasible prior certificate" and _prior_is_minimum(problem):
        return _result(problem, "central", problem.prior, tolerances, True, "converged", 0, [], start, feasibility)
    (equalities, equality_bounds), (inequalities, inequality_bounds) = _constraint_system(problem)
    matrix = sparse.vstack((equalities, inequalities), format="csc")
    bounds = np.r_[equality_bounds, inequality_bounds]
    cones = [clarabel.ZeroConeT(len(equality_bounds)), clarabel.NonnegativeConeT(len(inequality_bounds))]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_feas = max(1e-12, min(tolerances.absolute, tolerances.relative or tolerances.absolute) * 0.01)
    settings.tol_gap_abs = settings.tol_feas
    settings.tol_gap_rel = settings.tol_feas
    if max_iterations is not None:
        settings.max_iter = int(max_iterations)
    iterations = 0
    objective_matrix, objective_linear = _quadratic_objective(problem)
    while True:
        solver = clarabel.DefaultSolver(objective_matrix, objective_linear, matrix, bounds, cones, settings)
        if cancel is not None:
            solver.set_termination_callback(lambda info: bool(cancel()))
        solution = solver.solve()
        iterations += int(solution.iterations)
        if max_iterations is not None or str(solution.status) != "MaxIterations":
            break
        settings.max_iter *= 2
    state = np.asarray(solution.x, dtype=float) if solution.x is not None else problem.prior.copy()
    if state.shape != problem.prior.shape or not np.isfinite(state).all():
        return _result(problem, "central", problem.prior, tolerances, False, "invalid_solver_output", iterations, [], start, feasibility)
    diagnostics = _diagnostics(problem, state, tolerances)
    converged = str(solution.status) in ("Solved", "AlmostSolved") and diagnostics["constraints_satisfied"]
    status = "interrupted" if cancel is not None and cancel() else "converged" if converged else str(solution.status)
    converged = converged and status != "interrupted"
    return _result(problem, "central", state, tolerances, converged, status, iterations, [], start, feasibility)


def solve_consensus(problem, rho=None, tolerances=None, max_iterations=None, cancel: Callable[[], bool] | None = None, adaptive_rho=True,
                    *, progress=None, _workspace=None):
    start = perf_counter()
    problem = _shared_problem(problem)
    tolerances = tolerances or CorrectorTolerances()
    _limit(max_iterations)
    rho = float(np.median(problem.prior_weights)) if rho is None else float(rho)
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive")
    if _workspace is not None:
        _workspace.clear()
        _workspace.update(state=problem.prior.copy(), scaled_dual=np.zeros(sum(len(factor.indices) for factor in problem.factors)),
                          rho=rho, factor_names=tuple(factor.name for factor in problem.factors))
    if cancel is not None and cancel():
        return _result(problem, "consensus", problem.prior, tolerances, False, "interrupted", 0, [], start)
    if max_iterations == 0:
        return _result(problem, "consensus", problem.prior, tolerances, False, "maximum_iterations", 0, [], start)
    feasibility = _feasibility(problem, tolerances)
    if feasibility["status"] != "feasible":
        status = "PrimalInfeasible" if feasibility["status"] == "infeasible" else "feasibility_unknown"
        return _result(problem, "consensus", problem.prior, tolerances, False, status, 0, [], start, feasibility)
    if feasibility["method"] == "direct feasible prior certificate" and _prior_is_minimum(problem):
        return _result(problem, "consensus", problem.prior, tolerances, True, "converged", 0, [], start, feasibility)
    state = problem.prior.copy()
    indices = np.concatenate([factor.indices for factor in problem.factors])
    matrix = sparse.block_diag([factor.matrix for factor in problem.factors], format="csr")
    transpose = matrix.T.tocsr()
    inverse_gram = _factor_inverse_gram(problem, rho)
    bound = np.concatenate([factor.bound for factor in problem.factors])
    inequality = np.concatenate([np.full(len(factor.bound), factor.inequality, dtype=bool) for factor in problem.factors])
    soft_rows = np.concatenate([np.full(len(factor.bound), factor.penalty is not None, dtype=bool) for factor in problem.factors])
    local = state[indices].copy()
    dual = np.zeros_like(local)
    counts = np.bincount(indices, minlength=len(state))
    local_dimension = len(indices)
    history = []
    iteration = 0
    converged = False
    status = "maximum_iterations"
    previous = []
    try:
        while max_iterations is None or iteration < max_iterations:
            if cancel is not None and cancel():
                status = "interrupted"
                break
            old = state.copy()
            local = state[indices] - dual
            projection_residual = bound - matrix @ local
            projection_residual[inequality] = np.maximum(projection_residual[inequality], 0)
            local += transpose @ (inverse_gram @ projection_residual)
            totals = np.bincount(indices, weights=local + dual, minlength=len(state))
            state = (problem.prior_weights * problem.prior + rho * totals) / (problem.prior_weights + rho * counts)
            if not np.isfinite(state).all():
                state = old
                status = "nonfinite_iterate"
                break
            selected = state[indices]
            residual = local - selected
            dual += residual
            primal_squared = float(residual @ residual)
            local_squared = float(local @ local)
            selected_squared = float(selected @ selected)
            dual_squared = float(dual @ dual)
            iteration += 1
            primal = float(np.sqrt(primal_squared))
            dual_residual = float(rho * np.sqrt(np.dot(counts, (state - old)**2)))
            primal_tolerance = float(np.sqrt(local_dimension) * tolerances.absolute + tolerances.relative * max(np.sqrt(local_squared), np.sqrt(selected_squared)))
            dual_tolerance = float(np.sqrt(local_dimension) * tolerances.absolute + tolerances.relative * rho * np.sqrt(dual_squared))
            constraint_residual = matrix @ selected - bound
            constraint_residual[inequality] = np.minimum(constraint_residual[inequality], 0)
            constraint_residual[soft_rows] = 0
            violation = float(np.max(np.abs(constraint_residual), initial=0))
            constraint_tolerance = tolerances.absolute + tolerances.relative * max(float(np.max(np.abs(state), initial=0)), 1.0)
            history.append(CorrectorIteration(iteration, problem.objective(state), primal, dual_residual, primal_tolerance, dual_tolerance, violation, rho))
            if progress and (iteration == 1 or iteration % 25 == 0):
                progress(dict(event="corrector_iteration", method="consensus", stage="linear_constraints", iteration=iteration,
                              objective=history[-1].objective, primal=primal, dual=dual_residual,
                              primal_tolerance=primal_tolerance, dual_tolerance=dual_tolerance,
                              constraint_violation=violation, constraint_tolerance=constraint_tolerance,
                              rho=rho, runtime_seconds=perf_counter() - start))
            if primal <= primal_tolerance and dual_residual <= dual_tolerance and violation <= constraint_tolerance:
                converged, status = True, "converged"
                break
            if adaptive_rho and iteration % 25 == 0:
                multiplier = 2.0 if primal > 10 * dual_residual else 0.5 if dual_residual > 10 * primal else 1.0
                rho *= multiplier
                dual /= multiplier
                if multiplier != 1.0:
                    inverse_gram = _factor_inverse_gram(problem, rho)
            snapshot = (state.copy(), dual.copy(), rho)
            if any(snapshot[2] == prior[2] and np.array_equal(snapshot[0], prior[0]) and np.array_equal(snapshot[1], prior[1]) for prior in previous):
                status = "numerical_cycle"
                break
            previous = (previous + [snapshot])[-2:]
    except KeyboardInterrupt:
        status = "interrupted"
    if _workspace is not None:
        _workspace.update(state=state.copy(), scaled_dual=dual.copy(), rho=rho)
    return _result(problem, "consensus", state, tolerances, converged, status, iteration, history, start, feasibility)
