from dataclasses import dataclass, field

import numpy as np

from .dynamics_energy import native_corotational_energy
from .dynamics_mechanics import FEMFX_SOURCE, FrozenElasticBalance


FROZEN_ENERGY_SOURCE = f"{FEMFX_SOURCE}/SparseSolvers/FEMFXMpcgSolverSetup.cpp"
FROZEN_ENERGY_LAW = "frozen-rotation quadratic elastic + kinetic + gravitational potential"


@dataclass
class FrozenQuadraticEnergy:
    balance_model: FrozenElasticBalance
    nonlinear_energy: object | None = None
    mesh: object = field(init=False)
    density: float = field(init=False)
    youngs_modulus: float = field(init=False)
    poissons_ratio: float = field(init=False)
    gravity: np.ndarray = field(init=False, repr=False)
    masses: np.ndarray = field(init=False, repr=False)
    supports_zero_energy_shortcut: bool = field(default=False, init=False)
    supports_nonnegative_energy_shortcut: bool = field(default=False, init=False)

    def __post_init__(self):
        if not isinstance(self.balance_model, FrozenElasticBalance):
            raise TypeError("frozen energy requires a FrozenElasticBalance")
        source = self.balance_model
        self.mesh = source.mesh
        self.density, self.youngs_modulus, self.poissons_ratio = (
            float(np.float32(source.material[key]))
            for key in ("rest_density", "youngs_modulus", "poissons_ratio"))
        self.gravity = np.array(source.gravity, dtype=float, copy=True)
        corner_masses = np.array(source.corner_masses, dtype=float, copy=True).reshape(-1)
        self.groups = np.array(source.groups, dtype=np.int64, copy=True)
        self.node_masses = np.array(source.node_masses, dtype=float, copy=True)
        self.reference_positions = np.array(source.positions, dtype=float, copy=True).reshape(-1, 4, 3)
        self.reference_velocities = np.array(source.velocities, dtype=float, copy=True).reshape(-1, 4, 3)
        corner_sum = np.bincount(self.groups, weights=corner_masses, minlength=len(self.node_masses))
        if (not np.isfinite(corner_masses).all() or (corner_masses <= 0).any() or
                not np.isfinite(self.node_masses).all() or (self.node_masses <= 0).any() or (corner_sum <= 0).any()):
            raise ValueError("frozen energy requires finite positive material masses")
        self.corner_weights = corner_masses / corner_sum[self.groups]
        self.masses = (self.node_masses[self.groups] * self.corner_weights).reshape(-1, 4)
        self.linear_term = np.array(source.elastic_gradient, dtype=float, copy=True).reshape(-1)
        raw_stiffness = source.stiffness.tocsr(copy=True)
        difference = raw_stiffness - raw_stiffness.T
        self.stiffness_asymmetry = float(np.max(np.abs(difference.data), initial=0.0))
        self.stiffness = (0.5 * (raw_stiffness + raw_stiffness.T)).tocsr()
        if (not np.isfinite(self.masses).all() or (self.masses <= 0).any() or
                not np.isfinite(self.node_masses).all() or (self.node_masses <= 0).any() or
                not np.isfinite(self.linear_term).all() or not np.isfinite(self.stiffness.data).all()):
            raise ValueError("frozen energy requires finite coefficients and positive material masses")
        baseline = native_corotational_energy(self.mesh, self.density, self.youngs_modulus,
                                              self.poissons_ratio, self.gravity)
        self.reference_elastic_energy = baseline.components(
            self.reference_positions, np.zeros_like(self.reference_velocities))["elastic"]
        self.metadata = dict(law=FROZEN_ENERGY_LAW, source=FROZEN_ENERGY_SOURCE,
                             geometry="current-step frozen material connectivity and rotation",
                             corner_reduction="mass-weighted displacement average on shared material nodes",
                             kinetic_mass="actual source nodal mass distributed by normalized material-corner weights",
                             stiffness_quadratic_form="symmetric part; no eigenvalue clipping or PSD assumption",
                             maximum_stiffness_asymmetry=self.stiffness_asymmetry,
                             nonlinear_energy_role="independent diagnostic; never credited as allowed work",
                             engine_calls=0)

    def _states(self, positions, velocities):
        shape = (len(self.mesh.tets), 4, 3)
        positions, velocities = np.asarray(positions, dtype=float), np.asarray(velocities, dtype=float)
        if positions.shape not in (shape, (4 * len(self.mesh.tets), 3)) or velocities.shape != positions.shape:
            raise ValueError("frozen energy states require equal material-corner position and velocity arrays")
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise ValueError("frozen energy states must be finite")
        return positions.reshape(shape), velocities.reshape(shape)

    def _nodal_average(self, values):
        weighted = self.corner_weights[:, None] * np.asarray(values).reshape(-1, 3)
        return np.stack([np.bincount(self.groups, weights=weighted[:, axis], minlength=len(self.node_masses))
                         for axis in range(3)], axis=-1)

    def _displacement(self, positions):
        return self._nodal_average(positions - self.reference_positions).ravel()

    def _elastic_value(self, displacement):
        return float(self.reference_elastic_energy + self.linear_term @ displacement
                     + 0.5 * displacement @ (self.stiffness @ displacement))

    def components(self, positions, velocities):
        positions, velocities = self._states(positions, velocities)
        elastic = self._elastic_value(self._displacement(positions))
        kinetic = float(0.5 * np.sum(self.masses[..., None] * velocities**2))
        gravitational = float(-np.sum(self.masses[..., None] * positions * self.gravity))
        return dict(elastic=elastic, kinetic=kinetic, gravitational=gravitational,
                    total=float(elastic + kinetic + gravitational))

    def value(self, positions, velocities):
        return self.components(positions, velocities)["total"]

    def balance(self, previous_positions, previous_velocities, positions, velocities, velocity_scale, *, budget=None,
                energy_scale=None, absolute_tolerance=1e-8, relative_tolerance=1e-7, reference_components=None):
        previous_positions, previous_velocities = self._states(previous_positions, previous_velocities)
        positions, velocities = self._states(positions, velocities)
        settings = [velocity_scale, absolute_tolerance, relative_tolerance]
        if not np.isfinite(settings).all() or min(velocity_scale, absolute_tolerance) <= 0 or relative_tolerance < 0:
            raise ValueError("frozen energy balance requires finite positive scales and nonnegative tolerances")
        reference = self.components(previous_positions, previous_velocities) if reference_components is None else reference_components
        components = self.components(positions, velocities)
        budget = reference["total"] if budget is None else float(budget)
        if energy_scale is None:
            energy_scale = max(reference["elastic"] + reference["kinetic"],
                               float(self.masses.sum() * velocity_scale**2), np.finfo(float).tiny)
        if not np.isfinite([budget, energy_scale]).all() or energy_scale <= 0:
            raise ValueError("frozen energy budget and scale must be finite with positive scale")
        previous_displacement, displacement = self._displacement(previous_positions), self._displacement(positions)
        delta = displacement - previous_displacement
        elastic_change = float(self.linear_term @ delta
                               + 0.5 * (displacement + previous_displacement) @ (self.stiffness @ delta))
        kinetic_change = float(0.5 * np.sum(self.masses[..., None] *
                                          (velocities - previous_velocities) * (velocities + previous_velocities)))
        gravitational_change = float(-np.sum(self.masses[..., None] * (positions - previous_positions) * self.gravity))
        change = dict(elastic=elastic_change, kinetic=kinetic_change, gravitational=gravitational_change)
        change["total"] = float(sum(change.values()))
        allowed_work = budget - reference["total"]
        residual = (change["total"] - allowed_work) / energy_scale
        tolerance = absolute_tolerance + relative_tolerance
        return dict(components=components, budget=float(budget), energy_scale=float(energy_scale), energy_change=change,
                    allowed_external_work=float(allowed_work), normalized_violation=float(max(residual, 0.0)),
                    normalized_residual=float(residual), normalized_tolerance=float(tolerance),
                    energy_satisfied=bool(residual <= tolerance), law=FROZEN_ENERGY_LAW,
                    certificate_scope="current-step frozen quadratic work balance; full nonlinear energy is independent",
                    source=FROZEN_ENERGY_SOURCE, engine_calls=0)

    def gradient(self, positions, velocities):
        original_shape = np.shape(positions)
        positions, velocities = self._states(positions, velocities)
        nodal_gradient = (self.linear_term + self.stiffness @ self._displacement(positions)).reshape(-1, 3)
        position_gradient = (self.corner_weights[:, None] * nodal_gradient[self.groups]).reshape(positions.shape)
        position_gradient -= self.masses[..., None] * self.gravity
        velocity_gradient = self.masses[..., None] * velocities
        return position_gradient.reshape(original_shape), velocity_gradient.reshape(original_shape)

    def hessian_vector(self, positions, velocities, position_direction, velocity_direction):
        original_shape = np.shape(positions)
        positions, _ = self._states(positions, velocities)
        position_direction, velocity_direction = self._states(position_direction, velocity_direction)
        nodal_product = (self.stiffness @ self._nodal_average(position_direction).ravel()).reshape(-1, 3)
        position_product = (self.corner_weights[:, None] * nodal_product[self.groups]).reshape(positions.shape)
        velocity_product = self.masses[..., None] * velocity_direction
        return position_product.reshape(original_shape), velocity_product.reshape(original_shape)

    def stabilization_work(self, positions, stabilization):
        positions, stabilization = self._states(positions, stabilization)
        transported = positions - stabilization
        displacement = self._displacement(transported)
        nodal_shift = self._nodal_average(stabilization).ravel()
        elastic = float((self.linear_term + self.stiffness @ displacement) @ nodal_shift
                        + 0.5 * nodal_shift @ (self.stiffness @ nodal_shift))
        gravitational = float(-np.sum(self.masses[..., None] * stabilization * self.gravity))
        return dict(elastic=elastic, gravitational=gravitational, total=float(elastic + gravitational),
                    scope="numerical position stabilization work; not physical external work", engine_calls=0)

    def nonlinear_diagnostics(self, positions, velocities):
        if self.nonlinear_energy is None:
            return dict(status="unknown", reason="independent nonlinear energy model unavailable", engine_calls=0)
        positions, velocities = self._states(positions, velocities)
        frozen = self.components(positions, velocities)
        nonlinear = self.nonlinear_energy.components(positions, velocities)
        return dict(status="diagnostic_only", components=nonlinear, frozen_components=frozen,
                    elastic_linearization_defect=float(nonlinear["elastic"] - frozen["elastic"]),
                    total_energy_difference=float(nonlinear["total"] - frozen["total"]),
                    defect_credited_as_allowed_work=False, engine_calls=0)


@dataclass
class TransportEnergy:
    energy: FrozenQuadraticEnergy
    stabilization: np.ndarray
    supports_zero_energy_shortcut: bool = field(default=False, init=False)
    supports_nonnegative_energy_shortcut: bool = field(default=False, init=False)

    def __post_init__(self):
        if not isinstance(self.energy, FrozenQuadraticEnergy):
            raise TypeError("transport energy requires FrozenQuadraticEnergy")
        stabilization, _ = self.energy._states(self.stabilization, np.zeros_like(self.stabilization))
        self.stabilization = np.array(stabilization, copy=True)
        self.mesh = self.energy.mesh
        self.masses = self.energy.masses
        self.gravity = self.energy.gravity
        self.density = self.energy.density
        self.youngs_modulus = self.energy.youngs_modulus
        self.poissons_ratio = self.energy.poissons_ratio
        self.metadata = {**self.energy.metadata,
                         "law": "frozen quadratic transport energy; position stabilization excluded from mechanical transport",
                         "position_map": "transport position = endpoint position - fixed stabilization",
                         "stabilization_work_role": "separate numerical diagnostic; no allowed-work credit"}

    def _transport(self, positions, velocities):
        positions, velocities = self.energy._states(positions, velocities)
        return positions - self.stabilization, velocities

    def initial_components(self, positions, velocities):
        return self.energy.components(positions, velocities)

    def components(self, positions, velocities):
        return self.energy.components(*self._transport(positions, velocities))

    def value(self, positions, velocities):
        return self.components(positions, velocities)["total"]

    def balance(self, previous_positions, previous_velocities, positions, velocities, velocity_scale, **options):
        transported, velocities = self._transport(positions, velocities)
        reference = self.initial_components(previous_positions, previous_velocities)
        report = self.energy.balance(previous_positions, previous_velocities, transported, velocities, velocity_scale,
                                     **{**options, "reference_components": reference})
        report.update(law=self.metadata["law"],
                      certificate_scope="frozen quadratic energy of transport before numerical position stabilization",
                      position_stabilization_excluded=True)
        return report

    def gradient(self, positions, velocities):
        shape = np.shape(positions)
        position_gradient, velocity_gradient = self.energy.gradient(*self._transport(positions, velocities))
        return position_gradient.reshape(shape), velocity_gradient.reshape(shape)

    def hessian_vector(self, positions, velocities, position_direction, velocity_direction):
        shape = np.shape(positions)
        transported, velocities = self._transport(positions, velocities)
        position_direction, velocity_direction = self.energy._states(position_direction, velocity_direction)
        position_product, velocity_product = self.energy.hessian_vector(transported, velocities,
                                                                       position_direction, velocity_direction)
        return position_product.reshape(shape), velocity_product.reshape(shape)

    def stabilization_diagnostics(self, positions, velocities):
        positions, velocities = self.energy._states(positions, velocities)
        return dict(transport_components=self.components(positions, velocities),
                    endpoint_components=self.energy.components(positions, velocities),
                    numerical_stabilization_work=self.energy.stabilization_work(positions, self.stabilization),
                    nonlinear_endpoint=self.energy.nonlinear_diagnostics(positions, velocities),
                    work_credited_as_allowed_work=False, engine_calls=0)
