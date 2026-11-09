from time import perf_counter

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .integration2d import velocity_verlet
from .neural import build_graph
from .topology import lift_state, lumped_mass, mesh_edges, rectangle, split_edges


def component_indices(mesh):
    edges = mesh_edges(mesh)
    adjacency = coo_matrix(
        (np.ones(2 * len(edges)),
         (np.concatenate((edges[:, 0], edges[:, 1])),
          np.concatenate((edges[:, 1], edges[:, 0])))),
        shape=(len(mesh.rest), len(mesh.rest)),
    ).tocsr()
    count, labels = connected_components(adjacency, directed=False)
    return [np.flatnonzero(labels == index) for index in range(count)]


def component_force_balance(mesh, position, force):
    position = np.asarray(position, dtype=np.float64)
    force = np.asarray(force, dtype=np.float64)
    balances = []
    for nodes in component_indices(mesh):
        arm = position[nodes] - position[nodes].mean(axis=0)
        torque = arm[:, 0] * force[nodes, 1] - arm[:, 1] * force[nodes, 0]
        balances.append({
            "node_count": len(nodes),
            "net_force_norm": float(np.linalg.norm(force[nodes].sum(axis=0))),
            "net_torque_abs": float(abs(torque.sum())),
        })
    return {
        "component_count": len(balances),
        "max_net_force_norm": max(row["net_force_norm"] for row in balances),
        "max_net_torque_abs": max(row["net_torque_abs"] for row in balances),
        "components": balances,
    }


def teacher_connectivity_probe():
    from .teacher2d import InternalForce, Material

    mesh = rectangle(2, 2)
    cut = split_edges(mesh, [[1, 4], [4, 7]])
    material = Material()
    deformation = np.array([[1.03, 0.02], [0.0, 0.99]])
    position = mesh.rest @ deformation.T
    velocity = np.column_stack((0.07 + 0.02 * mesh.rest[:, 1], -0.03 * mesh.rest[:, 0]))
    cut_position, cut_velocity = lift_state(mesh, cut, position, velocity)
    force = InternalForce(mesh, material)(position)
    cut_force = InternalForce(cut, material)(cut_position)
    mass = lumped_mass(mesh, material.density, material.thickness)
    cut_mass = lumped_mass(cut, material.density, material.thickness)
    acceleration = force / mass[:, None]
    cut_acceleration = cut_force / cut_mass[:, None]
    corner_difference = cut_acceleration[cut.triangles] - acceleration[mesh.triangles]
    aggregate_force = np.zeros_like(force)
    np.add.at(aggregate_force, cut.original_vertex, cut_force)
    old_momentum = (mass[:, None] * velocity).sum(axis=0)
    new_momentum = (cut_mass[:, None] * cut_velocity).sum(axis=0)
    moved = cut_position.copy()
    groups = component_indices(cut)
    moved[groups[0]] += [0.4, -0.15]
    moved_force = InternalForce(cut, material)(moved)
    other = groups[1]
    return {
        "status": "completed",
        "teacher": "FElupe StVenantKirchhoff plane strain",
        "same_material_corner_position_max_error": float(np.max(np.abs(
            cut_position[cut.triangles] - position[mesh.triangles]))),
        "same_material_corner_velocity_max_error": float(np.max(np.abs(
            cut_velocity[cut.triangles] - velocity[mesh.triangles]))),
        "mass_transfer_abs_error": float(abs(cut_mass.sum() - mass.sum())),
        "momentum_transfer_error_norm": float(np.linalg.norm(new_momentum - old_momentum)),
        "aggregated_force_max_error": float(np.max(np.abs(aggregate_force - force))),
        "corner_acceleration_response_rms": float(np.sqrt(np.mean(corner_difference ** 2))),
        "unmoved_component_force_change_max": float(np.max(np.abs(moved_force[other] - cut_force[other]))),
        "uncut": component_force_balance(mesh, position, force),
        "cut": component_force_balance(cut, cut_position, cut_force),
        "scope": "Identical deformed material-corner states before and after a prescribed full cut; no learned fracture criterion",
    }


def integrate_teacher_force(mesh, material, time, amplitude, ramp_time, method):
    from .teacher2d import InternalForce, Pull

    time = np.asarray(time, dtype=np.float64)
    force = InternalForce(mesh, material)
    mass = torch.as_tensor(lumped_mass(mesh, material.density, material.thickness), dtype=torch.float64)
    mask, boundary_position, boundary_velocity = Pull(amplitude, ramp_time).boundary(mesh, time)
    boundary_mask = torch.as_tensor(mask)
    position = torch.as_tensor(mesh.rest.copy(), dtype=torch.float64)
    velocity = torch.zeros_like(position)
    positions, velocities = [position.numpy().copy()], [velocity.numpy().copy()]

    def evaluate(current):
        return torch.as_tensor(force(current.detach().numpy()), dtype=current.dtype)

    current_force = evaluate(position)
    for index, dt in enumerate(np.diff(time), start=1):
        target_position = torch.as_tensor(boundary_position[index])
        target_velocity = torch.as_tensor(boundary_velocity[index])
        if method == "velocity_verlet":
            result = velocity_verlet(
                evaluate, mass, position, velocity, float(dt),
                boundary_mask=boundary_mask,
                boundary_position_next=target_position,
                boundary_velocity_next=target_velocity,
                force_start=current_force,
            )
            position, velocity, current_force = result.position, result.velocity, result.end_force
        elif method == "legacy_constant_acceleration":
            next_velocity = velocity + float(dt) * current_force / mass[:, None]
            next_position = position + 0.5 * float(dt) * (velocity + next_velocity)
            position = torch.where(boundary_mask, target_position, next_position)
            velocity = torch.where(boundary_mask, target_velocity, next_velocity)
            current_force = evaluate(position)
        else:
            raise ValueError("Unknown teacher-force integration method")
        positions.append(position.numpy().copy())
        velocities.append(velocity.numpy().copy())
    return np.stack(positions), np.stack(velocities)


def teacher_integration_probe(nx=2, ny=2, column=1, depth=1):
    from .teacher2d import Material, simulate

    if not 1 <= column < nx or not 1 <= depth <= ny:
        raise ValueError("Integration probe requires an interior cut column and a valid cut depth")
    mesh = split_edges(rectangle(nx, ny), [
        [column + (nx + 1) * row, column + (nx + 1) * (row + 1)]
        for row in range(depth)
    ])
    material = Material()
    time = np.linspace(0, 0.2, 101)
    amplitude, ramp_time = 0.015, 0.05
    reference = simulate(mesh, material, time, amplitude, ramp_time,
                         rtol=1e-10, atol=1e-12, max_step=0.0005)
    refined_reference = simulate(mesh, material, time, amplitude, ramp_time,
                                 rtol=1e-11, atol=1e-13, max_step=0.00025)
    free = ~reference.arrays["boundary_mask"]

    def errors(position, velocity):
        return {
            "free_position_rmse": float(np.sqrt(np.mean(
                (position[1:, free] - reference.arrays["position"][1:, free]) ** 2))),
            "free_velocity_rmse": float(np.sqrt(np.mean(
                (velocity[1:, free] - reference.arrays["velocity"][1:, free]) ** 2))),
        }

    methods = {}
    for method in ("legacy_constant_acceleration", "velocity_verlet"):
        position, velocity = integrate_teacher_force(mesh, material, time, amplitude, ramp_time, method)
        methods[method] = errors(position, velocity)
    refined_time = np.linspace(0, 0.2, 201)
    position, velocity = integrate_teacher_force(
        mesh, material, refined_time, amplitude, ramp_time, "velocity_verlet")
    methods["velocity_verlet_half_dt"] = errors(position[::2], velocity[::2])
    return {
        "status": "completed",
        "mesh": {"nx": nx, "ny": ny, "cut_column": column, "cut_depth": depth},
        "dt": float(time[1] - time[0]),
        "steps": len(time) - 1,
        "teacher_refinement": {
            key + "_max_difference": float(np.max(np.abs(
                reference.arrays[key] - refined_reference.arrays[key])))
            for key in ("position", "velocity")
        },
        "methods": methods,
        "scope": "Identical instantaneous FElupe forces, mesh, initial states and prescribed grips; independent DOP853 reference; not neural accuracy",
    }


def oscillator_probe():
    omega, dt, steps = 100.0, 0.002, 100
    methods = {}
    for method in ("legacy_constant_acceleration", "velocity_verlet"):
        position = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        velocity = torch.zeros_like(position)
        mass = torch.ones(1, dtype=torch.float64)
        energy = []
        current_force = -omega ** 2 * position
        for _ in range(steps):
            if method == "velocity_verlet":
                result = velocity_verlet(
                    lambda value: -omega ** 2 * value, mass, position, velocity, dt,
                    force_start=current_force,
                )
                position, velocity, current_force = result.position, result.velocity, result.end_force
            else:
                next_velocity = velocity + dt * current_force
                position = position + 0.5 * dt * (velocity + next_velocity)
                velocity = next_velocity
                current_force = -omega ** 2 * position
            energy.append(float((velocity.square().sum() + omega ** 2 * position.square().sum()) / omega ** 2))
        methods[method] = {
            "final_energy_ratio": energy[-1],
            "min_energy_ratio": min(energy),
            "max_energy_ratio": max(energy),
        }
    return {
        "omega": omega, "dt": dt, "steps": steps, "methods": methods,
        "scope": "Analytic undamped oscillator with exact restoring force; a numerical integration audit",
    }


def neural_structure_probe(method="stress", device="cpu", hidden=16, layers=2):
    from .equilibrium import EquilibriumDynamicsModel

    mesh = split_edges(rectangle(2, 2), [[1, 4], [4, 7]])
    graph = build_graph(
        mesh, lumped_mass(mesh), np.zeros_like(mesh.rest, dtype=bool),
        material=(100.0, 0.25, 1.0), dtype=torch.float64, device=device,
    )
    devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(17)
        model = EquilibriumDynamicsModel(method, hidden_size=hidden, layers=layers).to(device).double().eval()
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=0.1)
    angle = 0.73
    rotation = torch.tensor([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]], dtype=torch.float64, device=device)
    deformation = torch.tensor([[1.03, 0.02], [0.0, 0.99]], dtype=torch.float64, device=device)
    with torch.inference_mode():
        position = graph.rest @ deformation.T
        force = model.instantaneous_force(graph, position)
        rotated = position @ rotation.T + position.new_tensor([0.31, -0.22])
        rotated_force = model.instantaneous_force(graph, rotated)
        rest_force = model.instantaneous_force(graph, graph.rest)
        rigid_force = model.instantaneous_force(graph, graph.rest @ rotation.T + 0.3)
        groups = component_indices(mesh)
        component_rotated = position.clone()
        component_rigid = graph.rest.clone()
        expected_component_force = force.clone()
        for index, nodes in enumerate(groups):
            component_angle = angle * (index + 1)
            component_rotation = position.new_tensor([
                [np.cos(component_angle), -np.sin(component_angle)],
                [np.sin(component_angle), np.cos(component_angle)],
            ])
            translation = position.new_tensor([0.2 * index, -0.17 * (index + 1)])
            component_rotated[nodes] = position[nodes] @ component_rotation.T + translation
            component_rigid[nodes] = graph.rest[nodes] @ component_rotation.T + translation
            expected_component_force[nodes] = force[nodes] @ component_rotation.T
        component_force = model.instantaneous_force(graph, component_rotated)
        component_rigid_force = model.instantaneous_force(graph, component_rigid)
        changed = position.clone()
        changed[groups[0], 0] += 0.05 * changed[groups[0], 1]
        changed_force = model.instantaneous_force(graph, changed)
        other = groups[1]
        uncut_mesh = rectangle(2, 2)
        uncut_graph = build_graph(
            uncut_mesh, lumped_mass(uncut_mesh), np.zeros_like(uncut_mesh.rest, dtype=bool),
            material=(100.0, 0.25, 1.0), dtype=torch.float64, device=device,
        )
        uncut_position = uncut_graph.rest @ deformation.T
        uncut_force = model.instantaneous_force(uncut_graph, uncut_position)
        acceleration = force / graph.mass[:, None]
        uncut_acceleration = uncut_force / uncut_graph.mass[:, None]
        response = acceleration[graph.triangles] - uncut_acceleration[uncut_graph.triangles]
    force_numpy = force.cpu().numpy()
    return {
        "method": method,
        "rest_force_max_abs": float(rest_force.abs().max()),
        "rigid_motion_force_max_abs": float(rigid_force.abs().max()),
        "rotation_covariance_max_error": float((rotated_force - force @ rotation.T).abs().max()),
        "independent_component_rotation_max_error": float((component_force - expected_component_force).abs().max()),
        "independent_component_rigid_force_max_abs": float(component_rigid_force.abs().max()),
        "deformed_force_max_abs": float(force.abs().max()),
        "modified_component_force_change_max": float((changed_force[groups[0]] - force[groups[0]]).abs().max()),
        "unmodified_component_force_change_max": float((changed_force[other] - force[other]).abs().max()),
        "same_state_cut_acceleration_response_rms": float(response.square().mean().sqrt()),
        "component_balance": component_force_balance(mesh, position.cpu().numpy(), force_numpy),
        "rotation_scope": "Superposed spatial rotations/translations with material reference coordinates held fixed; coordinate-frame covariance is not claimed",
        "scope": "Random nonzero model weights; structural invariants and disconnected-component isolation, not learned accuracy",
    }


def validate_equilibrium(device="cpu", hidden=16, layers=2, include_teacher=True):
    started = perf_counter()
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    neural = {method: neural_structure_probe(method, device, hidden, layers)
              for method in ("direct", "stress")}
    result = {
        "format": "cut-dynamics-equilibrium-validation-v1",
        "training_executed": False,
        "weights": "Seeded random nonzero weights; no checkpoint overwritten",
        "device": str(device),
        "neural_structure": neural,
        "oscillator": oscillator_probe(),
    }
    if include_teacher:
        result["teacher_connectivity"] = teacher_connectivity_probe()
        result["teacher_integration"] = teacher_integration_probe()
        result["teacher_integration_scaled"] = {
            f"8x8-column4-depth{depth}": teacher_integration_probe(nx=8, ny=8, column=4, depth=depth)
            for depth in (7, 8)
        }
    result["runtime_seconds"] = perf_counter() - started
    result["scope"] = "Structural and numerical tests only; trained rollout accuracy requires separate held-out evaluation"
    return result
