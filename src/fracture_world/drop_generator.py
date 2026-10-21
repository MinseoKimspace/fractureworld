from contextlib import chdir
from itertools import product
from importlib.metadata import version
import json
from pathlib import Path
import re
import tarfile
from tempfile import TemporaryDirectory
import urllib.request

import numpy as np
from scipy.spatial.transform import Rotation

from .femfx import SCHEMA as FEMFX_SCHEMA, reference_surface, validate_specification
from .io import file_hash, new_directory, write_json
from .mesh import FEMFX_MAX_TETRAHEDRON_ASPECT_RATIO, VolumeMesh, rigid_pose, tetrahedron_aspect_ratios


SCHEMA = "fracture-world.drop-suite.v1"
BUNNY_URL = "https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"


def _tetrahedralize(surface, settings, output):
    import tetgen

    target = settings.get("target_tetrahedra")
    if target is not None and (isinstance(target, bool) or not isinstance(target, int) or target < 1):
        raise ValueError("target_tetrahedra must be a positive integer")
    limit = float(settings.get("maximum_tetrahedron_aspect_ratio", FEMFX_MAX_TETRAHEDRON_ASPECT_RATIO))
    if not np.isfinite(limit) or limit <= np.sqrt(1.5) or limit > FEMFX_MAX_TETRAHEDRON_ASPECT_RATIO:
        raise ValueError("maximum_tetrahedron_aspect_ratio must be above the regular-tetrahedron ratio and at most 25")
    acceptance = float(limit * (1 - 256 * float(np.finfo(np.float32).eps)))
    ratio = float(settings["radius_edge_ratio"])
    angle = float(settings["minimum_dihedral_degrees"])
    if not np.isfinite([ratio, angle]).all() or ratio < 1 or not 0 <= angle <= 30:
        raise ValueError("mesher quality settings require radius_edge_ratio >= 1 and dihedral angle in [0, 30]")
    maximum_volume = float(surface.volume) / target if target is not None else None
    history, previous_violation = [], np.inf
    while True:
        options = dict(order=1, minratio=ratio, mindihedral=angle, opt_max_asp_ratio=acceptance,
                       steinerleft=-1)
        if maximum_volume is not None:
            options.update(fixedvolume=True, maxvolume=maximum_volume)
        mesher = tetgen.TetGen(surface)
        with TemporaryDirectory(prefix="tetgen-", dir=output) as scratch, chdir(scratch):
            result = mesher.tetrahedralize(**options)
        mesh = VolumeMesh(result[0], result[1])
        native_ratios = tetrahedron_aspect_ratios(mesh.nodes.astype(np.float32), mesh.tets)
        ratios = np.maximum(mesh.tetrahedron_aspect_ratios, native_ratios)
        if not np.isfinite(ratios).all():
            raise ValueError("tetrahedralizer produced elements that degenerate in native float precision")
        maximum_ratio = float(ratios.max())
        violation = max(0.0, maximum_ratio / acceptance - 1)
        attempt = dict(tetrahedra=len(mesh.tets), maximum_aspect_ratio=maximum_ratio,
                       maximum_double_precision_aspect_ratio=float(mesh.tetrahedron_aspect_ratios.max()),
                       maximum_native_coordinate_aspect_ratio=float(native_ratios.max()),
                       failing_tetrahedra=int(np.sum(ratios > acceptance)), options=options)
        history.append(attempt)
        quality = dict(metric="maximum_edge_length / minimum_vertex_to_opposite_face_height",
                       femfx_limit=FEMFX_MAX_TETRAHEDRON_ASPECT_RATIO, requested_limit=limit,
                       acceptance_limit=acceptance, passed=violation == 0, attempts=history)
        if violation == 0:
            return mesh, quality
        progress_tolerance = 256 * np.finfo(np.float32).eps * max(1.0, previous_violation)
        if np.isfinite(previous_violation) and violation >= previous_violation - progress_tolerance:
            write_json(output / "mesh-quality-failure.json", quality)
            raise ValueError(f"FEMFX mesh quality refinement stagnated: maximum aspect ratio {maximum_ratio:.8g} exceeds {acceptance:.8g}")
        previous_violation = violation
        ratio = 1 + (ratio - 1) / 2
        angle = min(30.0, angle + 5.0)
        maximum_volume = float(mesh.volumes.max()) / 2


def _surface(shape, output, surface_path=None):
    import pyvista as pv

    if surface_path is not None:
        path = Path(surface_path).resolve(strict=True)
        return pv.read(path).extract_surface().triangulate(), {
            "kind": "provided_surface", "filename": path.name, "sha256": file_hash(path),
            "shape_family": "asset-" + file_hash(path)}
    if shape == "bunny":
        archive = output / "bunny.tar.gz"
        with urllib.request.urlopen(BUNNY_URL, timeout=60) as response, archive.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        with tarfile.open(archive, "r:gz") as bundle:
            member = bundle.getmember("bunny/reconstruction/bun_zipper.ply")
            stream = bundle.extractfile(member)
            if stream is None or not member.isfile():
                raise ValueError("official Bunny archive has no reconstruction surface")
            path = output / "source.ply"
            with stream, path.open("wb") as target:
                while chunk := stream.read(1024 * 1024):
                    target.write(chunk)
        return pv.read(path).triangulate(), {
            "kind": "stanford_bunny", "url": BUNNY_URL, "archive_sha256": file_hash(archive),
            "surface_sha256": file_hash(path), "shape_family": "stanford_bunny"}
    constructors = {
        "box": lambda: pv.Cube().triangulate(),
        "sphere": lambda: pv.Sphere(theta_resolution=20, phi_resolution=20),
        "cylinder": lambda: pv.Cylinder(direction=(0, 1, 0), resolution=24).triangulate(),
        "torus": lambda: pv.ParametricTorus(u_res=24, v_res=16).triangulate(),
    }
    if shape not in constructors:
        raise ValueError("shape must select bunny, box, sphere, cylinder, torus or an explicit surface file")
    return constructors[shape](), {"kind": "pyvista_primitive", "name": shape, "shape_family": "primitive_" + shape}


def prepare_volume(shape, output, settings, surface_path=None):
    import pymeshfix
    import pyvista as pv

    if settings.get("geometry_mode") != "solid":
        raise ValueError("the first drop study requires an explicitly selected solid volume")
    height = settings.get("height_m")
    faces_target = settings.get("surface_triangles")
    if not isinstance(height, (int, float)) or isinstance(height, bool) or not np.isfinite(height) or height <= 0:
        raise ValueError("height_m must be finite and positive")
    if not isinstance(faces_target, int) or isinstance(faces_target, bool) or faces_target < 20:
        raise ValueError("surface_triangles must be an integer of at least twenty")
    surface, source = _surface(shape, output, surface_path)
    surface = surface.clean().triangulate()
    original_points = np.asarray(surface.points).copy()
    original_faces = surface.n_cells
    vertices, faces = pymeshfix.clean_from_arrays(np.asarray(surface.points), surface.faces.reshape(-1, 4)[:, 1:])
    surface = pv.PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces)).ravel())
    if surface.n_cells > faces_target:
        if settings.get("surface_remeshing") == "uniform":
            import pyacvd

            clustering = pyacvd.Clustering(surface)
            clustering.cluster(max(12, faces_target // 2 + 2))
            surface = clustering.create_mesh()
        else:
            surface = surface.decimate_pro(1 - faces_target / surface.n_cells, preserve_topology=True,
                                           boundary_vertex_deletion=False)
    surface = surface.clean().triangulate()
    vertices, faces = pymeshfix.clean_from_arrays(np.asarray(surface.points), surface.faces.reshape(-1, 4)[:, 1:])
    surface = pv.PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces)).ravel()).clean().triangulate()
    if surface.n_open_edges or not surface.is_manifold:
        raise ValueError("repaired surface must be watertight and manifold")
    points = np.asarray(surface.points, dtype=float).copy()
    extent = np.ptp(points, axis=0)
    if extent[1] <= 0 or not np.isfinite(points).all():
        raise ValueError("surface must have finite extent along the Y-up direction")
    scale = height / extent[1]
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    points = (points - center) * scale
    surface.points = points
    reference_points = pv.PolyData((original_points - center) * scale)
    distances = np.asarray(reference_points.compute_implicit_distance(surface)["implicit_distance"])
    mesh, quality = _tetrahedralize(surface, settings, output)
    if len(np.unique(mesh.tets)) != len(mesh.nodes):
        raise ValueError("tetrahedralizer returned unreferenced volume vertices")
    mesh.save(output / "mesh.npz")
    surface.save(output / "surface.ply")
    report = dict(source=source, geometry_mode="solid", height_m=height, scale=scale,
                  source_surface_vertices=len(original_points), source_surface_triangles=original_faces,
                  processed_surface_triangles=surface.n_cells, volume_vertices=len(mesh.nodes),
                  tetrahedra=len(mesh.tets), volume_m3=float(mesh.volumes.sum()),
                  minimum_tetrahedron_volume_m3=float(mesh.volumes.min()),
                  native_mesh_quality=quality,
                  surface_repair_applied=True, surface_simplification_target=faces_target,
                  surface_vertex_fit_rms_m=float(np.sqrt(np.mean(distances**2))),
                  surface_vertex_fit_maximum_m=float(np.max(np.abs(distances))),
                  libraries={name: version(name) for name in ("pyvista", "pymeshfix", "tetgen")},
                  mesh_sha256=file_hash(output / "mesh.npz"), settings=settings)
    if settings.get("surface_remeshing") == "uniform":
        report["libraries"]["pyacvd"] = version("pyacvd")
    write_json(output / "geometry.json", report)
    return mesh, report


def ballistic_drop_state(mesh, height, rotation, gravity, preimpact_time, floor):
    impact_time = np.sqrt(2 * height / gravity)
    elapsed = max(0.0, float(impact_time - preimpact_time))
    remaining = height - 0.5 * gravity * elapsed**2
    center = np.average(mesh.nodes[mesh.tets].mean(axis=1), axis=0, weights=mesh.volumes)
    rotated = rotation.apply(mesh.nodes - center)
    translation = np.array([0.0, floor + remaining - rotated[:, 1].min(), 0.0]) - rotation.apply(center)
    return dict(rotation_xyzw=rotation.as_quat().tolist(), translation=translation.tolist()), [0.0, -gravity * elapsed, 0.0], dict(
        original_drop_height_m=float(height), elapsed_ballistic_time_s=elapsed,
        remaining_clearance_m=float(remaining), nominal_time_until_first_floor_contact_s=float(impact_time - elapsed),
        free_flight_assumptions="uniform gravity, no air drag, zero angular velocity, no contact before the recorded window")


def create_drop_suite(output_dir, specification_path, shape="bunny", surface_path=None):
    settings = json.loads(Path(specification_path).read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or settings.get("schema") != SCHEMA:
        raise ValueError("unsupported drop suite schema")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", shape):
        raise ValueError("shape identifiers must be safe case-name prefixes")
    heights = np.asarray(settings["drop_heights_m"], dtype=float)
    orientations, steps, dt = settings["orientations"], settings["steps"], settings["dt"]
    gravity, before, floor = (float(settings[key]) for key in ("gravity_m_per_s2", "preimpact_time_s", "floor_height_m"))
    if heights.ndim != 1 or not len(heights) or not np.isfinite(heights).all() or (heights <= 0).any() or len(np.unique(heights)) != len(heights):
        raise ValueError("drop heights must be distinct finite positive values")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (orientations, steps)):
        raise ValueError("orientations and steps must be positive integers")
    if not np.isfinite([gravity, before, floor, dt]).all() or min(gravity, before, dt) <= 0 or steps * dt <= before:
        raise ValueError("the recording window must extend beyond the positive preimpact interval")
    output = new_directory(output_dir)
    mesh, geometry = prepare_volume(shape, output, settings["geometry"], surface_path)
    random_rotations = Rotation.random(max(orientations - 1, 0), random_state=np.random.default_rng(settings["seed"])) if orientations > 1 else None
    rotations = [Rotation.identity(), *([] if random_rotations is None else list(random_rotations))]
    cases, parameters = [], []
    for index, (height, rotation) in enumerate(product(heights, rotations), start=1):
        pose, velocity, approach = ballistic_drop_state(mesh, height, rotation, gravity, before, floor)
        case_id = f"{shape}-{index:04d}"
        cases.append(dict(id=case_id, vertex_impulses=[], initial_pose=pose,
                          initial_linear_velocity=velocity, ballistic_approach=approach))
        parameters.append(dict(id=case_id, **approach, initial_pose=pose, initial_linear_velocity=velocity))
    options = {"rayleigh_mass_damping": 0.0, "rayleigh_stiffness_damping": 0.0,
               **settings.get("scene_options", {}), "collision_bounds": [-1e6, 1e6, floor, 1e6, -1e6, 1e6]}
    specification = dict(schema=FEMFX_SCHEMA, parameter_status="uncalibrated_trial", material=settings["material"],
                         loading="gravity_driven_drop", shape_id=shape, shape_family=geometry["source"]["shape_family"],
                         geometry_source=geometry["source"], floor_height_m=floor, dt=dt, steps=steps, record_stride=1,
                         gravity=[0.0, -gravity, 0.0], fixed_vertices=[], threads=settings.get("threads", 1),
                         scene_options=options, cases=cases,
                         purpose="solid free-fall impact-window fracture teacher; no fixed supports")
    validate_specification(mesh, specification)
    write_json(output / "cases.json", specification)
    manifest = dict(schema=SCHEMA, configuration=settings, configuration_sha256=file_hash(specification_path),
                    shape_id=shape, shape_family=specification["shape_family"], geometry=geometry,
                    case_count=len(cases), cases=parameters, mesh="mesh.npz", specification="cases.json",
                    recording_scope="ballistically initialized preimpact window and native postimpact evolution",
                    simulation_executed=False, parameter_status="uncalibrated_trial")
    write_json(output / "manifest.json", manifest)
    first_geometry = reference_surface(mesh, rigid_pose(mesh.nodes, cases[0]["initial_pose"])[mesh.tets],
                                       np.zeros(len(mesh.internal_faces), dtype=bool))
    write_json(output / "preview.json", dict(title=f"{shape} initial drop geometry", source="prepared_solid_geometry",
               metadata=dict(simulation_executed=False, artifact_role="initial_geometry_preview", up_axis="y", floor_height_m=floor),
               bodies=[dict(id=0, **first_geometry)],
               frames=[dict(time=0.0, poses=[dict(position=[0, 0, 0], orientation=[0, 0, 0, 1])])]))
    return dict(mesh=str((output / "mesh.npz").resolve()), specification=str((output / "cases.json").resolve()),
                manifest=str((output / "manifest.json").resolve()), case_count=len(cases),
                tetrahedra=len(mesh.tets), simulation_executed=False, length_scale=1.0)
