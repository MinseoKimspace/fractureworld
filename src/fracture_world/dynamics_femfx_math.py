from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .mesh import VolumeMesh
from .partition import component_labels


FEMFX_REVISION = "95cf656731a2465ae1d400138170af2d6575b5bb"
FEMFX_SOURCE = f"https://github.com/GPUOpen-Effects/FEMFX/tree/{FEMFX_REVISION}/amd_femfx"
FEMFX_MATH_LICENSE = 'Copyright (c) 2019 Advanced Micro Devices, Inc.\n \nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n \nThe above copyright notice and this permission notice shall be included in\nall copies or substantial portions of the Software.\n \nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\nTHE SOFTWARE.'


def _multiply(left, right):
    product = left[..., :, :1] * right[..., :1, :]
    for index in range(1, left.shape[-1]):
        product = product + left[..., :, index:index + 1] * right[..., index:index + 1, :]
    return product


def fma32(left, right, offset):
    left, right, offset = np.broadcast_arrays(np.asarray(left, dtype=np.float32),
                                             np.asarray(right, dtype=np.float32),
                                             np.asarray(offset, dtype=np.float32))
    product = left.astype(np.float64) * right.astype(np.float64)
    offset = offset.astype(np.float64)
    total = product + offset
    recovered_offset = total - product
    error = (product - (total - recovered_offset)) + (offset - recovered_offset)
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = total.astype(np.float32)
        rounded64 = rounded.astype(np.float64)
        lower = np.where(rounded64 > total, np.nextafter(rounded, np.float32(-np.inf)), rounded)
        upper = np.where(rounded64 < total, np.nextafter(rounded, np.float32(np.inf)), rounded)
        midpoint = (lower.astype(np.float64) + upper.astype(np.float64)) * .5
        tie = np.isfinite(midpoint) & (lower != upper) & (total == midpoint) & (error != 0)
        return np.where(tie, np.where(error > 0, upper, lower), rounded)


def _matrix_vector(matrix, vector):
    product = matrix[..., :, 0] * vector[..., 0, None]
    for index in range(1, matrix.shape[-1]):
        product = fma32(matrix[..., :, index], vector[..., index, None], product)
    return product


def shape_parameters(rest):
    rest = np.asarray(rest, dtype=np.float32)
    if rest.ndim != 3 or rest.shape[1:] != (4, 3) or not np.isfinite(rest).all():
        raise ValueError("rest tetrahedra require finite arrays of shape (T, 4, 3)")
    x0 = rest[:, 0, 0]
    y0 = rest[:, 0, 1]
    z0 = rest[:, 0, 2]
    x1 = rest[:, 1, 0]
    y1 = rest[:, 1, 1]
    z1 = rest[:, 1, 2]
    x2 = rest[:, 2, 0]
    y2 = rest[:, 2, 1]
    z2 = rest[:, 2, 2]
    x3 = rest[:, 3, 0]
    y3 = rest[:, 3, 1]
    z3 = rest[:, 3, 2]
    x01 = x0 - x1
    x02 = x0 - x2
    x03 = x0 - x3
    x10 = -x01
    x12 = x1 - x2
    x13 = x1 - x3
    x20 = -x02
    x21 = -x12
    x23 = x2 - x3
    x30 = -x03
    x31 = -x13
    x32 = -x23
    y01 = y0 - y1
    y02 = y0 - y2
    y03 = y0 - y3
    y10 = -y01
    y12 = y1 - y2
    y13 = y1 - y3
    y20 = -y02
    y21 = -y12
    y23 = y2 - y3
    y30 = -y03
    y31 = -y13
    y32 = -y23
    det = (x10*y2 + x02*y1 + x21*y0)*z3 + (x01*y3 + x30*y1 + x13*y0)*z2 + (x20*y3 + x03*y2 + x32*y0)*z1 + (x12*y3 + x31*y2 + x23*y1)*z0
    m00 = (x1*y2 - x2*y1)*z3 + (x3*y1 - x1*y3)*z2 + (x2*y3 - x3*y2)*z1
    m01 = y12*z3 + y31*z2 + y23*z1
    m02 = x21*z3 + x13*z2 + x32*z1
    m03 = x12*y3 + x31*y2 + x23*y1
    m10 = (x2*y0 - x0*y2)*z3 + (x0*y3 - x3*y0)*z2 + (x3*y2 - x2*y3)*z0
    m11 = y20*z3 + y03*z2 + y32*z0
    m12 = x02*z3 + x30*z2 + x23*z0
    m13 = x20*y3 + x03*y2 + x32*y0
    m20 = (x0*y1 - x1*y0)*z3 + (x3*y0 - x0*y3)*z1 + (x1*y3 - x3*y1)*z0
    m21 = y01*z3 + y30*z1 + y13*z0
    m22 = x10*z3 + x03*z1 + x31*z0
    m23 = x01*y3 + x30*y1 + x13*y0
    m30 = (x1*y0 - x0*y1)*z2 + (x0*y2 - x2*y0)*z1 + (x2*y1 - x1*y2)*z0
    m31 = y10*z2 + y02*z1 + y21*z0
    m32 = x01*z2 + x20*z1 + x12*z0
    m33 = x10*y2 + x02*y1 + x21*y0
    invDet = np.float32(1.0) / det
    if (det <= 0).any() or not np.isfinite(invDet).all():
        raise ValueError("float32 rest tetrahedra must preserve positive orientation")
    barycentric = np.stack((np.stack((m01, m02, m03, m00), axis=-1),
                            np.stack((m11, m12, m13, m10), axis=-1),
                            np.stack((m21, m22, m23, m20), axis=-1),
                            np.stack((m31, m32, m33, m30), axis=-1)), axis=-2) * invDet[:, None, None]
    return barycentric, det * np.float32(1. / 6.)


def _inverse4(matrix):
    mA = matrix[:, 0, 0]
    mB = matrix[:, 1, 0]
    mC = matrix[:, 2, 0]
    mD = matrix[:, 3, 0]
    mE = matrix[:, 0, 1]
    mF = matrix[:, 1, 1]
    mG = matrix[:, 2, 1]
    mH = matrix[:, 3, 1]
    mI = matrix[:, 0, 2]
    mJ = matrix[:, 1, 2]
    mK = matrix[:, 2, 2]
    mL = matrix[:, 3, 2]
    mM = matrix[:, 0, 3]
    mN = matrix[:, 1, 3]
    mO = matrix[:, 2, 3]
    mP = matrix[:, 3, 3]
    tmp0 = ((mK * mD) - (mC * mL))
    tmp1 = ((mO * mH) - (mG * mP))
    tmp2 = ((mB * mK) - (mJ * mC))
    tmp3 = ((mF * mO) - (mN * mG))
    tmp4 = ((mJ * mD) - (mB * mL))
    tmp5 = ((mN * mH) - (mF * mP))
    res0X = (((mJ * tmp1) - (mL * tmp3)) - (mK * tmp5))
    res0Y = (((mN * tmp0) - (mP * tmp2)) - (mO * tmp4))
    res0Z = (((mD * tmp3) + (mC * tmp5)) - (mB * tmp1))
    res0W = (((mH * tmp2) + (mG * tmp4)) - (mF * tmp0))
    detInv = (np.float32(1.0) / ((((mA * res0X) + (mE * res0Y)) + (mI * res0Z)) + (mM * res0W)))
    res1X = (mI * tmp1)
    res1Y = (mM * tmp0)
    res1Z = (mA * tmp1)
    res1W = (mE * tmp0)
    res3X = (mI * tmp3)
    res3Y = (mM * tmp2)
    res3Z = (mA * tmp3)
    res3W = (mE * tmp2)
    res2X = (mI * tmp5)
    res2Y = (mM * tmp4)
    res2Z = (mA * tmp5)
    res2W = (mE * tmp4)
    tmp0 = ((mI * mB) - (mA * mJ))
    tmp1 = ((mM * mF) - (mE * mN))
    tmp2 = ((mI * mD) - (mA * mL))
    tmp3 = ((mM * mH) - (mE * mP))
    tmp4 = ((mI * mC) - (mA * mK))
    tmp5 = ((mM * mG) - (mE * mO))
    res2X = (((mL * tmp1) - (mJ * tmp3)) + res2X)
    res2Y = (((mP * tmp0) - (mN * tmp2)) + res2Y)
    res2Z = (((mB * tmp3) - (mD * tmp1)) - res2Z)
    res2W = (((mF * tmp2) - (mH * tmp0)) - res2W)
    res3X = (((mJ * tmp5) - (mK * tmp1)) + res3X)
    res3Y = (((mN * tmp4) - (mO * tmp0)) + res3Y)
    res3Z = (((mC * tmp1) - (mB * tmp5)) - res3Z)
    res3W = (((mG * tmp0) - (mF * tmp4)) - res3W)
    res1X = (((mK * tmp3) - (mL * tmp5)) - res1X)
    res1Y = (((mO * tmp2) - (mP * tmp4)) - res1Y)
    res1Z = (((mD * tmp5) - (mC * tmp3)) + res1Z)
    res1W = (((mH * tmp4) - (mG * tmp2)) + res1W)
    return np.stack(tuple(np.stack(column, axis=-1) for column in (
        (res0X, res0Y, res0Z, res0W), (res1X, res1Y, res1Z, res1W),
        (res2X, res2Y, res2Z, res2W), (res3X, res3Y, res3Z, res3W))), axis=-1) * detInv[:, None, None]


def _givens(first, offdiagonal, second):
    difference = first - second
    difference2 = difference * difference
    offdiagonal2 = offdiagonal * offdiagonal
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = np.float32(1.) / np.sqrt(offdiagonal2 + difference2)
    selected = offdiagonal2 < difference2
    with np.errstate(invalid="ignore"):
        sine = np.where(selected, omega * offdiagonal, np.float32(.7071067811865475244))
        cosine = np.where(selected, omega * difference, np.float32(.7071067811865475244))
    return sine, cosine


def _eigen_symmetric(symmetric):
    s00, s01, s02, s11, s12, s22 = (symmetric[:, row, column].copy() for row, column in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)))
    vectors = np.broadcast_to(np.eye(3, dtype=np.float32), symmetric.shape).copy()
    for _ in range(4):
        sine, cosine = _givens(s00, s01, s11)
        block = np.broadcast_to(np.eye(3, dtype=np.float32), symmetric.shape).copy()
        block[:, 0, 0], block[:, 0, 1], block[:, 1, 0], block[:, 1, 1] = cosine, -sine, sine, cosine
        vectors = _multiply(vectors, block)
        s00, s01, s02, s11, s12 = (sine*sine*s11 + sine*cosine*s01 + cosine*sine*s01 + cosine*cosine*s00,
                                  sine*cosine*s11 - sine*sine*s01 + cosine*cosine*s01 - cosine*sine*s00,
                                  sine*s12 + cosine*s02,
                                  cosine*cosine*s11 - cosine*sine*s01 - sine*cosine*s01 + sine*sine*s00,
                                  cosine*s12 - sine*s02)
        sine, cosine = _givens(s00, s02, s22)
        block = np.broadcast_to(np.eye(3, dtype=np.float32), symmetric.shape).copy()
        block[:, 0, 0], block[:, 0, 2], block[:, 2, 0], block[:, 2, 2] = cosine, -sine, sine, cosine
        vectors = _multiply(vectors, block)
        s00, s01, s02, s12, s22 = (sine*sine*s22 + sine*cosine*s02 + cosine*sine*s02 + cosine*cosine*s00,
                                  sine*s12 + cosine*s01,
                                  sine*cosine*s22 - sine*sine*s02 + cosine*cosine*s02 - cosine*sine*s00,
                                  cosine*s12 - sine*s01,
                                  cosine*cosine*s22 - cosine*sine*s02 - sine*cosine*s02 + sine*sine*s00)
        sine, cosine = _givens(s11, s12, s22)
        block = np.broadcast_to(np.eye(3, dtype=np.float32), symmetric.shape).copy()
        block[:, 1, 1], block[:, 1, 2], block[:, 2, 1], block[:, 2, 2] = cosine, -sine, sine, cosine
        vectors = _multiply(vectors, block)
        s01, s02, s11, s12, s22 = (sine*s02 + cosine*s01,
                                  cosine*s02 - sine*s01,
                                  sine*sine*s22 + sine*cosine*s12 + cosine*sine*s12 + cosine*cosine*s11,
                                  sine*cosine*s22 - sine*sine*s12 + cosine*cosine*s12 - cosine*sine*s11,
                                  cosine*cosine*s22 - cosine*sine*s12 - sine*cosine*s12 + sine*sine*s11)
    eigenvalues = np.stack((s00, s11, s22), axis=-1)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        selected = np.abs(eigenvalues[:, first]) < np.abs(eigenvalues[:, second])
        temporary = -vectors[:, :, first].copy()
        vectors[:, :, first] = np.where(selected[:, None], vectors[:, :, second], vectors[:, :, first])
        vectors[:, :, second] = np.where(selected[:, None], temporary, vectors[:, :, second])
        temporary = -eigenvalues[:, first].copy()
        eigenvalues[:, first] = np.where(selected, eigenvalues[:, second], eigenvalues[:, first])
        eigenvalues[:, second] = np.where(selected, temporary, eigenvalues[:, second])
    return eigenvalues, vectors


def _rotation(current, barycentric):
    transform = _multiply(np.concatenate((current.transpose(0, 2, 1), np.ones((len(current), 1, 4), dtype=np.float32)), axis=1), barycentric)
    matrix = transform[:, :3, :3]
    symmetric = _multiply(matrix.transpose(0, 2, 1), matrix)
    _, vectors = _eigen_symmetric(symmetric)
    triangular = _multiply(matrix, vectors)
    orthogonal = np.broadcast_to(np.eye(3, dtype=np.float32), matrix.shape).copy()
    for first, second in ((0, 1), (0, 2), (1, 2)):
        offdiagonal, diagonal = triangular[:, second, first], triangular[:, first, first]
        norm2 = offdiagonal * offdiagonal + diagonal * diagonal
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse = np.float32(1.) / np.sqrt(norm2)
        with np.errstate(invalid="ignore"):
            sine = np.where(norm2 < np.finfo(np.float32).eps, np.float32(0.), offdiagonal * inverse)
            cosine = np.where(norm2 < np.finfo(np.float32).eps, np.float32(1.), diagonal * inverse)
        block = np.broadcast_to(np.eye(3, dtype=np.float32), matrix.shape).copy()
        block[:, first, first], block[:, first, second], block[:, second, first], block[:, second, second] = cosine, -sine, sine, cosine
        orthogonal = _multiply(orthogonal, block)
        triangular = _multiply(block.transpose(0, 2, 1), triangular)
    return _multiply(orthogonal, vectors.transpose(0, 2, 1))


def _stiffness(barycentric, volumes, material):
    gradients = barycentric[:, :, :3]
    young, poisson = np.float32(material["youngs_modulus"]), np.float32(material["poissons_ratio"])
    scale = young / ((np.float32(1.) + poisson) * (np.float32(1.) - np.float32(2.) * poisson))
    v0, v1, v2 = scale * (np.float32(1.) - poisson), scale * poisson, scale * (np.float32(.5) - poisson)
    blocks = np.zeros((len(volumes), 4, 4, 3, 3), dtype=np.float32)
    for row in range(4):
        ai, bi, ci = gradients[:, row].T
        a0, a1, a2, b0, b1, b2, c0, c1, c2 = ai*v0, ai*v1, ai*v2, bi*v0, bi*v1, bi*v2, ci*v0, ci*v1, ci*v2
        for column in range(row + 1):
            aj, bj, cj = gradients[:, column].T
            jb1, jb2, jc1, jc2 = bj*v1, bj*v2, cj*v1, cj*v2
            block = np.stack((np.stack((cj*c2+bj*b2+aj*a0, aj*b2+ai*jb1, aj*c2+ai*jc1), axis=-1),
                              np.stack((ai*jb2+aj*b1, cj*c2+aj*a2+bj*b0, bj*c2+bi*jc1), axis=-1),
                              np.stack((ai*jc2+aj*c1, bi*jc2+bj*c1, bj*b2+aj*a2+cj*c0), axis=-1)), axis=-2)
            blocks[:, row, column] = volumes[:, None, None] * block
            blocks[:, column, row] = blocks[:, row, column].transpose(0, 2, 1)
    return blocks


@dataclass(frozen=True)
class NativeFrozenOperator:
    matrix: sparse.csr_matrix
    rhs: np.ndarray
    stiffness: sparse.csr_matrix
    elastic_gradient: np.ndarray
    corner_masses: np.ndarray
    node_masses: np.ndarray
    groups: np.ndarray
    representatives: np.ndarray
    rotations: np.ndarray
    volumes: np.ndarray
    barycentric: np.ndarray
    centers_of_mass: np.ndarray
    tet_centers_of_mass: np.ndarray
    ordering: dict


def _permutation(values, size, name):
    values = np.asarray(values)
    if values.shape != (size,) or not np.issubdtype(values.dtype, np.integer) or not np.array_equal(np.sort(values), np.arange(size)):
        raise ValueError(f"{name} must be a permutation of its current indices")
    return values.astype(np.int64, copy=True)


def _blocks_to_sparse(keys, values, size):
    rows = 3 * (keys // size)[:, None, None] + np.arange(3)[None, :, None]
    columns = 3 * (keys % size)[:, None, None] + np.arange(3)[None, None, :]
    matrix = sparse.coo_matrix((values.astype(float).ravel(),
                               (np.broadcast_to(rows, values.shape).ravel(), np.broadcast_to(columns, values.shape).ravel())),
                              shape=(3 * size, 3 * size)).tocsr()
    matrix.eliminate_zeros()
    return matrix


def native_corner_masses(mesh, density):
    density = np.float32(density)
    if not isinstance(mesh, VolumeMesh) or not np.isfinite(density) or density <= 0:
        raise ValueError("native masses require a volume mesh and positive finite density")
    _, volumes = shape_parameters(mesh.nodes.astype(np.float32)[mesh.tets])
    masses = np.float32(.25) * (density * volumes)
    if not np.isfinite(masses).all() or (masses <= 0).any():
        raise ValueError("float32 material-corner masses must be finite and positive")
    return np.repeat(masses.astype(float), 4)


def native_frozen_operator(mesh, material, positions, velocities, dt, gravity, broken, fixed, *,
                           vertex_order=None, tetrahedron_order=None, current_rotations=None,
                           current_centers_of_mass=None, current_nodal_masses=None):
    if not isinstance(mesh, VolumeMesh):
        raise TypeError("native frozen mechanics requires a VolumeMesh")
    shape = (len(mesh.tets), 4, 3)
    positions, velocities = np.asarray(positions), np.asarray(velocities)
    if positions.shape not in (shape, (4 * len(mesh.tets), 3)) or velocities.shape != positions.shape:
        raise ValueError("native frozen states require equal material-corner position and velocity arrays")
    with np.errstate(over="ignore", invalid="ignore"):
        positions = positions.reshape(shape).astype(np.float32)
        velocities = velocities.reshape(shape).astype(np.float32)
        gravity = np.asarray(gravity, dtype=np.float32)
        timestep = np.float32(dt)
        density, young, poisson = (np.float32(material[key]) for key in ("rest_density", "youngs_modulus", "poissons_ratio"))
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all() or gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("native frozen states and gravity must remain finite in float32")
    if not np.isfinite([timestep, density, young, poisson]).all() or min(timestep, density, young) <= 0 or not -1 < poisson < .5:
        raise ValueError("native frozen timestep and material are outside their finite physical ranges")
    broken, fixed = np.asarray(broken), np.asarray(fixed)
    if broken.shape != (len(mesh.internal_faces),) or broken.dtype != np.dtype(bool):
        raise ValueError("native frozen fracture mask must be Boolean over internal faces")
    if fixed.shape not in ((4 * len(mesh.tets),), (len(mesh.tets), 4)) or fixed.dtype != np.dtype(bool):
        raise ValueError("native frozen supports must be Boolean over material corners")
    barycentric, volumes = shape_parameters(mesh.nodes.astype(np.float32)[mesh.tets])
    groups = mesh.corner_groups(broken)
    _, representatives = np.unique(groups, return_index=True)
    count = len(representatives)
    cells = component_labels(len(mesh.tets), mesh.dual_edges, broken)
    node_cells = cells[representatives // 4]
    node_positions = positions.reshape(-1, 3)[representatives]
    node_velocities = velocities.reshape(-1, 3)[representatives]
    tet_order = np.arange(len(mesh.tets)) if tetrahedron_order is None else _permutation(tetrahedron_order, len(mesh.tets), "tetrahedron order")
    if vertex_order is None:
        node_order = np.argsort(mesh.tets.ravel()[representatives], kind="stable")
    else:
        node_order = _permutation(vertex_order, count, "vertex order")
    corner_mass = np.float32(.25) * (density * volumes)
    node_masses = np.zeros(count, dtype=np.float32)
    nodes = groups.reshape(-1, 4)
    np.add.at(node_masses, nodes[tet_order].ravel(), np.repeat(corner_mass[tet_order], 4))
    if current_nodal_masses is not None:
        supplied_mass = np.asarray(current_nodal_masses, dtype=np.float32)
        if supplied_mass.shape in ((len(mesh.tets), 4), (4 * len(mesh.tets),)):
            supplied_mass = supplied_mass.ravel()
            if not np.array_equal(supplied_mass, supplied_mass[representatives][groups]):
                raise ValueError("current nodal masses disagree within a shared material node")
            supplied_mass = supplied_mass[representatives]
        if supplied_mass.shape != (count,):
            raise ValueError("current nodal masses must index current nodes or repeated material corners")
        node_masses = supplied_mass.copy()
    if not np.isfinite(node_masses).all() or (node_masses <= 0).any():
        raise ValueError("native frozen nodal masses must be finite and positive")
    centers = np.zeros((int(cells.max()) + 1, 3), dtype=np.float32)
    total_mass = np.zeros(len(centers), dtype=np.float32)
    for node in node_order:
        component = node_cells[node]
        centers[component] += node_positions[node] * node_masses[node]
        total_mass[component] += node_masses[node]
    centers /= total_mass[:, None]
    tet_centers = centers[cells]
    if current_centers_of_mass is not None:
        tet_centers = np.asarray(current_centers_of_mass, dtype=np.float32)
        if tet_centers.shape != (len(mesh.tets), 3) or not np.isfinite(tet_centers).all():
            raise ValueError("current cached centers of mass must be finite original-tetrahedron three-vectors")
        tet_centers = tet_centers.copy()
        _, component_representatives = np.unique(cells, return_index=True)
        centers = tet_centers[component_representatives]
    current = positions - tet_centers[:, None, :]
    if current_rotations is None:
        rotations = _rotation(current, barycentric)
    else:
        rotations = np.asarray(current_rotations, dtype=np.float32)
        if rotations.shape != (len(mesh.tets), 3, 3) or not np.isfinite(rotations).all() or (np.linalg.det(rotations.astype(float)) <= 0).any():
            raise ValueError("current cached rotations must be finite positively oriented original-tetrahedron matrices")
        rotations = rotations.copy()
    recovered_rest = _inverse4(barycentric)[:, :3, :].transpose(0, 2, 1)
    recovered_rest -= recovered_rest[:, 3:4]
    rest_stiffness = _stiffness(barycentric, volumes, material)
    rotated = np.empty_like(rest_stiffness)
    offset = np.empty((len(mesh.tets), 4, 3), dtype=np.float32)
    for row in range(4):
        products = []
        for column in range(4):
            left = _multiply(rotations, rest_stiffness[:, row, column])
            products.append(_multiply(left, (-recovered_rest[:, column])[:, :, None])[:, :, 0])
            rotated[:, row, column] = _multiply(left, rotations.transpose(0, 2, 1))
        offset[:, row] = ((products[0] + products[1]) + products[2]) + products[3]
    for row in range(4):
        for column in range(row):
            rotated[:, column, row] = rotated[:, row, column].transpose(0, 2, 1)
    forces = np.empty_like(positions)
    for row in range(4):
        products = [_matrix_vector(rotated[:, row, column], current[:, column]) for column in range(4)]
        forces[:, row] = (((products[0] + products[1]) + products[2]) + products[3]) + offset[:, row]
    node_fixed = np.zeros(count, dtype=bool)
    node_fixed[np.unique(groups[fixed.ravel()])] = True
    ordered_nodes = nodes[tet_order]
    pair_keys = (ordered_nodes[:, :, None] * count + ordered_nodes[:, None, :]).ravel()
    diagonal_keys = np.arange(count) * (count + 1)
    keys, inverse = np.unique(np.r_[diagonal_keys, pair_keys], return_inverse=True)
    diagonal_indices, block_indices = inverse[:count], inverse[count:]
    stiffness_values = np.zeros((len(keys), 3, 3), dtype=np.float32)
    ordered_blocks = rotated[tet_order].reshape(-1, 3, 3)
    np.add.at(stiffness_values, block_indices, ordered_blocks)
    matrix_values = np.zeros_like(stiffness_values)
    diagonal_mass = np.where(node_fixed, np.float32(1.), node_masses)
    matrix_values[diagonal_indices] = diagonal_mass[:, None, None] * np.eye(3, dtype=np.float32)
    contributions = ordered_blocks * (timestep * timestep)
    contribution_nodes = np.repeat(ordered_nodes.ravel(), 4)
    contributions[node_fixed[contribution_nodes]] = 0
    np.add.at(matrix_values, block_indices, contributions)
    elastic_gradient = np.zeros((count, 3), dtype=np.float32)
    np.add.at(elastic_gradient, ordered_nodes.ravel(), forces[tet_order].reshape(-1, 3))
    rhs = node_velocities * node_masses[:, None] + (gravity * node_masses[:, None]) * timestep
    changes = forces[tet_order].reshape(-1, 3) * timestep
    changes[node_fixed[ordered_nodes.ravel()]] = 0
    np.subtract.at(rhs, ordered_nodes.ravel(), changes)
    if not all(np.isfinite(value).all() for value in (rhs, rotations, matrix_values, stiffness_values, elastic_gradient)):
        raise ValueError("native frozen float32 arithmetic overflowed or produced an invalid state")
    ordering = dict(tetrahedra="original cell order" if tetrahedron_order is None else "supplied current cell order",
                    vertices="stable original vertex order within current components" if vertex_order is None else "supplied current vertex order",
                    postfracture_native_order_verified=False,
                    cached_rotations_supplied=current_rotations is not None,
                    cached_centers_of_mass_supplied=current_centers_of_mass is not None,
                    current_nodal_masses_supplied=current_nodal_masses is not None,
                    arithmetic="float32 source operation ordering; explicit correctly rounded float32 fused multiply-add",
                    source=FEMFX_SOURCE, revision=FEMFX_REVISION,
                    engine_calls=0, future_observation_inputs=False)
    return NativeFrozenOperator(_blocks_to_sparse(keys, matrix_values, count), rhs.astype(float).ravel(),
                                _blocks_to_sparse(keys, stiffness_values, count), elastic_gradient.astype(float),
                                np.repeat(corner_mass.astype(float), 4), node_masses.astype(float),
                                groups, representatives, rotations.astype(float), volumes.astype(float),
                                barycentric.astype(float), centers.astype(float), tet_centers.astype(float), ordering)


def native_fracture_stress(mesh, material, next_positions, previous_broken, *, endpoint_rotations=None,
                           endpoint_centers_of_mass=None, current_nodal_masses=None, vertex_order=None,
                           tetrahedron_order=None):
    positions = np.asarray(next_positions)
    frame = native_frozen_operator(mesh, material, positions, np.zeros_like(positions), 1., [0., 0., 0.],
                                   previous_broken, np.zeros(4 * len(mesh.tets), dtype=bool),
                                   vertex_order=vertex_order, tetrahedron_order=tetrahedron_order,
                                   current_rotations=endpoint_rotations, current_centers_of_mass=endpoint_centers_of_mass,
                                   current_nodal_masses=current_nodal_masses)
    current = positions.reshape(-1, 4, 3).astype(np.float32) - frame.tet_centers_of_mass.astype(np.float32)[:, None, :]
    rotations = frame.rotations.astype(np.float32)
    rest = mesh.nodes.astype(np.float32)[mesh.tets]
    gradients = frame.barycentric.astype(np.float32)[:, :, :3]
    young, poisson = np.float32(material["youngs_modulus"]), np.float32(material["poissons_ratio"])
    threshold = np.float32(material["fracture_stress_threshold"])
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("native fracture stress requires a finite positive float32 stress threshold")
    scale = young / ((np.float32(1.) + poisson) * (np.float32(1.) - np.float32(2.) * poisson))
    v0, v1, v2 = scale * (np.float32(1.) - poisson), scale * poisson, scale * (np.float32(.5) - poisson)
    normal_parts, shear_parts = [], []
    for corner in range(4):
        a, b, c = gradients[:, corner].T
        a0, a1, a2 = a*v0, a*v1, a*v2
        b0, b1, b2 = b*v0, b*v1, b*v2
        c0, c1, c2 = c*v0, c*v1, c*v2
        zero = np.zeros_like(a)
        normal_matrix = np.stack((np.stack((a0, b1, c1), axis=-1), np.stack((a1, b0, c1), axis=-1),
                                  np.stack((a1, b1, c0), axis=-1)), axis=-2)
        shear_matrix = np.stack((np.stack((b2, a2, zero), axis=-1), np.stack((zero, c2, b2), axis=-1),
                                 np.stack((c2, zero, a2), axis=-1)), axis=-2)
        offset = _multiply(rotations.transpose(0, 2, 1), current[:, corner, :, None])[:, :, 0] - rest[:, corner]
        normal_parts.append(_multiply(normal_matrix, offset[:, :, None])[:, :, 0])
        shear_parts.append(_multiply(shear_matrix, offset[:, :, None])[:, :, 0])
    normal = ((normal_parts[0] + normal_parts[1]) + normal_parts[2]) + normal_parts[3]
    shear = ((shear_parts[0] + shear_parts[1]) + shear_parts[2]) + shear_parts[3]
    tensor = np.stack((np.stack((normal[:, 0], shear[:, 2], shear[:, 1]), axis=-1),
                       np.stack((shear[:, 2], normal[:, 1], shear[:, 0]), axis=-1),
                       np.stack((shear[:, 1], shear[:, 0], normal[:, 2]), axis=-1)), axis=-2)
    eigenvalues, _ = _eigen_symmetric(tensor)
    maximum = np.max(np.abs(eigenvalues), axis=-1)
    if not np.isfinite(maximum).all():
        raise ValueError("native float32 fracture stress produced a nonfinite eigenvalue")
    return dict(stress_normal=normal.astype(float), stress_shear=shear.astype(float), fracture_tensor=tensor.astype(float),
                maximum_stress_eigenvalue=maximum.astype(float), stress_ratios=maximum.astype(float) / float(threshold),
                rotations=frame.rotations, tet_centers_of_mass=frame.tet_centers_of_mass,
                phase="endpoint stress before the proposed new fracture using previous connectivity and masses",
                source=FEMFX_SOURCE, revision=FEMFX_REVISION, engine_calls=0,
                endpoint_cache_supplied=endpoint_rotations is not None and endpoint_centers_of_mass is not None,
                full_topology_certified=False)


