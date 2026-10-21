#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "AMD_FEMFX.h"
#include "SampleTaskSystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <malloc.h>

namespace py = pybind11;
using AMD::uint;

void* FmAlignedMalloc(size_t size, size_t alignment)
{
    return _aligned_malloc(size, alignment);
}

void FmAlignedFree(void* pointer)
{
    _aligned_free(pointer);
}

namespace
{
std::mutex scene_mutex;
bool scene_active = false;

template <typename T>
T option(const py::dict& options, const char* key, T fallback)
{
    return options.contains(key) ? py::cast<T>(options[key]) : fallback;
}

float finite_float(double value, const std::string& name)
{
    const float converted = static_cast<float>(value);
    if (!std::isfinite(value) || !std::isfinite(converted))
        throw std::invalid_argument(name + " must be finite in FEMFX float32 precision");
    return converted;
}

uint checked_count(uint64_t value, const std::string& name)
{
    if (value == 0 || value >= 0x80000000ULL)
        throw std::invalid_argument(name + " must be positive and below 2^31");
    return static_cast<uint>(value);
}

struct InternalFace
{
    std::array<uint, 3> vertices;
    std::array<uint, 2> tets;
    std::array<uint, 2> local_faces;
};

class Scene
{
    AMD::FmScene* scene_ = nullptr;
    AMD::FmTetMeshBuffer* buffer_ = nullptr;
    std::vector<InternalFace> internal_faces_;
    std::vector<double> fracture_times_;
    std::vector<double> vertex_masses_;
    py::dict settings_;
    uint tet_count_ = 0;
    uint vertex_count_ = 0;
    uint64_t step_ = 0;
    float dt_ = 0;
    bool owns_task_system_ = false;
    bool added_to_scene_ = false;
    bool valid_ = true;

    void close()
    {
        if (scene_ && buffer_ && added_to_scene_)
            AMD::FmRemoveTetMeshBufferFromScene(scene_, AMD::FmGetTetMeshBufferId(*buffer_));
        added_to_scene_ = false;
        if (buffer_)
            AMD::FmDestroyTetMeshBuffer(buffer_);
        buffer_ = nullptr;
        if (scene_)
            AMD::FmDestroyScene(scene_);
        scene_ = nullptr;
        if (owns_task_system_)
        {
            AMD::SampleDestroyTaskSystem();
            std::lock_guard<std::mutex> lock(scene_mutex);
            scene_active = false;
            owns_task_system_ = false;
        }
    }

    void require_open() const
    {
        if (!scene_ || !buffer_ || !valid_)
            throw std::runtime_error("The FEMFX scene is closed or invalid after a failed update");
    }

public:
    Scene(py::array_t<double, py::array::c_style | py::array::forcecast> nodes,
          py::array_t<int64_t, py::array::c_style> tets,
          const py::dict& material,
          py::array_t<double, py::array::c_style | py::array::forcecast> velocities,
          py::array_t<bool, py::array::c_style> kinematic,
          double dt,
          const std::array<double, 3>& gravity,
          int threads,
          const py::dict& options,
          py::object deformed_positions,
          double initial_time)
    {
        const std::set<std::string> material_keys = {
            "rest_density", "youngs_modulus", "poissons_ratio", "fracture_stress_threshold"
        };
        for (const auto& item : material)
            if (!material_keys.count(py::cast<std::string>(item.first)))
                throw std::invalid_argument("Unsupported constitutive material key: " + py::cast<std::string>(item.first));
        for (const auto& key : material_keys)
            if (!material.contains(py::str(key)))
                throw std::invalid_argument("Missing explicit material value: " + key);
        const std::set<std::string> option_keys = {
            "max_unconstrained_iterations", "friction_coefficient", "max_distance_contacts",
            "max_volume_contacts", "max_volume_contact_vertices", "max_broad_phase_pairs",
            "max_constraint_solver_bytes", "epsilon_cg", "rayleigh_mass_damping",
            "rayleigh_stiffness_damping", "dist_contact_bias", "dist_contact_threshold",
            "collision_bounds", "enable_fracture", "enable_self_collision"
        };
        for (const auto& item : options)
            if (!option_keys.count(py::cast<std::string>(item.first)))
                throw std::invalid_argument("Unsupported FEMFX scene option: " + py::cast<std::string>(item.first));
        if (nodes.ndim() != 2 || nodes.shape(1) != 3 || tets.ndim() != 2 || tets.shape(1) != 4)
            throw std::invalid_argument("nodes and tets must have shapes [N,3] and [M,4]");
        vertex_count_ = checked_count(nodes.shape(0), "vertex count");
        tet_count_ = checked_count(tets.shape(0), "tet count");
        if (velocities.ndim() != 2 || velocities.shape(0) != vertex_count_ || velocities.shape(1) != 3 ||
            kinematic.ndim() != 1 || kinematic.shape(0) != vertex_count_)
            throw std::invalid_argument("velocities and kinematic must have shapes [N,3] and [N]");
        dt_ = finite_float(dt, "dt");
        if (dt_ <= 0 || threads < 1)
            throw std::invalid_argument("dt and threads must be positive");
        const float start_time = finite_float(initial_time, "initial_time");
        if (start_time < 0 || !std::isfinite(start_time + dt_) || start_time + dt_ <= start_time)
            throw std::invalid_argument("initial_time must be nonnegative and dt must advance finite float32 time");
        const bool has_deformation = !deformed_positions.is_none();
        py::array_t<double, py::array::c_style | py::array::forcecast> deformed_array;
        if (has_deformation)
        {
            deformed_array = decltype(deformed_array)::ensure(deformed_positions);
            if (!deformed_array || deformed_array.ndim() != 2 || deformed_array.shape(0) != vertex_count_ || deformed_array.shape(1) != 3)
                throw std::invalid_argument("deformed_positions must have shape [N,3]");
        }

        AMD::FmTetMaterialParams constitutive;
        constitutive.restDensity = finite_float(py::cast<double>(material["rest_density"]), "rest_density");
        constitutive.youngsModulus = finite_float(py::cast<double>(material["youngs_modulus"]), "youngs_modulus");
        constitutive.poissonsRatio = finite_float(py::cast<double>(material["poissons_ratio"]), "poissons_ratio");
        constitutive.fractureStressThreshold = finite_float(py::cast<double>(material["fracture_stress_threshold"]), "fracture_stress_threshold");
        constitutive.maxUnconstrainedSolveIterations = checked_count(option<uint64_t>(options, "max_unconstrained_iterations", FM_DEFAULT_MAX_CG_ITERATIONS), "max_unconstrained_iterations");
        if (constitutive.restDensity <= 0 || constitutive.youngsModulus <= 0 || constitutive.poissonsRatio <= -1 ||
            constitutive.poissonsRatio >= 0.5f || constitutive.fractureStressThreshold <= 0)
            throw std::invalid_argument("Material requires density>0, Young modulus>0, -1<Poisson ratio<0.5, fracture threshold>0");
        const float friction = finite_float(option<double>(options, "friction_coefficient", 0.0), "friction_coefficient");
        if (friction < 0)
            throw std::invalid_argument("friction_coefficient must be nonnegative");
        const bool enable_fracture = option<bool>(options, "enable_fracture", true);
        const bool self_collision = option<bool>(options, "enable_self_collision", false);

        const auto node_data = nodes.unchecked<2>();
        const auto tet_data = tets.unchecked<2>();
        const auto velocity_data = velocities.unchecked<2>();
        const auto kinematic_data = kinematic.unchecked<1>();
        std::vector<AMD::FmVector3> positions(vertex_count_);
        std::vector<AMD::FmVector3> current_positions(vertex_count_);
        std::vector<AMD::FmVector3> initial_velocities(vertex_count_);
        std::vector<AMD::FmTetVertIds> connectivity(tet_count_);
        std::vector<AMD::FmArray<uint>> incidents(vertex_count_);
        std::vector<uint16_t> flags(tet_count_, 0);
        std::map<std::array<uint, 3>, std::vector<std::array<uint, 2>>> faces;
        for (uint vertex = 0; vertex < vertex_count_; ++vertex)
        {
            positions[vertex] = AMD::FmInitVector3(
                finite_float(node_data(vertex, 0), "node"), finite_float(node_data(vertex, 1), "node"), finite_float(node_data(vertex, 2), "node"));
            current_positions[vertex] = has_deformation ? AMD::FmInitVector3(
                finite_float(*deformed_array.data(vertex, 0), "deformed position"),
                finite_float(*deformed_array.data(vertex, 1), "deformed position"),
                finite_float(*deformed_array.data(vertex, 2), "deformed position")) : positions[vertex];
            initial_velocities[vertex] = AMD::FmInitVector3(
                finite_float(velocity_data(vertex, 0), "velocity"), finite_float(velocity_data(vertex, 1), "velocity"), finite_float(velocity_data(vertex, 2), "velocity"));
        }
        std::set<std::array<uint, 4>> unique_tets;
        for (uint tet = 0; tet < tet_count_; ++tet)
        {
            std::array<uint, 4> sorted;
            for (uint corner = 0; corner < 4; ++corner)
            {
                const int64_t id = tet_data(tet, corner);
                if (id < 0 || id >= vertex_count_)
                    throw std::invalid_argument("Tetrahedron vertex index out of range");
                connectivity[tet].ids[corner] = static_cast<uint>(id);
                sorted[corner] = static_cast<uint>(id);
                incidents[id].Add(tet);
            }
            std::sort(sorted.begin(), sorted.end());
            if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end() || !unique_tets.insert(sorted).second)
                throw std::invalid_argument("Repeated tetrahedron or repeated tetrahedron corner");
            const auto& ids = connectivity[tet].ids;
            const auto a = positions[ids[1]] - positions[ids[0]];
            const auto b = positions[ids[2]] - positions[ids[0]];
            const auto c = positions[ids[3]] - positions[ids[0]];
            if (!(FmVectormath::dot(a, FmVectormath::cross(b, c)) > 0))
                throw std::invalid_argument("FEMFX requires positively oriented nondegenerate tetrahedra after float32 conversion");
            const auto deformed_a = current_positions[ids[1]] - current_positions[ids[0]];
            const auto deformed_b = current_positions[ids[2]] - current_positions[ids[0]];
            const auto deformed_c = current_positions[ids[3]] - current_positions[ids[0]];
            const float deformed_determinant = FmVectormath::dot(deformed_a, FmVectormath::cross(deformed_b, deformed_c));
            if (!std::isfinite(deformed_determinant) || deformed_determinant <= 0)
                throw std::invalid_argument("deformed_positions must preserve positive nondegenerate tetrahedra in float32 precision");
            for (uint face = 0; face < 4; ++face)
            {
                AMD::FmFaceVertIds ids;
                AMD::FmGetFaceVertIds(&ids, face, connectivity[tet]);
                std::array<uint, 3> key = {ids.ids[0], ids.ids[1], ids.ids[2]};
                std::sort(key.begin(), key.end());
                faces[key].push_back({tet, face});
            }
        }
        for (uint vertex = 0; vertex < vertex_count_; ++vertex)
            if (incidents[vertex].GetNumElems() == 0 || incidents[vertex].GetNumElems() > FM_MAX_VERT_INCIDENT_TETS)
                throw std::invalid_argument("Every vertex must have between 1 and 64 incident tetrahedra, as required by upstream FEMFX");
        for (const auto& entry : faces)
        {
            if (entry.second.size() > 2)
                throw std::invalid_argument("Nonmanifold tetrahedral face has more than two incident cells");
            if (entry.second.size() == 2)
                internal_faces_.push_back({entry.first, {entry.second[0][0], entry.second[1][0]}, {entry.second[0][1], entry.second[1][1]}});
        }
        fracture_times_.assign(internal_faces_.size(), std::numeric_limits<double>::quiet_NaN());

        AMD::FmTetMeshBufferBounds bounds;
        std::vector<AMD::FmFractureGroupCounts> group_counts(tet_count_);
        std::vector<uint> group_ids(tet_count_);
        AMD::FmComputeTetMeshBufferBounds(&bounds, group_counts.data(), group_ids.data(), incidents.data(),
            connectivity.data(), flags.data(), vertex_count_, tet_count_, enable_fracture);
        AMD::FmTetMeshBufferSetupParams mesh_setup;
        mesh_setup.numVerts = bounds.numVerts;
        mesh_setup.numTets = bounds.numTets;
        mesh_setup.numVertIncidentTets = bounds.numVertIncidentTets;
        mesh_setup.maxVertAdjacentVerts = bounds.maxVertAdjacentVerts;
        mesh_setup.maxVerts = bounds.maxVerts;
        mesh_setup.maxExteriorFaces = bounds.maxExteriorFaces;
        mesh_setup.maxTetMeshes = bounds.maxTetMeshes;
        mesh_setup.enableFracture = enable_fracture;
        mesh_setup.enablePlasticity = false;

        AMD::FmSceneSetupParams setup;
        setup.maxTetMeshBuffers = 1;
        setup.maxTetMeshes = bounds.maxTetMeshes;
        setup.maxSceneVerts = bounds.maxVerts;
        setup.maxTetMeshBufferFeatures = checked_count(std::max({uint64_t(bounds.maxVerts) * (self_collision ? 2 : 1),
            uint64_t(bounds.maxExteriorFaces), uint64_t(tet_count_)}), "max_tet_mesh_buffer_features");
        setup.maxDistanceContacts = checked_count(option<uint64_t>(options, "max_distance_contacts", std::max<uint64_t>(64, uint64_t(bounds.maxExteriorFaces) * 4)), "max_distance_contacts");
        setup.maxVolumeContacts = checked_count(option<uint64_t>(options, "max_volume_contacts", std::max<uint64_t>(16, tet_count_)), "max_volume_contacts");
        setup.maxVolumeContactVerts = checked_count(option<uint64_t>(options, "max_volume_contact_vertices", std::max<uint64_t>(64, uint64_t(bounds.maxVerts) * 4)), "max_volume_contact_vertices");
        setup.maxBroadPhasePairs = checked_count(option<uint64_t>(options, "max_broad_phase_pairs", std::max<uint64_t>(64, uint64_t(bounds.maxTetMeshes) * 16)), "max_broad_phase_pairs");
        setup.maxDeformationConstraints = checked_count(uint64_t(tet_count_) * 2, "max_deformation_constraints");
        setup.numWorkerThreads = threads;
        setup.maxConstraintSolverDataSize = option<size_t>(options, "max_constraint_solver_bytes", AMD::FmEstimateSceneConstraintSolverDataSize(setup));
        {
            std::lock_guard<std::mutex> lock(scene_mutex);
            if (scene_active)
                throw std::runtime_error("The official sample task scheduler is global; use one active Scene per process or separate worker processes");
            scene_active = true;
            owns_task_system_ = true;
        }
        try
        {
            AMD::SampleInitTaskSystem(threads);
            AMD::SampleWaitForAllThreadsToStart();
            scene_ = AMD::FmCreateScene(setup);
            if (!scene_)
                throw std::runtime_error("FEMFX scene allocation failed");
            AMD::FmTaskSystemCallbacks callbacks;
            callbacks.SetCallbacks(AMD::SampleGetTaskSystemNumThreads, AMD::SampleGetTaskSystemWorkerIndex,
                AMD::SampleAsyncTask, AMD::SampleCreateSyncEvent, AMD::SampleDestroySyncEvent,
                AMD::SampleWaitForSyncEvent, AMD::SampleTriggerSyncEvent);
            AMD::FmSetSceneTaskSystemCallbacks(scene_, callbacks);
            AMD::FmSceneControlParams control;
            control.gravityVector = AMD::FmInitVector3(finite_float(gravity[0], "gravity"), finite_float(gravity[1], "gravity"), finite_float(gravity[2], "gravity"));
            control.numThreads = threads;
            control.timestep = dt_;
            control.simTime = start_time;
            control.defaultMaxCgIterations = constitutive.maxUnconstrainedSolveIterations;
            control.epsilonCg = finite_float(option<double>(options, "epsilon_cg", control.epsilonCg), "epsilon_cg");
            control.kRayleighMassDamping = finite_float(option<double>(options, "rayleigh_mass_damping", control.kRayleighMassDamping), "rayleigh_mass_damping");
            control.kRayleighStiffnessDamping = finite_float(option<double>(options, "rayleigh_stiffness_damping", control.kRayleighStiffnessDamping), "rayleigh_stiffness_damping");
            control.distContactBias = finite_float(option<double>(options, "dist_contact_bias", control.distContactBias), "dist_contact_bias");
            control.distContactThreshold = finite_float(option<double>(options, "dist_contact_threshold", control.distContactThreshold), "dist_contact_threshold");
            if (control.epsilonCg <= 0 || control.kRayleighMassDamping < 0 || control.kRayleighStiffnessDamping < 0 ||
                control.distContactBias < 0 || control.distContactThreshold < control.distContactBias)
                throw std::invalid_argument("Invalid CG tolerance, damping, or contact distance settings");
            std::array<double, 6> collision_bounds = {-double(FLT_MAX), double(FLT_MAX), -double(FLT_MAX), double(FLT_MAX), -double(FLT_MAX), double(FLT_MAX)};
            if (options.contains("collision_bounds"))
                collision_bounds = py::cast<std::array<double, 6>>(options["collision_bounds"]);
            float* bounds_fields[] = {&control.collisionPlanes.minX, &control.collisionPlanes.maxX,
                &control.collisionPlanes.minY, &control.collisionPlanes.maxY, &control.collisionPlanes.minZ, &control.collisionPlanes.maxZ};
            for (uint coordinate = 0; coordinate < 6; ++coordinate)
                *bounds_fields[coordinate] = finite_float(collision_bounds[coordinate], "collision_bounds");
            for (uint axis = 0; axis < 3; ++axis)
                if (collision_bounds[2 * axis] >= collision_bounds[2 * axis + 1])
                    throw std::invalid_argument("Collision plane minimum must be below maximum");
            AMD::FmSetSceneControlParams(scene_, control);
            AMD::FmTetMesh* mesh = nullptr;
            buffer_ = AMD::FmCreateTetMeshBuffer(mesh_setup, group_counts.data(), group_ids.data(), &mesh);
            if (!buffer_ || !mesh)
                throw std::runtime_error("FEMFX tet mesh allocation failed");
            AMD::FmInitVertState(mesh, positions.data(), AMD::FmMatrix3::identity(), AMD::FmInitVector3(0.0f));
            AMD::FmInitTetState(mesh, connectivity.data(), constitutive, friction);
            AMD::FmComputeMeshConstantMatrices(mesh);
            AMD::FmSetMassesFromRestDensities(mesh);
            if (!AMD::FmInitConnectivity(mesh, incidents.data()))
                throw std::runtime_error("FEMFX connectivity initialization failed");
            vertex_masses_.resize(vertex_count_);
            for (uint vertex = 0; vertex < vertex_count_; ++vertex)
            {
                AMD::FmSetVertPosition(nullptr, mesh, vertex, current_positions[vertex]);
                AMD::FmSetVertVelocity(nullptr, mesh, vertex, initial_velocities[vertex]);
                if (kinematic_data(vertex))
                    AMD::FmSetVertFlags(mesh, vertex, FM_VERT_FLAG_KINEMATIC);
                vertex_masses_[vertex] = AMD::FmGetVertMass(*mesh, vertex);
                if (!std::isfinite(vertex_masses_[vertex]) || vertex_masses_[vertex] <= 0)
                    throw std::runtime_error("FEMFX produced an invalid vertex mass");
            }
            AMD::FmEnableSelfCollision(mesh, self_collision);
            const int aspect_warnings = AMD::FmFinishTetMeshInit(mesh);
            if (aspect_warnings != 0)
                throw std::invalid_argument("FEMFX rejected tetrahedron quality: FmFinishTetMeshInit returned " + std::to_string(aspect_warnings));
            if (AMD::FmAddTetMeshBufferToScene(scene_, buffer_) == FM_INVALID_ID)
                throw std::runtime_error("FEMFX could not add the mesh buffer to the scene");
            added_to_scene_ = true;
            settings_["revision"] = FEMFX_REVISION;
            settings_["dt"] = dt_;
            settings_["initial_time"] = start_time;
            settings_["deformed_positions_supplied"] = has_deformation;
            settings_["restart_semantics"] = "rest-mesh elastic reinitialization with supplied positions and velocities; cold contact cache; no plasticity or solver-history checkpoint";
            settings_["rest_geometry_preserved"] = true;
            settings_["input_vertex_welding"] = false;
            settings_["nodal_state_order"] = "current mesh-piece order then current local vertex; corner_vertex_ids index these arrays within each state";
            settings_["corner_mass_semantics"] = "full current nodal mass repeated at each incident tetrahedron corner; sum nodal_masses once for total mass";
            settings_["threads"] = threads;
            settings_["gravity"] = gravity;
            settings_["material"] = material;
            settings_["plasticity"] = false;
            settings_["enable_fracture"] = enable_fracture;
            settings_["enable_self_collision"] = self_collision;
            settings_["friction_coefficient"] = friction;
            settings_["collision_bounds"] = collision_bounds;
            settings_["epsilon_cg"] = control.epsilonCg;
            settings_["max_unconstrained_iterations"] = constitutive.maxUnconstrainedSolveIterations;
            settings_["rayleigh_mass_damping"] = control.kRayleighMassDamping;
            settings_["rayleigh_stiffness_damping"] = control.kRayleighStiffnessDamping;
            settings_["dist_contact_bias"] = control.distContactBias;
            settings_["dist_contact_threshold"] = control.distContactThreshold;
            settings_["initial_aspect_warnings"] = aspect_warnings;
            settings_["max_distance_contacts"] = setup.maxDistanceContacts;
            settings_["max_volume_contacts"] = setup.maxVolumeContacts;
            settings_["max_volume_contact_vertices"] = setup.maxVolumeContactVerts;
            settings_["max_broad_phase_pairs"] = setup.maxBroadPhasePairs;
            settings_["max_constraint_solver_bytes"] = setup.maxConstraintSolverDataSize;
            settings_["max_tet_mesh_buffer_features"] = setup.maxTetMeshBufferFeatures;
        }
        catch (...)
        {
            close();
            throw;
        }
    }

    ~Scene()
    {
        close();
    }

    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    py::dict state()
    {
        require_open();
        py::array_t<double> positions({py::ssize_t(tet_count_), py::ssize_t(4), py::ssize_t(3)});
        py::array_t<double> velocities({py::ssize_t(tet_count_), py::ssize_t(4), py::ssize_t(3)});
        py::array_t<double> rest_positions({py::ssize_t(tet_count_), py::ssize_t(4), py::ssize_t(3)});
        py::array_t<double> corner_masses({py::ssize_t(tet_count_), py::ssize_t(4)});
        py::array_t<bool> corner_kinematic({py::ssize_t(tet_count_), py::ssize_t(4)});
        py::array_t<int64_t> corner_vertex_ids({py::ssize_t(tet_count_), py::ssize_t(4)});
        py::array_t<int64_t> corner_original_vertex_ids({py::ssize_t(tet_count_), py::ssize_t(4)});
        py::array_t<int64_t> piece_ids(tet_count_);
        py::array_t<int64_t> original_ids(tet_count_);
        py::array_t<int64_t> mesh_tet_ids(tet_count_);
        py::array_t<int64_t> adjacency({py::ssize_t(tet_count_), py::ssize_t(4)});
        auto position_output = positions.mutable_unchecked<3>();
        auto velocity_output = velocities.mutable_unchecked<3>();
        auto rest_output = rest_positions.mutable_unchecked<3>();
        auto corner_mass_output = corner_masses.mutable_unchecked<2>();
        auto corner_kinematic_output = corner_kinematic.mutable_unchecked<2>();
        auto corner_vertex_output = corner_vertex_ids.mutable_unchecked<2>();
        auto corner_original_vertex_output = corner_original_vertex_ids.mutable_unchecked<2>();
        auto piece_output = piece_ids.mutable_unchecked<1>();
        auto original_output = original_ids.mutable_unchecked<1>();
        auto local_output = mesh_tet_ids.mutable_unchecked<1>();
        auto adjacency_output = adjacency.mutable_unchecked<2>();
        const uint piece_count = AMD::FmGetNumTetMeshes(*buffer_);
        std::vector<py::ssize_t> vertex_offsets(piece_count + 1, 0);
        for (uint piece = 0; piece < piece_count; ++piece)
            vertex_offsets[piece + 1] = vertex_offsets[piece] + AMD::FmGetNumVerts(*AMD::FmGetTetMesh(*buffer_, piece));
        const auto nodal_count = vertex_offsets.back();
        py::array_t<double> nodal_positions({nodal_count, py::ssize_t(3)});
        py::array_t<double> nodal_rest_positions({nodal_count, py::ssize_t(3)});
        py::array_t<double> nodal_velocities({nodal_count, py::ssize_t(3)});
        py::array_t<double> nodal_masses(nodal_count);
        py::array_t<uint16_t> nodal_flags(nodal_count);
        py::array_t<bool> nodal_kinematic(nodal_count);
        py::array_t<int64_t> nodal_piece_ids(nodal_count);
        py::array_t<int64_t> nodal_original_ids(nodal_count);
        auto nodal_position_output = nodal_positions.mutable_unchecked<2>();
        auto nodal_rest_output = nodal_rest_positions.mutable_unchecked<2>();
        auto nodal_velocity_output = nodal_velocities.mutable_unchecked<2>();
        auto nodal_mass_output = nodal_masses.mutable_unchecked<1>();
        auto nodal_flags_output = nodal_flags.mutable_unchecked<1>();
        auto nodal_kinematic_output = nodal_kinematic.mutable_unchecked<1>();
        auto nodal_piece_output = nodal_piece_ids.mutable_unchecked<1>();
        auto nodal_original_output = nodal_original_ids.mutable_unchecked<1>();
        for (uint piece = 0; piece < piece_count; ++piece)
        {
            const auto* mesh = AMD::FmGetTetMesh(*buffer_, piece);
            for (uint vertex = 0; vertex < AMD::FmGetNumVerts(*mesh); ++vertex)
            {
                const auto index = vertex_offsets[piece] + vertex;
                const auto position = AMD::FmGetVertPosition(*mesh, vertex);
                const auto rest = AMD::FmGetVertRestPosition(*mesh, vertex);
                const auto velocity = AMD::FmGetVertVelocity(*mesh, vertex);
                const float p[3] = {position.x, position.y, position.z};
                const float r[3] = {rest.x, rest.y, rest.z};
                const float v[3] = {velocity.x, velocity.y, velocity.z};
                for (uint axis = 0; axis < 3; ++axis)
                {
                    if (!std::isfinite(p[axis]) || !std::isfinite(r[axis]) || !std::isfinite(v[axis]))
                        throw std::runtime_error("FEMFX returned nonfinite nodal positions or velocities");
                    nodal_position_output(index, axis) = p[axis];
                    nodal_rest_output(index, axis) = r[axis];
                    nodal_velocity_output(index, axis) = v[axis];
                }
                const auto mass = AMD::FmGetVertMass(*mesh, vertex);
                if (!std::isfinite(mass) || mass <= 0)
                    throw std::runtime_error("FEMFX returned an invalid current nodal mass");
                const auto flags = AMD::FmGetVertFlags(*mesh, vertex);
                nodal_mass_output(index) = mass;
                nodal_flags_output(index) = flags;
                nodal_kinematic_output(index) = (flags & FM_VERT_FLAG_KINEMATIC) != 0;
                nodal_piece_output(index) = piece;
                nodal_original_output(index) = AMD::FmGetVertIndex0(*mesh, vertex);
            }
        }
        std::vector<std::vector<int64_t>> local_to_original(piece_count);
        for (uint piece = 0; piece < piece_count; ++piece)
            local_to_original[piece].assign(AMD::FmGetNumTets(*AMD::FmGetTetMesh(*buffer_, piece)), -1);
        for (uint original = 0; original < tet_count_; ++original)
        {
            uint local, piece;
            auto* mesh = AMD::FmGetTetMeshContainingTet(&local, &piece, *buffer_, original);
            if (!mesh || piece >= piece_count || local >= local_to_original[piece].size() || local_to_original[piece][local] != -1)
                throw std::runtime_error("FEMFX original tetrahedron mapping is not bijective");
            local_to_original[piece][local] = original;
            piece_output(original) = piece;
            local_output(original) = local;
            original_output(original) = original;
            const auto ids = AMD::FmGetTetVertIds(*mesh, local);
            for (uint corner = 0; corner < 4; ++corner)
            {
                const auto vertex = vertex_offsets[piece] + ids.ids[corner];
                corner_vertex_output(original, corner) = vertex;
                corner_original_vertex_output(original, corner) = nodal_original_output(vertex);
                corner_mass_output(original, corner) = nodal_mass_output(vertex);
                corner_kinematic_output(original, corner) = nodal_kinematic_output(vertex);
                for (uint axis = 0; axis < 3; ++axis)
                {
                    position_output(original, corner, axis) = nodal_position_output(vertex, axis);
                    rest_output(original, corner, axis) = nodal_rest_output(vertex, axis);
                    velocity_output(original, corner, axis) = nodal_velocity_output(vertex, axis);
                }
            }
        }
        for (uint original = 0; original < tet_count_; ++original)
        {
            const uint piece = static_cast<uint>(piece_output(original));
            const uint local = static_cast<uint>(local_output(original));
            auto* mesh = AMD::FmGetTetMesh(*buffer_, piece);
            const auto neighbors = AMD::FmGetTetFaceIncidentTetIds(*mesh, local);
            for (uint face = 0; face < 4; ++face)
            {
                const uint neighbor = neighbors.ids[face];
                if (AMD::FmIsExteriorFaceId(neighbor))
                    adjacency_output(original, face) = -1;
                else if (neighbor >= local_to_original[piece].size() || local_to_original[piece][neighbor] < 0)
                    throw std::runtime_error("FEMFX adjacency references an unmapped tetrahedron");
                else
                    adjacency_output(original, face) = local_to_original[piece][neighbor];
            }
        }
        const py::ssize_t face_count = internal_faces_.size();
        py::array_t<int64_t> face_cells({face_count, py::ssize_t(2)});
        py::array_t<int64_t> face_local_ids({face_count, py::ssize_t(2)});
        py::array_t<int64_t> face_vertices({face_count, py::ssize_t(3)});
        py::array_t<int8_t> cuts(face_count);
        py::array_t<double> times(face_count);
        auto cell_output = face_cells.mutable_unchecked<2>();
        auto face_output = face_local_ids.mutable_unchecked<2>();
        auto vertex_output = face_vertices.mutable_unchecked<2>();
        auto cut_output = cuts.mutable_unchecked<1>();
        auto time_output = times.mutable_unchecked<1>();
        for (py::ssize_t index = 0; index < face_count; ++index)
        {
            const auto& face = internal_faces_[index];
            const bool cut_a = adjacency_output(face.tets[0], face.local_faces[0]) != face.tets[1];
            const bool cut_b = adjacency_output(face.tets[1], face.local_faces[1]) != face.tets[0];
            if (cut_a != cut_b)
                throw std::runtime_error("FEMFX fracture adjacency is asymmetric");
            if (!cut_a && std::isfinite(fracture_times_[index]))
                throw std::runtime_error("FEMFX healed an irreversible recorded fracture face");
            if (cut_a && !std::isfinite(fracture_times_[index]))
                fracture_times_[index] = AMD::FmGetSceneControlParams(*scene_).simTime;
            for (uint side = 0; side < 2; ++side)
            {
                cell_output(index, side) = face.tets[side];
                face_output(index, side) = face.local_faces[side];
            }
            for (uint corner = 0; corner < 3; ++corner)
                vertex_output(index, corner) = face.vertices[corner];
            cut_output(index) = cut_a ? 1 : 0;
            time_output(index) = fracture_times_[index];
        }
        py::array_t<double> initial_masses(vertex_count_);
        std::copy(vertex_masses_.begin(), vertex_masses_.end(), initial_masses.mutable_data());
        const auto& warnings = AMD::FmGetSceneWarningsReportRef(scene_);
        py::dict result;
        result["corner_positions"] = positions;
        result["corner_velocities"] = velocities;
        result["corner_rest_positions"] = rest_positions;
        result["corner_masses"] = corner_masses;
        result["corner_kinematic"] = corner_kinematic;
        result["corner_vertex_ids"] = corner_vertex_ids;
        result["corner_original_vertex_ids"] = corner_original_vertex_ids;
        result["nodal_positions"] = nodal_positions;
        result["nodal_rest_positions"] = nodal_rest_positions;
        result["nodal_velocities"] = nodal_velocities;
        result["nodal_masses"] = nodal_masses;
        result["nodal_flags"] = nodal_flags;
        result["nodal_kinematic"] = nodal_kinematic;
        result["nodal_piece_ids"] = nodal_piece_ids;
        result["nodal_original_ids"] = nodal_original_ids;
        result["tet_piece_ids"] = piece_ids;
        result["original_tet_ids"] = original_ids;
        result["mesh_tet_ids"] = mesh_tet_ids;
        result["tet_adjacency"] = adjacency;
        result["internal_faces"] = face_cells;
        result["internal_face_local_ids"] = face_local_ids;
        result["face_vertices"] = face_vertices;
        result["cut_labels"] = cuts;
        result["fracture_times"] = times;
        result["initial_vertex_masses"] = initial_masses;
        result["warnings"] = warnings.flags.val;
        result["warning_mesh_id"] = warnings.tetMeshId.val;
        result["time"] = AMD::FmGetSceneControlParams(*scene_).simTime;
        result["step"] = step_;
        result["piece_count"] = piece_count;
        return result;
    }

    py::dict step()
    {
        require_open();
        AMD::FmUpdateScene(scene_, dt_);
        ++step_;
        try
        {
            return state();
        }
        catch (...)
        {
            valid_ = false;
            throw;
        }
    }

    py::dict settings() const
    {
        return settings_;
    }

    void dispose()
    {
        close();
    }
};
}

PYBIND11_MODULE(_femfx, module)
{
    module.attr("revision") = FEMFX_REVISION;
    py::dict capabilities;
    capabilities["native_fracture"] = true;
    capabilities["original_tet_mapping"] = true;
    capabilities["face_fracture_history"] = true;
    capabilities["stress_tensor"] = false;
    capabilities["elastic_strain_energy"] = false;
    capabilities["kinetic_energy_from_unique_nodal_masses"] = true;
    capabilities["checkpoint"] = false;
    capabilities["deformed_elastic_initialization"] = true;
    capabilities["distinct_rest_and_current_geometry"] = true;
    capabilities["initial_time"] = true;
    capabilities["current_nodal_masses_and_kinematic_flags"] = true;
    capabilities["input_vertex_welding"] = false;
    capabilities["restart_contact_cache"] = "cold";
    capabilities["plastic_history_restore"] = false;
    capabilities["platform"] = "windows_msvc_cpu";
    capabilities["law"] = "upstream_corotational_fem_absolute_largest_eigenvalue_threshold_and_topology_split";
    capabilities["deterministic_seed_control"] = false;
    capabilities["fracture_time_semantics"] = "first_observed_at_end_of_native_update";
    capabilities["upstream_minimum_fracture_side_tetrahedra"] = 3;
    capabilities["upstream_minimum_fracture_mesh_tetrahedra"] = 6;
    capabilities["fracture_direction_randomization"] = true;
    capabilities["fracture_direction_seed"] = "native object ID";
    module.def("capabilities", [capabilities]() { return capabilities; });
    py::class_<Scene>(module, "Scene")
        .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>,
                      py::array_t<int64_t, py::array::c_style>, const py::dict&,
                      py::array_t<double, py::array::c_style | py::array::forcecast>,
                      py::array_t<bool, py::array::c_style>, double, const std::array<double, 3>&, int, const py::dict&, py::object, double>(),
            py::arg("nodes"), py::arg("tets"), py::arg("material"), py::arg("velocities"),
            py::arg("kinematic"), py::arg("dt"), py::arg("gravity"), py::arg("threads") = 1,
            py::arg("scene_options") = py::dict(), py::arg("deformed_positions") = py::none(),
            py::arg("initial_time") = 0.0)
        .def("step", &Scene::step)
        .def("state", &Scene::state)
        .def("settings", &Scene::settings)
        .def("close", &Scene::dispose);
}
