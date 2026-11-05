#include "fracture_world_observer.h"

#include "FEMFXConstraintSolver.h"
#include "FEMFXMpcgSolver.h"
#include "FEMFXSolverMath.h"
#include "FEMFXTetMesh.h"

#include <cmath>
#include <stdexcept>

namespace FractureWorldTelemetry
{
namespace
{
std::atomic<Observation*> active{nullptr};

void record(const AMD::FmTetMesh& mesh, const AMD::FmConstraintSolverData* solver)
{
    auto* observation = active.load(std::memory_order_acquire);
    if (!observation)
        return;
    std::lock_guard<std::mutex> lock(observation->mutex);
    for (AMD::uint local = 0; local < mesh.numTets; ++local)
    {
        const auto original = mesh.tetsIndex0[local];
        if (original >= observation->tetrahedra)
        {
            observation->invalid.store(true);
            continue;
        }
        const auto ids = mesh.tetsVertIds[local];
        for (AMD::uint corner = 0; corner < 4; ++corner)
        {
            const auto vertex = ids.ids[corner];
            const auto index = 4 * original + corner;
            const auto velocity = mesh.vertsVel[vertex];
            auto deltaVelocity = AMD::FmInitVector3(0.0f);
            auto deltaPosition = AMD::FmInitVector3(0.0f);
            if (solver && !(mesh.vertsFlags[vertex] & FM_VERT_FLAG_KINEMATIC))
            {
                const auto offset = mesh.solverData->solverStateOffset + vertex;
                deltaVelocity = AMD::FmInitVector3(solver->deltaVel[offset]);
#if FM_CONSTRAINT_STABILIZATION_SOLVE
                deltaPosition = AMD::FmInitVector3(solver->deltaPos[offset]);
#endif
            }
            const float values[3] = {velocity.x, velocity.y, velocity.z};
            const float deltaValues[3] = {deltaVelocity.x, deltaVelocity.y, deltaVelocity.z};
            const float positionValues[3] = {deltaPosition.x, deltaPosition.y, deltaPosition.z};
            for (AMD::uint axis = 0; axis < 3; ++axis)
            {
                observation->preContactVelocities[3 * index + axis] = values[axis];
                observation->appliedDeltaVelocities[3 * index + axis] = deltaValues[axis];
                observation->stabilizationDeltaPositions[3 * index + axis] = positionValues[axis];
            }
            observation->cornerObserved[index] = 1;
            observation->constraintSolveActive[index] = solver ? 1 : 0;
        }
    }
}
}

void Observation::reset(size_t count)
{
    tetrahedra = count;
    invalid.store(false);
    preContactVelocities.assign(12 * count, 0.0);
    appliedDeltaVelocities.assign(12 * count, 0.0);
    stabilizationDeltaPositions.assign(12 * count, 0.0);
    cornerObserved.assign(4 * count, 0);
    constraintSolveActive.assign(4 * count, 0);
    fractureStressEigenvalues.assign(count, 0.0);
    fractureStressThresholds.assign(count, 0.0);
    fractureStressObserved.assign(count, 0);
    mpcgSolution.assign(12 * count, 0.0);
    mpcgRhs.assign(12 * count, 0.0);
    mpcgResidual.assign(12 * count, 0.0);
    mpcgMasses.assign(4 * count, 0.0);
    mpcgObserved.assign(4 * count, 0);
    mpcgKinematic.assign(4 * count, 0);
    mpcgRecursiveNormSquared.assign(count, 0.0);
    mpcgTrueNormSquared.assign(count, 0.0);
    mpcgRhsNormSquared.assign(count, 0.0);
    mpcgToleranceSquared.assign(count, 0.0);
    mpcgEpsilon.assign(count, 0.0);
    mpcgIterations.assign(count, 0.0);
    mpcgMaxIterations.assign(count, 0.0);
    mpcgStoppingObserved.assign(count, 0);
    mpcgRhsPositive.assign(count, 0);
}

Scope::Scope(Observation* observation)
{
    Observation* expected = nullptr;
    if (!active.compare_exchange_strong(expected, observation, std::memory_order_acq_rel))
        throw std::runtime_error("A native observation scope is already active");
}

Scope::~Scope()
{
    active.store(nullptr, std::memory_order_release);
}

void observeConstraints(const AMD::FmTetMesh& mesh, const AMD::FmConstraintSolverData& solver)
{
    record(mesh, &solver);
}

void observeNoConstraints(const AMD::FmTetMesh& mesh)
{
    record(mesh, nullptr);
}

void observeFractureStress(const AMD::FmTetMesh* mesh, AMD::uint tetrahedron, float eigenvalue, float threshold)
{
    auto* observation = active.load(std::memory_order_acquire);
    if (!observation)
        return;
    std::lock_guard<std::mutex> lock(observation->mutex);
    const auto original = mesh->tetsIndex0[tetrahedron];
    if (original >= observation->tetrahedra)
    {
        observation->invalid.store(true);
        return;
    }
    observation->fractureStressEigenvalues[original] = eigenvalue;
    observation->fractureStressThresholds[original] = threshold;
    observation->fractureStressObserved[original] = 1;
}

void observeMpcgSolve(const AMD::FmTetMesh& mesh, const AMD::FmMpcgSolverData& solver,
    const AMD::FmMpcgSolverDataTemps& temps, float epsilon, AMD::uint maxIterations, AMD::uint iterations)
{
    auto* observation = active.load(std::memory_order_acquire);
    if (!observation)
        return;
    if (solver.A.numRows != mesh.numVerts || !temps.solution || !temps.r || !solver.b ||
        !solver.PInvDiag || !solver.A.submats || !solver.A.indices || !solver.A.rowStarts ||
        !solver.kinematicFlags || !solver.mass)
        return;

    std::vector<AMD::FmVector3> residuals(mesh.numVerts);
    std::vector<uint8_t> finiteRows(mesh.numVerts, 0);
    float recursiveNormSquared = 0.0f;
    float trueNormSquared = 0.0f;
    float rhsNormSquared = 0.0f;
    bool finiteNormInputs = true;
    for (AMD::uint row = 0; row < mesh.numVerts; ++row)
    {
        const auto rowStart = solver.A.rowStarts[row];
        const auto rowEnd = solver.A.rowStarts[row + 1];
        if (rowEnd < rowStart)
            return;
        auto residual = solver.b[row];
        auto adjustedRhs = solver.b[row];
        const auto diagonalProduct = mul(solver.A.submats[row], temps.solution[row]);
        residual -= diagonalProduct;
        if (solver.hasKinematicVerts && solver.kinematicFlags[row])
            adjustedRhs -= diagonalProduct;
        for (AMD::uint entry = rowStart; entry < rowEnd; ++entry)
        {
            const auto column = solver.A.indices[entry];
            if (column >= mesh.numVerts)
                return;
            const auto product = mul(solver.A.submats[entry], temps.solution[column]);
            residual -= product;
            if (solver.hasKinematicVerts && solver.kinematicFlags[column])
                adjustedRhs -= product;
        }
        residuals[row] = AMD::FmInitVector3(residual);
        const auto solution = AMD::FmInitVector3(temps.solution[row]);
        const auto rhs = AMD::FmInitVector3(solver.b[row]);
        const auto recursiveResidual = AMD::FmInitVector3(temps.r[row]);
        const auto trueResidual = residuals[row];
        finiteRows[row] = std::isfinite(solution.x) && std::isfinite(solution.y) && std::isfinite(solution.z) &&
            std::isfinite(rhs.x) && std::isfinite(rhs.y) && std::isfinite(rhs.z) &&
            std::isfinite(trueResidual.x) && std::isfinite(trueResidual.y) && std::isfinite(trueResidual.z) &&
            std::isfinite(recursiveResidual.x) && std::isfinite(recursiveResidual.y) && std::isfinite(recursiveResidual.z) &&
            std::isfinite(solver.mass[row]) && solver.mass[row] > 0.0f;
        if (!solver.hasKinematicVerts || !solver.kinematicFlags[row])
        {
            recursiveNormSquared += dot(temps.r[row], mul(solver.PInvDiag[row], temps.r[row]));
            trueNormSquared += dot(residual, mul(solver.PInvDiag[row], residual));
            rhsNormSquared += dot(adjustedRhs, mul(solver.PInvDiag[row], adjustedRhs));
            finiteNormInputs = finiteNormInputs && finiteRows[row];
        }
    }
    const float toleranceSquared = epsilon * epsilon * rhsNormSquared;
    const bool stoppingObserved = finiteNormInputs && std::isfinite(epsilon) && epsilon > 0.0f &&
        std::isfinite(recursiveNormSquared) && recursiveNormSquared >= 0.0f &&
        std::isfinite(trueNormSquared) && trueNormSquared >= 0.0f &&
        std::isfinite(rhsNormSquared) && rhsNormSquared >= 0.0f &&
        std::isfinite(toleranceSquared) && toleranceSquared >= 0.0f && iterations <= maxIterations;

    std::lock_guard<std::mutex> lock(observation->mutex);
    for (AMD::uint local = 0; local < mesh.numTets; ++local)
    {
        const auto original = mesh.tetsIndex0[local];
        if (original >= observation->tetrahedra)
        {
            observation->invalid.store(true);
            continue;
        }
        observation->mpcgRecursiveNormSquared[original] = recursiveNormSquared;
        observation->mpcgTrueNormSquared[original] = trueNormSquared;
        observation->mpcgRhsNormSquared[original] = rhsNormSquared;
        observation->mpcgToleranceSquared[original] = toleranceSquared;
        observation->mpcgEpsilon[original] = epsilon;
        observation->mpcgIterations[original] = iterations;
        observation->mpcgMaxIterations[original] = maxIterations;
        observation->mpcgStoppingObserved[original] = stoppingObserved;
        observation->mpcgRhsPositive[original] = rhsNormSquared > 0.0f;
        const auto ids = mesh.tetsVertIds[local];
        for (AMD::uint corner = 0; corner < 4; ++corner)
        {
            const auto vertex = ids.ids[corner];
            if (vertex >= mesh.numVerts)
            {
                observation->invalid.store(true);
                continue;
            }
            const auto index = 4 * original + corner;
            const auto solution = AMD::FmInitVector3(temps.solution[vertex]);
            const auto rhs = AMD::FmInitVector3(solver.b[vertex]);
            const auto residual = residuals[vertex];
            const float solutions[3] = {solution.x, solution.y, solution.z};
            const float rightHandSides[3] = {rhs.x, rhs.y, rhs.z};
            const float residualValues[3] = {residual.x, residual.y, residual.z};
            for (AMD::uint axis = 0; axis < 3; ++axis)
            {
                observation->mpcgSolution[3 * index + axis] = solutions[axis];
                observation->mpcgRhs[3 * index + axis] = rightHandSides[axis];
                observation->mpcgResidual[3 * index + axis] = residualValues[axis];
            }
            observation->mpcgMasses[index] = solver.mass[vertex];
            observation->mpcgKinematic[index] = solver.kinematicFlags[vertex];
            observation->mpcgObserved[index] = stoppingObserved && finiteRows[vertex];
        }
    }
}
}
