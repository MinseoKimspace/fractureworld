#pragma once

#include "AMD_FEMFX.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace AMD
{
struct FmConstraintSolverData;
struct FmMpcgSolverData;
struct FmMpcgSolverDataTemps;
}

namespace FractureWorldTelemetry
{
struct Observation
{
    std::mutex mutex;
    std::atomic<bool> invalid{false};
    size_t tetrahedra = 0;
    std::vector<double> preContactVelocities;
    std::vector<double> appliedDeltaVelocities;
    std::vector<double> stabilizationDeltaPositions;
    std::vector<uint8_t> cornerObserved;
    std::vector<uint8_t> constraintSolveActive;
    std::vector<double> fractureStressEigenvalues;
    std::vector<double> fractureStressThresholds;
    std::vector<uint8_t> fractureStressObserved;
    std::vector<double> mpcgSolution;
    std::vector<double> mpcgRhs;
    std::vector<double> mpcgResidual;
    std::vector<double> mpcgMasses;
    std::vector<uint8_t> mpcgObserved;
    std::vector<uint8_t> mpcgKinematic;
    std::vector<double> mpcgRecursiveNormSquared;
    std::vector<double> mpcgTrueNormSquared;
    std::vector<double> mpcgRhsNormSquared;
    std::vector<double> mpcgToleranceSquared;
    std::vector<double> mpcgEpsilon;
    std::vector<double> mpcgIterations;
    std::vector<double> mpcgMaxIterations;
    std::vector<uint8_t> mpcgStoppingObserved;
    std::vector<uint8_t> mpcgRhsPositive;

    void reset(size_t count);
};

class Scope
{
public:
    explicit Scope(Observation* observation);
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

void observeConstraints(const AMD::FmTetMesh& mesh, const AMD::FmConstraintSolverData& solver);
void observeNoConstraints(const AMD::FmTetMesh& mesh);
void observeFractureStress(const AMD::FmTetMesh* mesh, AMD::uint tetrahedron, float eigenvalue, float threshold);
void observeMpcgSolve(const AMD::FmTetMesh& mesh, const AMD::FmMpcgSolverData& solver,
    const AMD::FmMpcgSolverDataTemps& temps, float epsilon, AMD::uint maxIterations, AMD::uint iterations);
}
