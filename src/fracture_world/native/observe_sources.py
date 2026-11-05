import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "fracture-world.femfx-observation.v1"
SOLVER_SCHEMA = "fracture-world.femfx-mpcg-observation.v1"
PATCHES = {
    "ConstraintSolver/FEMFXConstraintSolver.cpp": (
        ("        FmTetMesh& tetMesh = *inTetMesh;",
         "        FmTetMesh& tetMesh = *inTetMesh;\n        FractureWorldTelemetry::observeConstraints(tetMesh, constraintSolverData);"),
        ("            FmTetMesh& tetMesh = *FmGetTetMeshPtrById(*scene, constraintIsland.tetMeshIds[islandMeshIdx]);\n\n            float maxSpeedThreshold",
         "            FmTetMesh& tetMesh = *FmGetTetMeshPtrById(*scene, constraintIsland.tetMeshIds[islandMeshIdx]);\n            FractureWorldTelemetry::observeNoConstraints(tetMesh);\n\n            float maxSpeedThreshold"),
    ),
    "Simulation/FEMFXUpdateTetState.cpp": (
        ("        if (updatedState.testFracture)\n        {",
         "        if (updatedState.testFracture)\n        {\n            FractureWorldTelemetry::observeFractureStress(tetMesh, tetId, updatedState.maxStressEigenvalue, updatedState.fractureStressThreshold);"),
    ),
    "Simulation/FEMFXSimulate.cpp": (
        ("        FmRunMpcgSolve(solverTemps.solution, solverData, &solverTemps, epsilonCG, tetMesh->maxUnconstrainedSolveIterations);",
         "        const auto observedMpcgIterations = FmRunMpcgSolve(solverTemps.solution, solverData, &solverTemps, epsilonCG, tetMesh->maxUnconstrainedSolveIterations);\n        FractureWorldTelemetry::observeMpcgSolve(*tetMesh, *solverData, solverTemps, epsilonCG, tetMesh->maxUnconstrainedSolveIterations, observedMpcgIterations);"),
    ),
}


def create_observation_sources(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    files = {}
    for relative, patches in PATCHES.items():
        upstream = source / "amd_femfx" / "src" / relative
        original = upstream.read_bytes()
        text = original.decode("utf-8-sig")
        text = text.replace("\r\n", "\n")
        for before, after in patches:
            if text.count(before) != 1:
                raise ValueError(f"observation hook requires one exact pinned-source match: {relative}")
            text = text.replace(before, after, 1)
        instrumented = '#include "fracture_world_observer.h"\n' + text
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(instrumented, encoding="utf-8", newline="\n")
        files[relative] = dict(upstream_sha256=hashlib.sha256(original).hexdigest(),
                               instrumented_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    manifest = dict(schema=SCHEMA, solver_observation_schema=SOLVER_SCHEMA,
                    source_files=files, upstream_checkout_modified=False,
                    compiled_source_instrumented=True, purpose="observation only; no solver arithmetic or branching modified",
                    worker_python_callbacks=False)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    create_observation_sources(args.source, args.output)
