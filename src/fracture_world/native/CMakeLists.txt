cmake_minimum_required(VERSION 3.24)
project(fracture_world_femfx LANGUAGES CXX)

if(NOT WIN32 OR NOT MSVC)
    message(FATAL_ERROR "The pinned upstream FEMFX requires Windows and MSVC. Generate on Windows and train in WSL if needed.")
endif()

set(FEMFX_SOURCE_DIR "" CACHE PATH "Pinned official FEMFX checkout")
set(FEMFX_ROOT "${FEMFX_SOURCE_DIR}" CACHE PATH "Pinned official FEMFX checkout")
if(NOT EXISTS "${FEMFX_ROOT}/amd_femfx/inc/AMD_FEMFX.h")
    message(FATAL_ERROR "FEMFX_ROOT must point to the pinned official checkout")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE FEMFX_SOURCES CONFIGURE_DEPENDS "${FEMFX_ROOT}/amd_femfx/src/*.cpp")
add_library(femfx STATIC ${FEMFX_SOURCES})
target_compile_features(femfx PUBLIC cxx_std_17)
target_compile_definitions(femfx PUBLIC WIN32 NOMINMAX)
target_compile_options(femfx PUBLIC /permissive /Zc:twoPhase-)
target_compile_options(femfx PRIVATE /W3 /wd4324 /wd4244 /wd4267)
target_include_directories(femfx PUBLIC
    "${FEMFX_ROOT}/amd_femfx/inc"
    "${FEMFX_ROOT}/amd_femfx/inc/Vectormath"
    "${FEMFX_ROOT}/amd_femfx/src/Bvh"
    "${FEMFX_ROOT}/amd_femfx/src/Common"
    "${FEMFX_ROOT}/amd_femfx/src/ConstraintSolver"
    "${FEMFX_ROOT}/amd_femfx/src/FindContacts"
    "${FEMFX_ROOT}/amd_femfx/src/PrimitiveCollision"
    "${FEMFX_ROOT}/amd_femfx/src/Scene"
    "${FEMFX_ROOT}/amd_femfx/src/Simulation"
    "${FEMFX_ROOT}/amd_femfx/src/SparseSolvers"
    "${FEMFX_ROOT}/amd_femfx/src/Threading"
)

add_library(femfx_tasks STATIC
    "${FEMFX_ROOT}/samples/sample_task_system/SampleTaskSystem.cpp"
    "${FEMFX_ROOT}/samples/sample_task_system/TLGetProcessorInfo.cpp"
    "${FEMFX_ROOT}/samples/sample_task_system/TLTaskSystem.cpp"
)
target_compile_features(femfx_tasks PUBLIC cxx_std_17)
target_compile_definitions(femfx_tasks PUBLIC WIN32 NOMINMAX)
target_include_directories(femfx_tasks PUBLIC "${FEMFX_ROOT}/samples/sample_task_system")

pybind11_add_module(_femfx MODULE femfx_bridge.cpp)
target_compile_features(_femfx PRIVATE cxx_std_17)
target_link_libraries(_femfx PRIVATE femfx femfx_tasks)
target_compile_definitions(_femfx PRIVATE FEMFX_REVISION="95cf656731a2465ae1d400138170af2d6575b5bb")
install(TARGETS _femfx LIBRARY DESTINATION . RUNTIME DESTINATION .)
