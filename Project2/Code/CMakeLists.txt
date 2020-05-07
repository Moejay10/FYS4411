project(project_1_fys4411)

# Add directories containing header-files
include_directories(
    .
    Hamiltonians
    InitialStates
    Math
    WaveFunctions
)

# Add sources from all directories
file(
    GLOB SOURCES "*.cpp"
    GLOB SOURCES "Hamiltonians/*.cpp"
    GLOB SOURCES "InitialStates/*.cpp"
    GLOB SOURCES "Math/*.cpp"
    GLOB SOURCES "WaveFunctions/*.cpp"
)

set (CMAKE_CXX_FLAGS "-openmp -std=c++11")
SET (CMAKE_CXX_COMPILER "/usr/bin/clang++")

# Create executable called "vmc"
add_executable(vmc ${SOURCES})
