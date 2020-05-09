project(project_2_fys4411)

# Add directories containing header-files
include_directories(
    .
    Hamiltonians
    InitialStates
    Math
    WaveFunctions
    NeuralNetworks
)

# Add sources from all directories
file(
    GLOB SOURCES "*.cpp"
    GLOB SOURCES "Hamiltonians/*.cpp"
    GLOB SOURCES "InitialStates/*.cpp"
    GLOB SOURCES "Math/*.cpp"
    GLOB SOURCES "WaveFunctions/*.cpp"
    GLOB SOURCES "NeuralNetworks/*.cpp"

)

set (CMAKE_CXX_FLAGS "-fopenmp")
#set (CMAKE_CXX_FLAGS "-Wall")

#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})


# Create executable called "vmc"
add_executable(vmc ${SOURCES})
target_link_libraries(vmc ${ARMADILLO_LIBRARIES})
