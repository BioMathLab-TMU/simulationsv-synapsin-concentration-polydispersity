cmake_minimum_required(VERSION 3.20)
project(SimulationSVConcentration VERSION 1.0.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(Boost REQUIRED COMPONENTS system filesystem)
add_executable(SimulationSV
    main.cpp
    Environment.cpp
    MyUtility.cpp
    DiscreteAgent.cpp
    Vesicles.cpp
    ProjectManager.cpp
    Synapsin.cpp
    GP.cpp
    Interaction.cpp
    Converter.cpp
    Saver.cpp
    GF.cpp
)
target_include_directories(SimulationSV PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/include ${HDF5_INCLUDE_DIRS})
target_link_libraries(SimulationSV PRIVATE OpenMP::OpenMP_CXX ${HDF5_LIBRARIES} Boost::system Boost::filesystem)
