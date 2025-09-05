cmake_minimum_required(VERSION 3.15)
project(test_convex_hull_FCL LANGUAGES CXX)

# Suggest C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# If building inside a Conda env, help CMake find packages and set RPATHs
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(CMAKE_MACOSX_RPATH ON)
  list(APPEND CMAKE_BUILD_RPATH "$ENV{CONDA_PREFIX}/lib")
  list(APPEND CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
endif()

find_package(pinocchio REQUIRED)

# hpp-fcl is typically a transitive dependency of pinocchio; keep optional
find_package(hpp-fcl QUIET)
find_package(assimp QUIET)

add_executable(test_convex_hull_FCL test_convex_hull_FCL.cpp)
target_link_libraries(test_convex_hull_FCL PRIVATE pinocchio::pinocchio)

if(hpp-fcl_FOUND)
  target_link_libraries(test_convex_hull_FCL PRIVATE hpp-fcl::hpp-fcl)
endif()

if(assimp_FOUND)
  target_link_libraries(test_convex_hull_FCL PRIVATE assimp::assimp)
endif()

# On Unix, also include link directories in runtime paths
if(UNIX AND NOT APPLE)
  set_property(TARGET test_convex_hull_FCL PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

