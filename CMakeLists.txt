###########################################################################
# Project created for benchmarking APR pipeline CPU vs GPU
###########################################################################
cmake_minimum_required(VERSION 3.9)
project(benchmarkThings)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


##########################################################################
# Configure and use submodules
##########################################################################

# APR
set(APR_TESTS OFF CACHE BOOL "" FORCE) 
set(APR_BUILD_STATIC_LIB ON CACHE BOOL "" FORCE)
set(APR_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(APR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(APR_USE_CUDA ON CACHE BOOL "" FORCE)
add_subdirectory("LibAPR")
set(APR_LIB staticLib)

# Google TEST
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
add_subdirectory("googletest")
set(GTEST_LIBRARIES gtest)
include_directories(LibAPR/src)

# Benchmark LIB
include_directories(APRBench)


##########################################################################
# APP stuff
##########################################################################

enable_language(CUDA)

add_executable(benchmark src/main.cpp)
target_link_libraries(benchmark ${APR_LIB} ${GTEST_LIBRARIES})
