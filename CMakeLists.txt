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
include_directories(LibAPR/src)

# Google TEST
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
add_subdirectory("googletest")
set(GTEST_LIBRARIES gtest)

# Benchmark LIB
include_directories(APRBench)

# HDF5
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})

# Handle OpenMP
find_package(OpenMP)
if(NOT OPENMP_FOUND OR DISABLE_OPENMP)
    message(WARNING "OpenMP support not found or disabled with current compiler. While APR can compile like this, performance might not be optimal. Please see README.md for instructions.")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_OPENMP ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_OPENMP ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -pedantic")
if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_RELEASE "-O4 -ffast-math")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Bdynamic")
    if(NOT WIN32)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ldl")
    endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
endif()

##########################################################################
# APP stuff
##########################################################################

# Setup CUDA
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "--cudart shared -g -lineinfo -Xptxas -O3,-v -use_fast_math -DAPR_USE_CUDA")
enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# SRC/BIN for app goes here
add_executable(benchmark src/main.cpp)
target_link_libraries(benchmark ${APR_LIB} ${GTEST_LIBRARIES} ${HDF5_LIBRARIES}  )
add_executable(ConvTest src/ConvTest.cpp src/Conv.cu)
target_link_libraries(ConvTest ${APR_LIB} ${GTEST_LIBRARIES} ${HDF5_LIBRARIES} )
