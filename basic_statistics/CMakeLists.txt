cmake_minimum_required(VERSION 3.8)
project(basic_statistics VERSION 1.0 LANGUAGES CXX CUDA)
## ----------------------------------------------------
## Source Files of library
## ----------------------------------------------------
set(SOURCE_FILES
    include/basic_statistics.h
    src/basic_statistics.cpp
    src/calculation.cu
    src/interface.cpp
    src/cuda_utility.h
    src/cuda_kernel_function.cuh
    src/cuda_parameter.h
   )

## ----------------------------------------------------------
## Flags
## ----------------------------------------------------------

## if you want to use float instead of double
#add_definitions(-DUSE_32_BIT_FLOAT)  ## uncoment this line if use 32-bit float instead of 64-bit float


set(CMAKE_CXX_STANDARD 11)

# OPENMP
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
ENDIF()

IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mllvm -inline-threshold=100000")
ELSEIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -finline-limit=100000")
ENDIF()


## -----------------------------------------------------
## External Library
## -----------------------------------------------------
# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND LIB_LIBRARIES ${CUDA_LIBRARIES})
list(APPEND LIB_LIBRARIES ${CUDA_CUBLAS_LIBRARIES})
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

# sub projects
add_subdirectory(../measure_time ./measure_time)
include_directories("../measure_time/include")
list(APPEND EXAMPLE_LIBRARIES measure_time)

## ---------------------------------------------------------
## Build Objects
## ---------------------------------------------------------
set(LIBNAME ${PROJECT_NAME})
if(TARGET ${LIBNAME})
else()
cuda_add_library(${LIBNAME} SHARED ${SOURCE_FILES})
target_link_libraries(${LIBNAME} ${LIB_LIBRARIES})

add_executable(basic_stat_example "example/example.cpp")
target_link_libraries(basic_stat_example ${LIBNAME})
target_link_libraries(basic_stat_example ${EXAMPLE_LIBRARIES})
endif()

## ---------------------------------------------------------
## Properties
## ---------------------------------------------------------
set_target_properties(${LIBNAME} PROPERTIES PUBLIC_HEADER include/${LIBNAME}.h)
set_target_properties(${LIBNAME} PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(${LIBNAME} PROPERTIES CXX_STANDARD 11)
set_target_properties(${LIBNAME} PROPERTIES CMAKE_CUDA_STANDARD 11)
set_target_properties(${LIBNAME} PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(${LIBNAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(basic_stat_example PROPERTIES CXX_STANDARD 11)
set_target_properties(basic_stat_example PROPERTIES CMAKE_CUDA_STANDARD 11)
set_target_properties(basic_stat_example PROPERTIES CXX_STANDARD_REQUIRED ON)

## ---------------------------------------------------------
## Install
## ---------------------------------------------------------

# To install this library, use something like
# mkdir build && cd build/ && cmake {this folder} && sudo make install

include(GNUInstallDirs)
if(TARGET ${LIBNAME})
else()
install(TARGETS ${LIBNAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
