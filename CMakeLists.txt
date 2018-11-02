cmake_minimum_required(VERSION 3.8)
project(conditional_moment_ineq_v2)

## ----------------------------------------------------
## Source Files of library
## ----------------------------------------------------
set(SOURCE_FILES
    "src/eigen_notation.h"
    "src/conditional_moment_ineq.cpp"
    "src/cube_instrument.cpp"
    "src/moment_selection.cpp"
    "src/bootstrap_multiplier.cpp"
    "src/test_statistics.cpp"
    "src/quadprog.h"
    "include/conditional_moment_ineq.h"
    "include/notation.h"
   )
include_directories("./include")
include_directories("./src")


## ------------------------------------------------------
## Precision
## ------------------------------------------------------
#add_definitions(-DUSE_32_BIT_FLOAT) ## if you want to use 32-bit float instead of double

## ----------------------------------------------------------
## Flags
## ----------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
find_package(OpenMP)
IF(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
ENDIF()

IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS}  -mllvm -inline-threshold=10000")
ELSEIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
ENDIF()

## -----------------------------------------------------
## External Library
## -----------------------------------------------------
if(DEFINED ENV{CODE_LIBRARY})
    set (CODE_LIBRARY $ENV{CODE_LIBRARY})
else()
    set (CODE_LIBRARY $ENV{HOME}/Sync/Documents/code_library)
endif()

# Common Library Path
include_directories($ENV{HOME}/.local/include)

# Basic statistics
add_subdirectory(./basic_statistics)
include_directories("./basic_statistics/include")
list(APPEND LIB_LIBRARIES basic_statistics)

# EIGEN
include_directories("$ENV{HOME}/.local/include/eigen3")

## ---------------------------------------------------------
## Build Objects
## ---------------------------------------------------------
set(LIBNAME ${PROJECT_NAME})
if (TARGET ${LIBNAME})
else()
    add_library(${LIBNAME} SHARED ${SOURCE_FILES})
    target_link_libraries(${LIBNAME} ${LIB_LIBRARIES})

    add_executable(${LIBNAME}_example "example/example.cpp")
    target_link_libraries(${LIBNAME}_example ${LIBNAME})
    target_link_libraries(${LIBNAME}_example ${LIB_LIBRARIES})
    target_link_libraries(${LIBNAME}_example ${EXAMPLE_LIBRARIES})


endif()

