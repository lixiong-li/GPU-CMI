cmake_minimum_required(VERSION 3.8)
project(measure_time VERSION 1.0)
## ----------------------------------------------------
## Source Files of library
## ----------------------------------------------------
set(SOURCE_FILES
    include/timer.h
    src/timer.cpp
   )

## ----------------------------------------------------------
## Flags
## ----------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

## ---------------------------------------------------------
## Build Objects
## ---------------------------------------------------------
set(LIBNAME ${PROJECT_NAME})
if(TARGET ${LIBNAME})
else()
add_library(${LIBNAME} SHARED ${SOURCE_FILES})

add_executable(timer_example "example/example.cpp")
target_link_libraries(timer_example ${LIBNAME})
endif()


## ---------------------------------------------------------
## Properties
## ---------------------------------------------------------
set_target_properties(${LIBNAME} PROPERTIES PUBLIC_HEADER include/${LIBNAME}.h)
set_target_properties(${LIBNAME} PROPERTIES CXX_STANDARD 11)
set_target_properties(${LIBNAME} PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(${LIBNAME} PROPERTIES VERSION ${PROJECT_VERSION})

set_target_properties(timer_example PROPERTIES CXX_STANDARD 11)
set_target_properties(timer_example PROPERTIES CXX_STANDARD_REQUIRED ON)
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
