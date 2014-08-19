cmake_minimum_required(VERSION 2.8)
find_package(Equelle REQUIRED)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  add_definitions(-DEQUELLE_DEBUG)
endif(CMAKE_BUILD_TYPE MATCHES "Debug")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

#include_directories( ${EQUELLE_INCLUDE_DIRS} "../../../backends/cuda/cuda_include" "../../../backends/cuda/include" "/usr/local/cuda-5.5/include" )
include_directories( ${EQUELLE_INCLUDE_DIRS} )

add_executable(simulator simulator.cpp)
target_link_libraries(simulator ${EQUELLE_LIBRARIES})
