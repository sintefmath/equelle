cmake_minimum_required(VERSION 2.8)
find_package(Equelle REQUIRED)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  add_definitions(-DEQUELLE_DEBUG)
endif(CMAKE_BUILD_TYPE MATCHES "Debug")

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

include_directories( ${EQUELLE_INCLUDE_DIRS} "../../../../build/backends/cuda" "../../../backends/cuda/cuda_include/" "../../../backends/cuda/include" "/usr/local/cuda-5.5/include" "../../../backends/serial/"  "../../../backends/serial/include" "../../../../build/backends/serial" )

add_executable(cpu_swe cpu_swe.cpp)
target_link_libraries( cpu_swe ${EQUELLE_LIBRARIES}  )

add_executable(gpu_swe gpu_swe.cpp)
target_link_libraries( gpu_swe ${EQUELLE_LIBRARIES} )

# If cmake can't find equelle package, run:
# cmake -DEquelle_DIR=/home/havahol/sintefEquelle/build/
# for also building in Debug mode:
# -DCMAKE_BUILD_TYPE=Debug