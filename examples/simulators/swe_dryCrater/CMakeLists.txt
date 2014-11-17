cmake_minimum_required(VERSION 2.8)
find_package(Equelle REQUIRED)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  add_definitions(-DEQUELLE_DEBUG)
endif(CMAKE_BUILD_TYPE MATCHES "Debug")

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

include_directories( ${EQUELLE_INCLUDE_DIRS} "../../../backends/cuda/cuda_include" "../../../backends/cuda/include" "/usr/local/cuda-6.0/include" )

add_executable(swe swe.cpp)
target_link_libraries( swe ${EQUELLE_LIBRARIES}  )

add_executable(serial_swe serial_swe.cpp)
target_link_libraries( serial_swe ${EQUELLE_LIBRARIES} )

# If cmake can't find equelle package, run:
# cmake -DEquelle_DIR=/home/havahol/sintefEquelle/build/
