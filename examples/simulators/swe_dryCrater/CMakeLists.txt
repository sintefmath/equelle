cmake_minimum_required(VERSION 2.8)
find_package(Equelle REQUIRED)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  add_definitions(-DEQUELLE_DEBUG)
endif(CMAKE_BUILD_TYPE MATCHES "Debug")

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

include_directories( ${EQUELLE_INCLUDE_DIRS} )

file( GLOB cpp_files "./*.cpp") 
foreach ( file ${cpp_files} )
  get_filename_component( prog ${file} NAME_WE )
  add_executable( ${prog} ${file} )
  target_link_libraries( ${prog} ${EQUELLE_LIBRARIES} )
endforeach()


# If cmake can't find equelle package, run:
# cmake -DEquelle_DIR=<path to build directory>
# for also building in Debug mode:
# -DCMAKE_BUILD_TYPE=Debug
