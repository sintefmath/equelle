project(standalone_partition)
cmake_minimum_required(VERSION 2.8)

find_package( Zoltan REQUIRED )
find_package( MPI REQUIRED )

set( CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++0x -Wall -Wextra -Wno-sign-compare" )
# set( CMAKE_CXX_FLAGS "-std=c++0x -Wall -Wextra -Wno-sign-compare" )

include_directories(
    ${MPI_INCLUDE_PATH}
    ${Zoltan_INCLUDE_DIRS} # JSE: I get a segfault if Zoltan include paths come before MPI_include path :(
    "../../backends/serial/include" "/usr/include/eigen3" "../../backends/MPI/include" )

add_executable(standalone_partition standalonepartition.cpp )

target_link_libraries(standalone_partition
    equelle_mpi equelle_rt
    ${MPI_C_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${Zoltan_LIBRARIES}
    opmautodiff opmcore dunecommon
 )
