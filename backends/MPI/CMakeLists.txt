project(equelle_mpi_backend)
cmake_minimum_required( VERSION 2.8 )

set( CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++0x -Wall -Wextra -Wno-sign-compare" )

find_package( Zoltan REQUIRED )
find_package( MPI REQUIRED )

file( GLOB mpi_src "src/*.cpp" )
file( GLOB mpi_inc "include/equelle/*.hpp" )

set( MPI_INCLUDE_DIRS "/usr/include/eigen3" ${EQUELLE_EXTRA_INCLUDE_DIRS})

include_directories( "include"
    ${MPI_INCLUDE_PATH}
    ${Zoltan_INCLUDE_DIRS}
    "../serial/include" "/usr/include/eigen3" "../include" )


add_library( equelle_mpi ${mpi_src} ${mpi_inc} )

set_target_properties( equelle_mpi PROPERTIES
        PUBLIC_HEADER "${mpi_inc}" )

add_subdirectory(test)

