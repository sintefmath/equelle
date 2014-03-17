project(equelle_mpi_backend)
cmake_minimum_required( VERSION 2.8 )

set( CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++0x -Wall -Wextra -Wno-sign-compare" )
add_definitions(-DEQUELLE_DEBUG)

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

# Below are commands needed to make find_package(Equelle) work
# These CMake-variables must be exported into the parent scope (using the PARENT_SCOPE clause)!

# CONF_INCLUDE_DIRS is used for building/linking against the build-tree (without make install)
# Must be set up for the include-directory of the backend. ${PROJECT_SOURCE_DIR} is relative to the
# "nearest" CMakeLists.txt containng a project.
set(CONF_INCLUDE_DIRS "${CONF_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}" PARENT_SCOPE)

set(EQUELLE_LIBS_FOR_CONFIG ${EQUELLE_LIBS_FOR_CONFIG}
    equelle_rt opmautodiff opmcore dunecommon 
    ${MPI_C_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${Zoltan_LIBRARIES}
    ${EQUELLE_EXTRA_LIBS}
    PARENT_SCOPE)

set(EQUELLE_LIB_DIRS_FOR_CONFIG ${EQUELLE_LIB_DIRS_FOR_CONFIG}
    ${EQUELLE_EXTRA_LIB_DIRS}
    PARENT_SCOPE)

# This is used for building/linking against the installed tree
set(EQUELLE_INCLUDE_DIRS_FOR_CONFIG ${EQUELLE_INCLUDE_DIRS_FOR_CONFIG}
    ${SERIAL_INCLUDE_DIRS}
    PARENT_SCOPE )

install(TARGETS equelle_mpi
	EXPORT EquelleTargets
	RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
	LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
	ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
	PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/equelle" COMPONENT dev)
