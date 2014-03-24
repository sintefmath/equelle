project(equelle_serial_backend)
cmake_minimum_required( VERSION 2.8 )

if(NOT MSVC)
	set( CMAKE_CXX_FLAGS "-std=c++0x -Wall -Wextra -Wno-sign-compare" )
ENDIF()

file( GLOB serial_src "src/*.cpp" )
file( GLOB serial_inc "include/*.hpp" )

set( SERIAL_INCLUDE_DIRS "/usr/include/eigen3" ${EQUELLE_EXTRA_INCLUDE_DIRS})


include_directories( "include"
	${SERIAL_INCLUDE_DIRS} )

add_library( equelle_rt ${serial_src} ${serial_inc} )

set_target_properties( equelle_rt PROPERTIES
	PUBLIC_HEADER "${serial_inc}" )

# Below are commands needed to make find_package(Equelle) work
# These CMake-variables must be exported into the parent scope (using the PARENT_SCOPE clause)!

# CONF_INCLUDE_DIRS is used for building/linking against the build-tree (without make install)
# Must be set up for the include-directory of the backend. ${PROJECT_SOURCE_DIR} is relative to the
# "nearest" CMakeLists.txt containng a project.
set(CONF_INCLUDE_DIRS "${CONF_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}" PARENT_SCOPE)

set(EQUELLE_LIBS_FOR_CONFIG ${EQUELLE_LIBS_FOR_CONFIG}
    equelle_rt opmautodiff opmcore dunecommon
    ${EQUELLE_EXTRA_LIBS}
    PARENT_SCOPE)

set(EQUELLE_LIB_DIRS_FOR_CONFIG ${EQUELLE_LIB_DIRS_FOR_CONFIG}
    ${EQUELLE_EXTRA_LIB_DIRS}
    PARENT_SCOPE)

# This is used for building/linking against the installed tree
set(EQUELLE_INCLUDE_DIRS_FOR_CONFIG ${EQUELLE_INCLUDE_DIRS_FOR_CONFIG}
    ${SERIAL_INCLUDE_DIRS}
    PARENT_SCOPE )

install(TARGETS equelle_rt
	EXPORT EquelleTargets
	RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
	LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
	ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
	PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/equelle" COMPONENT dev)


