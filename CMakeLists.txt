project(equelle)
cmake_minimum_required(VERSION 2.8)

# For running "make test"
enable_testing()

find_package(Eigen3 REQUIRED)

set(EQUELLE_MAJOR_VERSION 0)
set(EQUELLE_MINOR_VERSION 1)
set(EQUELLE_PATCH_VERSION 0)
set(EQUELLE_VERSION
    ${EQUELLE_MAJOR_VERSION}.${EQUELLE_MINOR_VERSION}.${EQUELLE_PATCH_VERSION} )

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH
  "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/cmake/Equelle)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

option(EQUELLE_BUILD_MPI "Build MPI backend and tools (requires MPI and Zoltan from Trilinos)" OFF)
option(EQUELLE_BUILD_CUDA "Build CUDA backend and tools (requires CUDA)" OFF)
option(EQUELLE_DEBUG "Enable extra debugging messages and tests" ON)

if(EQUELLE_DEBUG)
   add_definitions(-DEQUELLE_DEBUG)
   #add_definitions(-D_GLIBCXX_DEBUG)
   #add_definitions(-D_GLIBCXX_DEBUG_PEDANTIC)
endif()

add_subdirectory(compiler)
add_subdirectory(backends)
add_subdirectory(tools)
#add_subdirectory(experimental/cartesian)



# --- Stuff for making non-source code show up in IDEs

file( GLOB_RECURSE EQUELLE_README "README.*" )
file( GLOB_RECURSE EQUELLE_EXAMPLE_SOURCES "*.equelle" )
file( GLOB EQUELLE_VAGRANT "vagrant/*" )
source_group( "Assets" FILES ${EQUELLE_README} ${EQUELLE_EXAMPLE_SOURCES} ${EQUELLE_VAGRANT} )
add_custom_target( Equelle_DEPS SOURCES ${EQUELLE_README} ${EQUELLE_EXAMPLE_SOURCES} ${EQUELLE_VAGRANT} )


# --- Generation of package file (that allows for using find_package Equelle in other projects).
set(EQUELLE_TARGETS ec el equelle_rt)
if(EQUELLE_BUILD_CUDA)
  set(EQUELLE_TARGETS ${EQUELLE_TARGETS} equelle_cuda )
endif()
if(EQUELLE_BUILD_MPI)
  set(EQUELLE_TARGETS ${EQUELLE_TARGETS} equelle_mpi )
endif()

export( TARGETS ${EQUELLE_TARGETS} 
	FILE "${PROJECT_BINARY_DIR}/EquelleTargets.cmake")

export(PACKAGE equelle)


# Create the EquelleConfig.cmake and EquelleConfigVersion files
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
   "${INSTALL_INCLUDE_DIR}")

# ... for the build tree
set(CONF_INCLUDE_DIRS "${CONF_INCLUDE_DIRS}" "${EIGEN3_INCLUDE_DIR}")
configure_file(EquelleConfig.cmake.in
  "${PROJECT_BINARY_DIR}/EquelleConfig.cmake" @ONLY)

# ... for the install tree. 
set(CONF_INCLUDE_DIRS "\${EQUELLE_CMAKE_DIR}/${REL_INCLUDE_DIR}/equelle" "${EIGEN3_INCLUDE_DIR}")
configure_file(EquelleConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/EquelleConfig.cmake" @ONLY)

# ... for both
configure_file(EquelleConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/EquelleConfigVersion.cmake" @ONLY)
 
# Install the EquelleConfig.cmake and EquelleConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/EquelleConfig.cmake"
  "${PROJECT_BINARY_DIR}/EquelleConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
 
# Install the export set for use with the install-tree
install(EXPORT EquelleTargets DESTINATION
  "${INSTALL_CMAKE_DIR}" COMPONENT dev)

