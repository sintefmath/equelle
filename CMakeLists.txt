project(equelle)
cmake_minimum_required(VERSION 2.8)

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

option(EQUELLE_BUILD_MPI "Build MPI backend and tools (requires MPI and Zoltan from Trilinos)" OFF )

add_subdirectory(compiler)
add_subdirectory(backends)
add_subdirectory(tools)

export( TARGETS ec el equelle_rt 
	FILE "${PROJECT_BINARY_DIR}/EquelleTargets.cmake")

export(PACKAGE equelle)

# Create the EquelleConfig.cmake and EquelleConfigVersion files
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
   "${INSTALL_INCLUDE_DIR}")

# ... for the build tree
#set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file(EquelleConfig.cmake.in
  "${PROJECT_BINARY_DIR}/EquelleConfig.cmake" @ONLY)

# ... for the install tree. 
set(CONF_INCLUDE_DIRS "\${EQUELLE_CMAKE_DIR}/${REL_INCLUDE_DIR}/equelle")
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

