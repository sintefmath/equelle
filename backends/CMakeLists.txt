project( equelle_backends )
cmake_minimum_required( VERSION 2.8 )

add_subdirectory(serial)

# Needed to export libraries and include dirs for various backends out into EquelleConfig.cmake
set(EQUELLE_LIBS_FOR_CONFIG ${EQUELLE_LIBS_FOR_CONFIG} PARENT_SCOPE)
set(EQUELLE_LIB_DIRS_FOR_CONFIG ${EQUELLE_LIB_DIRS_FOR_CONFIG} PARENT_SCOPE)
set(EQUELLE_INCLUDE_DIRS_FOR_CONFIG ${EQUELLE_INCLUDE_DIRS_FOR_CONFIG} PARENT_SCOPE)
set(CONF_INCLUDE_DIRS ${CONF_INCLUDE_DIRS} PARENT_SCOPE)






