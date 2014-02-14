project(equelle_serial_backend)
cmake_minimum_required( VERSION 2.8 )

set( CMAKE_CXX_FLAGS "-std=c++0x -Wall -Wextra -Wno-sign-compare" )

file( GLOB serial_src "src/*.cpp" )
file( GLOB serial_inc "include/*.hpp" )

set( SERIAL_INCLUDE_DIRECTORIES "/usr/include/eigen3" )

include_directories( "include" 
	${SERIAL_INCLUDE_DIRECTORIES} )

add_library( equelle_rt ${serial_src} ${serial_inc} )

set_target_properties( equelle_rt PROPERTIES
	PUBLIC_HEADER "${serial_inc}" )


set(EQUELLE_LIBRARIES_FOR_CONFIG ${EQUELLE_LIBRARIES_FOR_CONFIG} equelle_rt opmautodiff opmcore dunecommon PARENT_SCOPE)
set(EQUELLE_INCLUDE_DIRS_FOR_CONFIG ${EQUELLE_INCLUDE_DIRS_FOR_CONFIG} /usr/include/eigen3 PARENT_SCOPE )


install(TARGETS equelle_rt
	EXPORT EquelleTargets
	RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
	LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
	ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
	PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/equelle" COMPONENT dev)


