project( equelle_backends )
cmake_minimum_required( VERSION 2.8 )

set( CMAKE_CXX_FLAGS "-std=c++0x -Wall -Wextra -Wno-sign-compare" )

file( GLOB serial_src "serial/src/*.cpp" )
file( GLOB serial_inc "serial/include/*.hpp" )

set( SERIAL_INCLUDE_DIRECTORIES "/usr/include/eigen3" )

include_directories( "serial/include" 
	${SERIAL_INCLUDE_DIRECTORIES} )

add_library( equelle_rt ${serial_src} ${serial_inc} )

install(TARGETS equelle_rt
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib)



