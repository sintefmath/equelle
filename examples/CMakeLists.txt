project( Equelle_examples )
cmake_minimum_required(VERSION 2.8)

# We need C++11 features.
if(NOT MSVC)
	set( CMAKE_CXX_FLAGS "-std=c++1y -Wall -Wextra ${CMAKE_CXX_FLAGS}")
endif()

find_package( Equelle REQUIRED )

option(EQUELLE_BUILD_MPI "Build MPI backend and tools (requires MPI and Zoltan from Trilinos)" OFF)

link_directories(
  ${EQUELLE_LIB_DIRS}
  ${EXTRA_LIBDIRS_FOR_EQUELLE}
)

# --- Stuff for making non-source code show up in IDEs

file( GLOB_RECURSE EQUELLE_README "README.*" )
file( GLOB_RECURSE EQUELLE_EXAMPLE_SOURCES "*.equelle" )
source_group( "Assets" FILES ${EQUELLE_README} ${EQUELLE_EXAMPLE_SOURCES} ${EQUELLE_VAGRANT} )
add_custom_target( Equelle_EXAMPLES_DEPS SOURCES ${EQUELLE_README} ${EQUELLE_EXAMPLE_SOURCES} )

add_subdirectory( benchmarks )

file( GLOB sources     "src/*.cpp" )
file( GLOB app_sources "app/*.cpp" )
file( GLOB headers     "include/*.hpp" )
file( GLOB DSL_sources "dsl/*.equelle" )
file( GLOB cuda_files  "src/*.cu" )
file( GLOB cuda_include "cuda_include/*.hpp" )

include_directories( ${EQUELLE_INCLUDE_DIRS} )

foreach( app ${app_sources} )
  get_filename_component( target ${app} NAME_WE )

  add_executable( ${target}
  	${app} ${headers} ${DSL_sources}
    )

  target_link_libraries( ${target}
    ${EQUELLE_LIBRARIES}
    ${EXTRA_LIBS_FOR_EQUELLE}
    opmcore
  )
endforeach()

