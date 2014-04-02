cmake_minimum_required(VERSION 2.8)

# We need C++11 features.
if(NOT MSVC)
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall -Wextra")
endif()

find_package( Equelle REQUIRED )

option(EQUELLE_BUILD_MPI "Build MPI backend and tools (requires MPI and Zoltan from Trilinos)" OFF)

link_directories(
  ${EQUELLE_LIB_DIRS}
  ${EQUELLE_INCLUDE_DIRS}
  ${EXTRA_LIBDIRS_FOR_EQUELLE}
)

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

