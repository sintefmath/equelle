cmake_minimum_required(VERSION 2.8)

# We need C++11 features.
set( CMAKE_CXX_FLAGS "-std=c++0x -Wall -Wextra" )

# The /usr/include/eigen3 path is default installed location of eigen3
# on Ubuntu, you may need to add your own location in the EXTRA_... variable.
include_directories(
  "./include"
  "/usr/include/eigen3"
  ${EXTRA_INCLUDES_FOR_EQUELLE}
)

link_directories(
  ${EXTRA_LIBDIRS_FOR_EQUELLE}
)

file( GLOB sources     "src/*.cpp" )
file( GLOB app_sources "app/*.cpp" )
file( GLOB headers     "include/*.hpp" )
file( GLOB DSL_sources "dsl/*.equelle" )

add_library( equelle_rt ${sources} ${headers})

foreach( app ${app_sources} )
  get_filename_component( target ${app} NAME_WE )
  add_executable( ${target}
    ${app} ${headers} ${DSL_sources}
    )
  target_link_libraries( ${target}
    "equelle_rt"
    "opmautodiff"
    "opmcore"
    "dunecommon"
    ${EXTRA_LIBS_FOR_EQUELLE}
  )
endforeach()

# -rdynamic



















