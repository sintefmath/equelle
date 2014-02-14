cmake_minimum_required(VERSION 2.8)

# We need C++11 features.
set( CMAKE_CXX_FLAGS "-std=c++0x -Wall -Wextra" )

find_package(Equelle REQUIRED)

link_directories(
  ${EXTRA_LIBDIRS_FOR_EQUELLE}
)

file( GLOB sources     "src/*.cpp" )
file( GLOB app_sources "app/*.cpp" )
file( GLOB headers     "include/*.hpp" )
file( GLOB DSL_sources "dsl/*.equelle" )

include_directories( ${EQUELLE_INCLUDE_DIRS} )

foreach( app ${app_sources} )
  get_filename_component( target ${app} NAME_WE )
  add_executable( ${target}
    ${app} ${headers} ${DSL_sources}
    )
  target_link_libraries( ${target}
    ${EQUELLE_LIBRARIES}
    ${EXTRA_LIBS_FOR_EQUELLE}
  )
endforeach()


















