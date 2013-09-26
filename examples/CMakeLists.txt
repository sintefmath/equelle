cmake_minimum_required(VERSION 2.8)

# A minmal CMakeList.txt for building early Equelle-examples
# Non-standard requirements: See README.txt

# Copied these from the 'examples' run done in the opm-autodiff 'examples' directory.

add_definitions( -DDUNE_COMMON_FIELDVECTOR_SIZE_IS_METHOD=1 )
add_definitions( -DENABLE_SUPERLU=1 )
add_definitions( -DHAVE_CONFIG_H=1 )
add_definitions( -DNDEBUG ) # A bit strange that this was combined with -ggdb3?!

set( CMAKE_CXX_FLAGS "-std=c++0x -O1 -Wall -Wno-unknown-pragmas" )

# As an example, this is what I (jny, Ubuntu 12.10) use as a preloaded cache.

#set( OPM_CORE_ROOT        "$ENV{HOME}/new_system/prosjekter/opm-core/opm-core"           CACHE PATH "" )
#set( OPM_CORE_LIB_DIR     "$ENV{HOME}/new_system/prosjekter/opm-core/opm-core/build/lib" CACHE PATH "" )
#set( OPM_AUTODIFF_ROOT    "$ENV{HOME}/new_system/prosjekter/opm_ad/opm-autodiff"           CACHE PATH "" )
#set( OPM_AUTODIFF_LIB_DIR "$ENV{HOME}/new_system/prosjekter/opm_ad/opm-autodiff/build/lib" CACHE PATH "" )
#set( EIGEN3_ROOT          "$ENV{HOME}/new_system/prosjekter/opm_ad/opm-autodiff/build/eigen3-installed/include/eigen3" CACHE PATH "" )
#set( SUPERLU_LIB          "superlu" CACHE PATH "" )


#==================================================================================================================================


# With the above variables, the following should not really need local modification

include_directories(
  ${OPM_AUTODIFF_ROOT}
  ${OPM_CORE_ROOT}
  ${EIGEN3_ROOT}
  ${DUNE_CORNERPOINT_ROOT}
  "/usr/include/superlu"
  "/usr/include/suitesparse"
  "./include"
)

link_directories(
  ${OPM_CORE_LIB_DIR}
  ${OPM_AUTODIFF_LIB_DIR}
  ${DUNE_CORNERPOINT_LIB_DIR}
)

file( GLOB sources     "src/*.cpp" )
file( GLOB app_sources "app/*.cpp" )
file( GLOB headers     "include/*.hpp" )
file( GLOB DSL_sources "dsl/*.equelle" )
file( GLOB other_files "../../Equelle_0_1.txt" "../../README.txt")


foreach( app ${app_sources} )
  get_filename_component( target ${app} NAME_WE )
  add_executable( ${target}
    ${app} ${sources} ${headers} ${DSL_sources} ${other_files}
    )
  target_link_libraries( ${target}
    "libdunecornerpoint.a"
    "libopmautodiff.a"
    "libopmcore.a"
    "boost_filesystem-mt"
    "boost_system-mt"
    "dunecommon"
    ${SUPERLU_LIB}
    "umfpack"
    ${EXTRA_LIBS_FOR_OPM}
  )
endforeach()

# -rdynamic



















