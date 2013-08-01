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
  "/usr/include/superlu"
  "/usr/include/suitesparse"
  "./include"
)

link_directories(
  ${OPM_CORE_LIB_DIR}
  ${OPM_AUTODIFF_LIB_DIR}
)

file( GLOB sources     "src/*.cpp" )
file( GLOB app_sources "app/*.cpp" )
file( GLOB headers     "include/*.hpp" )
file( GLOB DSL_sources "dsl/*.equelle" )

set( optional_libs 
  lapack
  blas
)

set( afr_libs 
  /usr/local/lib/libumfpack.a
  /usr/local/lib/libcholmod.a
  /usr/local/lib/libamd.a
  /usr/local/lib/libcolamd.a
  /usr/local/lib/libcamd.a
  /usr/local/lib/libecl.dylib
  /usr/local/lib/libert_geometry.dylib
  /usr/local/lib/libert_util.dylib
  /usr/lib/libz.dylib
  /usr/lib/libm.dylib
  /usr/local/lib/libdunecommon.a
  /Users/atgeirr/install/SuperLU_4.3/lib/libsuperlu_4.3.a
)

foreach( app ${app_sources} )
  get_filename_component( target ${app} NAME_WE )
  add_executable( ${target}
    ${app} ${sources} ${headers} ${DSL_sources}
    )
  target_link_libraries( ${target}
    "libopmautodiff.a"
    "libopmcore.a"
    "boost_filesystem-mt"
    "boost_system-mt"
    "dunecommon"
    ${SUPERLU_LIB}
    "umfpack"
#    ${afr_libs}
#    ${optional_libs}
  )
endforeach()

# -rdynamic



















