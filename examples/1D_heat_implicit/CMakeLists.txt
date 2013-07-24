cmake_minimum_required(VERSION 2.8)

# A minmal CMakeList.txt for building the opm_ad-based C++-examples in this directory
# Requirements: See README.txt


# Copied these from the 'examples' in opm-autodiff. Don't know what they do or how important they are...

add_definitions( -DDUNE_COMMON_FIELDVECTOR_SIZE_IS_METHOD=1 )
add_definitions( -DENABLE_SUPERLU=1 )
add_definitions( -DHAVE_CONFIG_H=1 )
add_definitions( -DNDEBUG ) # A bit strange that this was combined with -ggdb3?!


#set( CMAKE_CXX_FLAGS "-std=c++0x -O1 -Wall -Wno-unknown-pragmas" )
set( CMAKE_CXX_FLAGS "-std=c++0x -O0 -g -ggdb3 -Wall -Wno-unknown-pragmas" )

# Other flags that were used by default:
#
#-ggdb3
#-O3
#-flto
#-mtune=native
#-Wl,--as-needed


# Some paths that will depend on where you build (and optionally install) required packages:

set( OPM_CORE_ROOT        "$ENV{HOME}/new_system/prosjekter/opm-core/opm-core" )
set( OPM_AUTODIFF_ROOT    "$ENV{HOME}/new_system/prosjekter/opm_ad/opm-autodiff" )
set( OPM_AUTODIFF_LIB_DIR "$ENV{HOME}/new_system/prosjekter/opm_ad/opm-autodiff/build/lib" )
set( EIGEN3_ROOT          "$ENV{HOME}/new_system/prosjekter/opm_ad/opm-autodiff/build/eigen3-installed/include/eigen3" )


#==================================================================================================================================


# With the above variables, the following should not really need local modification
# Note that this simple CMakeLists.txt assumes all *.cpp files to contain a main function each.


include_directories(
  ${OPM_AUTODIFF_ROOT}
  ${OPM_CORE_ROOT}
  ${EIGEN3_ROOT}
  "/usr/include/superlu"
  "/usr/include/suitesparse"
  "./include"
)

link_directories(
  ${OPM_AUTODIFF_LIB_DIR}
)

file( GLOB sources     "src/*.cpp" )
file( GLOB app_sources "app/*.cpp" )
file( GLOB headers     "include/*.hpp" )
file( GLOB DSL_sources "dsl/*.equelle" )

foreach( app ${app_sources} )
  get_filename_component( target ${app} NAME_WE )
  add_executable( ${target}
    ${app} ${sources} ${headers} ${DSL_sources}
    )
  target_link_libraries( ${target}
    "libopmautodiff.a"
    "${OPM_CORE_ROOT}/build/lib/libopmcore.a"
#    "boost_date_time-mt"
    "boost_filesystem-mt"
    "boost_system-mt"
    "dunecommon"
    "superlu"
    "umfpack"
#    "amd"
#    "lapack"
#    "f77blas"
#    "atlas"
#    "m"
    )
endforeach()

# -rdynamic




















