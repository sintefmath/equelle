cmake_minimum_required(VERSION 2.8)

# This simple cmake build file works for the Ubuntu setup described at
# the Equelle wiki:
#
# https://github.com/sintefmath/equelle/wiki/Compiler-and-serial-back-end:-build-instructions-for-Ubuntu-12.04.
#
# If you have a different setup, you may need to customize it. You can
# do that by editing this file, but the recommended way is to use a
# custom cache file. For example, if your opm-core library is located
# in "/work/opm-core/build/lib", and your C++ compiler should be
# "/work/bin/g++-4.8" you can specify that by putting the following
# commands into a separate file, say custom.cmake:
# custom.cmake:
# ============
# set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-4.7.0" CACHE STRING "")
# set(OPM_CORE_LIB_DIR "/work/opm-core/build/lib" CACHE STRING "" )
# set(EXTRA_LIBDIRS_FOR_EQUELLE ${OPM_CORE_LIB_DIR} CACHE STRING "")
# ============
# Then you invoke that by using "cmake -C custom.cmake ../" instead of
# "cmake ../". In some cases, you may need to specify many extra
# libaries, for example if you have used static libraries. For
# example, the extra libs line in such a custom.cmake can look like this:
# set(EXTRA_LIBS_FOR_EQUELLE boost_filesystem-mt boost_system-mt superlu_4.3 umfpack cholmod amd colamd ecl lapack blas CACHE STRING "")

# Check for cuda, and set flag for double precision
find_package("CUDA" REQUIRED)
set( CUDA_NVCC_FLAGS "-arch=sm_13" CACHE STRING "nvcc flags" FORCE)


# We need C++11 features.
#set( CMAKE_CXX_FLAGS "-std=c++0x -Wall -Wextra" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

set( SUPERLU_LIB          "superlu" CACHE PATH "" ) 	# havahol



# The /usr/include/eigen3 path is default installed location of eigen3
# on Ubuntu, you may need to add your own location in the EXTRA_... variable.
include_directories(
  "./include"
  "./cuda_include"
  "/usr/local/include/Eigen" 		# havahol
  "/usr/local/superlu"				# havahol
  "/usr/include/suitesparse"		# havahol
  ${EXTRA_INCLUDES_FOR_EQUELLE}
)

link_directories(
  ${EXTRA_LIBDIRS_FOR_EQUELLE}
)

file( GLOB sources     "src/*.cpp" )
file( GLOB app_sources "app/*.cpp" )
file( GLOB headers     "include/*.hpp" )
file( GLOB DSL_sources "dsl/*.equelle" )
file( GLOB cuda_files  "src/*.cu" )
file( GLOB cuda_include "cuda_include/*.hpp" )

# Here we make the equelle_rt library which we list below!
add_library( equelle_rt ${sources} ${headers} )

# Building the cuda files as a library
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
#include_directories("/usr/local/cuda-5.5/include/")
cuda_add_library(cuda_object ${cuda_files} ${cuda_include} )

foreach( app ${app_sources} )
  get_filename_component( target ${app} NAME_WE )
  add_executable( ${target}
  	${app} 
  	#${headers} ${DSL_sources}
    )
    # Do we actually need headers and DSL_sources in the add_exec?
  target_link_libraries( ${target}
    "equelle_rt"
    "libopmautodiff.a"		# havahol
    "libopmcore.a"			# havahol
    "opmautodiff"
    "opmcore"
    "dunecommon"
    "boost_filesystem-mt"	# havahol
    "boost_system-mt"		# havahol
    ${EXTRA_LIBS_FOR_EQUELLE}
    ${SUPERLU_LIB}			# havahol
    "umfpack"				# havahol
    "tinyxml"				# havahol
    "cuda_object"
    ${CUDA_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_cublas_LIBRARY}
  )
endforeach()




#find_package( Equelle REQUIRED )
#add_executable( heat heatCUDA.cpp )
#target_link_libraries( ${EQULLE_LIBRARIES} )


