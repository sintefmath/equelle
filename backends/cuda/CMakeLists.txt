project(equelle_cuda_backend)
cmake_minimum_required( VERSION 2.8 )

find_package("CUDA" REQUIRED)

set( CMAKE_CXX_FLAGS "-std=c++0x -Wall -Wextra -Wno-sign-compare" )

file( GLOB cuda_src "src/*.cpp" )
file( GLOB cuda_inc "include/*.hpp" )
file( GLOB cuda_cuda_src "src/*.cu" )
file( GLOB cuda_cuda_inc "cuda_include/*.hpp" )

set( SERIAL_INCLUDE_DIRS "/usr/local/include/Eigen" ${EQUELLE_EXTRA_INCLUDE_DIRS})

include_directories( "include" "cuda_include"
	${SERIAL_INCLUDE_DIRS} )

add_library( equelle_cuda_cpu ${cuda_src} ${cuda_inc} )

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set( CUDA_NVCC_FLAGS "-arch=sm_13" CACHE STRING "nvcc flags" FORCE)
cuda_add_library( equelle_cuda_cu ${cuda_cuda_src} ${cuda_cuda_inc} )

set_target_properties( equelle_cuda_cpu PROPERTIES
	PUBLIC_HEADER "${cuda_inc}" )

set_target_properties( equelle_cuda_cu PROPERTIES
  PUBLIC_HEADER "${cuda_cuda_inc}" )

set(EQUELLE_LIBS_FOR_CONFIG ${EQUELLE_LIBS_FOR_CONFIG}
    equelle_cuda_cpu equelle_cuda_cu  opmautodiff opmcore dunecommon
    umfpack tinyxml
    ${EQUELLE_EXTRA_LIBS} ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} 
    ${CUDA_cublas_LIBRARY}
    PARENT_SCOPE)

set(EQUELLE_LIB_DIRS_FOR_CONFIG ${EQUELLE_LIB_DIRS_FOR_CONFIG}
    ${EQUELLE_EXTRA_LIB_DIRS}
    PARENT_SCOPE)

set(EQUELLE_INCLUDE_DIRS_FOR_CONFIG ${EQUELLE_INCLUDE_DIRS_FOR_CONFIG}
    ${SERIAL_INCLUDE_DIRS}
    PARENT_SCOPE )


install(TARGETS equelle_cuda_cpu equelle_cuda_cu
	EXPORT EquelleTargets
	RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
	LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
	ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
	PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/equelle" COMPONENT dev)


