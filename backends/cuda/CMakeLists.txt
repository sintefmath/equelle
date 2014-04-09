project(equelle_cuda_backend)
cmake_minimum_required( VERSION 2.8 )

find_package("CUDA" REQUIRED)

# find boost for testing:
find_package(Boost)


set( CMAKE_CXX_FLAGS "-std=c++0x -Wall -Wextra -Wno-sign-compare -Wno-unused-parameter" )
#set( CMAKE_CXX_FLAGS "-std=c++0x -Wall 	" )

# Define 'EQUELLE_DEBUG' if we are in debug mode.
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  add_definitions(-DEQUELLE_DEBUG)
  #cuda_add_definitions(-DEQUELLE_DEBUG)
endif(CMAKE_BUILD_TYPE MATCHES "Debug")
# From FindCUDA.cmake
# Flags passed into add_definitions with -D or /D are passed along to nvcc.


file( GLOB cuda_src "src/*.cpp" )
file( GLOB cuda_inc "include/*.hpp" )
file( GLOB cuda_cuda_src "src/*.cu" )
file( GLOB cuda_cuda_inc "cuda_include/*.hpp" )
#file( GLOB cuda_cuda_cuh "cuda_include/*.cuh" )
file( GLOB cuda_test_src "test/*.cpp" )
file( GLOB cuda_test_src_cu "test/*.cu" )
set( SUPERLU_LIB          "superlu" CACHE PATH "" ) 	# havahol

set( SERIAL_INCLUDE_DIRS "/usr/local/include/Eigen" ${EQUELLE_EXTRA_INCLUDE_DIRS})

# Create function for printing text:
# Use: CleanMessage("My message")
function(CleanMessage)
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo "${ARGN}")
endfunction()

#"/usr/local/cuda-5.5/include" 
include_directories( "include" "cuda_include" "test"
	${SERIAL_INCLUDE_DIRS}  )

# make cuda part of run-time library
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set( CUDA_NVCC_FLAGS "-arch=sm_13" CACHE STRING "nvcc flags" FORCE)
cuda_add_library( equelle_cuda_cu ${cuda_cuda_src} ${cuda_cuda_inc}  )
				  #${cuda_cuda_cuh} )
target_link_libraries ( equelle_cuda_cu
  ${CUDA_LIBRARIES}
  ${CUDA_cusparse_LIBRARY}
)

if( CMAKE_BUILD_TYPE MATCHES "Debug")
  set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DEQUELLE_DEBUG")
endif( CMAKE_BUILD_TYPE MATCHES "Debug")


# make non-cuda part of run-time library
add_library( equelle_cuda_cpu ${cuda_src} ${cuda_inc} )

# make the unit-testing 
if (Boost_FOUND)
	enable_testing()
	foreach ( single_test ${cuda_test_src} )
		# get name without extension
		get_filename_component( test_name ${single_test} NAME_WE )
		set( cuda_name "${test_name}_cuda" )
		set( cuda_file_name "${CMAKE_CURRENT_SOURCE_DIR}/test/${test_name}.cu" )
		set( cuda_h_file_name "${CMAKE_CURRENT_SOURCE_DIR}/test/${test_name}.h" )
		CleanMessage("cuda_file_name: ${cuda_file_name} ")
		CleanMessage("cuda_h_file_name: ${cuda_h_file_name} ")
		CleanMessage("single_test: ${single_test} " )
		cuda_add_library( ${cuda_name} ${cuda_file_name} ${cuda_h_file_name} )
		add_executable( ${test_name} ${single_test})
		target_link_libraries( ${test_name}  
			${cuda_name}
			${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
			equelle_cuda_cpu
			equelle_cuda_cu
			opmcore boost_filesystem-mt boost_system-mt 
			tinyxml umfpack ${SUPERLU_LIB} dunecommon
		)
		if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/${test_name}.param")
			add_test( ${test_name} optirun ./${test_name}
			${CMAKE_CURRENT_SOURCE_DIR}/test/${test_name}.param)
		else()
			add_test( ${test_name} optirun ./${test_name})
		endif()
	endforeach()
endif(Boost_FOUND)

set_target_properties( equelle_cuda_cpu PROPERTIES
	PUBLIC_HEADER "${cuda_inc}" )

set_target_properties( equelle_cuda_cu PROPERTIES
  PUBLIC_HEADER "${cuda_cuda_inc}" )

set(EQUELLE_LIBS_FOR_CONFIG ${EQUELLE_LIBS_FOR_CONFIG}
    equelle_cuda_cpu equelle_cuda_cu  opmautodiff opmcore dunecommon
    umfpack tinyxml boost_filesystem-mt	boost_system-mt ${SUPERLU_LIB}
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


