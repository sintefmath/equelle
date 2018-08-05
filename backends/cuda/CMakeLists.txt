project(equelle_cuda_backend)
cmake_minimum_required( VERSION 2.8 )

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS unit_test_framework)


set( CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wextra -Wno-sign-compare -Wno-unused-parameter" )

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
file( GLOB cuda_cuda_cuh "cuda_include/*.cuh" )
file( GLOB cuda_test_src "test/*.cpp" )
file( GLOB cuda_test_src_cu "test/*.cu" )
set( SUPERLU_LIB          "superlu" CACHE PATH "" )

set( SERIAL_INCLUDE_DIRS ${SERIAL_INCLUDE_DIRS}
  "../serial/include" ${EIGEN3_INCLUDE_DIR} ${EQUELLE_EXTRA_INCLUDE_DIRS})

# Create function for printing text:
# Use: CleanMessage("My message")
function(CleanMessage)
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo "${ARGN}")
endfunction()

#"/usr/local/cuda-x.x/include"
include_directories( "include" "cuda_include" "test"
	${SERIAL_INCLUDE_DIRS}  )

if( CMAKE_BUILD_TYPE MATCHES "Debug")
  set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DEQUELLE_DEBUG")
endif( CMAKE_BUILD_TYPE MATCHES "Debug")

# make cuda part of run-time library
# compile only, add to library later
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set( CUDA_NVCC_FLAGS "-arch=sm_61 -std=c++11 -Xcudafe=--diag_suppress=boolean_controlling_expr_is_constant -Xcudafe=--diag_suppress=unrecognized_gcc_pragma -Xcudafe=--diag_suppress=code_is_unreachable --expt-relaxed-constexpr" CACHE STRING "nvcc flags" FORCE)
cuda_compile ( equelle_cuda_cu ${cuda_cuda_src} ${cuda_cuda_inc} ${cuda_cuda_cuh} )


# Compile the non-cuda part of the library with the cuda compiled files
add_library( equelle_cuda ${equelle_cuda_cu} ${cuda_src} ${cuda_inc} )

# Link with the cuda libraries
target_link_libraries ( equelle_cuda ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} )

set(optirun_command "")
option(EQUELLE_CUDA_OPTIRUN "Run CUDA tests with optirun from Bumblebee" OFF)
if ( EQUELLE_CUDA_OPTIRUN )
  set(optirun_command "optirun" )
endif( EQUELLE_CUDA_OPTIRUN )

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
		CleanMessage("cuda_h_file_name: ${cuda_h_file_name} ")
		CleanMessage("single_test: ${single_test} " )
		cuda_add_library( ${cuda_name} ${cuda_file_name} ${cuda_h_file_name} )
		add_executable( ${test_name} ${single_test})
		target_link_libraries( ${test_name}  
			${cuda_name}
			${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
			equelle_cuda
			equelle_rt
			opmsimulators opmgrid opmcommon
			boost_filesystem boost_system boost_regex
			tinyxml ecl umfpack ${SUPERLU_LIB} dunecommon
		)
		if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/${test_name}.param")
		  add_test( NAME ${test_name}
		          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test/
		          COMMAND ${optirun_command} 
			  ${CMAKE_BINARY_DIR}/backends/cuda/${test_name} ${test_name}.param
		          #${CMAKE_CURRENT_SOURCE_DIR}/test/${test_name}.param
			)
		else()
			add_test( ${test_name} ${optirun_command} ./${test_name})
		endif()
	endforeach()
endif(Boost_FOUND)
CleanMessage("CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
CleanMessage("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

set_target_properties( equelle_cuda PROPERTIES
  PUBLIC_HEADER "${cuda_cuda_inc} ${cuda_inc}")


# The following commands are for making find_package(Equelle) work
# These CMake-variables must be exported into the parent scope (using the PARENT_SCOPE clause)!

# CONF_INCLUDE_DIRS is used for building/linking against the build-tree (without make install)
# Must be set up for the include-directory of the backend. ${PROJECT_SOURCE_DIR} is relative to the
# "nearest" CMakeLists.txt containng a project.
set(CONF_INCLUDE_DIRS "${CONF_INCLUDE_DIRS}" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/cuda_include" "${PROJECT_BINARY_DIR}" PARENT_SCOPE)

set(EQUELLE_LIBS_FOR_CONFIG ${EQUELLE_LIBS_FOR_CONFIG}
    equelle_cuda  opmsimulators opmparser dunecommon
    umfpack tinyxml boost_filesystem boost_system ${SUPERLU_LIB}
    ${EQUELLE_EXTRA_LIBS} ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} 
    ${CUDA_cublas_LIBRARY}
    PARENT_SCOPE)

set(EQUELLE_LIB_DIRS_FOR_CONFIG ${EQUELLE_LIB_DIRS_FOR_CONFIG}
    ${EQUELLE_EXTRA_LIB_DIRS} ${CUDA_LIBRARIES}
    PARENT_SCOPE)

set(EQUELLE_INCLUDE_DIRS_FOR_CONFIG ${EQUELLE_INCLUDE_DIRS_FOR_CONFIG}
    ${SERIAL_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS}
    PARENT_SCOPE )

install(TARGETS equelle_cuda
	EXPORT EquelleTargets
	RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
	LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
	ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
	PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/equelle" COMPONENT dev)


