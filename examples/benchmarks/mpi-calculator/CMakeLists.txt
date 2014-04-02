project(equelle-mpi-calculator)
cmake_minimum_required(VERSION 2.8)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/calculator.cpp
    COMMAND ${EQUELLE_COMPILER} 5  < ${CMAKE_CURRENT_SOURCE_DIR}/calculator.equelle > ${CMAKE_CURRENT_BINARY_DIR}/calculator.cpp
    DEPENDS  ${CMAKE_CURRENT_SOURCE_DIR}/calculator.equelle
)

include_directories( ${EQUELLE_INCLUDE_DIRS} )

add_executable(mpi-calculator ${CMAKE_CURRENT_BINARY_DIR}/calculator.cpp calculator.equelle )

target_link_libraries( mpi-calculator
    ${EQUELLE_LIBRARIES}
    ${EXTRA_LIBS_FOR_EQUELLE} )
