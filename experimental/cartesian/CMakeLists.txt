project(cartesian_test)
cmake_minimum_required(VERSION 2.8)

find_package(Boost REQUIRED COMPONENTS unit_test_framework)
add_definitions(-DBOOST_TEST_DYN_LINK)

file(GLOB test_src "src/*.cpp")
file(GLOB test_inc "include/equelle/*.hpp")

include_directories( "include"
    "../../serial/include" "/usr/include/eigen3" "../include" )

add_executable(cartesian_test ${test_src}
                              ${test_inc} )

target_link_libraries(cartesian_test equelle_mpi equelle_rt
    ${Boost_LIBRARIES}
    opmautodiff opmcore dunecommon )
