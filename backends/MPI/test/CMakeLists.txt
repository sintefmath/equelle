project(zoltanintegration_test)
cmake_minimum_required(VERSION 2.8)

find_package(Boost REQUIRED COMPONENTS unit_test_framework)
add_definitions(-DBOOST_TEST_DYN_LINK)

add_definitions(-DEQUELLE_DEBUG)

file(GLOB test_src "src/*.cpp")
file(GLOB test_inc "include/*.hpp")

find_package( Zoltan REQUIRED )
find_package( MPI REQUIRED )

include_directories( "include"
    ${Zoltan_INCLUDE_DIRS}
    ${MPI_INCLUDE_PATH}
    "../../serial/include" "/usr/include/eigen3" "../include" )

add_executable(zoltanintegration_test "src/SubGridBuilderTest.cpp" "src/zoltanIntegration.cpp" ${test_inc})

add_executable( subgridvalidator "src/subgridvalidator.cpp" )

target_link_libraries(zoltanintegration_test equelle_rt equelle_mpi
    ${MPI_C_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${Boost_LIBRARIES}
    ${Zoltan_LIBRARIES}
    opmautodiff opmcore dunecommon )

target_link_libraries(subgridvalidator equelle_rt equelle_mpi
    ${MPI_C_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${Boost_LIBRARIES}
    ${Zoltan_LIBRARIES}
    opmautodiff opmcore dunecommon )
