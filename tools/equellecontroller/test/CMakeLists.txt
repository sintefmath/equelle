project(equellecontroller_test)
cmake_minimum_required(VERSION 2.8)

find_package(Boost REQUIRED COMPONENTS unit_test_framework)
add_definitions(-DBOOST_TEST_DYN_LINK)

file(GLOB test_src "src/*.cpp")
file(GLOB test_inc "include/*.hpp")

add_executable(equellecontroller_test ${test_src} ${test_inc})
target_link_libraries(equellecontroller_test ${Boost_LIBRARIES}) 
