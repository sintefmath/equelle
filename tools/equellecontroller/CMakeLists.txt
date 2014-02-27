project(equellecontroller)
cmake_minimum_required(VERSION 2.8)

file(GLOB equellecontroller_src "src/*.cpp")
file(GLOB equellecontroller_inc "include/equelle/*.hpp")

include_directories(include)
add_subdirectory(test)

add_library(equellecontroller ${equellecontroller_src} ${equellecontroller_inc})
