project(runequelle)
cmake_minimum_required(VERSION 2.8)



configure_file( runequelle/ebs.py.in runequelle/ebs )
configure_file( runequelle/CMakeLists.txt.in runequelle/CMakeLists.txt @ONLY )
	
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/runequelle/ebs DESTINATION bin)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/runequelle/CMakeLists.txt DESTINATION shared/equelle)
 
