add_subdirectory(runequelle)
add_subdirectory(equellecontroller)
if(EQUELLE_BUILD_MPI)
    add_subdirectory(standalonepartition)
endif()


