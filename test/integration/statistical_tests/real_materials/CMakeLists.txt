#  This application was used to build the reference data for the NiO test.
#add_executable(NiO_fullDCA NiO_fullDCA.cpp)
#target_link_libraries(NiO_fullDCA ${TEST_LIBRARIES})
#target_include_directories(NiO_fullDCA PRIVATE ${TEST_INCLUDES})
#target_compile_definitions(NiO_fullDCA PRIVATE TEST_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}/")


dca_add_gtest(NiO_stattest
    STOCHASTIC
    # Run with more ranks for better error detection.
    MPI MPI_NUMPROC 1
    INCLUDE_DIRS ${DCA_INCLUDE_DIRS};${PROJECT_SOURCE_DIR}
    LIBS  ${DCA_LIBS};statistical_testing
    )

include(FetchContent)
  
FetchContent_Declare(
    niotestdata
    URL https://www.dropbox.com/s/0jbzz6e12vgu3go/NiO.data.ed.hdf5
    URL_HASH SHA256=c788632b768c8e3545772315445caae9f8be49e6b25cb64a3860cb6fb7f361e3
    DOWNLOAD_NO_EXTRACT 1
    )

FetchContent_MakeAvailable(niotestdata)
  
configure_file(t_ij_NiO.txt ${CMAKE_CURRENT_BINARY_DIR}/t_ij_NiO.txt COPYONLY)
configure_file(U_ij_NiO_8_lit.txt ${CMAKE_CURRENT_BINARY_DIR}/U_ij_NiO_8_lit.txt COPYONLY)

