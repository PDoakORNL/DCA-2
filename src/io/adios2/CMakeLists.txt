# ADIOS2

add_library(dca_adios2 STATIC adios2_writer.cpp adios2_reader.cpp)
target_link_libraries(dca_adios2 PUBLIC adios2::adios2 MPI::MPI_CXX)
target_include_directories(dca_adios2 PUBLIC adios2::adios2)

if(DCA_HAVE_GPU)
  target_link_libraries(dca_adios2 PUBLIC gpu_utils)
endif(DCA_HAVE_GPU)
