# domains/cluster unit tests

dca_add_gtest(cluster_operations_test
  GTEST_MAIN
  LIBS ${LAPACK_LIBRARIES} ${DCA_GPU_LIBS})

dca_add_gtest(momentum_exchage_domain_test
  GTEST_MPI_MAIN
  LIBS ${DCA_LIBS}
  INCLUDE_DIRS ${DCA_INCLUDE_DIRS};${PROJECT_SOURCE_DIR})

dca_add_gtest(cluster_domain_test
  GTEST_MAIN
  LIBS ${LAPACK_LIBRARIES} function enumerations quantum_domains cluster_domains dca_io FFTW::Double ${DCA_GPU_LIBS})

# deprecated (requires NFFT)
# add_subdirectory(interpolation/wannier_interpolation)
