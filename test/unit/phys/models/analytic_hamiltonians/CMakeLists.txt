# Analytic Hamiltonians unit tests

dca_add_gtest(kagome_hubbard_test
  GTEST_MAIN
  INCLUDE_DIRS ${FFTW_INCLUDE_DIR}
  LIBS function function_transform ${LAPACK_LIBRARIES} ${DCA_GPU_LIBS} ${UTIL_LIBS})

dca_add_gtest(bilayer_lattice_test
  GTEST_MAIN
  INCLUDE_DIRS ${FFTW_INCLUDE_DIR}
  LIBS function function_transform ${LAPACK_LIBRARIES} ${DCA_GPU_LIBS} ${UTIL_LIBS})

dca_add_gtest(square_lattice_test
  GTEST_MAIN
  INCLUDE_DIRS ${FFTW_INCLUDE_DIR}
  LIBS function function_transform ${LAPACK_LIBRARIES} ${DCA_GPU_LIBS}  ${UTIL_LIBS})

dca_add_gtest(triangular_lattice_test
  GTEST_MAIN
  INCLUDE_DIRS ${FFTW_INCLUDE_DIR}
  LIBS function function_transform ${LAPACK_LIBRARIES} ${DCA_GPU_LIBS}  ${UTIL_LIBS})
