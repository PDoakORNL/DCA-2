dca_add_gtest(tp_equal_time_accumulator_test
  FAST
  GTEST_MPI_MAIN
  INCLUDE_DIRS ${DCA_INCLUDE_DIRS};${PROJECT_SOURCE_DIR}
  LIBS FFTW::Double ${DCA_LIBS} ${KERNELS_LIB} parallel_no_concurrency
  )
