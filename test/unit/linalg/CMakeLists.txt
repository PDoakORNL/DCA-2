# linalg unit tests

dca_add_gtest(vector_cpu_test GTEST_MAIN LIBS ${DCA_LIBS})

dca_add_gtest(vector_cpu_gpu_test
              GTEST_MAIN
              CUDA
              LIBS ${DCA_LIBS})

dca_add_gtest(vector_gpu_test
              GTEST_MAIN
              CUDA
              LIBS ${DCA_LIBS} gpu_utils)

dca_add_gtest(matrix_cpu_test GTEST_MAIN LIBS ${DCA_LIBS})

dca_add_gtest(matrix_view_test GTEST_MAIN LIBS ${DCA_LIBS})

dca_add_gtest(matrix_cpu_gpu_test
              GTEST_MAIN
              CUDA
              LIBS ${DCA_LIBS})

dca_add_gtest(matrix_gpu_test
              CUDA
              LIBS ${DCA_LIBS} ${DCA_KERNEL_LIBS})

# dca_add_gtest(matrixop_complex_cpu_test
#               GTEST_MAIN
#               LIBS ${DCA_THREADING_LIBS} lapack ${MAGMA_LIBRARY} ${DCA_EXTERNAL_LIBS})

# dca_add_gtest(matrixop_real_cpu_test
#               GTEST_MAIN
#               LIBS ${DCA_LIBS})

dca_add_gtest(matrixop_cpu_gpu_test
              GTEST_MAIN
              CUDA
              LIBS ${DCA_LIBS})

dca_add_gtest(matrixop_real_gpu_test
              GTEST_MAIN
              CUDA
              LIBS ${DCA_LIBS} lapack gpu_utils magma::magma lapack_kernels blas_kernels)

# lapack_kernels blas_kernels  lapack_kernels
dca_add_gtest(matrixop_complex_gpu_test
              GTEST_MAIN
              CUDA
              LIBS ${DCA_LIBS} lapack lapack_kernels blas_kernels gpu_utils ${MAGMA_LIBARY} lapack_kernels)

dca_add_gtest(reshapable_matrix_cpu_test GTEST_MAIN LIBS ${UTIL_LIBS} ${DCA_LIBS})

dca_add_gtest(reshapable_matrix_gpu_test CUDA GTEST_MAIN LIBS ${DCA_LIBS} gpu_utils magma::magma)

dca_add_gtest(multi_vector_test GTEST_MAIN LIBS ${DCA_LIBS} ${UTIL_LIBS})

add_subdirectory(blas)
add_subdirectory(util)
