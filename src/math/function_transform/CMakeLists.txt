# Function transform

add_library(function_transform STATIC
  basis_expansions.cpp boundary_conditions.cpp domain_representations.cpp element_spacings.cpp)

target_link_libraries(function_transform PUBLIC FFTW::Double)

if (DCA_HAVE_GPU)
  add_library(special_transform_kernels special_transforms_kernels.cu)
  dca_gpu_device_link(special_transform_kernels)
  target_link_libraries(function_transform PUBLIC special_transform_kernels)
endif()
