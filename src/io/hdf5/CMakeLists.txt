# HDF5

add_library(dca_hdf5 STATIC hdf5_reader.cpp hdf5_writer.cpp)
target_link_libraries(dca_hdf5 PUBLIC hdf5::hdf5 hdf5::hdf5_cpp)
target_include_directories(dca_hdf5 PUBLIC hdf5::hdf5 hdf5::hdf5_cpp)

target_link_libraries(dca_hdf5 PRIVATE
	std::filesystem
	)
