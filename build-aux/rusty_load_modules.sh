#!/bin/bash

module purge
module load cmake/3.19.1
module load gcc/9.3.0
module load openmpi4
module load lib/hdf5/1.12.0
module load lib/boost/1.70-gcc7
module load intel/mkl/2020-4
module load lib/openblas/0.3.7-single
module load cuda/11.1.0_455.23.05
module load lib/fftw3/3.3.8-openmpi4

export BOOST_ROOT=${BOOST_BASE}
export BLAS_ROOT=${MKLROOT}
export LAPACK_ROOT=${MKLROOT}
export CUDA_ROOT=${CUDA_BASE}
export MAGMA_ROOT=/mnt/home/mmorales/Libraries/MAGMA/magma-2.6.1_MKL_GCC9.3_CUDA11.1/

export PATH=/mnt/home/mmorales/libexec/git-core/:${PATH}
export path=/mnt/home/mmorales/libexec/git-core/:${path}

export CC=mpicc
export CXX=mpicxx

#cmake -C ../rusty.cmake ../DCA/ -DDCA_WITH_CUDA=ON -DDCA_HAVE_LAPACK=OFF

#make 
