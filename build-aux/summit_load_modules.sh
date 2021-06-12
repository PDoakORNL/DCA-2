#!/bin/bash
#
# Loads all modules that are required to build DCA++ on ORNL's Summit supercomputer.
# A reset is done at the beginning to restore to the default programming environment on Summit.
#
# Usage: source summit_load_modules.sh

module reset
module load gcc/8.1.1
module load cuda/11.1.0
module load magma/2.5.4-cuda11.1
module load hdf5
module load fftw
module load cmake
module load netlib-lapack
module load essl
module load adios2/2.7.0

export CC=mpicc
export CXX=mpicxx
