// Copyright (C) 2026 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Peter W. Doak, Oak Ridge National Lab, (doakpw@ornl.gov)
//
// Here we apply a disorder configuration and disorder potential to a g0

#ifndef DCA_PHYS_DCA_LOOP_DISORDER_APPLY_DISORDER_HPP
#define DCA_PHYS_DCA_LOOP_DISORDER_APPLY_DISORDER_HPP
#include "dca/linalg/matrixop.hpp"
#include "dca/phys/dca_data/dca_data.hpp"
#include "dca/linalg/linalg.hpp"

template <class Parameters, DistType DT>
void DcaData<Parameters, DT>::makeDisorderedG0(
    const DisorderConfiguration disorder_configuration,
    const decltype(dca::phys::DcaData<Parameters, DIST>::G0_r_t_cluster_excluded)& g0_r_t_cl_exl,
    decltype(dca::phys::DcaData<Parameters, DIST>::disordered_G0_r_t_cluster_excluded)&
        disordered_g0_r_t_cl_exl) {
  int matrix_dim = dca::phys::DcaData::NuDmn::dmn_size();
  dca::linalg::Matrix<std::complex<double, dca::linalg::CPU>> g0_rtcex_inverse(matrix_dim);
  dca::linalg::Vector<int, dca::linalg::CPU> ipiv;
  dca::linalg::Vector<std::complex<double>, dca::linalg::CPU> work;

  for (int ir = 0; ir < RClusterDmn::dmn_size(); ++ir)
    for (int it = 0; it < TDmn::dmn_size(); ++it) {
      dca::linalg::matrixop::copyArrayToMatrix(matrix_dim, matrix_dim, g0_r_t_cl_exl(0, 0, ir, it),
                                               matrix_dim, g0_rtcex_inverse);
      dca::linalg::matrixop::inverse(g0_rtcex_inverse, ipiv, work);
      for (int imd = 0; imd < matrix_dim; ++imd) {
        g0_rctex_inverse(imd, imd) += disorder_configuration(imd, ir);
      }
      // Then apply disorder potential to diagonal according to the
      // configuration
      dca::linalg::matrixop::inverse(g0_rtcex_inverse, ipiv, work);
      dca::linalg::matrixop::copyMatrixToArray(g0_rtcex_inverse,
                                               disordered_g0_r_t_cl_exl(0, 0, ir, it), matrix_dim);
    }
}

#endif
