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
#include "dca/phys/dca_data/dca_data.hpp"
#include <dca_data.hpp>

template <class Parameters, DistType DIST>
auto makeDisorderedG0(decltype(dca::phys::DcaData::G0_r_t_cluster_excluded)& g0_r_t_cl_exl) {
  int matrix_dim = b::dmn_size() * s::dmn_size();

  for (int ir = 0; ir < RClusterDmn::dmn_size(); ++ir)
    for (int it = 0; it < TDmn::dmn_size(); ++it) {
      dca::linalg::matrixop::copyArrayToMatrix()
    }
}

#endif
