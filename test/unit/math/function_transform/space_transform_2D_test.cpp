// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Urs R. Haehner (haehneru@itp.phys.ethz.ch)
//
// This file tests SpaceTransform2D.

#include <array>
#include <complex>
#include <vector>

#include "gtest/gtest.h"

#include "dca/function/domains.hpp"
#include "dca/function/function.hpp"
#include "dca/math/function_transform/special_transforms/space_transform_2D.hpp"
#include "dca/phys/domains/cluster/cluster_domain_aliases.hpp"
#include "dca/phys/domains/cluster/cluster_domain_initializer.hpp"

using namespace dca;

TEST(SpaceTransform2DTest, f_r_r_otherDmns) {
  const int dimension = 2;

  const std::array<double, 4> basis{1., 0., 0., 1.};
  const std::vector<std::vector<int>> superbasis{{2, 2}, {2, -2}};

  using RDmn = phys::ClusterDomainAliases<dimension>::RClusterDmn;
  using KDmn = phys::ClusterDomainAliases<dimension>::KClusterDmn;
  phys::domains::cluster_domain_initializer<RDmn>::execute(basis.data(), superbasis);

  using OtherDmns =
      func::dmn_variadic<func::dmn_0<func::dmn<2, double>>, func::dmn_0<func::dmn<3, int>>>;

  func::function<std::complex<double>, func::dmn_variadic<RDmn, RDmn, OtherDmns>> f_r_r;
  func::function<std::complex<double>, func::dmn_variadic<KDmn, KDmn, OtherDmns>> f_k_k;

  // Simplest test function:
  //     f(\vec{r1}, \vec{r2}, j) = j + j*j i, if \vec{r1} = \vec{r2} = 0,
  //                              = 0, otherwise.
  // The Fourier transform of this function is a constant,
  //    \hat{f}(\vec{k1}, \vec{k2}, j) = j/Nc + j*j/Nc i,
  // where Nc is the number of vectors \vec{R}.
  for (int j = 0; j < OtherDmns::dmn_size(); ++j) {
    f_r_r(0, 0, j) = std::complex<double>(j, j * j);
  }

  math::transform::SpaceTransform2D<RDmn, KDmn>::execute(f_r_r, f_k_k);

  for (int j = 0; j < OtherDmns::dmn_size(); ++j) {
    for (int k2 = 0; k2 < KDmn::dmn_size(); ++k2) {
      for (int k1 = 0; k1 < KDmn::dmn_size(); ++k1) {
        EXPECT_DOUBLE_EQ((1. * j) / RDmn::dmn_size(), f_k_k(k1, k2, j).real());
        EXPECT_DOUBLE_EQ((1. * j * j) / RDmn::dmn_size(), f_k_k(k1, k2, j).imag());
      }
    }
  }

  // Check plus sign in exponential of first argument.
  f_r_r = 0.;
  for (int j = 0; j < OtherDmns::dmn_size(); ++j) {
    f_r_r(1, 0, j) = std::complex<double>(j, j * j);
  }

  math::transform::SpaceTransform2D<RDmn, KDmn>::execute(f_r_r, f_k_k);

  const std::complex<double> i(0., 1.);
  const auto r1 = RDmn::get_elements()[1];
  const auto k1 = KDmn::get_elements()[1];

  for (int j = 0; j < OtherDmns::dmn_size(); ++j) {
    const auto expected =
        1. / RDmn::dmn_size() * f_r_r(1, 0, j) * std ::exp(i * (k1[0] * r1[0] + k1[1] * r1[1]));
    EXPECT_DOUBLE_EQ(expected.real(), f_k_k(1, 0, j).real());
    EXPECT_DOUBLE_EQ(expected.imag(), f_k_k(1, 0, j).imag());
  }

  // Check minus sign in exponential of second argument.
  f_r_r = 0.;
  for (int j = 0; j < OtherDmns::dmn_size(); ++j) {
    f_r_r(0, 1, j) = std::complex<double>(j, j * j);
  }

  math::transform::SpaceTransform2D<RDmn, KDmn>::execute(f_r_r, f_k_k);

  for (int j = 0; j < OtherDmns::dmn_size(); ++j) {
    const auto expected =
        1. / RDmn::dmn_size() * f_r_r(0, 1, j) * std ::exp(-i * (k1[0] * r1[0] + k1[1] * r1[1]));
    EXPECT_DOUBLE_EQ(expected.real(), f_k_k(0, 1, j).real());
    EXPECT_DOUBLE_EQ(expected.imag(), f_k_k(0, 1, j).imag());
  }
}