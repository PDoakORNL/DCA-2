// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Peter Staar (taa@zurich.ibm.com)
//         Urs R. Haehner (haehneru@itp.phys.ethz.ch)
//         Seher Karakuzu (karakuzu.seher@gmail.com)
//         Thomas Maier (maierta@ornl.gov)
//
// Moire Hubbard model on triangular lattice.

#ifndef DCA_PHYS_MODELS_ANALYTIC_HAMILTONIANS_MOIRE_HUBBARD_HPP
#define DCA_PHYS_MODELS_ANALYTIC_HAMILTONIANS_MOIRE_HUBBARD_HPP

#include <cmath>
#include <stdexcept>
#include <vector>

#include "dca/function/domains.hpp"
#include "dca/function/function.hpp"
#include "dca/phys/domains/cluster/symmetries/point_groups/no_symmetry.hpp"
#include "dca/phys/models/analytic_hamiltonians/util.hpp"

namespace dca {
namespace phys {
namespace models {
// dca::phys::models::

//template <typename DCA_point_group_type>
template <typename PointGroup>
class moire_hubbard {
public:
  static constexpr bool complex_g0 = true;
  static constexpr bool spin_symmetric = false;

  using LDA_point_group = domains::no_symmetry<2>;
  using DCA_point_group = PointGroup;
  //typedef domains::no_symmetry<2> LDA_point_group;
  //typedef DCA_point_group_type DCA_point_group;

  const static int DIMENSION = 2;
  const static int BANDS = 2;

  static const double* initializeRDCABasis();
  static const double* initializeRLDABasis();
  
  constexpr static int transformationSignOfR(int, int, int) {
    return 1;
  }
  constexpr static int transformationSignOfK(int, int, int) {
    return 1;
  }

  static std::vector<int> flavors();
  static std::vector<std::vector<double>> aVectors();

  // Initializes the interaction part of the real space Hubbard Hamiltonian.
  template <typename BandDmn, typename SpinDmn, typename RDmn, typename parameters_type>
  static void initializeHInteraction(
      func::function<double, func::dmn_variadic<func::dmn_variadic<BandDmn, SpinDmn>,
                                                func::dmn_variadic<BandDmn, SpinDmn>, RDmn>>& H_interaction,
      const parameters_type& parameters);

  template <class domain>
  static void initializeHSymmetry(func::function<int, domain>& H_symmetry);

  // Initializes the tight-binding (non-interacting) part of the momentum space Hamiltonian.
  // Preconditions: The elements of KDmn are two-dimensional (access through index 0 and 1).
  template <typename ParametersType, typename ScalarType, typename BandDmn, typename SpinDmn, typename KDmn>
  static void initializeH0(
      const ParametersType& parameters,
      func::function<ScalarType, func::dmn_variadic<func::dmn_variadic<BandDmn, SpinDmn>,
                                                    func::dmn_variadic<BandDmn, SpinDmn>, KDmn>>& H_0);
};

template <typename DCA_point_group_type>
const double* moire_hubbard<DCA_point_group_type>::initializeRDCABasis() {
  static double* r_DCA = new double[4];

  r_DCA[0] = std::cos(M_PI / 3.);
  r_DCA[1] = std::sin(M_PI / 3.);
  r_DCA[2] = std::cos(-M_PI / 3.);
  r_DCA[3] = std::sin(-M_PI / 3.);

  return r_DCA;
}

template <typename DCA_point_group_type>
const double* moire_hubbard<DCA_point_group_type>::initializeRLDABasis() {
  static double* r_LDA = new double[4];

  r_LDA[0] = std::cos(M_PI / 3.);
  r_LDA[1] = std::sin(M_PI / 3.);
  r_LDA[2] = std::cos(-M_PI / 3.);
  r_LDA[3] = std::sin(-M_PI / 3.);

  return r_LDA;
}

template <typename DCA_point_group_type>
std::vector<int> moire_hubbard<DCA_point_group_type>::flavors() {
  static std::vector<int> flavors(BANDS);

  for (int i = 0; i < BANDS; i++)
    flavors[i] = i;

  return flavors;
}

template <typename DCA_point_group_type>
std::vector<std::vector<double>> moire_hubbard<DCA_point_group_type>::aVectors() {
  static std::vector<std::vector<double>> a_vecs(BANDS, std::vector<double>(DIMENSION, 0.));
  return a_vecs;
}

template <typename point_group_type>
template <typename BandDmn, typename SpinDmn, typename RDmn, typename parameters_type>
void moire_hubbard<point_group_type>::initializeHInteraction(
    func::function<double, func::dmn_variadic<func::dmn_variadic<BandDmn, SpinDmn>,
                                              func::dmn_variadic<BandDmn, SpinDmn>, RDmn>>& H_interaction,
    const parameters_type& parameters) {
  if (BandDmn::dmn_size() != BANDS)
    throw std::logic_error("Moire lattice has two bands.");
  if (SpinDmn::dmn_size() != 2)
    throw std::logic_error("Spin domain size must be 2.");

  const std::vector<typename RDmn::parameter_type::element_type>& basis =
      RDmn::parameter_type::get_basis_vectors();

  assert(basis.size() == 2);

  // There are three different nearest neighbor (nn) pairs: along the basis vector a1, along the
  // basis vector a2, and along their sum a1+a2.
  std::vector<typename RDmn::parameter_type::element_type> nn_vec(3);
  nn_vec[0] = basis[0];
  nn_vec[1] = basis[1];
  nn_vec[2] = basis[0];
  nn_vec[2][0] += basis[1][0];
  nn_vec[2][1] += basis[1][1];

//I cancelled the line below since we will use bilayer model
//  util::initializeSingleBandHint(parameters, nn_vec, H_interaction);


   // Set all elements to zero.
   H_interaction = 0.;

  const int origin = RDmn::parameter_type::origin_index();
   // On-site interaction, store up-down interaction in the first sector.
   const double U = parameters.get_U();
   H_interaction(0, 0, 1, 0, origin) = U;
   H_interaction(1, 0, 0, 0, origin) = U;

}

template <typename DCA_point_group_type>
template <class domain>
void moire_hubbard<DCA_point_group_type>::initializeHSymmetry(
    func::function<int, domain>& H_symmetries) {
  H_symmetries = -1;
  
  //H_symmetries(0, 0) = 0;
  //H_symmetries(0, 1) = -1;
  //H_symmetries(1, 0) = -1;
  //H_symmetries(1, 1) = 0;
}

template <typename point_group_type>
template <typename ParametersType, typename ScalarType, typename BandDmn, typename SpinDmn, typename KDmn>
void moire_hubbard<point_group_type>::initializeH0(
    const ParametersType& parameters,
    func::function<ScalarType, func::dmn_variadic<func::dmn_variadic<BandDmn, SpinDmn>,
                                                  func::dmn_variadic<BandDmn, SpinDmn>, KDmn>>& H_0) {
  if (BandDmn::dmn_size() != BANDS)
    throw std::logic_error("Triangular lattice has one band.");
  if (SpinDmn::dmn_size() != 2)
    throw std::logic_error("Spin domain size must be 2.");

  const auto& k_vecs = KDmn::get_elements();

  const auto t = parameters.get_t();
  const auto h = parameters.get_h();
  const auto phi = parameters.get_phi();

  H_0 = ScalarType(0);
  dca::linalg::Matrix<ScalarType, dca::linalg::CPU> m(2);

  for (int k_ind = 0; k_ind < KDmn::dmn_size(); ++k_ind) {
    const auto& k = k_vecs[k_ind];

     // kinetic term
     //m(0, 0) = m(1, 1) = -2. * t * (std::cos(k[0]) + std::cos(k[1]));
     m(0,0) = -2. * t * std::cos(k[0] + phi) - 4. * t * std::cos(sqrt(3.) * k[1] / 2.) * std::cos(-k[0] / 2. + phi);
     m(1,1) = -2. * t * std::cos(k[0] - phi) - 4. * t * std::cos(sqrt(3.) * k[1] / 2.) * std::cos(-k[0] / 2. - phi);

     // Note: spin space is {e_DN, e_UP}
     // Zeeman field
     m(0, 0) += h;
     m(1, 1) += -h;
    
    for (int s1 = 0; s1 < 2; ++s1)
      for (int s2 = 0; s2 < 2; ++s2)
        H_0(s1, 0, s2, 0, k_ind) = m(s1, s2);



//    const auto val =
//        -2. * t * std::cos(k[0]) - 4. * t * std::cos(sqrt(3.) * k[1] / 2.) * std::cos(k[0] / 2.);

//    H_0(0, 0, 0, 0, k_ind) = val;
//    H_0(0, 1, 0, 1, k_ind) = val;
  }
}

}  // models
}  // phys
}  // dca

#endif  // DCA_PHYS_MODELS_ANALYTIC_HAMILTONIANS_MOIRE_HUBBARD_HPP
