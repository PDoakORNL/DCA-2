// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE.txt for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Authors: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Bilayer lattice with spin flip  and pair hopping term.

#ifndef DCA_PHYS_MODELS_ANALYTIC_HAMILTONIANS_HUND_LATTICE_HPP
#define DCA_PHYS_MODELS_ANALYTIC_HAMILTONIANS_HUND_LATTICE_HPP

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dca/function/domains.hpp"
#include "dca/function/function.hpp"
#include "dca/phys/dca_step/cluster_solver/ctint/structs/interaction_vertices.hpp"
#include "dca/phys/domains/cluster/symmetries/point_groups/no_symmetry.hpp"
#include "dca/phys/models/analytic_hamiltonians/bilayer_lattice.hpp"
#include "dca/phys/models/analytic_hamiltonians/cluster_shape_type.hpp"
#include "dca/phys/models/traits.hpp"
#include "dca/util/type_list.hpp"

namespace dca {
namespace phys {
namespace models {
// dca::phys::models::

template <typename point_group_type>
class HundLattice : public bilayer_lattice<point_group_type> {
public:
  static constexpr bool complex_g0 = false;
  static constexpr bool spin_symmetric = true;

  using BaseClass = bilayer_lattice<point_group_type>;
  constexpr static int BANDS = BaseClass::BANDS;
  constexpr static int DIMENSION = BaseClass::DIMENSION;
  
  // Initializes the interaction Hamiltonian non density-density local term.
  template <typename Scalar, class Parameters>
  static void initializeNonDensityInteraction(
      NonDensityIntHamiltonian<Scalar, Parameters>& non_density_interaction,
      const Parameters& parameters);

  template <class domain>
  static void initializeHSymmetry(func::function<int, domain>& H_symmetry);

  template <class Parameters>
  static void printNonDensityType([[maybe_unused]] Parameters& pars) {
    dca::util::print_type<NonDensityIntHamiltonian<typename Parameters::Scalar, Parameters>> printer;
    printer.print();
  }
  
};

template <typename point_group_type>
template <typename Scalar, class Parameters>
void HundLattice<point_group_type>::initializeNonDensityInteraction(
    NonDensityIntHamiltonian<Scalar, Parameters>& non_density_interaction,
    const Parameters& parameters) {
  const double Jh = parameters.get_Jh();
  const NuDmn nu;  // band-spin domain.
  constexpr int up(0), down(1);

  non_density_interaction = dca::util::TheZero<Scalar>::value;
  for (int b1 = 0; b1 < BANDS; b1++)
    for (int b2 = 0; b2 < BANDS; b2++) {
      if (b1 == b2)
        continue;
      non_density_interaction(nu(b1, up), nu(b2, up), nu(b2, down), nu(b1, down), 0) = Jh;
      non_density_interaction(nu(b1, up), nu(b2, up), nu(b1, down), nu(b2, down), 0) = Jh;
    }
}

// TODO: check.
template <typename point_group_type>
template <class domain>
void HundLattice<point_group_type>::initializeHSymmetry(func::function<int, domain>& H_symmetries) {
  H_symmetries = -1;

  H_symmetries(0, 0, 0, 0) = 0;
  H_symmetries(0, 1, 0, 1) = 0;

  H_symmetries(1, 0, 1, 0) = 1;
  H_symmetries(1, 1, 1, 1) = 1;
}

}  // namespace models
}  // namespace phys
}  // namespace dca

#endif  // DCA_PHYS_MODELS_ANALYTIC_HAMILTONIANS_HUND_LATTICE_HPP
