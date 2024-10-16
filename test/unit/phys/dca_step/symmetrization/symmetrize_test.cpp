// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Test the symmetries of the non interacting Greens function.

#include "dca/platform/dca_gpu.h"
#include "dca/phys/dca_step/symmetrization/symmetrize.hpp"

#include "dca/testing/gtest_h_w_warning_blocking.h"

#include "dca/config/threading.hpp"
#include "dca/io/json/json_reader.hpp"
#include "dca/function/util/real_complex_conversion.hpp"
#include "dca/parallel/no_concurrency/no_concurrency.hpp"
#include "dca/phys/dca_algorithms/compute_free_greens_function.hpp"
#include "dca/phys/domains/cluster/symmetries/point_groups/no_symmetry.hpp"
#include "dca/phys/parameters/parameters.hpp"
#include "dca/profiling/null_profiler.hpp"
#include "dca/phys/models/analytic_hamiltonians/twoband_chain.hpp"
#include "dca/phys/models/tight_binding_model.hpp"
#include "dca/phys/parameters/num_traits.hpp"

const std::string input_dir = DCA_SOURCE_DIR "/test/unit/phys/dca_step/symmetrization/";

using Concurrency = dca::parallel::NoConcurrency;
using Model = dca::phys::models::TightBindingModel<
    dca::phys::models::twoband_chain<dca::phys::domains::no_symmetry<2>>>;

using dca::func::dmn_0;
using dca::func::dmn_variadic;
using dca::func::function;
using BDmn = dmn_0<dca::phys::domains::electron_band_domain>;
using SDmn = dmn_0<dca::phys::domains::electron_spin_domain>;
using NuDmn = dmn_variadic<BDmn, SDmn>;

template<typename T>
class SymmetrizeTest : public ::testing::Test {
protected:
  using Parameters = dca::phys::params::Parameters<Concurrency, Threading, dca::profiling::NullProfiler,
						   Model, void, dca::ClusterSolverId::CT_AUX, dca::NumericalTraits<dca::util::RealAlias<T>, T>>;

  using Lattice = typename Parameters::lattice_type;

  using RClusterDmn = typename Parameters::RClusterDmn;
  using KClusterDmn = typename Parameters::KClusterDmn;
  using TDmn = typename Parameters::TDmn;
  using WDmn = typename Parameters::WDmn;
  
  SymmetrizeTest() : concurrency_(0, nullptr), parameters_("", concurrency_) {
    parameters_.template read_input_and_broadcast<dca::io::JSONReader>(input_dir + "input.json");
    parameters_.update_model();

    static bool domain_initialized = false;
    if (!domain_initialized) {
      parameters_.update_domains();

      H0_.reset();
      H_symmetry_.reset();
      domain_initialized = true;
    }

    Parameters::model_type::initializeH0(parameters_, H0_);
    Parameters::model_type::initialize_H_symmetries(H_symmetry_);
  }

protected:
  Concurrency concurrency_;
  Parameters parameters_;

  function<std::complex<double>, dmn_variadic<NuDmn, NuDmn, KClusterDmn>> H0_;
  function<int, dmn_variadic<NuDmn, NuDmn>> H_symmetry_;
};

using MyTypes = ::testing::Types<double>;
TYPED_TEST_CASE(SymmetrizeTest, MyTypes);

TYPED_TEST(SymmetrizeTest, G0_t) {
  // Compute the Green's functions in imaginary time.
  using KClusterDmn = typename SymmetrizeTest<TypeParam>::KClusterDmn;
  using RClusterDmn = typename SymmetrizeTest<TypeParam>::RClusterDmn;
  using TDmn = typename SymmetrizeTest<TypeParam>::TDmn;
  
  function<std::complex<double>, dmn_variadic<NuDmn, NuDmn, KClusterDmn, TDmn>> G0_k_t;

  dca::phys::compute_G0_k_t(this->H0_, this->parameters_.get_chemical_potential(), this->parameters_.get_beta(),
                            G0_k_t);

  function<std::complex<double>, dmn_variadic<NuDmn, NuDmn, RClusterDmn, TDmn>> G0_r_t_cmplx;
  dca::math::transform::FunctionTransform<KClusterDmn, RClusterDmn>::execute(G0_k_t, G0_r_t_cmplx);
  function<double, dmn_variadic<NuDmn, NuDmn, RClusterDmn, TDmn>> G0_r_t;
  G0_r_t = dca::func::util::real(G0_r_t_cmplx, dca::ImagCheck::FAIL);

  using Parameters = typename SymmetrizeTest<TypeParam>::Parameters;
  // Test the symmetrization.
  dca::phys::Symmetrize<Parameters>::execute(G0_k_t, this->H_symmetry_, true);
  dca::phys::Symmetrize<Parameters>::execute(G0_r_t, this->H_symmetry_, true);

  EXPECT_FALSE(dca::phys::Symmetrize<Parameters>::differenceDetected());
}

TYPED_TEST(SymmetrizeTest, G0_w) {
  using KClusterDmn = typename SymmetrizeTest<TypeParam>::KClusterDmn;
  using RClusterDmn = typename SymmetrizeTest<TypeParam>::RClusterDmn;
  using WDmn = typename SymmetrizeTest<TypeParam>::WDmn;

  // Compute the Green's functions in imaginary time.
  function<std::complex<double>, dmn_variadic<NuDmn, NuDmn, KClusterDmn, WDmn>> G0_k_w;

  dca::phys::compute_G0_k_w(this->H0_, this->parameters_.get_chemical_potential(), 1, G0_k_w);

  function<std::complex<double>, dmn_variadic<NuDmn, NuDmn, RClusterDmn, WDmn>> G0_r_w;
  dca::math::transform::FunctionTransform<KClusterDmn, RClusterDmn>::execute(G0_k_w, G0_r_w);

  using Parameters = typename SymmetrizeTest<TypeParam>::Parameters;
  // Test the symmetrization.
  dca::phys::Symmetrize<Parameters>::execute(G0_k_w, this->H_symmetry_, true);
  dca::phys::Symmetrize<Parameters>::execute(G0_r_w, this->H_symmetry_, true);

  EXPECT_FALSE(dca::phys::Symmetrize<Parameters>::differenceDetected());
}

