// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
// See LICENSE.txt for terms of usage./
//  See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Wrapper to access protected members of the CT-INT walker inside the tests.

#ifndef TEST_UNIT_PHYS_DCA_STEP_CLUSTER_SOLVER_CTINT_WALKER_WALKER_WRAPPER_HPP
#define TEST_UNIT_PHYS_DCA_STEP_CLUSTER_SOLVER_CTINT_WALKER_WALKER_WRAPPER_HPP

#include "dca/distribution/dist_types.hpp"
#include "dca/linalg/device_type.hpp"
#include "dca/phys/dca_step/cluster_solver/ctint/walker/ctint_walker_cpu.hpp"
#include "dca/phys/dca_data/dca_data.hpp"

namespace testing {
namespace phys {
namespace solver {
namespace ctint {
// testing::phys::solver::ctint::

using dca::DistType;

using namespace dca::phys::solver::ctint;
  template <typename SCALAR, class Parameters, DistType DIST = DistType::NONE>
class WalkerWrapper : public CtintWalker<dca::linalg::CPU, Parameters, DIST> {
public:
  using Base = CtintWalker<dca::linalg::CPU, Parameters, DIST>;
  using Scalar = SCALAR;
  using Real = dca::util::RealAlias<Scalar>;
  using Rng = typename Base::Rng;

    WalkerWrapper(Parameters& parameters_ref, Rng& rng_ref, DMatrixBuilder<dca::linalg::CPU, Scalar>& d_matrix_builder)
      : Base(parameters_ref, dca::phys::DcaData<Parameters>(parameters_ref), rng_ref, d_matrix_builder, 0) {
    Base::initialize(0);
  }

  using Base::doStep;

  bool tryVertexInsert() {
    Base::initializeStep();
    return Base::tryVertexInsert();
  }
  bool tryVertexRemoval() {
    Base::initializeStep();
    return Base::tryVertexRemoval();
  }

  using Base::setMFromConfig;
  using Base::getM;

  using Matrix = dca::linalg::Matrix<Scalar, dca::linalg::CPU>;

  void setM(const Matrix& m) {
    Base::getM() = m;
  }

  Real getRatio() const {
    return Base::det_ratio_[0] * Base::det_ratio_[1];
  }

  auto getAcceptanceProbability() const {
    return Base::acceptance_prob_;
  }

  const auto& getWalkerConfiguration() const {
    return Base::configuration_;
  }

private:
  using Base::configuration_;
};

}  // namespace ctint
}  // namespace solver
}  // namespace phys
}  // namespace testing

#endif  //  TEST_UNIT_PHYS_DCA_STEP_CLUSTER_SOLVER_CTINT_WALKER_WALKER_WRAPPER_HPP
