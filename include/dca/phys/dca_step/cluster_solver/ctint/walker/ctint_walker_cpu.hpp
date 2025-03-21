// Copyright (C) 2023 ETH Zurich
// Copyright (C) 2023 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE.txt for terms of usage.
//  See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// This class organizes the MC walker in the CT-INT QMC purely on the CPU.

#ifndef DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_CTINT_WALKER_CTINT_WALKER_CPU_HPP
#define DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_CTINT_WALKER_CTINT_WALKER_CPU_HPP

#include "dca/phys/dca_step/cluster_solver/ctint/walker/ctint_walker_base.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "dca/linalg/linalg.hpp"
#include "dca/phys/dca_step/cluster_solver/ctint/walker/tools/d_matrix_builder.hpp"
#include "dca/phys/dca_step/cluster_solver/ctint/walker/tools/walker_methods.hpp"
#include "dca/phys/dca_step/cluster_solver/ctint/structs/solver_configuration.hpp"
#include "dca/phys/dca_step/cluster_solver/shared_tools/interpolation/g0_interpolation.hpp"
#include "dca/util/integer_division.hpp"

namespace dca {
namespace phys {
namespace solver {
namespace ctint {
// dca::phys::solver::ctint::

template <class Parameters, DistType DIST>
class CtintWalker<linalg::CPU, Parameters, DIST> : public CtintWalkerBase<Parameters, DIST> {
public:
  using this_type = CtintWalker<linalg::CPU, Parameters,DIST>;
  using BaseClass = CtintWalkerBase<Parameters,DIST>;
  using typename BaseClass::Real;
  using typename BaseClass::Scalar;
  
  using typename BaseClass::Rng;
  using typename BaseClass::Data;

  using Matrix = typename BaseClass::Matrix;
  using MatrixPair = typename BaseClass::MatrixPair;
  using MatrixView = typename linalg::MatrixView<Scalar, linalg::CPU>;
  using ConstView = typename linalg::MatrixView<const Scalar, linalg::CPU>;

  using Resource = DMatrixBuilder<linalg::CPU, Scalar>;

public:
  CtintWalker(const Parameters& pars_ref, const Data& /*data*/, Rng& rng_ref, DMatrixBuilder<linalg::CPU, Scalar>& d_matrix_builder, int id = 0);

  void doSweep() override;

  void markThermalized() override;
protected:
  void doStep() override;
  /** try to insertVertex, if accepted do it.
   *  on accept applyInsertion is called.
   */
  bool tryVertexInsert();
  bool tryVertexRemoval();

  void moveRemovalToEnd();
  void popBack(int delta_vertices);

  // Test handle.
  const auto& getM() const {
    return M_;
  }

  void initializeStep();

  void setMFromConfig() override;
private:
  auto insertionProbability(int delta_vertices);

  void applyInsertion(const MatrixPair& S, const MatrixPair& Q, const MatrixPair& R);

  auto removalProbability();
  void applyRemoval();

  virtual void smallInverse(const ConstView& in, MatrixView& out, int s);
  virtual void smallInverse(MatrixView& in_out, int s);

protected:
  using BaseClass::parameters_;
  using BaseClass::configuration_;
  using BaseClass::rng_;
  DMatrixBuilder<linalg::CPU, Scalar>& d_matrix_builder_;
  using BaseClass::total_interaction_;
  using BaseClass::beta_;
  using BaseClass::phase_;
  using BaseClass::thread_id_;

  using BaseClass::thermalized_;
  using BaseClass::nb_steps_per_sweep_;
  using BaseClass::n_accepted_;

  using BaseClass::M_;
  // For testing purposes.
  using BaseClass::acceptance_prob_;
  std::array<Scalar, 2> det_ratio_;
  using BaseClass::sweeps_per_meas_;
  using BaseClass::partial_order_avg_;
  using BaseClass::thermalization_steps_;
  using BaseClass::order_avg_;
  using BaseClass::sign_avg_;
  using BaseClass::n_steps_;
  using BaseClass::mc_log_weight_;
private:
  std::array<linalg::Matrix<Scalar, linalg::CPU>, 2> S_, Q_, R_;
  // work spaces
  MatrixPair M_Q_;
  Matrix ws_dn_;
  linalg::Vector<Scalar, linalg::CPU> v_work_;
  linalg::Vector<int, linalg::CPU> ipiv_;

  std::array<linalg::util::HostVector<int>, 2> matrix_removal_list_;
  std::array<linalg::util::HostVector<int>, 2> matrix_source_list_;
  std::vector<int> removal_list_;

  // Number of random values to use to maintain testing consistency with the submatrix walker.
  const unsigned n_removal_rngs_;
};

template <class Parameters, DistType DIST>
CtintWalker<linalg::CPU, Parameters,DIST>::CtintWalker(const Parameters& parameters_ref,
						       const Data& /*data*/, Rng& rng_ref, DMatrixBuilder<linalg::CPU, Scalar>& d_matrix_builder, int id)
    : BaseClass(parameters_ref, rng_ref, id),
      det_ratio_{1, 1},
      d_matrix_builder_(d_matrix_builder),
      // If we perform double updates, we need at most 3 rng values for: selecting the first vertex,
      // deciding if we select a second one, select the second vertex. Otherwise only the first is
      // needed.
      n_removal_rngs_(configuration_.getDoubleUpdateProb() ? 3 : 1) {}

template <class Parameters, DistType DIST>
void CtintWalker<linalg::CPU, Parameters,DIST>::markThermalized() {
  thermalized_ = true;

  nb_steps_per_sweep_ = std::max(1., std::ceil(sweeps_per_meas_ * partial_order_avg_.mean()));
  thermalization_steps_ = n_steps_;

  order_avg_.reset();
  sign_avg_.reset();
  n_accepted_ = 0;

  // Recompute the Monte Carlo weight.
  setMFromConfig();
#ifndef NDEBUG
  //writeAlphas();
#endif
}
  
template <class Parameters, DistType DIST>
void CtintWalker<linalg::CPU, Parameters,DIST>::doSweep() {
  int nb_of_steps;
  if (nb_steps_per_sweep_ < 0)  // Not thermalized or fixed.
    nb_of_steps = BaseClass::avgOrder() + 1;
  else
    nb_of_steps = nb_steps_per_sweep_;

  for (int i = 0; i < nb_of_steps; i++) {
    doStep();
  }

  BaseClass::n_steps_ += nb_of_steps;
  BaseClass::updateSweepAverages();
}

template <class Parameters, DistType DIST>
void CtintWalker<linalg::CPU, Parameters,DIST>::doStep() {
  initializeStep();

  if (rng_() <= 0.5) {
    n_accepted_ += tryVertexInsert();
  }
  else {
    if (configuration_.size())
      n_accepted_ += tryVertexRemoval();
    else {
      // Burn random numbers for testing consistency.
      // One extra rng for the acceptance probability.
      for (unsigned i = 0; i < n_removal_rngs_ + 1; ++i)
        rng_();
    }
  }

  assert(configuration_.checkConsistency());
}

template <class Parameters, DistType DIST>
bool CtintWalker<linalg::CPU, Parameters,DIST>::tryVertexInsert() {
  configuration_.insertRandom(rng_);
  const int delta_vertices = configuration_.lastInsertionSize();

  // Compute the new pieces of the D(= M^-1) matrix.
  d_matrix_builder_.buildSQR(S_, Q_, R_, configuration_);

  acceptance_prob_ = insertionProbability(delta_vertices);

  const bool accept = rng_() < std::min(std::abs(acceptance_prob_), Real(1.));

  if (!accept) {
    popBack(delta_vertices);
  }

  else {
    phase_.multiply(acceptance_prob_);
    applyInsertion(S_, Q_, R_);

    auto mc_weight_term = det_ratio_[0] * det_ratio_[1];
    for (int i = 0; i < delta_vertices; ++i)
      mc_weight_term *= -configuration_.getStrength(configuration_.size() - 1 - i);

    BaseClass::mc_log_weight_ += std::log(std::abs(mc_weight_term));
  }
  return accept;
}

template <class Parameters, DistType DIST>
bool CtintWalker<linalg::CPU, Parameters,DIST>::tryVertexRemoval() {
  acceptance_prob_ = removalProbability();
  const bool accept = rng_() < std::min(std::abs(acceptance_prob_), Real(1.));

  if (accept) {
    phase_.multiply(acceptance_prob_);
    
    auto mc_weight_term = det_ratio_[0] * det_ratio_[1];
    for (auto idx : removal_list_)
      mc_weight_term /= -configuration_.getStrength(idx);

    BaseClass::mc_log_weight_ += std::log(std::abs(mc_weight_term));

    applyRemoval();
  }
  return accept;
}

template <class Parameters, DistType DIST>
auto CtintWalker<linalg::CPU, Parameters,DIST>::insertionProbability(const int delta_vertices) {
  const int old_size = configuration_.size() - delta_vertices;

  for (int s = 0; s < 2; ++s) {
    const auto& Q = Q_[s];
    if (Q.nrCols() == 0) {
      det_ratio_[s] = 1.;
      continue;
    }
    const auto& R = R_[s];
    auto& S = S_[s];
    auto& M = M_[s];

    if (M.nrRows()) {
      auto& M_Q = M_Q_[s];
      M_Q.resizeNoCopy(Q.size());
      linalg::matrixop::gemm(M, Q, M_Q);
      // S <- S_tilde^(-1) = S - R*M*Q
      linalg::matrixop::gemm(Scalar(-1.), R, M_Q, Scalar(1.), S);
    }

    det_ratio_[s] = details::smallDeterminant(S);
  }

  const Real combinatorial_factor =
      delta_vertices == 1 ? old_size + 1 : (old_size + 2) * (configuration_.nPartners(old_size) + 1);

  const Real strength_factor = delta_vertices == 1
                                   ? -beta_ * total_interaction_ * configuration_.getSign(old_size)
                                   : total_interaction_ * beta_ * beta_ *
                                         std::abs(configuration_.getStrength(old_size)) *
                                         configuration_.possiblePartners(old_size);

  const Scalar det_ratio = det_ratio_[0] * det_ratio_[1];

  return det_ratio * strength_factor / combinatorial_factor;

  //  return details::computeAcceptanceProbability(
  //      delta_vertices, det_ratio_[0] * det_ratio_[1], total_interaction_, beta_,
  //      configuration_.getStrength(old_size), combinatorial_factor, details::VERTEX_INSERTION);
}

template <class Parameters, DistType DIST>
auto CtintWalker<linalg::CPU, Parameters,DIST>::removalProbability() {
  std::array<double, 3> removal_rngs;
  for (unsigned i = 0; i < n_removal_rngs_; ++i)
    removal_rngs[i] = rng_();

  const auto candidates = configuration_.randomRemovalCandidate(removal_rngs);
  removal_list_.clear();
  for (int candidate : candidates) {
    if (candidate != -1)
      removal_list_.push_back(candidate);
  }

  const int n = configuration_.size();
  const int n_removed = removal_list_.size();

  for (int s = 0; s < 2; ++s) {
    matrix_removal_list_[s].clear();
    for (auto index : removal_list_) {
      configuration_.findIndices(matrix_removal_list_[s], index, s);
    }
    if (matrix_removal_list_[s].size() == 0) {
      det_ratio_[s] = 1.;
      continue;
    }
    std::sort(matrix_removal_list_[s].begin(), matrix_removal_list_[s].end());
    det_ratio_[s] = details::separateIndexDeterminant(M_[s], matrix_removal_list_[s]);
  }
  assert((matrix_removal_list_[0].size() + matrix_removal_list_[1].size()) % 2 == 0);

  const Real combinatorial_factor =
      n_removed == 1 ? n : n * configuration_.nPartners(removal_list_[0]);
  const Real strength_factor =
      n_removed == 1
          ? -beta_ * total_interaction_ * configuration_.getSign(removal_list_[0])
          : total_interaction_ * beta_ * beta_ * configuration_.possiblePartners(removal_list_[0]) *
                std::abs(configuration_.getStrength(removal_list_[0]));

  const Scalar det_ratio = det_ratio_[0] * det_ratio_[1];

  return det_ratio * combinatorial_factor / strength_factor;
}

template <class Parameters, DistType DIST>
void CtintWalker<linalg::CPU, Parameters,DIST>::applyInsertion(const MatrixPair& Sp,
                                                                const MatrixPair& Qp,
                                                                const MatrixPair& Rp) {
  for (int s = 0; s < 2; ++s) {
    const int delta = Qp[s].nrCols();
    if (not delta)
      continue;
    // update M matrix.
    const auto& R = Rp[s];
    const ConstView S(Sp[s]);
    auto& M = M_[s];
    const auto& M_Q = M_Q_[s];
    const int m_size = M.nrCols();

    if (not m_size) {
      M.resizeNoCopy(delta);
      auto M_view = MatrixView(M);
      smallInverse(S, M_view, s);
      continue;
    }

    auto& R_M = ws_dn_;
    R_M.resizeNoCopy(R.size());
    linalg::matrixop::gemm(R, M, R_M);

    M.resize(m_size + delta);
    //  S_tilde = S^-1.
    MatrixView S_tilde(M, m_size, m_size, delta, delta);
    smallInverse(S, S_tilde, s);

    // R_tilde = - S * R * M
    MatrixView R_tilde(M, m_size, 0, delta, m_size);
    linalg::matrixop::gemm(Scalar(-1.), S_tilde, R_M, Scalar(0.), R_tilde);

    // Q_tilde = -M * Q * S
    MatrixView Q_tilde(M, 0, m_size, m_size, delta);
    linalg::matrixop::gemm(Scalar(-1.), M_Q, S_tilde, Scalar(0.), Q_tilde);

    // update bulk: M += M*Q*S*R*M
    MatrixView M_bulk(M, 0, 0, m_size, m_size);
    linalg::matrixop::gemm(Scalar(-1.), Q_tilde, R_M, Scalar(1.), M_bulk);
  }

  const int delta_vertices = configuration_.lastInsertionSize();
  for (int i = 0; i < delta_vertices; ++i)
    configuration_.commitInsertion(configuration_.size() - delta_vertices + i);
}

template <class Parameters, DistType DIST>
void CtintWalker<linalg::CPU, Parameters,DIST>::applyRemoval() {
  for (auto idx : removal_list_)
    configuration_.markForRemoval(idx);

  std::array<int, 2> removal_size;
  for (int s = 0; s < 2; ++s)
    removal_size[s] = configuration_.getSector(s).size();

  moveRemovalToEnd();

  for (int s = 0; s < 2; ++s)
    removal_size[s] -= configuration_.getSector(s).size();

  for (int s = 0; s < 2; ++s) {
    auto& M = M_[s];
    const int delta = removal_size[s];
    if (delta == 0)  // Nothing to update.
      continue;
    const int m_size = M.nrCols() - delta;
    if (m_size == 0) {  // removing last vertex
      M.resize(0);
      continue;
    }

    MatrixView Q(M, 0, m_size, m_size, delta);
    MatrixView R(M, m_size, 0, delta, m_size);

    MatrixView S(M, m_size, m_size, delta, delta);
    smallInverse(S, s);

    auto& Q_S = M_Q_[s];
    Q_S.resizeNoCopy(Q.size());
    linalg::matrixop::gemm(Q, S, Q_S);

    // M -= Q*S^-1*R
    MatrixView M_bulk(M, 0, 0, m_size, m_size);
    linalg::matrixop::gemm(Scalar(-1.), Q_S, R, Scalar(1.), M_bulk);
    M.resize(m_size);
  }
}

template <class Parameters, DistType DIST>
void CtintWalker<linalg::CPU, Parameters,DIST>::popBack(int delta_vertices) {
  for (int i = configuration_.size() - delta_vertices; i < configuration_.size(); ++i) {
    removal_list_.push_back(i);
  }

  configuration_.moveAndShrink(matrix_source_list_, matrix_removal_list_, removal_list_);

  for (int s = 0; s < 2; ++s) {
    M_[s].resize(configuration_.getSector(s).size());
  }
}

template <class Parameters, DistType DIST>
void CtintWalker<linalg::CPU, Parameters,DIST>::moveRemovalToEnd() {
  configuration_.moveAndShrink(matrix_source_list_, matrix_removal_list_, removal_list_);

  for (int s = 0; s < 2; ++s) {
    for (int i = 0; i < matrix_source_list_[s].size(); ++i)
      linalg::matrixop::swapRowAndCol(M_[s], matrix_source_list_[s][i], matrix_removal_list_[s][i]);
  }
}

template <class Parameters, DistType DIST>
void CtintWalker<linalg::CPU, Parameters,DIST>::smallInverse(const ConstView& in, MatrixView& out,
                                                              const int s) {
  details::smallInverse(in, out, det_ratio_[s], ipiv_, v_work_);
}

template <class Parameters, DistType DIST>
void CtintWalker<linalg::CPU, Parameters,DIST>::smallInverse(MatrixView& in_out, const int s) {
  details::smallInverse(in_out, det_ratio_[s], ipiv_, v_work_);
}

template <class Parameters, DistType DIST>
void CtintWalker<linalg::CPU, Parameters,DIST>::initializeStep() {
  removal_list_.clear();
  for (int s = 0; s < 2; ++s) {
    matrix_removal_list_[s].clear();
    matrix_source_list_[s].clear();
  }
}

template <class Parameters, DistType DIST>
void CtintWalker<linalg::CPU, Parameters, DIST>::setMFromConfig() {
  BaseClass::setMFromConfigImpl(d_matrix_builder_);
}

  
}  // namespace ctint
}  // namespace solver
}  // namespace phys
}  // namespace dca

#endif  // DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_CTINT_WALKER_CTINT_WALKER_CPU_HPP
