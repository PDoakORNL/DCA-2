// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// A std::thread jacket that measures correlations in the MC walker independent of the MC method.

#ifndef DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_STDTHREAD_QMCI_STDTHREAD_QMCI_WALKER_HPP
#define DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_STDTHREAD_QMCI_STDTHREAD_QMCI_WALKER_HPP

#include <mutex>

#include "dca/io/writer.hpp"
#include "dca/math/statistics/autocorrelation.hpp"
#include "dca/phys/dca_step/cluster_solver/shared_tools/accumulation/time_correlator.hpp"
#include "dca/phys/dca_step/cluster_solver/ss_ct_hyb/ss_ct_hyb_walker.hpp"
#include "dca/phys/dca_step/cluster_solver/stdthread_qmci/qmci_autocorrelation_data.hpp"

namespace dca {
namespace phys {
namespace solver {
namespace stdthreadqmci {
// dca::phys::solver::stdthreadqmci::

template <class QmciWalker>
class StdThreadQmciWalker final : public QmciWalker, public QmciAutocorrelationData<QmciWalker> {
  using ThisType = StdThreadQmciWalker<QmciWalker>;
  using Parameters = typename QmciWalker::parameters_type;
  using Data = DcaData<Parameters>;
  using Concurrency = typename Parameters::concurrency_type;
  using Rng = typename Parameters::random_number_generator;
  using Real = typename QmciWalker::Scalar;

  constexpr static auto device = QmciWalker::device;
  constexpr static int bands = Parameters::bands;

public:
  StdThreadQmciWalker(const Parameters& parameters_ref, Data& data_ref, Rng& rng, int id,
                      const std::shared_ptr<io::Writer>& writer);

  void initialize(int iteration_);
  void doSweep();

  void markThermalized() {
    QmciAutocorrelationData<QmciWalker>::markThermalized();
    QmciWalker::markThermalized();
  }

private:
  void logConfiguration() const;

  const unsigned stamping_period_;
  int thread_id_;

  std::shared_ptr<io::Writer> writer_;
  std::size_t meas_id_ = 0;

  const int total_iterations_;

  bool last_iteration_ = false;
};

template <class QmciWalker>
StdThreadQmciWalker<QmciWalker>::StdThreadQmciWalker(const Parameters& parameters,
                                                     Data& data_ref, Rng& rng, const int id,
                                                     const std::shared_ptr<io::Writer>& writer)
    : QmciWalker(parameters, data_ref, rng, id),
      QmciAutocorrelationData<QmciWalker>(parameters, id),
      stamping_period_(parameters.stamping_period()),
      thread_id_(id),
      writer_(writer),
      total_iterations_(parameters.get_dca_iterations()) {}

template <class QmciWalker>
void StdThreadQmciWalker<QmciWalker>::initialize(int iteration) {
  QmciWalker::initialize(iteration);

  meas_id_ = 0;
  last_iteration_ = iteration == total_iterations_ - 1;
}

template <class QmciWalker>
void StdThreadQmciWalker<QmciWalker>::doSweep() {
  QmciWalker::doSweep();
  QmciAutocorrelationData<QmciWalker>::accumulateAutocorrelation(*this);

  if (QmciWalker::is_thermalized()) {
    if (last_iteration_)
      logConfiguration();
    ++meas_id_;
  }
}

template <class QmciWalker>
void StdThreadQmciWalker<QmciWalker>::logConfiguration() const {
  const bool print_to_log = writer_ && static_cast<bool>(*writer_);  // File exists and it is open.
  if (print_to_log && last_iteration_ && stamping_period_ && (meas_id_ % stamping_period_) == 0) {
    const std::string stamp_name =
        "w_" + std::to_string(thread_id_) + "_step_" + std::to_string(meas_id_);

    writer_->lock();

    writer_->open_group("Configurations");
    const auto& config = QmciWalker::get_configuration();
    config.write(*writer_, stamp_name);
    writer_->close_group();

    writer_->unlock();
  }
}

// No additional ops specialization for SS-HYB
template <linalg::DeviceType device, class Parameters, class Data>
class StdThreadQmciWalker<cthyb::SsCtHybWalker<device, Parameters, Data>>
    : public cthyb::SsCtHybWalker<device, Parameters, Data>,
      public QmciAutocorrelationData<cthyb::SsCtHybWalker<device, Parameters, Data>> {
  using QmciWalker = cthyb::SsCtHybWalker<device, Parameters, Data>;
  using ThisType = StdThreadQmciWalker<QmciWalker>;
  using Rng = typename Parameters::random_number_generator;

public:
  StdThreadQmciWalker(const Parameters& parameters_ref, Data& data_ref, Rng& rng, int id,
                      const std::shared_ptr<io::HDF5Writer>& /*writer*/)
      : QmciWalker(parameters_ref, data_ref, rng, id),
        QmciAutocorrelationData<cthyb::SsCtHybWalker<device, Parameters, Data>>(parameters_ref, id) {
  }

  static void write(io::HDF5Writer& /*writer*/) {}
};

}  // namespace stdthreadqmci
}  // namespace solver
}  // namespace phys
}  // namespace dca

#endif  // DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_STDTHREAD_QMCI_STDTHREAD_QMCI_WALKER_HPP