// Copyright (C) 2023 ETH Zurich
// Copyright (C) 2023 UT-Battelle, LLC
// All rights reserved.
// See LICENSE.txt for terms of usage./
// See CITATION.txt for citation guidelines if you use this code for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//         Weile Wei (wwei9@lsu.edu)
//         Peter Doak (doakpw@ornl.gov)
//
// Implementation of the two particle Green's function computation on the GPU.

#ifndef DCA_TP_ACCUMULATOR_GPU_HPP
#define DCA_TP_ACCUMULATOR_GPU_HPP

#include "dca/phys/dca_step/cluster_solver/shared_tools/accumulation/tp/tp_accumulator_base.hpp"
#include "dca/phys/dca_step/cluster_solver/shared_tools/accumulation/tp/tp_accumulator_gpu_base.hpp"

#if defined(DCA_HAVE_GPU)
#include "dca/platform/dca_gpu.h"
#else
#error "This file requires GPU."
#endif

#include <algorithm>
#include <mutex>
#include <vector>

// its expected that dca::config::McOptions will be provided in some manner before parameters.hpp is
// included
#include "dca/distribution/dist_types.hpp"
#include "dca/linalg/blas/use_device.hpp"
#include "dca/linalg/lapack/magma.hpp"
#include "dca/linalg/reshapable_matrix.hpp"
#include "dca/linalg/util/allocators/managed_allocator.hpp"
#include "dca/linalg/util/gpu_event.hpp"
#include "dca/linalg/util/magma_queue.hpp"
#include "dca/math/util/vector_operations.hpp"
#include "dca/math/function_transform/special_transforms/space_transform_2D_gpu.hpp"
#include "dca/parallel/util/call_once_per_loop.hpp"
#include "dca/parallel/util/get_workload.hpp"
#include "dca/phys/domains/cluster/cluster_operations.hpp"
#include "dca/phys/dca_step/cluster_solver/shared_tools/accumulation/tp/kernels_interface.hpp"
#include "dca/phys/dca_step/cluster_solver/shared_tools/accumulation/tp/ndft/cached_ndft_cpu.hpp"
#include "dca/phys/dca_step/cluster_solver/shared_tools/accumulation/tp/ndft/cached_ndft_gpu.hpp"
#include "dca/util/integer_division.hpp"

#ifdef DCA_HAVE_MPI
#include "dca/parallel/mpi_concurrency/mpi_concurrency.hpp"
#include "dca/parallel/mpi_concurrency/mpi_type_map.hpp"
#endif

#include "dca/util/integer_division.hpp"

namespace dca {
namespace phys {
namespace solver {
namespace accumulator {
// dca::phys::solver::accumulator::
#ifdef DCA_HAVE_MPI
using dca::parallel::MPITypeMap;
#endif

template <class Parameters, DistType DT>
class TpAccumulator<Parameters, DT, linalg::GPU> : public TpAccumulatorBase<Parameters, DT>,
                                                   public TpAccumulatorGpuBase<Parameters, DT> {
public:
  static constexpr DistType dist = DT;
  using Base = TpAccumulatorBase<Parameters, DT>;
  using BaseGpu = TpAccumulatorGpuBase<Parameters, DT>;
  using ThisType = TpAccumulator<Parameters, DT, linalg::GPU>;
  using typename Base::TpPrecision;
  using typename Base::TpComplex;

  using RDmn = typename Base::RDmn;
  using KDmn = typename Base::KDmn;
  using NuDmn = typename Base::NuDmn;
  using WDmn = typename Base::WDmn;
  using typename Base::WTpDmn;
  using typename Base::WTpExtDmn;
  using typename Base::WTpExtPosDmn;
  using typename Base::WTpPosDmn;
  using typename Base::BDmn;
  using typename Base::SDmn;
  using typename Base::TpGreensFunction;
  using typename Base::KExchangeDmn;
  using typename Base::WExchangeDmn;

  using TpDomain =
      func::dmn_variadic<BDmn, BDmn, BDmn, BDmn, KDmn, KDmn, KExchangeDmn, WTpDmn, WTpDmn, WExchangeDmn>;

protected:
  using BaseGpu::queues_;
  using BaseGpu::ndft_objs_;
  using BaseGpu::workspaces_;
  using BaseGpu::G_;
  using BaseGpu::space_trsf_objs_;
  using BaseGpu::nr_accumulators_;
  using BaseGpu::event_;
  using typename BaseGpu::RMatrix;
  using typename BaseGpu::RMatrixValueType;

public:
  // Constructor:
  // In: G0: non interacting greens function.
  // In: pars: parameters object.
  // In: thread_id: thread id, only used by the profiler.
  TpAccumulator(const func::function<TpComplex, func::dmn_variadic<NuDmn, NuDmn, KDmn, WDmn>>& G0,
                const Parameters& pars, int thread_id = 0);

  // Resets the object between DCA iterations.
  void resetAccumulation(unsigned int dca_loop);

  // Resets the object between DCA iterations.
  void resetAccumulation(unsigned int dca_loop, dca::util::OncePerLoopFlag&);

  // Computes the two particles Greens function from the M matrix and accumulates it internally.
  // In: M_array: stores the M matrix for each spin sector.
  // In: configs: stores the walker's configuration for each spin sector.
  // In: sign: sign of the configuration.
  // Returns: number of flop.

  template <class Configuration, typename SpScalar, typename SignType>
  double accumulate(const std::array<linalg::Matrix<SpScalar, linalg::GPU>, 2>& M,
                    const std::array<Configuration, 2>& configs, const SignType factor);

  // CPU input. For testing purposes.
  template <class Configuration, typename SpScalar, typename SignType>
  double accumulate(const std::array<linalg::Matrix<SpScalar, linalg::CPU>, 2>& M,
                    const std::array<Configuration, 2>& configs, const SignType factor);

  // Downloads the accumulation result to the host.
  void finalize();

  // Sums on the host the accumulated Green's function to the accumulated Green's function of
  // other_acc.
  void sumTo(ThisType& other_acc);

  const linalg::util::GpuStream* get_stream() {
    return &queues_[0].getStream();
  }

  void synchronizeCopy() {
    ndft_objs_[0].synchronizeCopy();
    ndft_objs_[1].synchronizeCopy();
  }

  void syncStreams(const linalg::util::GpuEvent& event) {
    for (const auto& stream : queues_)
      event.block(stream);
  }

  std::size_t deviceFingerprint() const {
    std::size_t res(0);
    for (const auto& work : workspaces_)
      res += work->deviceFingerprint();
    for (int s = 0; s < 2; ++s)
      res += ndft_objs_[s].deviceFingerprint() + G_[s].deviceFingerprint() +
             space_trsf_objs_[s].deviceFingerprint();
    return res;
  }

  static std::size_t staticDeviceFingerprint() {
    std::size_t res = 0;

    for (const auto& G4_channel : get_G4Dev())
      res += G4_channel.deviceFingerprint();

    res += BaseGpu::get_G0()[0].deviceFingerprint() + BaseGpu::get_G0()[1].deviceFingerprint();

    return res;
  }

  // Returns the accumulated Green's function.
  const std::vector<typename TpAccumulator<Parameters, DT, linalg::GPU>::Base::TpGreensFunction>& get_G4()
      const;

#ifndef NDEBUG
  // Returns the accumulated Green's function.
  const auto& get_G_Debug();
#endif

  // FOR TESTING: Returns the accumulated Green's function.
  std::vector<TpGreensFunction>& get_nonconst_G4();

  using G4DevType = linalg::Vector<TpComplex, linalg::GPU, config::McOptions::TpAllocator<TpComplex>>;
  // Returns the accumulated Green's function.
  static inline std::vector<G4DevType>& get_G4Dev();

  void set_multiple_accumulators(bool have_multiple_accumulators) {
    multiple_accumulators_ = true;
  }

protected:
  using typename BaseGpu::Matrix;
  using typename BaseGpu::MatrixHost;
  using typename BaseGpu::RMatrix;
  using Base::beta_;
  using Base::channels_;
  using Base::extension_index_offset_;
  using Base::G4_;
  using Base::G0_;
  using Base::multiple_accumulators_;
  using Base::n_bands_;

  // using Base::n_pos_frqs_;

  using Base::non_density_density_;
  using Base::thread_id_;

  using BaseGpu::initialized_;
  using BaseGpu::finalized_;

  using BaseGpu::n_ndft_queues_;

  using BaseGpu::get_G0;

  void initializeG0();
  void resetG4();

  void computeG();

  void computeGMultiband(int s);

  void computeGSingleband(int s);

  template <class Configuration, typename SpScalar>
  void prepareFiniteQMultibandPPData(const std::array<linalg::Matrix<SpScalar, linalg::GPU>, 2>& M,
                                     const std::array<Configuration, 2>& configs);

  template <typename SignType>
  double updateG4(const std::size_t channel_index, SignType factor);

  template <typename SignType>
  double updateG4FiniteQMultibandPP(const std::size_t channel_index, SignType factor);

  void synchronizeStreams();
#ifdef DCA_HAVE_MPI
  struct RingMessage {
    int target;
    int source;
    MPI_Request request;
  };

  std::array<RingMessage, 2> recv_requests_{RingMessage{-1, -1, MPI_REQUEST_NULL}, RingMessage {
                                              -1,
                                              -1,
                                              MPI_REQUEST_NULL
                                            }};
  std::array<RingMessage, 2> send_requests_{RingMessage{-1, -1, MPI_REQUEST_NULL}, RingMessage {
                                              -1,
                                              -1,
                                              MPI_REQUEST_NULL
                                            }};
  // For distributed G4's
  // Applies pipepline ring algorithm to move G matrices around all ranks
  template <typename SignType>
  void ringG(double& flop, SignType Factor);

  void send(const std::array<RMatrix, 2>& data, std::array<RingMessage, 2>& request);
  void receive(std::array<RMatrix, 2>& data, std::array<RingMessage, 2>& request);

  // send buffer for pipeline ring algorithm
  std::array<RMatrix, 2> sendbuff_G_;
#endif

#ifndef NDEBUG
  std::array<
      linalg::ReshapableMatrix<TpComplex, linalg::CPU, dca::linalg::util::PinnedAllocator<TpComplex>>, 2>
      G_debug_;
#endif

#ifndef DCA_HAVE_GPU_AWARE_MPI
  std::array<std::vector<TpComplex>, 2> sendbuffer_;
  std::array<std::vector<TpComplex>, 2> recvbuffer_;
#endif  // DCA_HAVE_GPU_AWARE_MPI

private:
  using BandBlockView = linalg::MatrixView<TpComplex, linalg::CPU>;
  using RealSpaceSpGreensFunction =
      func::function<TpComplex,
                     func::dmn_variadic<RDmn, RDmn, BDmn, BDmn, SDmn, WTpExtPosDmn, WTpExtDmn>>;
  using HostNdftType =
      CachedNdft<TpComplex, RDmn, WTpExtDmn, WTpExtPosDmn, linalg::CPU,
                 Base::non_density_density_>;

  bool needsFiniteQMultibandPP() const;
  void getGMultibandHost(int s, int k1, int k2, int w1, int w2, MatrixHost& G,
                         TpComplex sign = 0) const;
  void getGMultibandDirectHost(int s, const std::vector<double>& k1_vec,
                               const std::vector<double>& k2_vec, int w1, int w2, MatrixHost& G,
                               TpComplex sign = 0) const;
  void buildMomentumBlockFromRealSpaceHost(int s, const std::vector<double>& k1_vec,
                                           const std::vector<double>& k2_vec, int w1, int w2,
                                           MatrixHost& M_k) const;
  void buildUnfoldedG0BlockHost(int s, const std::vector<double>& k_vec, int w,
                                MatrixHost& G0_k) const;
  static void matrixOperationsGMultibandHost(const BandBlockView& G0_kw1,
                                             const BandBlockView& G0_kw2, BandBlockView& G_kkww,
                                             MatrixHost& G0_M);

  mutable HostNdftType ndft_host_;
  mutable MatrixHost G0_M_host_;
  mutable MatrixHost G_a_host_;
  mutable MatrixHost G_b_host_;
  mutable MatrixHost G0_a_host_;
  mutable MatrixHost G0_b_host_;
  mutable MatrixHost M_unfolded_host_;
  mutable std::array<linalg::ReshapableMatrix<TpComplex, linalg::CPU>, 2> G_host_;
  mutable RealSpaceSpGreensFunction M_r_r_w_w_host_;
};

template <class Parameters, DistType DT>
TpAccumulator<Parameters, DT, linalg::GPU>::TpAccumulator(
    const func::function<TpComplex, func::dmn_variadic<NuDmn, NuDmn, KDmn, WDmn>>& G0,
    const Parameters& pars, const int thread_id)
    : Base(G0, pars, thread_id),
      BaseGpu(G0, pars, WTpExtDmn::dmn_size(), thread_id),
      ndft_host_(),
      G0_M_host_(n_bands_),
      G_a_host_(n_bands_),
      G_b_host_(n_bands_),
      G0_a_host_(n_bands_),
      G0_b_host_(n_bands_),
      M_unfolded_host_(n_bands_),
      M_r_r_w_w_host_("tp_acc_gpu::M_r_r_w_w_host_") {}

template <class Parameters, DistType DT>
void TpAccumulator<Parameters, DT, linalg::GPU>::resetAccumulation(const unsigned int dca_loop,
                                                                   dca::util::OncePerLoopFlag& flag) {
  dca::util::callOncePerLoop(flag, dca_loop, [&]() {
    resetG4();
    BaseGpu::initializeG0();
  });

  BaseGpu::synchronizeStreams();

  initialized_ = true;
  finalized_ = false;
}

template <class Parameters, DistType DT>
void TpAccumulator<Parameters, DT, linalg::GPU>::resetAccumulation(const unsigned int dca_loop) {
  static dca::util::OncePerLoopFlag flag;
  dca::util::callOncePerLoop(flag, dca_loop, [&]() {
    resetG4();
    BaseGpu::initializeG0();
    BaseGpu::synchronizeStreams();
  });

  initialized_ = true;
  finalized_ = false;
}

template <class Parameters, DistType DT>
void TpAccumulator<Parameters, DT, linalg::GPU>::resetG4() {
  // Note: this method is not thread safe by itself.
  get_G4Dev().resize(G4_.size());

  for (auto& G4_channel : get_G4Dev()) {
    try {
      TpDomain tp_dmn;
      dca::linalg::util::GpuStream reset_stream(cudaStreamLegacy);
      if (!multiple_accumulators_)
        reset_stream = queues_[0].getStream();

      G4_channel.setStream(reset_stream);
      G4_channel.resizeNoCopy(G4_[0].size());
      G4_channel.setToZero(reset_stream);
    }
    catch (std::bad_alloc& err) {
      std::cerr << "Failed to allocate G4 on device.\n";
      throw(err);
    }
  }
}

template <class Parameters, DistType DT>
template <class Configuration, typename SpScalar, typename SignType>
double TpAccumulator<Parameters, DT, linalg::GPU>::accumulate(
    const std::array<linalg::Matrix<SpScalar, linalg::GPU>, 2>& M,
    const std::array<Configuration, 2>& configs, const SignType factor) {
  [[maybe_unused]] Profiler profiler("accumulate", "tp-accumulation", __LINE__, thread_id_);
  double flop = 0;

  if (!BaseGpu::initialized_)
    throw(std::logic_error("The accumulator is not ready to measure."));

  if (!(configs[0].size() + configs[0].size()))  // empty config
    return flop;

  // Base::phase_ = factor;
  flop += BaseGpu::computeM(M, configs);

  computeG();

  if (needsFiniteQMultibandPP())
    prepareFiniteQMultibandPPData(M, configs);

#ifndef NDEBUG
  G_debug_[0] = G_[0];
  G_debug_[1] = G_[1];
#endif

  for (int channel_index = 0; channel_index < G4_.size(); ++channel_index) {
    flop += updateG4(channel_index, factor);
  }

#ifdef DCA_HAVE_MPI
  if constexpr (dist == DistType::LINEAR || dist == DistType::BLOCKED) {
    ringG(flop, factor);
  }
#endif
  return flop;
}

template <class Parameters, DistType DT>
template <class Configuration, typename SpScalar, typename SignType>
double TpAccumulator<Parameters, DT, linalg::GPU>::accumulate(
    const std::array<linalg::Matrix<SpScalar, linalg::CPU>, 2>& M,
    const std::array<Configuration, 2>& configs, const SignType factor) {
  std::array<linalg::Matrix<SpScalar, linalg::GPU>, 2> M_dev;
  for (int s = 0; s < 2; ++s)
    M_dev[s].setAsync(M[s], queues_[0].getStream());

  return accumulate(M_dev, configs, factor);
}

template <class Parameters, DistType DT>
void TpAccumulator<Parameters, DT, linalg::GPU>::computeG() {
  if (n_ndft_queues_ == 1) {
    event_.record(queues_[0]);
    event_.block(queues_[1]);
  }
  {
    [[maybe_unused]] Profiler prf("ComputeG: HOST", "tp-accumulation", __LINE__, thread_id_);
    for (int s = 0; s < 2; ++s) {
      if (n_bands_ == 1) {
        computeGSingleband(s);
      }
      else
        computeGMultiband(s);
    }
  }

  event_.record(queues_[1]);
  event_.block(queues_[0]);
}

template <class Parameters, DistType DT>
void TpAccumulator<Parameters, DT, linalg::GPU>::computeGSingleband(const int s) {
  details::computeGSingleband(G_[s].ptr(), G_[s].leadingDimension(), get_G0()[s].ptr(),
                              KDmn::dmn_size(), WTpExtDmn::dmn_size(), beta_, queues_[s], s);
  assert(cudaPeekAtLastError() == cudaSuccess);
}

template <class Parameters, DistType DT>
void TpAccumulator<Parameters, DT, linalg::GPU>::computeGMultiband(const int s) {
  // std::cout << "WTpExtDmn::dmn_size(): " << WTpExtDmn::dmn_size() << '\n';
  details::computeGMultiband(G_[s].ptr(), G_[s].leadingDimension(), get_G0()[s].ptr(),
                             get_G0()[s].leadingDimension(), n_bands_, KDmn::dmn_size(),
                             WTpExtDmn::dmn_size(), beta_, queues_[s]);
  assert(cudaPeekAtLastError() == cudaSuccess);
}

template <class Parameters, DistType DT>
bool TpAccumulator<Parameters, DT, linalg::GPU>::needsFiniteQMultibandPP() const {
  if (n_bands_ <= 1)
    return false;

  const bool has_pp_channel =
      std::find(channels_.begin(), channels_.end(), FourPointType::PARTICLE_PARTICLE_UP_DOWN) !=
      channels_.end();
  if (!has_pp_channel)
    return false;

  const auto origin = KDmn::parameter_type::origin_index();
  for (const int k_ex : domains::MomentumExchangeDomain::get_elements()) {
    if (k_ex != origin)
      return true;
  }

  return false;
}

template <class Parameters, DistType DT>
template <class Configuration, typename SpScalar>
void TpAccumulator<Parameters, DT, linalg::GPU>::prepareFiniteQMultibandPPData(
    const std::array<linalg::Matrix<SpScalar, linalg::GPU>, 2>& M,
    const std::array<Configuration, 2>& configs) {
  M_r_r_w_w_host_ = TpComplex(0., 0.);

  std::array<linalg::Matrix<SpScalar, linalg::CPU>, 2> M_host;
  for (int s = 0; s < 2; ++s) {
    G_host_[s] = G_[s];
    M_host[s].set(M[s], queues_[0].getStream());
    ndft_host_.execute(configs[s], M_host[s], M_r_r_w_w_host_, s);
  }
}

template <class Parameters, DistType DT>
void TpAccumulator<Parameters, DT, linalg::GPU>::matrixOperationsGMultibandHost(
    const BandBlockView& G0_kw1, const BandBlockView& G0_kw2, BandBlockView& G_kkww,
    MatrixHost& G0_M) {
  G0_M.resize(G0_kw1.size());
  linalg::matrixop::gemm(G0_kw1, G_kkww, G0_M);
  linalg::matrixop::gemm(TpComplex(-1), G0_M, G0_kw2, TpComplex(0), G_kkww);
}

template <class Parameters, DistType DT>
void TpAccumulator<Parameters, DT, linalg::GPU>::getGMultibandHost(const int s, const int k1,
                                                                   const int k2, const int w1,
                                                                   const int w2, MatrixHost& G,
                                                                   const TpComplex sign) const {
  const int w1_ext = w1 + extension_index_offset_;
  const int w2_ext = w2 + extension_index_offset_;
  const int no = n_bands_ * KDmn::dmn_size();
  const int row = n_bands_ * k1 + no * w1_ext;
  const int col = n_bands_ * k2 + no * w2_ext;
  const auto* const G_ptr = G_host_[s].ptr(row, col);

  for (int b2 = 0; b2 < n_bands_; ++b2)
    for (int b1 = 0; b1 < n_bands_; ++b1)
      G(b1, b2) = sign * G(b1, b2) + G_ptr[b1 + b2 * n_bands_];
}

template <class Parameters, DistType DT>
void TpAccumulator<Parameters, DT, linalg::GPU>::buildMomentumBlockFromRealSpaceHost(
    const int s, const std::vector<double>& k1_vec, const std::vector<double>& k2_vec, const int w1,
    const int w2, MatrixHost& M_k) const {
  using dca::math::util::innerProduct;

  const int w1_ext = w1 + extension_index_offset_;
  const int w2_ext = w2 + extension_index_offset_;
  const TpComplex norm = TpComplex(1. / RDmn::dmn_size(), 0.);
  const auto& band_elements = BDmn::get_elements();
  const auto& r_elements = RDmn::parameter_type::get_elements();

  for (int b2 = 0; b2 < n_bands_; ++b2) {
    for (int b1 = 0; b1 < n_bands_; ++b1) {
      TpComplex value{0., 0.};
      const auto& a1 = band_elements[b1].a_vec;
      const auto& a2 = band_elements[b2].a_vec;
      const TpComplex phase_band_1 = std::exp(TpComplex(0., innerProduct(k1_vec, a1)));
      const TpComplex phase_band_2 = std::exp(TpComplex(0., -innerProduct(k2_vec, a2)));

      for (int r2 = 0; r2 < RDmn::dmn_size(); ++r2) {
        const TpComplex phase_r2 = std::exp(TpComplex(0., -innerProduct(k2_vec, r_elements[r2])));
        for (int r1 = 0; r1 < RDmn::dmn_size(); ++r1) {
          const TpComplex phase_r1 = std::exp(TpComplex(0., innerProduct(k1_vec, r_elements[r1])));
          value += phase_r1 * phase_r2 * M_r_r_w_w_host_(r1, r2, b1, b2, s, w1_ext, w2_ext);
        }
      }

      M_k(b1, b2) = norm * phase_band_1 * phase_band_2 * value;
    }
  }
}

template <class Parameters, DistType DT>
void TpAccumulator<Parameters, DT, linalg::GPU>::buildUnfoldedG0BlockHost(
    const int s, const std::vector<double>& k_vec, const int w, MatrixHost& G0_k) const {
  using dca::math::util::innerProduct;

  auto folded_k = domains::cluster_operations::translate_inside_cluster(
      k_vec, KDmn::parameter_type::get_super_basis_vectors());
  const int k_idx = domains::cluster_operations::index(
      folded_k, KDmn::parameter_type::get_elements(), KDmn::parameter_type::SHAPE);

  std::vector<double> reciprocal_shift(k_vec.size(), 0.);
  for (int d = 0; d < int(k_vec.size()); ++d)
    reciprocal_shift[d] = k_vec[d] - folded_k[d];

  const int w_ext = w + extension_index_offset_;
  const auto& band_elements = BDmn::get_elements();
  for (int b2 = 0; b2 < n_bands_; ++b2) {
    for (int b1 = 0; b1 < n_bands_; ++b1) {
      const TpComplex left_phase =
          std::exp(TpComplex(0., innerProduct(reciprocal_shift, band_elements[b1].a_vec)));
      const TpComplex right_phase =
          std::exp(TpComplex(0., -innerProduct(reciprocal_shift, band_elements[b2].a_vec)));
      G0_k(b1, b2) = left_phase * G0_(b1, b2, s, k_idx, w_ext) * right_phase;
    }
  }
}

template <class Parameters, DistType DT>
void TpAccumulator<Parameters, DT, linalg::GPU>::getGMultibandDirectHost(
    const int s, const std::vector<double>& k1_vec, const std::vector<double>& k2_vec, const int w1,
    const int w2, MatrixHost& G, const TpComplex sign) const {
  buildMomentumBlockFromRealSpaceHost(s, k1_vec, k2_vec, w1, w2, M_unfolded_host_);
  buildUnfoldedG0BlockHost(s, k1_vec, w1, G0_a_host_);
  buildUnfoldedG0BlockHost(s, k2_vec, w2, G0_b_host_);

  for (int b2 = 0; b2 < n_bands_; ++b2)
    for (int b1 = 0; b1 < n_bands_; ++b1)
      G(b1, b2) = sign * G(b1, b2) + M_unfolded_host_(b1, b2);

  BandBlockView G_view(G);
  matrixOperationsGMultibandHost(G0_a_host_, G0_b_host_, G_view, G0_M_host_);
  if (dca::math::util::distance2(k1_vec, k2_vec) < 1.e-6 && w1 == w2) {
    for (int b2 = 0; b2 < n_bands_; ++b2)
      for (int b1 = 0; b1 < n_bands_; ++b1)
        G(b1, b2) += G0_a_host_(b1, b2) * beta_;
  }
}

template <class Parameters, DistType DT>
template <typename SignType>
double TpAccumulator<Parameters, DT, linalg::GPU>::updateG4(const std::size_t channel_index,
                                                            SignType factor) {
  // G4 is stored with the following band convention:
  // b1 ------------------------ b3
  //        |           |
  //        |           |
  //        |           |
  // b2 ------------------------ b4

  //  TODO: set stream only if this thread gets exclusive access to G4.
  //  get_G4().setStream(queues_[0]);
  const FourPointType channel = Base::channels_[channel_index];
  if (channel == FourPointType::PARTICLE_PARTICLE_UP_DOWN && needsFiniteQMultibandPP())
    return updateG4FiniteQMultibandPP(channel_index, factor);

  uint64_t start = Base::G4_[0].get_start();
  uint64_t end =
      Base::G4_[0].get_end() + 1;  // because the kernel expects this to be one past the end index
  if constexpr (Base::spin_symmetric_) {
    switch (channel) {
      case FourPointType::PARTICLE_HOLE_TRANSVERSE:
        return details::updateG4<TpComplex, FourPointType::PARTICLE_HOLE_TRANSVERSE>(
            get_G4Dev()[channel_index].ptr(), G_[0].ptr(), G_[0].leadingDimension(), G_[1].ptr(),
            G_[1].leadingDimension(), factor, multiple_accumulators_, queues_[0], start, end);

      case FourPointType::PARTICLE_HOLE_MAGNETIC:
        return details::updateG4<TpComplex, FourPointType::PARTICLE_HOLE_MAGNETIC>(
            get_G4Dev()[channel_index].ptr(), G_[0].ptr(), G_[0].leadingDimension(), G_[1].ptr(),
            G_[1].leadingDimension(), factor, multiple_accumulators_, queues_[0], start, end);

      case FourPointType::PARTICLE_HOLE_CHARGE:
        return details::updateG4<TpComplex, FourPointType::PARTICLE_HOLE_CHARGE>(
            get_G4Dev()[channel_index].ptr(), G_[0].ptr(), G_[0].leadingDimension(), G_[1].ptr(),
            G_[1].leadingDimension(), factor, multiple_accumulators_, queues_[0], start, end);

        // case FourPointType::PARTICLE_HOLE_LONGITUDINAL_UP_UP:
        //   return details::updateG4<TpComplex, FourPointType::PARTICLE_HOLE_LONGITUDINAL_UP_UP>(
        //       get_G4Dev()[channel_index].ptr(), G_[0].ptr(), G_[0].leadingDimension(), G_[1].ptr(),
        //       G_[1].leadingDimension(), factor, multiple_accumulators_, queues_[0], start, end);

        // case FourPointType::PARTICLE_HOLE_LONGITUDINAL_UP_DOWN:
        //   return details::updateG4<TpComplex, FourPointType::PARTICLE_HOLE_LONGITUDINAL_UP_DOWN>(
        //       get_G4Dev()[channel_index].ptr(), G_[0].ptr(), G_[0].leadingDimension(), G_[1].ptr(),
        //       G_[1].leadingDimension(), factor, multiple_accumulators_, queues_[0], start, end);

      case FourPointType::PARTICLE_PARTICLE_UP_DOWN:
        return details::updateG4<TpComplex, FourPointType::PARTICLE_PARTICLE_UP_DOWN>(
            get_G4Dev()[channel_index].ptr(), G_[0].ptr(), G_[0].leadingDimension(), G_[1].ptr(),
            G_[1].leadingDimension(), factor, multiple_accumulators_, queues_[0], start, end);

      default:
        throw std::logic_error("Specified four point type not implemented by tp_accumulator_gpu." +
                               std::to_string(static_cast<int>(channel)));
    }
  }
  else {
    if (channel == FourPointType::PARTICLE_PARTICLE_UP_DOWN)
      return details::updateG4NoSpin<TpComplex, FourPointType::PARTICLE_PARTICLE_UP_DOWN>(
          get_G4Dev()[channel_index].ptr(), G_[0].ptr(), G_[0].leadingDimension(), factor,
          multiple_accumulators_, queues_[0], start, end);
  }
  throw std::logic_error("Specified four point type not implemented by tp_accumulator_gpu." +
                         std::to_string(static_cast<int>(channel)));
}

template <class Parameters, DistType DT>
template <typename SignType>
double TpAccumulator<Parameters, DT, linalg::GPU>::updateG4FiniteQMultibandPP(
    const std::size_t channel_index, SignType factor) {
  auto& delta = G4_[channel_index];
  delta = TpComplex(0., 0.);

  const auto complex_factor = TpComplex(factor);
  const auto sign_over_2 = TpComplex(0.5) * complex_factor;
  const auto& exchange_frq = domains::FrequencyExchangeDomain::get_elements();
  const auto& exchange_mom = domains::MomentumExchangeDomain::get_elements();

  auto q_minus_k = [](const int k, const int q) { return KDmn::parameter_type::subtract(k, q); };
  auto w_ex_minus_w = [](const int w, const int w_ex) { return w_ex + WTpDmn::dmn_size() - 1 - w; };

  if constexpr (Base::spin_symmetric_) {
    for (int w_ex_idx = 0; w_ex_idx < exchange_frq.size(); ++w_ex_idx) {
      const int w_ex = exchange_frq[w_ex_idx];
      for (int k_ex_idx = 0; k_ex_idx < exchange_mom.size(); ++k_ex_idx) {
        const int k_ex = exchange_mom[k_ex_idx];
        if (k_ex == KDmn::parameter_type::origin_index())
          continue;
        const auto& q_vec = KDmn::parameter_type::get_elements()[k_ex];
        for (int w2 = 0; w2 < WTpDmn::dmn_size(); ++w2)
          for (int k2 = 0; k2 < KDmn::dmn_size(); ++k2)
            for (int w1 = 0; w1 < WTpDmn::dmn_size(); ++w1)
              for (int k1 = 0; k1 < KDmn::dmn_size(); ++k1) {
                const auto& k1_vec = KDmn::parameter_type::get_elements()[k1];
                const auto& k2_vec = KDmn::parameter_type::get_elements()[k2];
                std::vector<double> q_minus_k1(k1_vec.size(), 0.);
                std::vector<double> q_minus_k2(k2_vec.size(), 0.);
                for (int d = 0; d < int(q_vec.size()); ++d) {
                  q_minus_k1[d] = q_vec[d] - k1_vec[d];
                  q_minus_k2[d] = q_vec[d] - k2_vec[d];
                }

                for (int s = 0; s < 2; ++s) {
                  getGMultibandHost(s, k1, k2, w1, w2, G_a_host_);
                  getGMultibandDirectHost(!s, q_minus_k1, q_minus_k2, w_ex_minus_w(w1, w_ex),
                                          w_ex_minus_w(w2, w_ex), G_b_host_);
                  for (int b4 = 0; b4 < BDmn::dmn_size(); ++b4)
                    for (int b3 = 0; b3 < BDmn::dmn_size(); ++b3)
                      for (int b2 = 0; b2 < BDmn::dmn_size(); ++b2)
                        for (int b1 = 0; b1 < BDmn::dmn_size(); ++b1)
                          delta(b1, b2, b3, b4, k1, w1, k2, w2, k_ex_idx, w_ex_idx) +=
                              sign_over_2 * G_a_host_(b1, b3) * G_b_host_(b2, b4);
                }
              }
      }
    }
  }
  else {
    for (int w_ex_idx = 0; w_ex_idx < exchange_frq.size(); ++w_ex_idx) {
      const int w_ex = exchange_frq[w_ex_idx];
      for (int k_ex_idx = 0; k_ex_idx < exchange_mom.size(); ++k_ex_idx) {
        const int k_ex = exchange_mom[k_ex_idx];
        if (k_ex == KDmn::parameter_type::origin_index())
          continue;
        const auto& q_vec = KDmn::parameter_type::get_elements()[k_ex];
        for (int w2 = 0; w2 < WTpDmn::dmn_size(); ++w2)
          for (int k2 = 0; k2 < KDmn::dmn_size(); ++k2)
            for (int w1 = 0; w1 < WTpDmn::dmn_size(); ++w1)
              for (int k1 = 0; k1 < KDmn::dmn_size(); ++k1) {
                const auto& k1_vec = KDmn::parameter_type::get_elements()[k1];
                const auto& k2_vec = KDmn::parameter_type::get_elements()[k2];
                std::vector<double> q_minus_k1(k1_vec.size(), 0.);
                std::vector<double> q_minus_k2(k2_vec.size(), 0.);
                for (int d = 0; d < int(q_vec.size()); ++d) {
                  q_minus_k1[d] = q_vec[d] - k1_vec[d];
                  q_minus_k2[d] = q_vec[d] - k2_vec[d];
                }

                getGMultibandHost(0, k1, k2, w1, w2, G_a_host_);
                getGMultibandDirectHost(0, q_minus_k1, q_minus_k2, w_ex_minus_w(w1, w_ex),
                                        w_ex_minus_w(w2, w_ex), G_b_host_);

                for (int b4 = 0; b4 < BDmn::dmn_size(); ++b4)
                  for (int b3 = 0; b3 < BDmn::dmn_size(); ++b3)
                    for (int b2 = 0; b2 < BDmn::dmn_size(); ++b2)
                      for (int b1 = 0; b1 < BDmn::dmn_size(); ++b1)
                        delta(b1, b2, b3, b4, k1, w1, k2, w2, k_ex_idx, w_ex_idx) +=
                            complex_factor * G_a_host_(b1, b3) * G_b_host_(b2, b4);

                getGMultibandDirectHost(0, k1_vec, q_minus_k2, w1, w_ex_minus_w(w2, w_ex),
                                        G_a_host_);
                getGMultibandDirectHost(0, q_minus_k1, k2_vec, w_ex_minus_w(w1, w_ex), w2,
                                        G_b_host_);

                for (int b4 = 0; b4 < BDmn::dmn_size(); ++b4)
                  for (int b3 = 0; b3 < BDmn::dmn_size(); ++b3)
                    for (int b2 = 0; b2 < BDmn::dmn_size(); ++b2)
                      for (int b1 = 0; b1 < BDmn::dmn_size(); ++b1)
                        delta(b1, b2, b3, b4, k1, w1, k2, w2, k_ex_idx, w_ex_idx) -=
                            complex_factor * G_a_host_(b1, b4) * G_b_host_(b2, b3);
              }
      }
    }
  }

  G4DevType delta_dev("tp_acc_gpu::finite_q_pp_delta");
  delta_dev.setAsync(delta, queues_[0].getStream());
  details::accumulateG4Device(get_G4Dev()[channel_index].ptr(), delta_dev.ptr(), delta_dev.size(),
                              multiple_accumulators_, queues_[0].getStream());
  BaseGpu::synchronizeStreams();

  return 2. * WTpDmn::dmn_size() * WTpDmn::dmn_size() * KDmn::dmn_size() * KDmn::dmn_size() *
         std::pow(n_bands_, 4);
}

template <class Parameters, DistType DT>
void TpAccumulator<Parameters, DT, linalg::GPU>::finalize() {
  if (finalized_)
    return;

  for (std::size_t channel = 0; channel < G4_.size(); ++channel) {
    get_G4Dev()[channel].copyTo(G4_[channel]);
  }
  // TODO: release memory if needed by the rest of the DCA loop.
  // get_G4().clear();

  finalized_ = true;
  initialized_ = false;
}

template <class Parameters, DistType DT>
void TpAccumulator<Parameters, DT, linalg::GPU>::sumTo(ThisType& other /*other_one*/) {
  // Nothing to do: G4 on the device is shared.
  BaseGpu::sumTo_(other);
  return;
}

template <class Parameters, DistType DT>
auto TpAccumulator<Parameters, DT, linalg::GPU>::get_G4Dev() -> std::vector<G4DevType>& {
  static std::vector<G4DevType> G4;
  return G4;
}

/** Return the G4 only as the correct type
 *
 *  the return type is quite a code smell
 */
template <class Parameters, DistType DT>
const std::vector<typename TpAccumulator<Parameters, DT, linalg::GPU>::Base::TpGreensFunction>& TpAccumulator<
    Parameters, DT, linalg::GPU>::get_G4() const {
  if (G4_.empty())
    throw std::logic_error("There is no G4 stored in this class.");

  return G4_;
}

#ifndef NDEBUG
template <class Parameters, DistType DT>
const auto& TpAccumulator<Parameters, DT, linalg::GPU>::get_G_Debug() {
  if (G_debug_.empty())
    throw std::logic_error("There is no G4 stored in this class.");

  return G_debug_;
}
#endif

template <class Parameters, DistType DT>
std::vector<typename TpAccumulator<Parameters, DT, linalg::GPU>::TpGreensFunction>& TpAccumulator<
    Parameters, DT, linalg::GPU>::get_nonconst_G4() {
  if (G4_.empty())
    throw std::logic_error("There is no G4 stored in this class.");
  return G4_;
}

#ifdef DCA_HAVE_MPI

template <class Parameters, DistType DT>
template <typename SignType>
void TpAccumulator<Parameters, DT, linalg::GPU>::ringG(double& flop, SignType factor) {
  // get ready for send and receive
  static_assert(dist != DistType::NONE);
  for (int s = 0; s < 2; ++s) {
    sendbuff_G_[s] = G_[s];
  }

  int my_concurrency_id, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_concurrency_id);

  // get rank index of left and right neighbor
  int left_neighbor = (my_concurrency_id - 1 + mpi_size) % mpi_size;
  int right_neighbor = (my_concurrency_id + 1 + mpi_size) % mpi_size;

  // Pipepline ring algorithm in the following for-loop:
  // 1) At each time step, local rank receives a new G2 from left hand neighbor,
  // makes a copy locally and uses this G2 to update G4, and
  // sends this G2 to right hand neighbor. In total, the algorithm performs (mpi_size - 1) steps.
  // 2) This algorithm currently requires parameters in input file, please refer to mci_parameters.hpp:
  //      a) #walker == #accumulator and shared-walk-and-accumulation-thread = true;
  //      b) and, local measurements are equal, and each accumulator should have same #measurement, i.e.
  //         measurements % ranks == 0 && local_measurement % threads == 0.
  for (int icount = 0; icount < (mpi_size - 1); icount++) {
    send_requests_.fill({right_neighbor, my_concurrency_id, MPI_REQUEST_NULL});
    send(sendbuff_G_, send_requests_);
    recv_requests_.fill({my_concurrency_id, left_neighbor, MPI_REQUEST_NULL});
    receive(G_, recv_requests_);

    // wait for G2 to be available again
    for (int s = 0; s < 2; ++s)
      MPI_Wait(&(recv_requests_[s].request), MPI_STATUSES_IGNORE);

    // use newly copied G2 to update G4
    for (std::size_t channel = 0; channel < G4_.size(); ++channel) {
      flop += updateG4(channel, factor);
    }

    // wait for sendbuf_G2 to be available again
    for (int s = 0; s < 2; ++s)
      MPI_Wait(&(send_requests_[s].request), MPI_STATUSES_IGNORE);

    // get ready for send again
    for (int s = 0; s < 2; ++s) {
      sendbuff_G_[s].swap(G_[s]);
    }
  }
}

template <class Parameters, DistType DT>
void TpAccumulator<Parameters, DT, linalg::GPU>::send(const std::array<RMatrix, 2>& data,
                                                      std::array<RingMessage, 2>& messages) {
  static_assert(dist != DistType::NONE);

  const auto g_size = data[0].size().first * data[0].size().second;

#ifdef DCA_HAVE_GPU_AWARE_MPI
  for (int s = 0; s < 2; ++s) {
    MPI_Isend(data[s].ptr(), g_size, MPITypeMap<TpComplex>::value(), messages[s].target,
              thread_id_ + 1, MPI_COMM_WORLD, &(messages[s].request));
  }
#else

  for (int s = 0; s < 2; ++s) {
    sendbuffer_[s].resize(g_size);
    checkRC(cudaMemcpy(sendbuffer_[s].data(), data[s].ptr(), g_size * sizeof(TpComplex),
                       cudaMemcpyDeviceToHost));

    MPI_Isend(sendbuffer_[s].data(), g_size, MPITypeMap<TpComplex>::value(), messages[s].target,
              thread_id_ + 1, MPI_COMM_WORLD, &(messages[s].request));
  }

#endif  // DCA_HAVE_GPU_AWARE_MPI
}

template <class Parameters, DistType DT>
void TpAccumulator<Parameters, DT, linalg::GPU>::receive(std::array<RMatrix, 2>& data,
                                                         std::array<RingMessage, 2>& messages) {
  static_assert(dist != DistType::NONE);
  using dca::parallel::MPITypeMap;
  const auto g_size = data[0].size().first * data[0].size().second;

#ifdef DCA_HAVE_GPU_AWARE_MPI
  for (int s = 0; s < 2; ++s) {
    MPI_Irecv(data[s].ptr(), g_size, MPITypeMap<TpComplex>::value(), messages[s].source,
              thread_id_ + 1, MPI_COMM_WORLD, &(messages[s].request));
  }

#else
  for (int s = 0; s < 2; ++s) {
    recvbuffer_[s].resize(g_size);
    MPI_Irecv(recvbuffer_[s].data(), g_size, MPITypeMap<TpComplex>::value(), messages[s].source,
              thread_id_ + 1, MPI_COMM_WORLD, &(messages[s].request));
  }
  for (int s = 0; s < 2; ++s) {
    MPI_Wait(&(messages[s].request), MPI_STATUSES_IGNORE);
    // Note: MPI can not access host memory allocated from CUDA, hence the usage of blocking communication.
    checkRC(cudaMemcpy(data[s].ptr(), recvbuffer_[s].data(), g_size * sizeof(TpComplex),
                       cudaMemcpyHostToDevice));
  }
#endif  // DCA_HAVE_GPU_AWARE_MPI
}
#endif  // DCA_WITH_MPI

}  // namespace accumulator
}  // namespace solver
}  // namespace phys
}  // namespace dca

#endif  // DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_SHARED_TOOLS_ACCUMULATION_TP_TP_ACUCMULATOR_GPU_HPP
