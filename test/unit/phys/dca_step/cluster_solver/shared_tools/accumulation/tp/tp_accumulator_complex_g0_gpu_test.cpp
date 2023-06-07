// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE.txt for terms of usage.
// See CITATION.txt for citation guidelines if you use this code for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// This file implements a no-change test for the two particles accumulation on the GPU with
// the Rashba model.

#include "dca/config/profiler.hpp"
#include <complex>

#include "dca/platform/dca_gpu.h"

#include "dca/phys/dca_step/cluster_solver/shared_tools/accumulation/tp/tp_accumulator_gpu.hpp"

#include <array>
#include <functional>
#include <string>
#include "dca/testing/gtest_h_w_warning_blocking.h"

#include "dca/function/util/difference.hpp"
#include "dca/linalg/util/util_cublas.hpp"
#include "dca/math/random/std_random_wrapper.hpp"
#include "dca/phys/four_point_type.hpp"
#include "test/unit/phys/dca_step/cluster_solver/shared_tools/accumulation/accumulation_test.hpp"
#include "test/unit/phys/dca_step/cluster_solver/test_setup.hpp"

constexpr bool write_G4s = true;

#ifdef DCA_HAVE_ADIOS2
adios2::ADIOS* adios_ptr;
#endif

#ifdef DCA_HAVE_MPI
#include "dca/parallel/mpi_concurrency/mpi_concurrency.hpp"
dca::parallel::MPIConcurrency* concurrency_ptr;
#else
#include "dca/parallel/no_concurrency/no_concurrency.hpp"
dca::parallel::NoConcurrency* concurrency_ptr;
#endif

#define INPUT_DIR \
  DCA_SOURCE_DIR "/test/unit/phys/dca_step/cluster_solver/shared_tools/accumulation/tp/"

constexpr char input_file[] = INPUT_DIR "input_4x4_complex.json";

template <typename SCALAR>
struct TpAccumulatorComplexG0GpuTest : public ::testing::Test {
  using G0Setup = dca::testing::G0SetupBare<SCALAR, dca::testing::LatticeRashba,
                                            dca::ClusterSolverId::CT_AUX, input_file>;
  virtual void SetUp() {
    host_setup.SetUp();
    gpu_setup.SetUp();
  }

  virtual void TearDown() {}
  G0Setup host_setup;
  G0Setup gpu_setup;
};

uint loop_counter = 0;

using TestTypes = ::testing::Types<std::complex<double>>;
TYPED_TEST_CASE(TpAccumulatorComplexG0GpuTest, TestTypes);

#define TYPING_PREFACE                                            \
  using Scalar = TypeParam;                                       \
  using ConfigGenerator = dca::testing::AccumulationTest<Scalar>; \
  using Configuration = typename ConfigGenerator::Configuration;  \
  using Sample = typename ConfigGenerator::Sample;

TYPED_TEST(TpAccumulatorComplexG0GpuTest, Accumulate) {
  TYPING_PREFACE
  dca::linalg::util::initializeMagma();

  const std::array<int, 2> n{18, 22};
  Sample M;
  Configuration config;
  ConfigGenerator::prepareConfiguration(
      config, M, TpAccumulatorComplexG0GpuTest<Scalar>::G0Setup::BDmn::dmn_size(),
      TpAccumulatorComplexG0GpuTest<Scalar>::G0Setup::RDmn::dmn_size(),
      this->host_setup.parameters_.get_beta(), n);

  using namespace dca::phys;
  std::vector<FourPointType> four_point_channels{FourPointType::PARTICLE_HOLE_MAGNETIC};
  this->host_setup.parameters_.set_four_point_channels(four_point_channels);
  this->gpu_setup.parameters_.set_four_point_channels(four_point_channels);

  dca::phys::solver::accumulator::TpAccumulator<decltype(this->host_setup.parameters_),
                                                dca::DistType::NONE, dca::linalg::CPU>
      accumulatorHost(this->host_setup.data_->G0_k_w_cluster_excluded, this->host_setup.parameters_);
  dca::phys::solver::accumulator::TpAccumulator<decltype(this->gpu_setup.parameters_),
                                                dca::DistType::NONE, dca::linalg::GPU>
      accumulatorDevice(this->gpu_setup.data_->G0_k_w_cluster_excluded, this->gpu_setup.parameters_);
  const Scalar sign{1, 0};

  accumulatorDevice.resetAccumulation(loop_counter);
  accumulatorDevice.accumulate(M, config, sign);
  accumulatorDevice.finalize();

  accumulatorHost.resetAccumulation(loop_counter);
  accumulatorHost.accumulate(M, config, sign);
  accumulatorHost.finalize();

  ++loop_counter;

#ifdef DCA_HAVE_ADIOS2
  if (write_G4s) {
    dca::io::Writer writer(*adios_ptr, *concurrency_ptr, "ADIOS2", true);
    dca::io::Writer writer_h5(*adios_ptr, *concurrency_ptr, "HDF5", true);

    writer.open_file("tp_gpu_test_G4.bp");
    writer_h5.open_file("tp_gpu_test_G4.hdf5");

    this->host_setup.parameters_.write(writer);
    this->host_setup.parameters_.write(writer_h5);
    this->host_setup.data_->write(writer);
    this->host_setup.data_->write(writer_h5);

    for (std::size_t channel = 0; channel < accumulatorHost.get_G4().size(); ++channel) {
      std::string channel_str =
          dca::phys::toString(this->host_setup.parameters_.get_four_point_channels()[channel]);
      writer.execute("accumulatorHOST_" + channel_str, accumulatorHost.get_G4()[channel]);
      writer.execute("accumulatorDevice_" + channel_str, accumulatorDevice.get_G4()[channel]);
      writer_h5.execute("accumulatorHOST_" + channel_str, accumulatorHost.get_G4()[channel]);
      writer_h5.execute("accumulatorDevice_" + channel_str, accumulatorDevice.get_G4()[channel]);
    }
    writer.close_file();
    writer_h5.close_file();
  }
#endif

  std::cout << "blocks: " << dca::util::ceilDiv(int(accumulatorHost.get_G4()[0].size()), 256) << '\n';

  for (std::size_t channel = 0; channel < accumulatorHost.num_channels(); ++channel) {
    const auto diff = dca::func::util::difference(accumulatorHost.get_G4()[channel],
                                                  accumulatorDevice.get_G4()[channel]);
    EXPECT_GT(5e-7, diff.l_inf);
  }
}

TYPED_TEST(TpAccumulatorComplexG0GpuTest, computeM) {
  TYPING_PREFACE
  const std::array<int, 2> n{18, 22};
  Sample M;
  std::array<dca::linalg::Matrix<Scalar, dca::linalg::GPU>, 2> M_dev;
  Configuration config;
  ConfigGenerator::prepareConfiguration(
      config, M, TpAccumulatorComplexG0GpuTest<Scalar>::G0Setup::BDmn::dmn_size(),
      TpAccumulatorComplexG0GpuTest<Scalar>::G0Setup::RDmn::dmn_size(),
      this->host_setup.parameters_.get_beta(), n);

  using namespace dca::phys;
  std::vector<FourPointType> four_point_channels{FourPointType::PARTICLE_HOLE_MAGNETIC};
  this->host_setup.parameters_.set_four_point_channels(four_point_channels);
  this->gpu_setup.parameters_.set_four_point_channels(four_point_channels);

  dca::phys::solver::accumulator::TpAccumulator<decltype(this->host_setup.parameters_),
                                                dca::DistType::NONE, dca::linalg::CPU>
      accumulatorHost(this->host_setup.data_->G0_k_w_cluster_excluded, this->host_setup.parameters_);
  dca::phys::solver::accumulator::TpAccumulator<decltype(this->gpu_setup.parameters_),
                                                dca::DistType::NONE, dca::linalg::GPU>
      accumulatorDevice(this->gpu_setup.data_->G0_k_w_cluster_excluded, this->gpu_setup.parameters_);
  const int8_t sign = 1;

  for (int s = 0; s < 2; ++s)
    M_dev[s].setAsync(M[s], *(accumulatorDevice.get_stream()));

  accumulatorDevice.resetAccumulation(loop_counter);
  accumulatorDevice.computeM(M_dev, config);

  accumulatorHost.resetAccumulation(loop_counter);
  accumulatorHost.computeM(M, config);

  Sample M_from_dev;

  for (int s = 0; s < 2; ++s)
    M_from_dev[s].setAsync(M_dev[s], *(accumulatorDevice.get_stream()));

  accumulatorDevice.synchronizeStreams();

  for (int s = 0; s < 2; ++s)
    for (int i = 0; i < M[s].nrCols(); ++i)
      for (int j = 0; j < M[s].nrRows(); ++j) {
        auto diff = M[s](i, j) - M_from_dev[s](i, j);
        auto diff_sq = diff * diff;
        EXPECT_GT(5e-7, diff_sq.real())
            << "M[" << s << "](i,j) !=  M_from_dev[" << s << "](i,j) for i:" << i << " j:" << j;
      }
}

int main(int argc, char** argv) {
#ifdef DCA_HAVE_MPI
  dca::parallel::MPIConcurrency concurrency(argc, argv);
  concurrency_ptr = &concurrency;
#else
  dca::parallel::NoConcurrency concurrency(argc, argv);
  concurrency_ptr = &concurrency;
#endif

  dca::linalg::util::initializeMagma();

#ifdef DCA_HAVE_ADIOS2
  // ADIOS expects MPI_COMM pointer or nullptr
  adios2::ADIOS adios("", concurrency_ptr->get(), false);
  adios_ptr = &adios;
#endif
  ::testing::InitGoogleTest(&argc, argv);

  // ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  // delete listeners.Release(listeners.default_result_printer());
  // listeners.Append(new dca::testing::MinimalistPrinter);

  int result = RUN_ALL_TESTS();
  return result;
}
