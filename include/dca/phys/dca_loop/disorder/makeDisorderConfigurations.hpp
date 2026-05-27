// Copyright (C) 2026 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Peter W. Doak (doakpw@ornl.gov)
//
// This function creates a set of disorder configurations based on the
// sites in the cluster and requested density

#ifndef DCA_PHYS_DCA_LOOP_MAKE_DISORDER_HPP
#define DCA_PHYS_DCA_LOOP_MAKE_DISORDER_HPP

#include <vector>
#include <unordered_set>
#include <random>
#include "dca/phys/types/dca_shared_types.hpp"

namespace dca::phys {

namespace detail {
struct VectorHash {
  size_t operator()(const std::vector<int>& v) const {
    size_t seed = 0;
    for (int x : v) {
      // XOR current hash with hash of element + constant + bit rotations
      // This helps avoid collisions for similar vectors (e.g.,
      // [1,-1] vs [-1,1])
      // Based on the boost hash_combine function.
      constexpr size_t golden_ratio = sizeof(int) == 8 ? 0x9e3779b97f4a7c15 : 0x9e3779b9;
      seed ^= std::hash<int>{}(x) + golden_ratio + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

}  // namespace detail

template <typename Parameters>
class MakeDisorderConfigurations {
  using DST = DcaSharedTypes<Parameters>;
  using DisorderConfiguration = typename DST::DisorderConfiguration;
  using DDA = typename DST::DDA;
  using RDmn = typename DDA::RClusterDmn;
  using BDmn = typename DDA::BDmn;

public:
  void operator()(Parameters& parameters, std::vector<DisorderConfiguration>& disorder_configurations,
                  std::vector<double>& disorder_weights) {
    auto m_configs = parameters.get_disorder_num_configurations();
    assert(m_configs > 0);
    disorder_configurations.resize(m_configs);
    disorder_weights.resize(m_configs);

    auto n_bands = BDmn::dmn_size();
    auto n_rsites = RDmn::dmn_size();
    auto n_sites = disorder_configurations[0].size();

    // right now we always make spin synmmetric configurations
    n_sites /= 2;

    // Check feasibility: only 2^n distinct ±1 vectors of length n exist
    if (m_configs > (1 << n_sites)) {
      throw std::runtime_error("More unique vectors requested than possible.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 1);

    // Set to track uniqueness using our custom hash
    std::unordered_set<std::vector<int>, detail::VectorHash> seen;

    // here we put the disorder potential in while preserving spin
    // symmetry for the potential
    auto insert_config = [n_rsites, n_bands](const auto& this_config, auto& config) {
      int i_disorder = 0;
      for (int isite = 0; isite < n_rsites; ++isite) {
        for (int iband = 0; iband < n_bands; ++iband, ++i_disorder) {
          config(iband, 0, isite) = this_config[i_disorder];
          config(iband, 1, isite) = this_config[i_disorder];
        }
      }
    };

    int max_ones = parameters.get_disorder_max_sites();
    int configs_generated = 0;
    int attempts = 0;
    int max_attempts = 1000;
    // Generate vectors until we have m unique ones
    while (configs_generated < m_configs && attempts < max_attempts) {
      auto& config = disorder_configurations[configs_generated];
      auto& weight = disorder_weights[configs_generated];
      std::vector<int> this_config(n_sites);
      int num_dis = 0;
      for (auto& site : this_config) {
        // Map 0 → -1 and 1 → 1
        site = dist(gen) ? 1 : -1;
        if (site == 1)
          ++num_dis;
        if (num_dis > max_ones)
          break;
      }

      if (num_dis <= max_ones) {
        // Insert returns true only if this vector is new (unique)
        if (seen.insert(this_config).second) {
          ++configs_generated;
          attempts = 0;
          insert_config(this_config, config);
          auto conc = parameters.get_disorder_density();
          weight = std::pow(conc, num_dis) * std::pow((1 - conc), (n_sites - num_dis));

          // If disorder didn't have spin symmetry
          // std::copy(this_config.begin(), this_config.end(), config.begin());
        }
        else {
          ++attempts;
        }
      }
      else {
        ++attempts;
      }
    }
    auto disorder_half_pot = parameters.get_disorder_potential() * 0.5;
    for (int ic = 0; ic < disorder_configurations.size(); ++ic) {
      auto& config = disorder_configurations[ic];
      auto weight = disorder_weights[ic];
      config *= disorder_half_pot;
      config.print_elements(std::cout);
      std::cout << "weight: " << weight << '\n';
    }
  }
};
}  // namespace dca::phys

#endif
