// Copyright (C) 2026 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Peter W. Doak, Oak Ridge National Lab, (doakpw@ornl.gov)
//
// This file tests disorder_parameters.hpp
//

#include "dca/phys/parameters/disorder_parameters.hpp"
#include "gtest/gtest.h"
#include "dca/io/json/json_reader.hpp"

TEST(DisorderParametersTest, DefaultValues) {
  dca::phys::params::DisorderParameters pars;

  EXPECT_EQ(0.0, pars.get_disorder_potential());
  EXPECT_EQ(0.0, pars.get_disorder_density());
  EXPECT_EQ(0, pars.get_disorder_num_configurations());
  EXPECT_EQ(1, pars.get_disorder_max_sites());
}

TEST(DomainsParametersTest, ReadAll) {
  dca::io::JSONReader reader;
  dca::phys::params::DisorderParameters pars;

  reader.open_file(DCA_SOURCE_DIR
                   "/test/unit/phys/parameters/disorder_parameters/input_read_all.json");
  pars.readWrite(reader);
  reader.close_file();

  EXPECT_EQ(1.0, pars.get_disorder_potential());
  EXPECT_EQ(0.25, pars.get_disorder_density());
  EXPECT_EQ(10, pars.get_disorder_num_configurations());
  EXPECT_EQ(1, pars.get_disorder_max_sites());
}
