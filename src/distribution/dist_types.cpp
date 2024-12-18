// Copyright (C) 2020 ETH Zurich
// Copyright (C) 2020 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// This file implements dist_types.hpp

#include "dca/distribution/dist_types.hpp"

#include <stdexcept>

namespace dca {
DistType stringToDistType(const std::string& name) {
  if (name == "BLOCKED")
    return DistType::BLOCKED;
  else if (name == "LINEAR")
    return DistType::LINEAR;
  else if (name == "NONE")
    return DistType::NONE;
  else
    throw std::logic_error("Invalid distribtion mode: " + name);
}

std::string toString(DistType type) {
  switch (type) {
    case DistType::BLOCKED:
      return "BLOCKED";
    case DistType::LINEAR:
      return "LINEAR";
    case DistType::NONE:
      return "NONE";
    default:
      throw std::logic_error("Invalid distribution mode.");
  }
}
}  // namespace dca
