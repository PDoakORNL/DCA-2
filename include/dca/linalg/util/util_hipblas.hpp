// Copyright (C) 2021 ETH Zurich
// Copyright (C) 2021 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Raffaele Solca' (rasolca@itp.phys.ethz.ch)
//         Peter Doak (doakpw@ornl.gov)
//
// TODO: This file has to be modified when the LAPACK functions are cleaned up.

#ifndef DCA_LINALG_UTIL_UTIL_HIPBLAS_HPP
#define DCA_LINALG_UTIL_UTIL_HIPBLAS_HPP

namespace dca {
namespace linalg {
namespace util {
// dca::linalg::util::

int getCublasVersion();

void initializeMagma();

}  // util
}  // linalg
}  // dca

#endif  // DCA_LINALG_UTIL_UTIL_CUBLAS_HPP
