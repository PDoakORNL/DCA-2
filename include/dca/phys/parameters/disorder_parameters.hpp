// Copyright (C) 2026 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Peter W. Doak (doakpw@ornl.gov)

#ifndef DCA_PHYS_PARAMETERS_DISORDER_PARAMETERS_HPP
#define DCA_PHYS_PARAMETERS_DISORDER_PARAMETERS_HPP

#include <stdexcept>

namespace dca {
namespace phys {
namespace params {
// dca::phys::params::

/** This class handles the disorder parameters
 */
class DisorderParameters {
public:
  DisorderParameters() {}

  template <typename Concurrency>
  int getBufferSize(const Concurrency& concurrency) const;
  template <typename Concurrency>
  void pack(const Concurrency& concurrency, char* buffer, int buffer_size, int& position) const;
  template <typename Concurrency>
  void unpack(const Concurrency& concurrency, char* buffer, int buffer_size, int& position);

  template <typename ReaderOrWriter>
  void readWrite(ReaderOrWriter& reader_or_writer);

  int get_num_disorder_configrations() const {
    return num_disorder_configurations_;
  }
  int get_max_disorder_sites() const {
    return max_disorder_sites_;
  }
  double get_disorder_density() const {
    return disorder_density_;
  }

private:
  int num_disorder_configurations_{0};
  int max_disorder_sites_{1};
  double disorder_density_{0.0};
};

template <typename Concurrency>
int DisorderParameters::getBufferSize(const Concurrency& concurrency) const {
  int buffer_size = 0;

  buffer_size += concurrency.get_buffer_size(num_disorder_configurations_);
  buffer_size += concurrency.get_buffer_size(max_disorder_sites_);
  buffer_size += concurrency.get_buffer_size(disorder_density_);

  return buffer_size;
}

template <typename Concurrency>
void DisorderParameters::pack(const Concurrency& concurrency, char* buffer, int buffer_size,
                              int& position) const {
  concurrency.pack(buffer, buffer_size, position, num_disorder_configurations_);
  concurrency.pack(buffer, buffer_size, position, max_disorder_sites_);
  concurrency.pack(buffer, buffer_size, position, disorder_density_);
}

template <typename Concurrency>
void DisorderParameters::unpack(const Concurrency& concurrency, char* buffer, int buffer_size,
                                int& position) {
  concurrency.unpack(buffer, buffer_size, position, num_disorder_configurations_);
  concurrency.unpack(buffer, buffer_size, position, max_disorder_sites_);
  concurrency.unpack(buffer, buffer_size, position, disorder_density_);
}

template <typename ReaderOrWriter>
void DisorderParameters::readWrite(ReaderOrWriter& reader_or_writer) {
  try {
    reader_or_writer.open_group("disorder");

    try {
      reader_or_writer.execute("num-disorder-configurations", num_disorder_configurations_);
    }
    catch (const std::exception& r_e) {
    }
    try {
      reader_or_writer.execute("max-disorder-sites", max_disorder_sites_);
    }
    catch (const std::exception& r_e) {
    }
    try {
      reader_or_writer.execute("disorder-density", disorder_density_);
    }
    catch (const std::exception& r_e) {
    }
    reader_or_writer.close_group();
  }
  catch (const std::exception& r_e) {
  }
}

}  // namespace params
}  // namespace phys
}  // namespace dca

#endif  // DCA_PHYS_PARAMETERS_DISORDER_PARAMETERS_HPP
