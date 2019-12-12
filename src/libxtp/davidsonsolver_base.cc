/*
 * Copyright 2009-2019 The VOTCA Development Team (http://www.votca.org)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <iostream>
#include <stdexcept>

#include <votca/xtp/davidsonsolver_base.h>
#include <votca/xtp/eigen.h>

using boost::format;
using std::flush;

namespace votca {
namespace xtp {

using namespace std;

DavidsonSolver_base::DavidsonSolver_base(Logger &log) : _log(log) {}

void DavidsonSolver_base::printTiming(
    const std::chrono::time_point<std::chrono::system_clock> &start) const {
  XTP_LOG(Log::error, _log)
      << TimeStamp() << "-----------------------------------" << flush;
  std::chrono::time_point<std::chrono::system_clock> end =
      std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time = end - start;
  XTP_LOG(Log::error, _log) << TimeStamp() << "- Davidson ran for "
                            << elapsed_time.count() << "secs." << flush;
  XTP_LOG(Log::error, _log)
      << TimeStamp() << "-----------------------------------" << flush;
}

void DavidsonSolver_base::checkOptions(Index operator_size) {
  //. search space exceeding the system size
  if (_max_search_space > operator_size) {
    XTP_LOG(Log::error, _log)
        << TimeStamp() << " == Warning : Max search space ("
        << _max_search_space << ") larger than system size (" << operator_size
        << ")" << flush;

    _max_search_space = operator_size;
    XTP_LOG(Log::error, _log)
        << TimeStamp() << " == Warning : Max search space set to "
        << operator_size << flush;

    XTP_LOG(Log::error, _log)
        << TimeStamp()
        << " == Warning : If problems appear, try asking for less than "
        << Index(operator_size / 10) << " eigenvalues" << flush;
  }
}

void DavidsonSolver_base::printOptions(Index operator_size) const {

  XTP_LOG(Log::error, _log) << TimeStamp() << " Davidson Solver using "
                            << OPENMP::getMaxThreads() << " threads." << flush;
  XTP_LOG(Log::error, _log) << TimeStamp() << " Tolerance : " << _tol << flush;

  XTP_LOG(Log::error, _log) << TimeStamp() << " Matrix size : " << operator_size
                            << 'x' << operator_size << flush;
}

void DavidsonSolver_base::set_tolerance(std::string tol) {
  if (tol == "loose") {
    this->_tol = 1E-3;
  } else if (tol == "normal") {
    this->_tol = 1E-4;
  } else if (tol == "strict") {
    this->_tol = 1E-5;
  } else {
    throw std::runtime_error(tol + " is not a valid Davidson tolerance");
  }
}

void DavidsonSolver_base::set_size_update(std::string update_size) {

  if (update_size == "min") {
    this->_davidson_update = UPDATE::MIN;
  } else if (update_size == "safe") {
    this->_davidson_update = UPDATE::SAFE;
  } else if (update_size == "max") {
    this->_davidson_update = UPDATE::MAX;
  } else {
    throw std::runtime_error(update_size + " is not a valid Davidson update");
  }
}

Index DavidsonSolver_base::getSizeUpdate(Index neigen) const {
  Index size_update;
  switch (this->_davidson_update) {
    case UPDATE::MIN:
      size_update = neigen;
      break;
    case UPDATE::SAFE:
      if (neigen < 20) {
        size_update = static_cast<Index>(1.5 * double(neigen));
      } else {
        size_update = neigen + 10;
      }
      break;
    case UPDATE::MAX:
      size_update = 2 * neigen;
      break;
    default:
      size_update = 2 * neigen;
      break;
  }
  return size_update;
}

}  // namespace xtp
}  // namespace votca