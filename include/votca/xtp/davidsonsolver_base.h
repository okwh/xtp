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

#pragma once
#ifndef __VOTCA_TOOLS_DAVIDSON_SOLVER_BASE_H
#define __VOTCA_TOOLS_DAVIDSON_SOLVER_BASE_H

#include <chrono>
#include <iostream>
#include <stdexcept>

#include <boost/format.hpp>
#include <votca/xtp/eigen.h>
#include <votca/xtp/logger.h>

using boost::format;
using std::flush;

namespace votca {
namespace xtp {

/**
* \brief Base class for Davidson algorithms to solve A*V=E*V

**/

class DavidsonSolver_base {

 public:
  DavidsonSolver_base(Logger &log);

  void set_iter_max(Index N) { this->_iter_max = N; }
  void set_max_search_space(Index N) { this->_max_search_space = N; }
  void set_tolerance(std::string tol);
  void set_size_update(std::string method);

  Eigen::ComputationInfo info() const { return _info; }
  Eigen::VectorXd eigenvalues() const { return this->_eigenvalues; }
  Eigen::MatrixXd eigenvectors() const { return this->_eigenvectors; }
  Index num_iterations() const { return this->_i_iter; }

 protected:
  Logger &_log;
  Index _iter_max = 50;
  Index _i_iter = 0;
  double _tol = 1E-4;
  Index _max_search_space = 0;

  enum UPDATE { MIN, SAFE, MAX };
  UPDATE _davidson_update = UPDATE::SAFE;

  Eigen::VectorXd _eigenvalues;
  Eigen::MatrixXd _eigenvectors;
  Eigen::VectorXd _preconditioner;
  Eigen::ComputationInfo _info = Eigen::ComputationInfo::NoConvergence;

  Index getSizeUpdate(Index neigen) const;

  void checkOptions(Index operator_size);

  void printOptions(Index operator_size) const;

  void printTiming(
      const std::chrono::time_point<std::chrono::system_clock> &start) const;
};

}  // namespace xtp
}  // namespace votca

#endif  // __VOTCA_TOOLS_DAVIDSON_SOLVER_H
