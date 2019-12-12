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
#ifndef __VOTCA_TOOLS_DAVIDSON_SOLVER_BTDA_H
#define __VOTCA_TOOLS_DAVIDSON_SOLVER_BTDA_H

#include <chrono>
#include <iostream>
#include <stdexcept>

#include <boost/format.hpp>
#include <votca/xtp/davidsonsolver_base.h>
#include <votca/xtp/eigen.h>
#include <votca/xtp/logger.h>

using boost::format;
using std::flush;

namespace votca {
namespace xtp {

/**
* \brief Use Davidson algorithm to solve H*V=E*V
with H=( A  B) and V=(X) and A and B symmetric
       (-B -A)       (Y)
using the Paper from Yang
Efficient Block Preconditioned Eigensolvers for Linear Response Time-dependet
Density Functional Theory
**/

class DavidsonSolver_BTDA : public DavidsonSolver_base {

 public:
  DavidsonSolver_BTDA(Logger &log) : DavidsonSolver_base(log){};

  template <typename MatrixReplacement1, typename MatrixReplacement2>
  void solve(const MatrixReplacement1 &ApB, const MatrixReplacement2 &AmB,
             Index neigen, Index size_initial_guess = 0) {

    // in the paper ApB and AmB are referred to as M and K respectively, we will
    // do the same internally
    const MatrixReplacement1 &M = ApB;
    const MatrixReplacement1 &K = AmB;

    // diagonal preconditioner
    _preconditioner = M.diagonal().cwiseProduct(K.diagonal());

    if (_max_search_space < neigen) {
      _max_search_space = neigen * 5;
    }
    std::chrono::time_point<std::chrono::system_clock> start =
        std::chrono::system_clock::now();
    Index op_size = M.rows();

    checkOptions(op_size);
    printOptions(op_size);

    // initial guess size
    if (size_initial_guess == 0) {
      size_initial_guess = 2 * neigen;
    }

    // target the lowest diagonal element
    Eigen::MatrixXd X = setupInitialEigenvectors(size_initial_guess);
    std::cout << X.transpose() * K * X << std::endl;
    X = KOrthogonalize(X, Eigen::MatrixXd(0, 0), K);
    std::cout << "after" << std::endl;
    std::cout << X.transpose() * K * X << std::endl;

    X = KOrthogonalize(X, Eigen::MatrixXd(0, 0), K);
    std::cout << "after2" << std::endl;
    std::cout << X.transpose() * K * X << std::endl;
  }

 private:
  template <typename MatrixReplacement>
  Eigen::MatrixXd KOrthogonalize(const Eigen::MatrixXd &Residues,
                                 const Eigen::MatrixXd &S,
                                 const MatrixReplacement &K) const {

    Eigen::MatrixXd TR = _preconditioner.cwiseInverse().asDiagonal() * Residues;
    Eigen::MatrixXd W;
    if (S.size() == 0) {
      W = TR;
    } else {
      W = TR - S * S.transpose() * K * TR;
    }
    Eigen::MatrixXd WW = W.transpose() * K * W;
    Eigen::LLT<Eigen::MatrixXd> cholesky(WW);
    Eigen::MatrixXd L = cholesky.matrixL();
    return W * L.inverse();
  }

  Eigen::MatrixXd setupInitialEigenvectors(Index size_initial_guess) const;
  ArrayXl argsort(const Eigen::VectorXd &V) const;
};

}  // namespace xtp
}  // namespace votca

#endif  // __VOTCA_TOOLS_DAVIDSON_SOLVER_H
