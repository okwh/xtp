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
using the Paper
https://www.sciencedirect.com/science/article/pii/S0010465517302370
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
    Eigen::MatrixXd Xm1 = Sm1(X.transpose() * (K * X));
    X = X * Xm1;
    Eigen::MatrixXd Y = K * X;
    Eigen::MatrixXd Ym = M * Y;

    Eigen::MatrixXd S = X;
    Eigen::MatrixXd KS = Y;
    Eigen::MatrixXd MKS = Ym;

    Eigen::VectorXd e = Eigen::VectorXd::Ones(K.rows());
    e.normalize();
    double MK_norm = (M * (K * e)).sum();
    std::cout << "MNorm:" << MK_norm << std::endl;
    std::cout << "Search space:" << _max_search_space
              << " initial guess:" << size_initial_guess << " neigen:" << neigen
              << std::endl;
    for (_i_iter = 0; _i_iter < _iter_max; _i_iter++) {

      Eigen::MatrixXd mat = (KS).transpose() * MKS;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(mat);
      X = S * es.eigenvectors().leftCols(neigen);
      Y = KS * es.eigenvectors().leftCols(neigen);
      Ym = MKS * es.eigenvectors().leftCols(neigen);
      _eigenvalues.resize(es.eigenvalues().size());
      _eigenvalues = es.eigenvalues().cwiseSqrt();
      Index nact = 0;
      Eigen::MatrixXd W;
      for (Index j = 0; j < neigen; j++) {
        Eigen::VectorXd r = Ym.col(j) - X.col(j) * es.eigenvalues()(j);
        if (r.norm() > _tol * (MK_norm + es.eigenvalues()(j))) {
          r = -r.array() / (_preconditioner.array() - es.eigenvalues()(j));
          W.conservativeResize(r.size(), W.cols() + 1);
          W.col(W.cols() - 1) = r;
          nact++;
        }
      }
      std::cout << "not converged:" << nact << " of " << neigen << std::endl;
      if (nact == 0) {
        break;
      }

      if (S.cols() + nact > _max_search_space) {
        S = X;
        KS = Y;
        MKS = Ym;
      }

      W = W - S * (KS.transpose() * W);
      W.colwise().normalize();
      W = W - S * (KS.transpose() * W);
      // twice is enough Gram Schmidt
      Eigen::MatrixXd Wk = K * W;
      Eigen::MatrixXd Wmk = M * Wk;

      Eigen::MatrixXd overlap = W.transpose() * Wk;
      Eigen::MatrixXd R = Sm1(overlap);
      W = W * R;
      Wk = Wk * R;
      Wmk = Wmk * R;

      S.conservativeResize(Eigen::NoChange, S.cols() + W.cols());
      S.rightCols(W.cols()) = W;
      KS.conservativeResize(Eigen::NoChange, KS.cols() + Wk.cols());
      KS.rightCols(Wk.cols()) = Wk;
      std::cout << S.transpose() * KS << std::endl;
      MKS.conservativeResize(Eigen::NoChange, MKS.cols() + Wmk.cols());
      MKS.rightCols(Wmk.cols()) = Wmk;
    }

    _eigenvalues.conservativeResize(neigen);
    std::cout << "X.transpose()*K*X" << std::endl;
    std::cout << X.transpose() * K * X << std::endl;
    Y *= (_eigenvalues.cwiseInverse().asDiagonal());
    std::cout << "Y.transpose()*M*Y" << std::endl;
    std::cout << Y.transpose() * M * Y << std::endl;
    _eigenvectors = (X + Y) / std::sqrt(2);
    _eigenvectors2 = (Y - X) / std::sqrt(2);
    std::cout << _eigenvalues << std::endl;
    std::cout << "overlap" << std::endl;
    std::cout << _eigenvectors.transpose() * _eigenvectors -
                     _eigenvectors2.transpose() * _eigenvectors2
              << std::endl;
    std::cout << "overlap2" << std::endl;
    std::cout << _eigenvectors.transpose() * _eigenvectors2 -
                     _eigenvectors2.transpose() * _eigenvectors
              << std::endl;
  }

  Eigen::MatrixXd eigenvectors2() const { return this->_eigenvectors2; }

 private:
  Eigen::MatrixXd Sm1(const Eigen::MatrixXd &m) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(m);
    Eigen::VectorXd diagonal = Eigen::VectorXd::Zero(es.eigenvalues().size());
    double etol = 1e-7;
    Index removedfunctions = 0;
    for (Index i = 0; i < diagonal.size(); ++i) {
      if (es.eigenvalues()(i) > etol) {
        diagonal(i) = 1.0 / std::sqrt(es.eigenvalues()(i));
      } else {
        removedfunctions++;
      }
    }
    return es.eigenvectors() * diagonal.asDiagonal() *
           es.eigenvectors().transpose().rightCols(es.eigenvalues().size() -
                                                   removedfunctions);
  }

  Eigen::MatrixXd setupInitialEigenvectors(Index size_initial_guess) const;
  ArrayXl argsort(const Eigen::VectorXd &V) const;

  Eigen::MatrixXd _eigenvectors2;
};

}  // namespace xtp
}  // namespace votca

#endif  // __VOTCA_TOOLS_DAVIDSON_SOLVER_H
