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

    std::chrono::time_point<std::chrono::system_clock> start =
        std::chrono::system_clock::now();

    // in the paper ApB and AmB are referred to as M and K respectively, we will
    // do the same internally
    const MatrixReplacement1 &M = ApB;
    const MatrixReplacement2 &K = AmB;

    // diagonal preconditioner
    _preconditioner = M.diagonal().cwiseProduct(K.diagonal());

    if (_max_search_space < neigen) {
      _max_search_space = neigen * 5;
    }
    Index op_size = M.rows();

    checkOptions(2 * op_size);
    printOptions(2 * op_size);

    // initial guess size
    if (size_initial_guess == 0) {
      size_initial_guess = 2 * neigen;
    }

    // target the lowest diagonal element
    Eigen::MatrixXd X =
        setupInitialEigenvectors(size_initial_guess + getSizeUpdate(neigen));
    Eigen::MatrixXd Xm1 = Sm1(X.transpose() * (K * X));
    X *= Xm1;
    Eigen::MatrixXd Y = K * X;
    Eigen::MatrixXd Ym = M * Y;

    Eigen::MatrixXd S = X;
    Eigen::MatrixXd KS = Y;
    Eigen::MatrixXd MKS = Ym;
    double MK_norm = ApproxMatrixNorm(M, K);
    XTP_LOG(Log::error, _log)
        << TimeStamp() << "Approximate Matrix norm of (A+B)*(A-B)=" << MK_norm
        << " Hrt^2" << flush;
    XTP_LOG(Log::error, _log)
        << TimeStamp() << " iter\tSearch Space\tNorm" << flush;

    for (_i_iter = 0; _i_iter < _iter_max; _i_iter++) {

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(KS.transpose() * MKS);
      Index sizeupdate = getSizeUpdate(neigen);
      if (sizeupdate > S.cols()) {
        sizeupdate = S.cols();
      }
      const auto eigvec = es.eigenvectors().leftCols(sizeupdate);
      X = S * eigvec;
      Y = KS * eigvec;
      Ym = MKS * eigvec;
      // MKL needs the resize
      _eigenvalues.resize(neigen);
      _eigenvalues = es.eigenvalues().head(neigen).cwiseSqrt();
      Index nact = 0;
      Eigen::MatrixXd W;
      double rmax = 0.0;
      for (Index j = 0; j < sizeupdate; j++) {
        Eigen::VectorXd r = Ym.col(j) - X.col(j) * es.eigenvalues()(j);
        rmax = std::max(rmax, r.norm() / (MK_norm + es.eigenvalues()(j)));
        if (r.norm() > _tol * (MK_norm + es.eigenvalues()(j))) {
          r = -r.array() / (_preconditioner.array() - es.eigenvalues()(j));
          W.conservativeResize(r.size(), W.cols() + 1);
          W.col(W.cols() - 1) = r;
          if (j < neigen) {
            nact++;
          }
        }
      }

      double percent_converged = 100 * double(neigen - nact) / double(neigen);
      XTP_LOG(Log::error, _log)
          << TimeStamp()
          << boost::format(" %1$4d %2$12d \t %3$4.2e \t %4$5.2f%% converged") %
                 _i_iter % S.cols() % rmax % percent_converged
          << std::flush;

      // converged
      if (nact == 0) {
        _info = Eigen::ComputationInfo::Success;
        break;
      }

      if (_i_iter == _iter_max - 1) {
        XTP_LOG(Log::error, _log)
            << TimeStamp() << "Diagonalisation did not converge after "
            << _iter_max
            << " iterations\n Try increasing the number of iterations"
            << std::flush;
      }

      // restart
      if (S.cols() + sizeupdate > _max_search_space) {
        S = X;
        KS = Y;
        MKS = Ym;
      }
      // orthogoanlize W with respect to S
      // twice is enough Gram Schmidt
      W -= S * (KS.transpose() * W);
      W.colwise().normalize();
      W -= S * (KS.transpose() * W);

      Eigen::MatrixXd Wk = K * W;
      Eigen::MatrixXd Wmk = M * Wk;
      // orthogonalize colums of W with respect to each other
      Eigen::MatrixXd R = Sm1(W.transpose() * Wk);
      W *= R;
      Wk *= R;
      Wmk *= R;
      // increase search space
      AppendMatrixToMatrix(S, W);
      AppendMatrixToMatrix(KS, Wk);
      AppendMatrixToMatrix(MKS, Wmk);
    }

    Y = Y.leftCols(neigen) * (_eigenvalues.cwiseInverse().asDiagonal());
    X.conservativeResize(Eigen::NoChange, neigen);
    _eigenvectors = 0.5 * (X + Y) * _eigenvalues.cwiseSqrt().asDiagonal();
    _eigenvectors2 = 0.5 * (Y - X) * _eigenvalues.cwiseSqrt().asDiagonal();

    printTiming(start);
  }

  Eigen::MatrixXd eigenvectors2() const { return this->_eigenvectors2; }

 private:
  template <typename MatrixReplacement1, typename MatrixReplacement2>
  double ApproxMatrixNorm(const MatrixReplacement1 &M,
                          const MatrixReplacement2 &K) const {
    Eigen::VectorXd e = Eigen::VectorXd::Ones(K.rows());
    e.normalize();
    e = K * e;
    return (M * e).sum();
  }

  Eigen::MatrixXd Sm1(const Eigen::MatrixXd &m) const;

  void AppendMatrixToMatrix(Eigen::MatrixXd &A, const Eigen::MatrixXd &B) const;

  Eigen::MatrixXd setupInitialEigenvectors(Index size_initial_guess) const;
  ArrayXl argsort(const Eigen::VectorXd &V) const;

  Eigen::MatrixXd _eigenvectors2;
};

}  // namespace xtp
}  // namespace votca

#endif  // __VOTCA_TOOLS_DAVIDSON_SOLVER_H
