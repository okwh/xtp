
#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE davidson_btda_test

#include <boost/test/unit_test.hpp>
#include <iostream>

#include <votca/xtp/davidsonsolver_btda.h>
#include <votca/xtp/eigen.h>
#include <votca/xtp/matrixfreeoperator.h>

using namespace votca::xtp;
using namespace votca;

Eigen::MatrixXd symm_matrix(Index N, double eps) {
  Eigen::MatrixXd matrix;
  matrix = eps * Eigen::MatrixXd::Random(N, N);
  Eigen::MatrixXd tmat = matrix.transpose();
  matrix = matrix + tmat;
  return matrix;
}

Eigen::MatrixXd init_matrix(Index N, double eps) {
  Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(N, N);
  for (Index i = 0; i < N; i++) {
    for (Index j = i; j < N; j++) {
      if (i == j) {
        matrix(i, i) = std::sqrt(static_cast<double>(1 + i));
      } else {
        matrix(i, j) = eps / std::pow(static_cast<double>(j - i), 2);
        matrix(j, i) = eps / std::pow(static_cast<double>(j - i), 2);
      }
    }
  }
  return matrix;
}

BOOST_AUTO_TEST_SUITE(davidson_btda_test)

Eigen::ArrayXi index_eval(Eigen::VectorXd ev, Index neigen) {

  Index nev = ev.rows();
  Index npos = nev / 2;

  Eigen::ArrayXi idx = Eigen::ArrayXi::Zero(npos);
  Index nstored = 0;

  // get only positives
  for (Index i = 0; i < nev; i++) {
    if (ev(i) > 0) {
      idx(nstored) = int(i);
      nstored++;
    }
  }

  // sort the epos eigenvalues
  std::sort(idx.data(), idx.data() + idx.size(),
            [&](Index i1, Index i2) { return ev[i1] < ev[i2]; });
  return idx.head(neigen);
}

Eigen::MatrixXd extract_eigenvectors(const Eigen::MatrixXd &V,
                                     const Eigen::ArrayXi &idx) {
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(V.rows(), idx.size());
  for (Index i = 0; i < idx.size(); i++) {
    W.col(i) = V.col(idx(i));
  }
  return W;
}

BOOST_AUTO_TEST_CASE(davidson_btda_matrix) {

  Index size = 60;
  Index neigen = 5;
  Logger log;

  Eigen::MatrixXd rmat = init_matrix(size, 0.01);
  Eigen::MatrixXd cmat = symm_matrix(size, 0.01);

  Eigen::MatrixXd ApB = rmat + cmat;
  Eigen::MatrixXd AmB = rmat - cmat;

  Eigen::MatrixXd large = Eigen::MatrixXd::Zero(2 * size, 2 * size);
  large.topLeftCorner(size, size) = rmat;
  large.topRightCorner(size, size) = cmat;
  large.bottomRightCorner(size, size) = -rmat;
  large.bottomLeftCorner(size, size) = -cmat;
  DavidsonSolver_BTDA DS(log);
  DS.set_tolerance("normal");
  DS.set_size_update("max");
  DS.set_iter_max(30);
  DS.solve(ApB, AmB, neigen, 30);
  auto lambda = DS.eigenvalues().real();
  std::sort(lambda.data(), lambda.data() + lambda.size());

  Eigen::EigenSolver<Eigen::MatrixXd> es(large);
  Eigen::ArrayXi idx = index_eval(es.eigenvalues().real(), neigen);
  Eigen::VectorXd lambda_ref = idx.unaryExpr(es.eigenvalues().real());
  bool check_eigenvalues = lambda.isApprox(lambda_ref.head(neigen), 1E-6);
  if (!check_eigenvalues) {
    std::cout << "Reference eigenvalues" << std::endl;
    std::cout << lambda_ref.head(neigen) << std::endl;
    std::cout << "Davidson eigenvalues" << std::endl;
    std::cout << lambda << std::endl;
  }
  BOOST_CHECK_EQUAL(check_eigenvalues, 1);
}

BOOST_AUTO_TEST_SUITE_END()
