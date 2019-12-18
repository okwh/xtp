
#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE davidson_test

#include <boost/test/unit_test.hpp>
#include <iostream>

#include <votca/xtp/davidsonsolver.h>
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

BOOST_AUTO_TEST_SUITE(davidson_test)

BOOST_AUTO_TEST_CASE(davidson_full_matrix) {

  Index size = 100;
  Index neigen = 10;
  double eps = 0.01;
  Eigen::MatrixXd A = init_matrix(size, eps);
  Logger log;
  DavidsonSolver DS(log);
  DS.solve(A, neigen);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);

  auto lambda = DS.eigenvalues();
  auto lambda_ref = es.eigenvalues().head(neigen);
  bool check_eigenvalues = lambda.isApprox(lambda_ref, 1E-6);
  if (!check_eigenvalues) {
    std::cout << "ref" << std::endl;
    std::cout << es.eigenvalues().head(neigen).transpose() << std::endl;
    std::cout << "result" << std::endl;
    std::cout << DS.eigenvalues().transpose() << std::endl;
  }

  BOOST_CHECK_EQUAL(check_eigenvalues, 1);
}

BOOST_AUTO_TEST_CASE(davidson_full_matrix_large) {

  Index size = 400;
  Index neigen = 10;
  double eps = 0.01;
  Eigen::MatrixXd A = init_matrix(size, eps);
  Logger log;
  DavidsonSolver DS(log);
  DS.solve(A, neigen);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);

  auto lambda = DS.eigenvalues();
  auto lambda_ref = es.eigenvalues().head(neigen);
  bool check_eigenvalues = lambda.isApprox(lambda_ref, 1E-6);
  if (!check_eigenvalues) {
    std::cout << "ref" << std::endl;
    std::cout << es.eigenvalues().head(neigen).transpose() << std::endl;
    std::cout << "result" << std::endl;
    std::cout << DS.eigenvalues().transpose() << std::endl;
  }

  BOOST_CHECK_EQUAL(check_eigenvalues, 1);
}

BOOST_AUTO_TEST_CASE(davidson_full_matrix_fail) {

  Index size = 100;
  Index neigen = 10;
  double eps = 0.01;
  Eigen::MatrixXd A = init_matrix(size, eps);

  Logger log;
  DavidsonSolver DS(log);
  DS.set_iter_max(1);
  DS.solve(A, neigen);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);

  auto lambda = DS.eigenvalues();
  auto lambda_ref = es.eigenvalues().head(neigen);
  bool check_eigenvalues = lambda.isApprox(lambda_ref, 1E-6);

  BOOST_CHECK_EQUAL(check_eigenvalues, 0);
}

class TestOperator : public MatrixFreeOperator {
 public:
  TestOperator() = default;
  Eigen::RowVectorXd OperatorRow(Index index) const override;

 private:
};

//  get a col of the operator
Eigen::RowVectorXd TestOperator::OperatorRow(Index index) const {
  Index lsize = this->size();
  Eigen::RowVectorXd row_out = Eigen::RowVectorXd::Zero(lsize);
  for (Index j = 0; j < lsize; j++) {
    if (j == index) {
      row_out(j) = std::sqrt(static_cast<double>(index + 1));
    } else {
      row_out(j) = 0.01 / std::pow(static_cast<double>(j - index), 2);
    }
  }
  return row_out;
}

BOOST_AUTO_TEST_CASE(davidson_matrix_free) {

  Index size = 100;
  Index neigen = 10;

  // Create Operator
  TestOperator Aop;
  Aop.set_size(size);

  Logger log;
  DavidsonSolver DS(log);
  DS.set_tolerance("normal");
  DS.set_size_update("safe");
  DS.solve(Aop, neigen);

  Eigen::MatrixXd A = Aop.get_full_matrix();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);

  auto lambda = DS.eigenvalues();
  auto lambda_ref = es.eigenvalues().head(neigen);
  bool check_eigenvalues = lambda.isApprox(lambda_ref, 1E-6);

  BOOST_CHECK_EQUAL(check_eigenvalues, 1);
  if (!check_eigenvalues) {
    std::cout << "ref" << std::endl;
    std::cout << es.eigenvalues().head(neigen).transpose() << std::endl;
    std::cout << "result" << std::endl;
    std::cout << DS.eigenvalues().transpose() << std::endl;
  }
}

BOOST_AUTO_TEST_CASE(davidson_matrix_free_large) {

  Index size = 400;
  Index neigen = 10;

  // Create Operator
  TestOperator Aop;
  Aop.set_size(size);

  Logger log;
  DavidsonSolver DS(log);
  DS.set_tolerance("normal");
  DS.set_size_update("safe");
  DS.solve(Aop, neigen);
  std::cout << log << std::endl;
  Eigen::MatrixXd A = Aop.get_full_matrix();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);

  auto lambda = DS.eigenvalues();
  auto lambda_ref = es.eigenvalues().head(neigen);
  bool check_eigenvalues = lambda.isApprox(lambda_ref, 1E-6);

  BOOST_CHECK_EQUAL(check_eigenvalues, 1);
  if (!check_eigenvalues) {
    std::cout << "ref" << std::endl;
    std::cout << es.eigenvalues().head(neigen).transpose() << std::endl;
    std::cout << "result" << std::endl;
    std::cout << DS.eigenvalues().transpose() << std::endl;
  }
}

class BlockOperator : public MatrixFreeOperator {
 public:
  BlockOperator() = default;
  Eigen::MatrixXd OperatorBlock(Index row, Index col) const override;

  bool useRow() const override { return false; }
  bool useBlock() const override { return true; }
  Index getBlocksize() const override { return size() / 10; }

 private:
};

//  get a block of the operator
Eigen::MatrixXd BlockOperator::OperatorBlock(Index row, Index col) const {
  Index blocksize = getBlocksize();
  Eigen::MatrixXd block = Eigen::MatrixXd::Zero(blocksize, blocksize);
  Index blocdisttodiagonal = std::abs(row - col) * blocksize;
  for (Index i_col = 0; i_col < blocksize; i_col++) {
    for (Index i_row = 0; i_row < blocksize; i_row++) {
      block(i_row, i_col) =
          0.01 / std::pow(static_cast<double>(std::abs(i_row - i_col) +
                                              blocdisttodiagonal),
                          2);
    }
  }
  if (blocdisttodiagonal == 0) {
    for (Index i = 0; i < blocksize; i++) {
      block(i, i) = std::sqrt(static_cast<double>(row * blocksize + i + 1));
    }
  }

  return block;
}

BOOST_AUTO_TEST_CASE(davidson_matrix_free_block) {

  Index size = 100;
  Index neigen = 10;

  // Create Operator
  BlockOperator Aop;
  Aop.set_size(size);

  Logger log;
  DavidsonSolver DS(log);
  DS.set_tolerance("normal");
  DS.set_size_update("safe");

  Eigen::MatrixXd A = Aop.get_full_matrix();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
  DS.solve(Aop, neigen);
  auto lambda = DS.eigenvalues();
  auto lambda_ref = es.eigenvalues().head(neigen);
  bool check_eigenvalues = lambda.isApprox(lambda_ref, 1E-6);

  BOOST_CHECK_EQUAL(check_eigenvalues, 1);
  if (!check_eigenvalues) {
    std::cout << "ref" << std::endl;
    std::cout << es.eigenvalues().head(neigen).transpose() << std::endl;
    std::cout << "result" << std::endl;
    std::cout << DS.eigenvalues().transpose() << std::endl;
  }
}

BOOST_AUTO_TEST_CASE(davidson_matrix_free_block_large) {

  Index size = 400;
  Index neigen = 10;

  // Create Operator
  BlockOperator Aop;
  Aop.set_size(size);

  Logger log;
  DavidsonSolver DS(log);
  DS.set_tolerance("normal");
  DS.set_size_update("safe");

  Eigen::MatrixXd A = Aop.get_full_matrix();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
  DS.solve(Aop, neigen);
  auto lambda = DS.eigenvalues();
  auto lambda_ref = es.eigenvalues().head(neigen);
  bool check_eigenvalues = lambda.isApprox(lambda_ref, 1E-6);

  BOOST_CHECK_EQUAL(check_eigenvalues, 1);
  if (!check_eigenvalues) {
    std::cout << "ref" << std::endl;
    std::cout << es.eigenvalues().head(neigen).transpose() << std::endl;
    std::cout << "result" << std::endl;
    std::cout << DS.eigenvalues().transpose() << std::endl;
  }
}

BOOST_AUTO_TEST_SUITE_END()
