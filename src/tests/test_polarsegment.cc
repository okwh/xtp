/*
 * Copyright 2009-2019 The VOTCA Development Team (http://www.votca.org)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
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
#define BOOST_TEST_MAIN

#define BOOST_TEST_MODULE polarsegment_test
#include <boost/test/unit_test.hpp>
#include <votca/xtp/classicalsegment.h>
#include <votca/tools/constants.h>

using namespace votca::xtp;
using namespace std;

BOOST_AUTO_TEST_SUITE(polarsegment_test)

BOOST_AUTO_TEST_CASE(constructors_test) { PolarSegment("seg1", 0); }

BOOST_AUTO_TEST_CASE(load_mps) {

  PolarSegment seg = PolarSegment("seg1", 0);
  std::ofstream mpsfile("polarsite.mps");
  mpsfile << "! One Site" << endl;
  mpsfile << "! N=1 " << endl;
  mpsfile << "Units angstrom" << endl;
  mpsfile << "  C +0 0 3 Rank 2" << endl;
  mpsfile << "+1" << endl;
  mpsfile << "10 0 0" << endl;
  mpsfile << "     100 0 0 0 0" << endl;
  mpsfile
      << "P +1.9445387 +0.0000000 +0.0000000 +1.9445387 +0.0000000 +1.9445387 "
      << endl;

  seg.LoadFromFile("polarsite.mps");
  Eigen::Vector3d ref_pos =
      Eigen::Vector3d(0, 0, 3 * votca::tools::conv::ang2bohr);

  bool is_equal = seg.getPos().isApprox(ref_pos, 0.0001);
  if (!is_equal) {
    std::cout << "result" << std::endl;
    std::cout << seg.getPos() << std::endl;
    std::cout << "reference" << std::endl;
    std::cout << ref_pos << std::endl;
  }

  BOOST_CHECK_EQUAL(is_equal, true);

  BOOST_CHECK_EQUAL(seg[0].getRank(), 2);
  BOOST_CHECK_EQUAL(seg[0].getElement(), "C");

  Eigen::VectorXd mul_ref = Eigen::VectorXd::Zero(9);
  mul_ref << 1, 10, 0, 0, 100, 0, 0, 0, 0;
  bool multipoles_equal = mul_ref.isApprox(seg[0].getPermMultipole(), 1e-5);
  if (!multipoles_equal) {
    std::cout << "result" << std::endl;
    std::cout << seg[0].getPermMultipole() << std::endl;
    std::cout << "reference" << std::endl;
    std::cout << mul_ref << std::endl;
  }

  std::string ref_string =
      "  C +0.0000000 +0.0000000 +3.0000000 Rank 2\n"
      "    +1.0000000\n"
      "    +10.0000000 +0.0000000 +0.0000000\n"
      "    +100.0000000 +0.0000000 +0.0000000 +0.0000000 +0.0000000\n"
      "     P +1.9445387 +0.0000000 +0.0000000 +1.9445387 +0.0000000 "
      "+1.9445387\n";
  bool string_equal = (ref_string == seg[0].WriteMpsLine("angstrom"));
  if (!string_equal) {
    std::string result = seg[0].WriteMpsLine("angstrom");
    std::cout << "result" << std::endl;
    std::cout << result << std::endl;
    std::cout << "reference" << std::endl;
    std::cout << ref_string << std::endl;
  }

  BOOST_CHECK_EQUAL(string_equal, true);
}

BOOST_AUTO_TEST_CASE(add_atom_test) {
  PolarSegment seg = PolarSegment("seg1", 0);
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  PolarSite site = PolarSite(0, "C", pos);
  Eigen::VectorXd poles = Vector9d::Ones(9);
  int rank = 2;
  site.setMultipole(poles, rank);
  seg.push_back(site);
}

BOOST_AUTO_TEST_SUITE_END()
