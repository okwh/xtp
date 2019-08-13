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

#define BOOST_TEST_MODULE aomatrix_test
#include "votca/xtp/orbitals.h"
#include <boost/test/unit_test.hpp>
#include <fstream>
#include <votca/xtp/sphere_lebedev_rule.h>
using namespace votca::xtp;
using namespace std;

BOOST_AUTO_TEST_SUITE(sphere_lebedev_rule_test)

BOOST_AUTO_TEST_CASE(setup_test) {

  ofstream xyzfile("molecule.xyz");
  xyzfile << " 5" << endl;
  xyzfile << " methane" << endl;
  xyzfile << " C            .000000     .000000     .000000" << endl;
  xyzfile << " H            .629118     .629118     .629118" << endl;
  xyzfile << " H           -.629118    -.629118     .629118" << endl;
  xyzfile << " H            .629118    -.629118    -.629118" << endl;
  xyzfile << " H           -.629118     .629118    -.629118" << endl;
  xyzfile.close();

  ofstream basisfile("3-21G.xml");
  basisfile << "<basis name=\"3-21G\">" << endl;
  basisfile << "  <element name=\"H\">" << endl;
  basisfile << "    <shell scale=\"1.0\" type=\"S\">" << endl;
  basisfile << "      <constant decay=\"5.447178e+00\">" << endl;
  basisfile << "        <contractions factor=\"1.562850e-01\" type=\"S\"/>"
            << endl;
  basisfile << "      </constant>" << endl;
  basisfile << "      <constant decay=\"8.245470e-01\">" << endl;
  basisfile << "        <contractions factor=\"9.046910e-01\" type=\"S\"/>"
            << endl;
  basisfile << "      </constant>" << endl;
  basisfile << "    </shell>" << endl;
  basisfile << "    <shell scale=\"1.0\" type=\"S\">" << endl;
  basisfile << "      <constant decay=\"1.831920e-01\">" << endl;
  basisfile << "        <contractions factor=\"1.000000e+00\" type=\"S\"/>"
            << endl;
  basisfile << "      </constant>" << endl;
  basisfile << "    </shell>" << endl;
  basisfile << "  </element>" << endl;
  basisfile << "  <element name=\"C\">" << endl;
  basisfile << "    <shell scale=\"1.0\" type=\"S\">" << endl;
  basisfile << "      <constant decay=\"1.722560e+02\">" << endl;
  basisfile << "        <contractions factor=\"6.176690e-02\" type=\"S\"/>"
            << endl;
  basisfile << "      </constant>" << endl;
  basisfile << "      <constant decay=\"2.591090e+01\">" << endl;
  basisfile << "        <contractions factor=\"3.587940e-01\" type=\"S\"/>"
            << endl;
  basisfile << "      </constant>" << endl;
  basisfile << "      <constant decay=\"5.533350e+00\">" << endl;
  basisfile << "        <contractions factor=\"7.007130e-01\" type=\"S\"/>"
            << endl;
  basisfile << "      </constant>" << endl;
  basisfile << "    </shell>" << endl;
  basisfile << "    <shell scale=\"1.0\" type=\"SP\">" << endl;
  basisfile << "      <constant decay=\"3.664980e+00\">" << endl;
  basisfile << "        <contractions factor=\"-3.958970e-01\" type=\"S\"/>"
            << endl;
  basisfile << "        <contractions factor=\"2.364600e-01\" type=\"P\"/>"
            << endl;
  basisfile << "      </constant>" << endl;
  basisfile << "      <constant decay=\"7.705450e-01\">" << endl;
  basisfile << "        <contractions factor=\"1.215840e+00\" type=\"S\"/>"
            << endl;
  basisfile << "        <contractions factor=\"8.606190e-01\" type=\"P\"/>"
            << endl;
  basisfile << "      </constant>" << endl;
  basisfile << "    </shell>" << endl;
  basisfile << "    <shell scale=\"1.0\" type=\"SP\">" << endl;
  basisfile << "      <constant decay=\"1.958570e-01\">" << endl;
  basisfile << "        <contractions factor=\"1.000000e+00\" type=\"S\"/>"
            << endl;
  basisfile << "        <contractions factor=\"1.000000e+00\" type=\"P\"/>"
            << endl;
  basisfile << "      </constant>" << endl;
  basisfile << "    </shell>" << endl;
  basisfile << "  </element>" << endl;
  basisfile << "</basis>" << endl;
  basisfile.close();

  Orbitals orbitals;
  orbitals.QMAtoms().LoadFromFile("molecule.xyz");
  BasisSet basis;
  basis.Load("3-21G.xml");
  AOBasis aobasis;
  aobasis.Fill(basis, orbitals.QMAtoms());

  LebedevGrid spheregrid;

  auto grid = spheregrid.CalculateSphericalGrids(orbitals.QMAtoms(), "medium");

  auto Hgrid = grid.at("H");
  auto Cgrid = grid.at("C");

  Eigen::VectorXd C_phi_ref = Eigen::VectorXd::Zero(434);
  C_phi_ref << 90, 90, 90, 90, 0, 180, 45, 45, 135, 135, 45, 45, 135, 135, 90,
      90, 90, 90, 54.7356, 54.7356, 54.7356, 54.7356, 125.264, 125.264, 125.264,
      125.264, 77.7225, 77.7225, 77.7225, 77.7225, 102.278, 102.278, 102.278,
      102.278, 46.2959, 46.2959, 46.2959, 46.2959, 133.704, 133.704, 133.704,
      133.704, 46.2959, 46.2959, 46.2959, 46.2959, 133.704, 133.704, 133.704,
      133.704, 14.5367, 14.5367, 14.5367, 14.5367, 165.463, 165.463, 165.463,
      165.463, 79.7768, 79.7768, 79.7768, 79.7768, 100.223, 100.223, 100.223,
      100.223, 79.7768, 79.7768, 79.7768, 79.7768, 100.223, 100.223, 100.223,
      100.223, 44.0267, 44.0267, 44.0267, 44.0267, 135.973, 135.973, 135.973,
      135.973, 60.5651, 60.5651, 60.5651, 60.5651, 119.435, 119.435, 119.435,
      119.435, 60.5651, 60.5651, 60.5651, 60.5651, 119.435, 119.435, 119.435,
      119.435, 65.9388, 65.9388, 65.9388, 65.9388, 114.061, 114.061, 114.061,
      114.061, 49.7843, 49.7843, 49.7843, 49.7843, 130.216, 130.216, 130.216,
      130.216, 49.7843, 49.7843, 49.7843, 49.7843, 130.216, 130.216, 130.216,
      130.216, 23.869, 23.869, 23.869, 23.869, 156.131, 156.131, 156.131,
      156.131, 73.3737, 73.3737, 73.3737, 73.3737, 106.626, 106.626, 106.626,
      106.626, 73.3737, 73.3737, 73.3737, 73.3737, 106.626, 106.626, 106.626,
      106.626, 6.14407, 6.14407, 6.14407, 6.14407, 173.856, 173.856, 173.856,
      173.856, 85.6597, 85.6597, 85.6597, 85.6597, 94.3403, 94.3403, 94.3403,
      94.3403, 85.6597, 85.6597, 85.6597, 85.6597, 94.3403, 94.3403, 94.3403,
      94.3403, 33.7382, 33.7382, 33.7382, 33.7382, 146.262, 146.262, 146.262,
      146.262, 66.8758, 66.8758, 66.8758, 66.8758, 113.124, 113.124, 113.124,
      113.124, 66.8758, 66.8758, 66.8758, 66.8758, 113.124, 113.124, 113.124,
      113.124, 90, 90, 90, 90, 90, 90, 90, 90, 61.8619, 61.8619, 118.138,
      118.138, 28.1381, 28.1381, 151.862, 151.862, 61.8619, 61.8619, 118.138,
      118.138, 28.1381, 28.1381, 151.862, 151.862, 90, 90, 90, 90, 90, 90, 90,
      90, 77.8617, 77.8617, 102.138, 102.138, 12.1383, 12.1383, 167.862,
      167.862, 77.8617, 77.8617, 102.138, 102.138, 12.1383, 12.1383, 167.862,
      167.862, 63.2414, 63.2414, 63.2414, 63.2414, 116.759, 116.759, 116.759,
      116.759, 29.6636, 29.6636, 29.6636, 29.6636, 150.336, 150.336, 150.336,
      150.336, 63.2414, 63.2414, 63.2414, 63.2414, 116.759, 116.759, 116.759,
      116.759, 78.1423, 78.1423, 78.1423, 78.1423, 101.858, 101.858, 101.858,
      101.858, 29.6636, 29.6636, 29.6636, 29.6636, 150.336, 150.336, 150.336,
      150.336, 78.1423, 78.1423, 78.1423, 78.1423, 101.858, 101.858, 101.858,
      101.858, 83.869, 83.869, 83.869, 83.869, 96.131, 96.131, 96.131, 96.131,
      36.8768, 36.8768, 36.8768, 36.8768, 143.123, 143.123, 143.123, 143.123,
      83.869, 83.869, 83.869, 83.869, 96.131, 96.131, 96.131, 96.131, 53.8064,
      53.8064, 53.8064, 53.8064, 126.194, 126.194, 126.194, 126.194, 36.8768,
      36.8768, 36.8768, 36.8768, 143.123, 143.123, 143.123, 143.123, 53.8064,
      53.8064, 53.8064, 53.8064, 126.194, 126.194, 126.194, 126.194, 71.915,
      71.915, 71.915, 71.915, 108.085, 108.085, 108.085, 108.085, 39.489,
      39.489, 39.489, 39.489, 140.511, 140.511, 140.511, 140.511, 71.915,
      71.915, 71.915, 71.915, 108.085, 108.085, 108.085, 108.085, 56.2882,
      56.2882, 56.2882, 56.2882, 123.712, 123.712, 123.712, 123.712, 39.489,
      39.489, 39.489, 39.489, 140.511, 140.511, 140.511, 140.511, 56.2882,
      56.2882, 56.2882, 56.2882, 123.712, 123.712, 123.712, 123.712, 84.3059,
      84.3059, 84.3059, 84.3059, 95.6941, 95.6941, 95.6941, 95.6941, 70.4617,
      70.4617, 70.4617, 70.4617, 109.538, 109.538, 109.538, 109.538, 84.3059,
      84.3059, 84.3059, 84.3059, 95.6941, 95.6941, 95.6941, 95.6941, 20.4166,
      20.4166, 20.4166, 20.4166, 159.583, 159.583, 159.583, 159.583, 70.4617,
      70.4617, 70.4617, 70.4617, 109.538, 109.538, 109.538, 109.538, 20.4166,
      20.4166, 20.4166, 20.4166, 159.583, 159.583, 159.583, 159.583;
  C_phi_ref *= votca::tools::conv::Pi / 180.0;

  Eigen::VectorXd C_theta_ref = Eigen::VectorXd::Zero(434);
  C_theta_ref << 0, 180, 90, -90, 0, 0, 90, -90, 90, -90, 0, 180, 0, 180, 45,
      135, -45, -135, 45, 135, -45, -135, 45, 135, -45, -135, 45, 135, -45,
      -135, 45, 135, -45, -135, 17.1066, 162.893, -17.1066, -162.893, 17.1066,
      162.893, -17.1066, -162.893, 72.8934, 107.107, -72.8934, -107.107,
      72.8934, 107.107, -72.8934, -107.107, 45, 135, -45, -135, 45, 135, -45,
      -135, 79.61, 100.39, -79.61, -100.39, 79.61, 100.39, -79.61, -100.39,
      10.39, 169.61, -10.39, -169.61, 10.39, 169.61, -10.39, -169.61, 45, 135,
      -45, -135, 45, 135, -45, -135, 55.6481, 124.352, -55.6481, -124.352,
      55.6481, 124.352, -55.6481, -124.352, 34.3519, 145.648, -34.3519,
      -145.648, 34.3519, 145.648, -34.3519, -145.648, 45, 135, -45, -135, 45,
      135, -45, -135, 32.2708, 147.729, -32.2708, -147.729, 32.2708, 147.729,
      -32.2708, -147.729, 57.7292, 122.271, -57.7292, -122.271, 57.7292,
      122.271, -57.7292, -122.271, 45, 135, -45, -135, 45, 135, -45, -135,
      72.6256, 107.374, -72.6256, -107.374, 72.6256, 107.374, -72.6256,
      -107.374, 17.3744, 162.626, -17.3744, -162.626, 17.3744, 162.626,
      -17.3744, -162.626, 45, 135, -45, -135, 45, 135, -45, -135, 85.6471,
      94.3529, -85.6471, -94.3529, 85.6471, 94.3529, -85.6471, -94.3529,
      4.35285, 175.647, -4.35285, -175.647, 4.35285, 175.647, -4.35285,
      -175.647, 45, 135, -45, -135, 45, 135, -45, -135, 64.7204, 115.28,
      -64.7204, -115.28, 64.7204, 115.28, -64.7204, -115.28, 25.2796, 154.72,
      -25.2796, -154.72, 25.2796, 154.72, -25.2796, -154.72, 28.1381, 151.862,
      -28.1381, -151.862, 61.8619, 118.138, -61.8619, -118.138, 0, 180, 0, 180,
      0, 180, 0, 180, 90, -90, 90, -90, 90, -90, 90, -90, 12.1383, 167.862,
      -12.1383, -167.862, 77.8617, 102.138, -77.8617, -102.138, 0, 180, 0, 180,
      0, 180, 0, 180, 90, -90, 90, -90, 90, -90, 90, -90, 76.6955, 103.305,
      -76.6955, -103.305, 76.6955, 103.305, -76.6955, -103.305, 65.4685,
      114.532, -65.4685, -114.532, 65.4685, 114.532, -65.4685, -114.532,
      13.3045, 166.695, -13.3045, -166.695, 13.3045, 166.695, -13.3045,
      -166.695, 27.3903, 152.61, -27.3903, -152.61, 27.3903, 152.61, -27.3903,
      -152.61, 24.5315, 155.468, -24.5315, -155.468, 24.5315, 155.468, -24.5315,
      -155.468, 62.6097, 117.39, -62.6097, -117.39, 62.6097, 117.39, -62.6097,
      -117.39, 53.5648, 126.435, -53.5648, -126.435, 53.5648, 126.435, -53.5648,
      -126.435, 10.2518, 169.748, -10.2518, -169.748, 10.2518, 169.748,
      -10.2518, -169.748, 36.4352, 143.565, -36.4352, -143.565, 36.4352,
      143.565, -36.4352, -143.565, 7.60483, 172.395, -7.60483, -172.395,
      7.60483, 172.395, -7.60483, -172.395, 79.7482, 100.252, -79.7482,
      -100.252, 79.7482, 100.252, -79.7482, -100.252, 82.3952, 97.6048,
      -82.3952, -97.6048, 82.3952, 97.6048, -82.3952, -97.6048, 54.2775,
      125.722, -54.2775, -125.722, 54.2775, 125.722, -54.2775, -125.722,
      29.2189, 150.781, -29.2189, -150.781, 29.2189, 150.781, -29.2189,
      -150.781, 35.7225, 144.278, -35.7225, -144.278, 35.7225, 144.278,
      -35.7225, -144.278, 21.912, 158.088, -21.912, -158.088, 21.912, 158.088,
      -21.912, -158.088, 60.7811, 119.219, -60.7811, -119.219, 60.7811, 119.219,
      -60.7811, -119.219, 68.088, 111.912, -68.088, -111.912, 68.088, 111.912,
      -68.088, -111.912, 19.6391, 160.361, -19.6391, -160.361, 19.6391, 160.361,
      -19.6391, -160.361, 6.04329, 173.957, -6.04329, -173.957, 6.04329,
      173.957, -6.04329, -173.957, 70.3609, 109.639, -70.3609, -109.639,
      70.3609, 109.639, -70.3609, -109.639, 16.5241, 163.476, -16.5241,
      -163.476, 16.5241, 163.476, -16.5241, -163.476, 83.9567, 96.0433,
      -83.9567, -96.0433, 83.9567, 96.0433, -83.9567, -96.0433, 73.4759,
      106.524, -73.4759, -106.524, 73.4759, 106.524, -73.4759, -106.524;
  C_theta_ref *= votca::tools::conv::Pi / 180.0;

  Eigen::VectorXd C_weight_ref = Eigen::VectorXd::Zero(434);
  C_weight_ref << 0.00661732, 0.00661732, 0.00661732, 0.00661732, 0.00661732,
      0.00661732, 0.0320219, 0.0320219, 0.0320219, 0.0320219, 0.0320219,
      0.0320219, 0.0320219, 0.0320219, 0.0320219, 0.0320219, 0.0320219,
      0.0320219, 0.0315707, 0.0315707, 0.0315707, 0.0315707, 0.0315707,
      0.0315707, 0.0315707, 0.0315707, 0.031798, 0.031798, 0.031798, 0.031798,
      0.031798, 0.031798, 0.031798, 0.031798, 0.031798, 0.031798, 0.031798,
      0.031798, 0.031798, 0.031798, 0.031798, 0.031798, 0.031798, 0.031798,
      0.031798, 0.031798, 0.031798, 0.031798, 0.031798, 0.031798, 0.0253122,
      0.0253122, 0.0253122, 0.0253122, 0.0253122, 0.0253122, 0.0253122,
      0.0253122, 0.0253122, 0.0253122, 0.0253122, 0.0253122, 0.0253122,
      0.0253122, 0.0253122, 0.0253122, 0.0253122, 0.0253122, 0.0253122,
      0.0253122, 0.0253122, 0.0253122, 0.0253122, 0.0253122, 0.0314376,
      0.0314376, 0.0314376, 0.0314376, 0.0314376, 0.0314376, 0.0314376,
      0.0314376, 0.0314376, 0.0314376, 0.0314376, 0.0314376, 0.0314376,
      0.0314376, 0.0314376, 0.0314376, 0.0314376, 0.0314376, 0.0314376,
      0.0314376, 0.0314376, 0.0314376, 0.0314376, 0.0314376, 0.0315826,
      0.0315826, 0.0315826, 0.0315826, 0.0315826, 0.0315826, 0.0315826,
      0.0315826, 0.0315826, 0.0315826, 0.0315826, 0.0315826, 0.0315826,
      0.0315826, 0.0315826, 0.0315826, 0.0315826, 0.0315826, 0.0315826,
      0.0315826, 0.0315826, 0.0315826, 0.0315826, 0.0315826, 0.0289365,
      0.0289365, 0.0289365, 0.0289365, 0.0289365, 0.0289365, 0.0289365,
      0.0289365, 0.0289365, 0.0289365, 0.0289365, 0.0289365, 0.0289365,
      0.0289365, 0.0289365, 0.0289365, 0.0289365, 0.0289365, 0.0289365,
      0.0289365, 0.0289365, 0.0289365, 0.0289365, 0.0289365, 0.0183783,
      0.0183783, 0.0183783, 0.0183783, 0.0183783, 0.0183783, 0.0183783,
      0.0183783, 0.0183783, 0.0183783, 0.0183783, 0.0183783, 0.0183783,
      0.0183783, 0.0183783, 0.0183783, 0.0183783, 0.0183783, 0.0183783,
      0.0183783, 0.0183783, 0.0183783, 0.0183783, 0.0183783, 0.0307295,
      0.0307295, 0.0307295, 0.0307295, 0.0307295, 0.0307295, 0.0307295,
      0.0307295, 0.0307295, 0.0307295, 0.0307295, 0.0307295, 0.0307295,
      0.0307295, 0.0307295, 0.0307295, 0.0307295, 0.0307295, 0.0307295,
      0.0307295, 0.0307295, 0.0307295, 0.0307295, 0.0307295, 0.0303785,
      0.0303785, 0.0303785, 0.0303785, 0.0303785, 0.0303785, 0.0303785,
      0.0303785, 0.0303785, 0.0303785, 0.0303785, 0.0303785, 0.0303785,
      0.0303785, 0.0303785, 0.0303785, 0.0303785, 0.0303785, 0.0303785,
      0.0303785, 0.0303785, 0.0303785, 0.0303785, 0.0303785, 0.0240137,
      0.0240137, 0.0240137, 0.0240137, 0.0240137, 0.0240137, 0.0240137,
      0.0240137, 0.0240137, 0.0240137, 0.0240137, 0.0240137, 0.0240137,
      0.0240137, 0.0240137, 0.0240137, 0.0240137, 0.0240137, 0.0240137,
      0.0240137, 0.0240137, 0.0240137, 0.0240137, 0.0240137, 0.030372, 0.030372,
      0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372,
      0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372,
      0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372,
      0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372,
      0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372,
      0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372,
      0.030372, 0.030372, 0.030372, 0.030372, 0.0315697, 0.0315697, 0.0315697,
      0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697,
      0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697,
      0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697,
      0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697,
      0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697,
      0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697,
      0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697,
      0.0315697, 0.0315697, 0.0315697, 0.0313738, 0.0313738, 0.0313738,
      0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738,
      0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738,
      0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738,
      0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738,
      0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738,
      0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738,
      0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738,
      0.0313738, 0.0313738, 0.0313738, 0.028106, 0.028106, 0.028106, 0.028106,
      0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106,
      0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106,
      0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106,
      0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106,
      0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106,
      0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106,
      0.028106, 0.028106;

  Eigen::VectorXd H_phi_ref = Eigen::VectorXd::Zero(434);
  H_phi_ref << 90, 90, 90, 90, 0, 180, 45, 45, 135, 135, 45, 45, 135, 135, 90,
      90, 90, 90, 54.7356, 54.7356, 54.7356, 54.7356, 125.264, 125.264, 125.264,
      125.264, 77.7225, 77.7225, 77.7225, 77.7225, 102.278, 102.278, 102.278,
      102.278, 46.2959, 46.2959, 46.2959, 46.2959, 133.704, 133.704, 133.704,
      133.704, 46.2959, 46.2959, 46.2959, 46.2959, 133.704, 133.704, 133.704,
      133.704, 14.5367, 14.5367, 14.5367, 14.5367, 165.463, 165.463, 165.463,
      165.463, 79.7768, 79.7768, 79.7768, 79.7768, 100.223, 100.223, 100.223,
      100.223, 79.7768, 79.7768, 79.7768, 79.7768, 100.223, 100.223, 100.223,
      100.223, 44.0267, 44.0267, 44.0267, 44.0267, 135.973, 135.973, 135.973,
      135.973, 60.5651, 60.5651, 60.5651, 60.5651, 119.435, 119.435, 119.435,
      119.435, 60.5651, 60.5651, 60.5651, 60.5651, 119.435, 119.435, 119.435,
      119.435, 65.9388, 65.9388, 65.9388, 65.9388, 114.061, 114.061, 114.061,
      114.061, 49.7843, 49.7843, 49.7843, 49.7843, 130.216, 130.216, 130.216,
      130.216, 49.7843, 49.7843, 49.7843, 49.7843, 130.216, 130.216, 130.216,
      130.216, 23.869, 23.869, 23.869, 23.869, 156.131, 156.131, 156.131,
      156.131, 73.3737, 73.3737, 73.3737, 73.3737, 106.626, 106.626, 106.626,
      106.626, 73.3737, 73.3737, 73.3737, 73.3737, 106.626, 106.626, 106.626,
      106.626, 6.14407, 6.14407, 6.14407, 6.14407, 173.856, 173.856, 173.856,
      173.856, 85.6597, 85.6597, 85.6597, 85.6597, 94.3403, 94.3403, 94.3403,
      94.3403, 85.6597, 85.6597, 85.6597, 85.6597, 94.3403, 94.3403, 94.3403,
      94.3403, 33.7382, 33.7382, 33.7382, 33.7382, 146.262, 146.262, 146.262,
      146.262, 66.8758, 66.8758, 66.8758, 66.8758, 113.124, 113.124, 113.124,
      113.124, 66.8758, 66.8758, 66.8758, 66.8758, 113.124, 113.124, 113.124,
      113.124, 90, 90, 90, 90, 90, 90, 90, 90, 61.8619, 61.8619, 118.138,
      118.138, 28.1381, 28.1381, 151.862, 151.862, 61.8619, 61.8619, 118.138,
      118.138, 28.1381, 28.1381, 151.862, 151.862, 90, 90, 90, 90, 90, 90, 90,
      90, 77.8617, 77.8617, 102.138, 102.138, 12.1383, 12.1383, 167.862,
      167.862, 77.8617, 77.8617, 102.138, 102.138, 12.1383, 12.1383, 167.862,
      167.862, 63.2414, 63.2414, 63.2414, 63.2414, 116.759, 116.759, 116.759,
      116.759, 29.6636, 29.6636, 29.6636, 29.6636, 150.336, 150.336, 150.336,
      150.336, 63.2414, 63.2414, 63.2414, 63.2414, 116.759, 116.759, 116.759,
      116.759, 78.1423, 78.1423, 78.1423, 78.1423, 101.858, 101.858, 101.858,
      101.858, 29.6636, 29.6636, 29.6636, 29.6636, 150.336, 150.336, 150.336,
      150.336, 78.1423, 78.1423, 78.1423, 78.1423, 101.858, 101.858, 101.858,
      101.858, 83.869, 83.869, 83.869, 83.869, 96.131, 96.131, 96.131, 96.131,
      36.8768, 36.8768, 36.8768, 36.8768, 143.123, 143.123, 143.123, 143.123,
      83.869, 83.869, 83.869, 83.869, 96.131, 96.131, 96.131, 96.131, 53.8064,
      53.8064, 53.8064, 53.8064, 126.194, 126.194, 126.194, 126.194, 36.8768,
      36.8768, 36.8768, 36.8768, 143.123, 143.123, 143.123, 143.123, 53.8064,
      53.8064, 53.8064, 53.8064, 126.194, 126.194, 126.194, 126.194, 71.915,
      71.915, 71.915, 71.915, 108.085, 108.085, 108.085, 108.085, 39.489,
      39.489, 39.489, 39.489, 140.511, 140.511, 140.511, 140.511, 71.915,
      71.915, 71.915, 71.915, 108.085, 108.085, 108.085, 108.085, 56.2882,
      56.2882, 56.2882, 56.2882, 123.712, 123.712, 123.712, 123.712, 39.489,
      39.489, 39.489, 39.489, 140.511, 140.511, 140.511, 140.511, 56.2882,
      56.2882, 56.2882, 56.2882, 123.712, 123.712, 123.712, 123.712, 84.3059,
      84.3059, 84.3059, 84.3059, 95.6941, 95.6941, 95.6941, 95.6941, 70.4617,
      70.4617, 70.4617, 70.4617, 109.538, 109.538, 109.538, 109.538, 84.3059,
      84.3059, 84.3059, 84.3059, 95.6941, 95.6941, 95.6941, 95.6941, 20.4166,
      20.4166, 20.4166, 20.4166, 159.583, 159.583, 159.583, 159.583, 70.4617,
      70.4617, 70.4617, 70.4617, 109.538, 109.538, 109.538, 109.538, 20.4166,
      20.4166, 20.4166, 20.4166, 159.583, 159.583, 159.583, 159.583;
  H_phi_ref *= votca::tools::conv::Pi / 180.0;

  Eigen::VectorXd H_theta_ref = Eigen::VectorXd::Zero(434);
  H_theta_ref << 0, 180, 90, -90, 0, 0, 90, -90, 90, -90, 0, 180, 0, 180, 45,
      135, -45, -135, 45, 135, -45, -135, 45, 135, -45, -135, 45, 135, -45,
      -135, 45, 135, -45, -135, 17.1066, 162.893, -17.1066, -162.893, 17.1066,
      162.893, -17.1066, -162.893, 72.8934, 107.107, -72.8934, -107.107,
      72.8934, 107.107, -72.8934, -107.107, 45, 135, -45, -135, 45, 135, -45,
      -135, 79.61, 100.39, -79.61, -100.39, 79.61, 100.39, -79.61, -100.39,
      10.39, 169.61, -10.39, -169.61, 10.39, 169.61, -10.39, -169.61, 45, 135,
      -45, -135, 45, 135, -45, -135, 55.6481, 124.352, -55.6481, -124.352,
      55.6481, 124.352, -55.6481, -124.352, 34.3519, 145.648, -34.3519,
      -145.648, 34.3519, 145.648, -34.3519, -145.648, 45, 135, -45, -135, 45,
      135, -45, -135, 32.2708, 147.729, -32.2708, -147.729, 32.2708, 147.729,
      -32.2708, -147.729, 57.7292, 122.271, -57.7292, -122.271, 57.7292,
      122.271, -57.7292, -122.271, 45, 135, -45, -135, 45, 135, -45, -135,
      72.6256, 107.374, -72.6256, -107.374, 72.6256, 107.374, -72.6256,
      -107.374, 17.3744, 162.626, -17.3744, -162.626, 17.3744, 162.626,
      -17.3744, -162.626, 45, 135, -45, -135, 45, 135, -45, -135, 85.6471,
      94.3529, -85.6471, -94.3529, 85.6471, 94.3529, -85.6471, -94.3529,
      4.35285, 175.647, -4.35285, -175.647, 4.35285, 175.647, -4.35285,
      -175.647, 45, 135, -45, -135, 45, 135, -45, -135, 64.7204, 115.28,
      -64.7204, -115.28, 64.7204, 115.28, -64.7204, -115.28, 25.2796, 154.72,
      -25.2796, -154.72, 25.2796, 154.72, -25.2796, -154.72, 28.1381, 151.862,
      -28.1381, -151.862, 61.8619, 118.138, -61.8619, -118.138, 0, 180, 0, 180,
      0, 180, 0, 180, 90, -90, 90, -90, 90, -90, 90, -90, 12.1383, 167.862,
      -12.1383, -167.862, 77.8617, 102.138, -77.8617, -102.138, 0, 180, 0, 180,
      0, 180, 0, 180, 90, -90, 90, -90, 90, -90, 90, -90, 76.6955, 103.305,
      -76.6955, -103.305, 76.6955, 103.305, -76.6955, -103.305, 65.4685,
      114.532, -65.4685, -114.532, 65.4685, 114.532, -65.4685, -114.532,
      13.3045, 166.695, -13.3045, -166.695, 13.3045, 166.695, -13.3045,
      -166.695, 27.3903, 152.61, -27.3903, -152.61, 27.3903, 152.61, -27.3903,
      -152.61, 24.5315, 155.468, -24.5315, -155.468, 24.5315, 155.468, -24.5315,
      -155.468, 62.6097, 117.39, -62.6097, -117.39, 62.6097, 117.39, -62.6097,
      -117.39, 53.5648, 126.435, -53.5648, -126.435, 53.5648, 126.435, -53.5648,
      -126.435, 10.2518, 169.748, -10.2518, -169.748, 10.2518, 169.748,
      -10.2518, -169.748, 36.4352, 143.565, -36.4352, -143.565, 36.4352,
      143.565, -36.4352, -143.565, 7.60483, 172.395, -7.60483, -172.395,
      7.60483, 172.395, -7.60483, -172.395, 79.7482, 100.252, -79.7482,
      -100.252, 79.7482, 100.252, -79.7482, -100.252, 82.3952, 97.6048,
      -82.3952, -97.6048, 82.3952, 97.6048, -82.3952, -97.6048, 54.2775,
      125.722, -54.2775, -125.722, 54.2775, 125.722, -54.2775, -125.722,
      29.2189, 150.781, -29.2189, -150.781, 29.2189, 150.781, -29.2189,
      -150.781, 35.7225, 144.278, -35.7225, -144.278, 35.7225, 144.278,
      -35.7225, -144.278, 21.912, 158.088, -21.912, -158.088, 21.912, 158.088,
      -21.912, -158.088, 60.7811, 119.219, -60.7811, -119.219, 60.7811, 119.219,
      -60.7811, -119.219, 68.088, 111.912, -68.088, -111.912, 68.088, 111.912,
      -68.088, -111.912, 19.6391, 160.361, -19.6391, -160.361, 19.6391, 160.361,
      -19.6391, -160.361, 6.04329, 173.957, -6.04329, -173.957, 6.04329,
      173.957, -6.04329, -173.957, 70.3609, 109.639, -70.3609, -109.639,
      70.3609, 109.639, -70.3609, -109.639, 16.5241, 163.476, -16.5241,
      -163.476, 16.5241, 163.476, -16.5241, -163.476, 83.9567, 96.0433,
      -83.9567, -96.0433, 83.9567, 96.0433, -83.9567, -96.0433, 73.4759,
      106.524, -73.4759, -106.524, 73.4759, 106.524, -73.4759, -106.524;

  H_theta_ref *= votca::tools::conv::Pi / 180.0;

  Eigen::VectorXd H_weight_ref = Eigen::VectorXd::Zero(434);
  H_weight_ref << 0.00661732, 0.00661732, 0.00661732, 0.00661732, 0.00661732,
      0.00661732, 0.0320219, 0.0320219, 0.0320219, 0.0320219, 0.0320219,
      0.0320219, 0.0320219, 0.0320219, 0.0320219, 0.0320219, 0.0320219,
      0.0320219, 0.0315707, 0.0315707, 0.0315707, 0.0315707, 0.0315707,
      0.0315707, 0.0315707, 0.0315707, 0.031798, 0.031798, 0.031798, 0.031798,
      0.031798, 0.031798, 0.031798, 0.031798, 0.031798, 0.031798, 0.031798,
      0.031798, 0.031798, 0.031798, 0.031798, 0.031798, 0.031798, 0.031798,
      0.031798, 0.031798, 0.031798, 0.031798, 0.031798, 0.031798, 0.0253122,
      0.0253122, 0.0253122, 0.0253122, 0.0253122, 0.0253122, 0.0253122,
      0.0253122, 0.0253122, 0.0253122, 0.0253122, 0.0253122, 0.0253122,
      0.0253122, 0.0253122, 0.0253122, 0.0253122, 0.0253122, 0.0253122,
      0.0253122, 0.0253122, 0.0253122, 0.0253122, 0.0253122, 0.0314376,
      0.0314376, 0.0314376, 0.0314376, 0.0314376, 0.0314376, 0.0314376,
      0.0314376, 0.0314376, 0.0314376, 0.0314376, 0.0314376, 0.0314376,
      0.0314376, 0.0314376, 0.0314376, 0.0314376, 0.0314376, 0.0314376,
      0.0314376, 0.0314376, 0.0314376, 0.0314376, 0.0314376, 0.0315826,
      0.0315826, 0.0315826, 0.0315826, 0.0315826, 0.0315826, 0.0315826,
      0.0315826, 0.0315826, 0.0315826, 0.0315826, 0.0315826, 0.0315826,
      0.0315826, 0.0315826, 0.0315826, 0.0315826, 0.0315826, 0.0315826,
      0.0315826, 0.0315826, 0.0315826, 0.0315826, 0.0315826, 0.0289365,
      0.0289365, 0.0289365, 0.0289365, 0.0289365, 0.0289365, 0.0289365,
      0.0289365, 0.0289365, 0.0289365, 0.0289365, 0.0289365, 0.0289365,
      0.0289365, 0.0289365, 0.0289365, 0.0289365, 0.0289365, 0.0289365,
      0.0289365, 0.0289365, 0.0289365, 0.0289365, 0.0289365, 0.0183783,
      0.0183783, 0.0183783, 0.0183783, 0.0183783, 0.0183783, 0.0183783,
      0.0183783, 0.0183783, 0.0183783, 0.0183783, 0.0183783, 0.0183783,
      0.0183783, 0.0183783, 0.0183783, 0.0183783, 0.0183783, 0.0183783,
      0.0183783, 0.0183783, 0.0183783, 0.0183783, 0.0183783, 0.0307295,
      0.0307295, 0.0307295, 0.0307295, 0.0307295, 0.0307295, 0.0307295,
      0.0307295, 0.0307295, 0.0307295, 0.0307295, 0.0307295, 0.0307295,
      0.0307295, 0.0307295, 0.0307295, 0.0307295, 0.0307295, 0.0307295,
      0.0307295, 0.0307295, 0.0307295, 0.0307295, 0.0307295, 0.0303785,
      0.0303785, 0.0303785, 0.0303785, 0.0303785, 0.0303785, 0.0303785,
      0.0303785, 0.0303785, 0.0303785, 0.0303785, 0.0303785, 0.0303785,
      0.0303785, 0.0303785, 0.0303785, 0.0303785, 0.0303785, 0.0303785,
      0.0303785, 0.0303785, 0.0303785, 0.0303785, 0.0303785, 0.0240137,
      0.0240137, 0.0240137, 0.0240137, 0.0240137, 0.0240137, 0.0240137,
      0.0240137, 0.0240137, 0.0240137, 0.0240137, 0.0240137, 0.0240137,
      0.0240137, 0.0240137, 0.0240137, 0.0240137, 0.0240137, 0.0240137,
      0.0240137, 0.0240137, 0.0240137, 0.0240137, 0.0240137, 0.030372, 0.030372,
      0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372,
      0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372,
      0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372,
      0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372,
      0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372,
      0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372, 0.030372,
      0.030372, 0.030372, 0.030372, 0.030372, 0.0315697, 0.0315697, 0.0315697,
      0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697,
      0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697,
      0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697,
      0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697,
      0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697,
      0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697,
      0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697, 0.0315697,
      0.0315697, 0.0315697, 0.0315697, 0.0313738, 0.0313738, 0.0313738,
      0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738,
      0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738,
      0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738,
      0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738,
      0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738,
      0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738,
      0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738, 0.0313738,
      0.0313738, 0.0313738, 0.0313738, 0.028106, 0.028106, 0.028106, 0.028106,
      0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106,
      0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106,
      0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106,
      0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106,
      0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106,
      0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106, 0.028106,
      0.028106, 0.028106;

  BOOST_CHECK_EQUAL(Cgrid.phi.size(), C_phi_ref.size());
  BOOST_CHECK_EQUAL(Cgrid.theta.size(), C_theta_ref.size());
  BOOST_CHECK_EQUAL(Cgrid.weight.size(), C_weight_ref.size());
  BOOST_CHECK_EQUAL(Hgrid.phi.size(), H_phi_ref.size());
  BOOST_CHECK_EQUAL(Hgrid.theta.size(), H_theta_ref.size());
  BOOST_CHECK_EQUAL(Hgrid.weight.size(), H_weight_ref.size());

  bool Cphi = C_phi_ref.isApprox(Cgrid.phi, 0.001);
  bool Ctheta = C_theta_ref.isApprox(Cgrid.theta, 0.001);
  if (!Cphi || !Ctheta) {
    std::cout << "phi_ref : Phi_comp | theta_ref : theta_comp" << std::endl;
    for (int i = 0; i < C_phi_ref.size(); i++) {
      std::cout << Cgrid.phi[i] << ":" << C_phi_ref[i] << " | "
                << Cgrid.theta[i] << ":" << C_theta_ref[i] << std::endl;
    }
  }

  bool Cweight = C_weight_ref.isApprox(Cgrid.weight, 0.0001);
  BOOST_CHECK_EQUAL(Cphi, true);
  BOOST_CHECK_EQUAL(Ctheta, true);
  BOOST_CHECK_EQUAL(Cweight, true);

  bool Hphi = H_phi_ref.isApprox(Hgrid.phi, 0.001);
  bool Htheta = H_theta_ref.isApprox(Hgrid.theta, 0.001);
  bool Hweight = H_weight_ref.isApprox(Hgrid.weight, 0.0001);
  BOOST_CHECK_EQUAL(Hphi, true);
  BOOST_CHECK_EQUAL(Htheta, true);
  BOOST_CHECK_EQUAL(Hweight, true);
}

BOOST_AUTO_TEST_SUITE_END()
