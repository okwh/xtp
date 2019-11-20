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

#define BOOST_TEST_MODULE threecenter_test
#include <boost/test/unit_test.hpp>
#include <votca/xtp/ERIs.h>
#include <votca/xtp/aomatrix.h>
#include <votca/xtp/orbitals.h>
#include <votca/xtp/threecenter.h>

using namespace votca::xtp;
using namespace std;

BOOST_AUTO_TEST_SUITE(threecenter_test)

BOOST_AUTO_TEST_CASE(threecenter_dft) {

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
  basis.LoadBasisSet("3-21G.xml");
  AOBasis aobasis;
  aobasis.AOBasisFill(basis, orbitals.QMAtoms());
  TCMatrix_dft threec;
  threec.Fill(aobasis, aobasis);

  Eigen::MatrixXcd Res0 =
      Eigen::MatrixXcd::Zero(aobasis.AOBasisSize(), aobasis.AOBasisSize());
  threec[0].AddtoEigenMatrix(Res0);

  Eigen::MatrixXcd Res4 =
      Eigen::MatrixXcd::Zero(aobasis.AOBasisSize(), aobasis.AOBasisSize());
  threec[4].AddtoEigenMatrix(Res4);

  Eigen::MatrixXd Ref0 =
      Eigen::MatrixXd::Zero(aobasis.AOBasisSize(), aobasis.AOBasisSize());
  Ref0 << 1.60933, 0.156026, 3.9986e-17, 2.08105e-18, 1.04957e-17, 0.203748,
      5.44261e-18, 3.17213e-19, 1.30909e-18, 0.0195044, 0.0905022, 0.0195044,
      0.0905022, 0.0195044, 0.0905022, 0.0195044, 0.0905022, 0.156026, 0.189689,
      1.14095e-16, 2.20804e-17, -2.99842e-18, 0.112228, 3.84692e-17,
      1.40062e-17, -1.44399e-17, 0.0156068, 0.0516639, 0.0156068, 0.0516639,
      0.0156068, 0.0516639, 0.0156068, 0.0516639, 3.9986e-17, 1.14095e-16,
      0.252489, -2.56122e-17, 4.14503e-18, 7.22824e-17, 0.0409547, -1.86454e-17,
      2.81609e-18, 0.00993143, 0.00923661, 0.00993143, 0.00923661, -0.00993143,
      -0.00923661, -0.00993143, -0.00923661, 2.08105e-18, 2.20804e-17,
      -2.56122e-17, 0.252489, -1.82397e-17, 2.0959e-17, -1.86454e-17, 0.0409547,
      -1.23571e-17, 0.00993143, 0.00923661, -0.00993143, -0.00923661,
      -0.00993143, -0.00923661, 0.00993143, 0.00923661, 1.04957e-17,
      -2.99842e-18, 4.14503e-18, -1.82397e-17, 0.252489, -1.66222e-17,
      2.81609e-18, -1.23571e-17, 0.0409547, 0.00993143, 0.00923661, -0.00993143,
      -0.00923661, 0.00993143, 0.00923661, -0.00993143, -0.00923661, 0.203748,
      0.112228, 7.22824e-17, 2.0959e-17, -1.66222e-17, 0.0739222, 2.45431e-17,
      2.83659e-17, -5.2652e-17, 0.0109019, 0.0339426, 0.0109019, 0.0339426,
      0.0109019, 0.0339426, 0.0109019, 0.0339426, 5.44261e-18, 3.84692e-17,
      0.0409547, -1.86454e-17, 2.81609e-18, 2.45431e-17, 0.00662462,
      -3.27429e-17, 1.50564e-18, 0.00383329, 0.00241774, 0.00383329, 0.00241774,
      -0.00383329, -0.00241774, -0.00383329, -0.00241774, 3.17213e-19,
      1.40062e-17, -1.86454e-17, 0.0409547, -1.23571e-17, 2.83659e-17,
      -3.27429e-17, 0.00662462, -1.18056e-17, 0.00383329, 0.00241774,
      -0.00383329, -0.00241774, -0.00383329, -0.00241774, 0.00383329,
      0.00241774, 1.30909e-18, -1.44399e-17, 2.81609e-18, -1.23571e-17,
      0.0409547, -5.2652e-17, 1.50564e-18, -1.18056e-17, 0.00662462, 0.00383329,
      0.00241774, -0.00383329, -0.00241774, 0.00383329, 0.00241774, -0.00383329,
      -0.00241774, 0.0195044, 0.0156068, 0.00993143, 0.00993143, 0.00993143,
      0.0109019, 0.00383329, 0.00383329, 0.00383329, 0.0143735, 0.0108626,
      0.000680431, 0.00420737, 0.000680431, 0.00420737, 0.000680431, 0.00420737,
      0.0905022, 0.0516639, 0.00923661, 0.00923661, 0.00923661, 0.0339426,
      0.00241774, 0.00241774, 0.00241774, 0.0108626, 0.019639, 0.00420737,
      0.0149384, 0.00420737, 0.0149384, 0.00420737, 0.0149384, 0.0195044,
      0.0156068, 0.00993143, -0.00993143, -0.00993143, 0.0109019, 0.00383329,
      -0.00383329, -0.00383329, 0.000680431, 0.00420737, 0.0143735, 0.0108626,
      0.000680431, 0.00420737, 0.000680431, 0.00420737, 0.0905022, 0.0516639,
      0.00923661, -0.00923661, -0.00923661, 0.0339426, 0.00241774, -0.00241774,
      -0.00241774, 0.00420737, 0.0149384, 0.0108626, 0.019639, 0.00420737,
      0.0149384, 0.00420737, 0.0149384, 0.0195044, 0.0156068, -0.00993143,
      -0.00993143, 0.00993143, 0.0109019, -0.00383329, -0.00383329, 0.00383329,
      0.000680431, 0.00420737, 0.000680431, 0.00420737, 0.0143735, 0.0108626,
      0.000680431, 0.00420737, 0.0905022, 0.0516639, -0.00923661, -0.00923661,
      0.00923661, 0.0339426, -0.00241774, -0.00241774, 0.00241774, 0.00420737,
      0.0149384, 0.00420737, 0.0149384, 0.0108626, 0.019639, 0.00420737,
      0.0149384, 0.0195044, 0.0156068, -0.00993143, 0.00993143, -0.00993143,
      0.0109019, -0.00383329, 0.00383329, -0.00383329, 0.000680431, 0.00420737,
      0.000680431, 0.00420737, 0.000680431, 0.00420737, 0.0143735, 0.0108626,
      0.0905022, 0.0516639, -0.00923661, 0.00923661, -0.00923661, 0.0339426,
      -0.00241774, 0.00241774, -0.00241774, 0.00420737, 0.0149384, 0.00420737,
      0.0149384, 0.00420737, 0.0149384, 0.0108626, 0.019639;

  Eigen::MatrixXd Ref4 =
      Eigen::MatrixXd::Zero(aobasis.AOBasisSize(), aobasis.AOBasisSize());
  Ref4 << 3.0437e-16, 5.55112e-17, 1.18148e-16, -7.26115e-17, 0.115582,
      4.43581e-17, 1.57656e-17, -9.58568e-18, 0.0159, 0.00332761, 0.00350516,
      -0.00332761, -0.00350516, 0.00332761, 0.00350516, -0.00332761,
      -0.00350516, 5.55112e-17, 1.66533e-16, 2.64065e-16, -1.32226e-16,
      0.400942, 1.98651e-16, 5.90928e-17, -1.05206e-17, 0.160429, 0.0438354,
      0.0379715, -0.0438354, -0.0379715, 0.0438354, 0.0379715, -0.0438354,
      -0.0379715, 1.18148e-16, 2.64065e-16, 1.49655e-16, -0.0118561,
      -6.93775e-17, 1.39834e-16, 1.43594e-16, -0.00838998, -5.51212e-17,
      0.0247852, 0.00498474, -0.0247852, -0.00498474, -0.0247852, -0.00498474,
      0.0247852, 0.00498474, -7.26115e-17, -1.32226e-16, -0.0118561,
      2.05166e-16, -7.11351e-17, -4.96284e-17, -0.00838998, 1.43594e-16,
      -4.76612e-17, 0.0247852, 0.00498474, 0.0247852, 0.00498474, -0.0247852,
      -0.00498474, -0.0247852, -0.00498474, 0.115582, 0.400942, -6.93775e-17,
      -7.11351e-17, 2.05166e-16, 0.283457, -5.51212e-17, -4.76612e-17,
      1.43594e-16, 0.0620134, 0.144146, 0.0620134, 0.144146, 0.0620134,
      0.144146, 0.0620134, 0.144146, 4.43581e-17, 1.98651e-16, 1.39834e-16,
      -4.96284e-17, 0.283457, 3.08944e-16, -4.814e-17, 1.66101e-16, 0.150009,
      0.0408784, 0.0355068, -0.0408784, -0.0355068, 0.0408784, 0.0355068,
      -0.0408784, -0.0355068, 1.57656e-17, 5.90928e-17, 1.43594e-16,
      -0.00838998, -5.51212e-17, -4.814e-17, 3.05311e-16, -0.0119247,
      -1.40833e-16, 0.0165925, 0.00164473, -0.0165925, -0.00164473, -0.0165925,
      -0.00164473, 0.0165925, 0.00164473, -9.58568e-18, -1.05206e-17,
      -0.00838998, 1.43594e-16, -4.76612e-17, 1.66101e-16, -0.0119247,
      3.05311e-16, -3.61091e-17, 0.0165925, 0.00164473, 0.0165925, 0.00164473,
      -0.0165925, -0.00164473, -0.0165925, -0.00164473, 0.0159, 0.160429,
      -5.51212e-17, -4.76612e-17, 1.43594e-16, 0.150009, -1.40833e-16,
      -3.61091e-17, 2.87567e-16, 0.0403597, 0.0808243, 0.0403597, 0.0808243,
      0.0403597, 0.0808243, 0.0403597, 0.0808243, 0.00332761, 0.0438354,
      0.0247852, 0.0247852, 0.0620134, 0.0408784, 0.0165925, 0.0165925,
      0.0403597, 0.0699398, 0.0446004, 3.06913e-18, 0.0118855, 0.00334514,
      0.0207078, 2.43424e-18, 0.0118855, 0.00350516, 0.0379715, 0.00498474,
      0.00498474, 0.144146, 0.0355068, 0.00164473, 0.00164473, 0.0808243,
      0.0446004, 0.0387592, -0.0118855, 7.97996e-17, 0.0207078, 0.0350755,
      -0.0118855, 1.81964e-16, -0.00332761, -0.0438354, -0.0247852, 0.0247852,
      0.0620134, -0.0408784, -0.0165925, 0.0165925, 0.0403597, 3.06913e-18,
      -0.0118855, -0.0699398, -0.0446004, 1.98493e-18, -0.0118855, -0.00334514,
      -0.0207078, -0.00350516, -0.0379715, -0.00498474, 0.00498474, 0.144146,
      -0.0355068, -0.00164473, 0.00164473, 0.0808243, 0.0118855, 7.97996e-17,
      -0.0446004, -0.0387592, 0.0118855, 2.42885e-17, -0.0207078, -0.0350755,
      0.00332761, 0.0438354, -0.0247852, -0.0247852, 0.0620134, 0.0408784,
      -0.0165925, -0.0165925, 0.0403597, 0.00334514, 0.0207078, 1.98493e-18,
      0.0118855, 0.0699398, 0.0446004, 1.1332e-18, 0.0118855, 0.00350516,
      0.0379715, -0.00498474, -0.00498474, 0.144146, 0.0355068, -0.00164473,
      -0.00164473, 0.0808243, 0.0207078, 0.0350755, -0.0118855, 2.42885e-17,
      0.0446004, 0.0387592, -0.0118855, 1.12575e-16, -0.00332761, -0.0438354,
      0.0247852, -0.0247852, 0.0620134, -0.0408784, 0.0165925, -0.0165925,
      0.0403597, 2.43424e-18, -0.0118855, -0.00334514, -0.0207078, 1.1332e-18,
      -0.0118855, -0.0699398, -0.0446004, -0.00350516, -0.0379715, 0.00498474,
      -0.00498474, 0.144146, -0.0355068, 0.00164473, -0.00164473, 0.0808243,
      0.0118855, 1.81964e-16, -0.0207078, -0.0350755, 0.0118855, 1.12575e-16,
      -0.0446004, -0.0387592;

  bool check_three1 = Res0.isApprox(Ref0, 0.00001);
  if (!check_three1) {
    cout << "Res0" << endl;
    cout << Res0 << endl;
    cout << "0_ref" << endl;
    cout << Ref0 << endl;
  }
  BOOST_CHECK_EQUAL(check_three1, true);
  bool check_three2 = Res4.isApprox(Ref4, 0.00001);
  if (!check_three2) {
    cout << "Res4" << endl;
    cout << Res4 << endl;
    cout << "4_ref" << endl;
    cout << Ref4 << endl;
  }
  BOOST_CHECK_EQUAL(check_three2, true);
}

BOOST_AUTO_TEST_CASE(threecenter_gwbse) {
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
  basis.LoadBasisSet("3-21G.xml");
  AOBasis aobasis;
  aobasis.AOBasisFill(basis, orbitals.QMAtoms());

  Eigen::MatrixXd MOs = Eigen::MatrixXd::Zero(17, 17);
  MOs << -0.00761992, -4.69664e-13, 8.35009e-15, -1.15214e-14, -0.0156169,
      -2.23157e-12, 1.52916e-14, 2.10997e-15, 8.21478e-15, 3.18517e-15,
      2.89043e-13, -0.00949189, 1.95787e-12, 1.22168e-14, -2.63092e-15,
      -0.22227, 1.00844, 0.233602, -3.18103e-12, 4.05093e-14, -4.70943e-14,
      0.1578, 4.75897e-11, -1.87447e-13, -1.02418e-14, 6.44484e-14, -2.6602e-14,
      6.5906e-12, -0.281033, -6.67755e-12, 2.70339e-14, -9.78783e-14, -1.94373,
      -0.36629, -1.63678e-13, -0.22745, -0.054851, 0.30351, 3.78688e-11,
      -0.201627, -0.158318, -0.233561, -0.0509347, -0.650424, 0.452606,
      -5.88565e-11, 0.453936, -0.165715, -0.619056, 7.0149e-12, 2.395e-14,
      -4.51653e-14, -0.216509, 0.296975, -0.108582, 3.79159e-11, -0.199301,
      0.283114, -0.0198557, 0.584622, 0.275311, 0.461431, -5.93732e-11,
      0.453057, 0.619523, 0.166374, 7.13235e-12, 2.56811e-14, -9.0903e-14,
      -0.21966, -0.235919, -0.207249, 3.75979e-11, -0.199736, -0.122681,
      0.255585, -0.534902, 0.362837, 0.461224, -5.91028e-11, 0.453245,
      -0.453298, 0.453695, 7.01644e-12, 2.60987e-14, 0.480866, 1.8992e-11,
      -2.56795e-13, 4.14571e-13, 2.2709, 4.78615e-10, -2.39153e-12,
      -2.53852e-13, -2.15605e-13, -2.80359e-13, 7.00137e-12, 0.145171,
      -1.96136e-11, -2.24876e-13, -2.57294e-14, 4.04176, 0.193617, -1.64421e-12,
      -0.182159, -0.0439288, 0.243073, 1.80753e-10, -0.764779, -0.600505,
      -0.885907, 0.0862014, 1.10077, -0.765985, 6.65828e-11, -0.579266,
      0.211468, 0.789976, -1.41532e-11, -1.29659e-13, -1.64105e-12, -0.173397,
      0.23784, -0.0869607, 1.80537e-10, -0.755957, 1.07386, -0.0753135,
      -0.989408, -0.465933, -0.78092, 6.72256e-11, -0.578145, -0.790571,
      -0.212309, -1.42443e-11, -1.31306e-13, -1.63849e-12, -0.17592, -0.188941,
      -0.165981, 1.79403e-10, -0.757606, -0.465334, 0.969444, 0.905262,
      -0.61406, -0.78057, 6.69453e-11, -0.578385, 0.578453, -0.578959,
      -1.40917e-11, -1.31002e-13, 0.129798, -0.274485, 0.00256652, -0.00509635,
      -0.0118465, 0.141392, -0.000497905, -0.000510338, -0.000526798,
      -0.00532572, 0.596595, 0.65313, -0.964582, -0.000361559, -0.000717866,
      -0.195084, 0.0246232, 0.0541331, -0.255228, 0.00238646, -0.0047388,
      -0.88576, 1.68364, -0.00592888, -0.00607692, -9.5047e-05, -0.000960887,
      0.10764, -0.362701, 1.53456, 0.000575205, 0.00114206, -0.793844,
      -0.035336, 0.129798, 0.0863299, -0.0479412, 0.25617, -0.0118465,
      -0.0464689, 0.0750316, 0.110468, -0.0436647, -0.558989, -0.203909,
      0.65313, 0.320785, 0.235387, 0.878697, -0.195084, 0.0246232, 0.0541331,
      0.0802732, -0.0445777, 0.238198, -0.88576, -0.553335, 0.893449, 1.31541,
      -0.00787816, -0.100855, -0.0367902, -0.362701, -0.510338, -0.374479,
      -1.39792, -0.793844, -0.035336, 0.129798, 0.0927742, -0.197727, -0.166347,
      -0.0118465, -0.0473592, 0.0582544, -0.119815, -0.463559, 0.320126,
      -0.196433, 0.65313, 0.321765, 0.643254, -0.642737, -0.195084, 0.0246232,
      0.0541331, 0.0862654, -0.183855, -0.154677, -0.88576, -0.563936, 0.693672,
      -1.42672, -0.0836372, 0.0577585, -0.0354411, -0.362701, -0.511897,
      -1.02335, 1.02253, -0.793844, -0.035336, 0.129798, 0.0953806, 0.243102,
      -0.0847266, -0.0118465, -0.0475639, -0.132788, 0.00985812, 0.507751,
      0.244188, -0.196253, 0.65313, 0.322032, -0.87828, -0.235242, -0.195084,
      0.0246232, 0.0541331, 0.088689, 0.226046, -0.0787824, -0.88576, -0.566373,
      -1.58119, 0.117387, 0.0916104, 0.0440574, -0.0354087, -0.362701,
      -0.512321, 1.39726, 0.374248, -0.793844, -0.035336;

  TCMatrix_gwbse tc;
  tc.Initialize(aobasis.AOBasisSize(), 0, 5, 0, 7);
  tc.Fill(aobasis, aobasis, MOs);

  Eigen::MatrixXd ref0b = Eigen::MatrixXd::Zero(8, 17);
  ref0b << 0.052453955915, 0.26945204403, -4.9641876153e-13, -4.3017169895e-13,
      -4.5508824269e-13, 0.27735882244, -7.0243260325e-13, -6.6612945222e-13,
      -6.7921572331e-13, 0.090104440628, 0.28902528964, 0.090104440629,
      0.28902528964, 0.090104440629, 0.28902528964, 0.090104440629,
      0.28902528964, -1.1155268557e-08, -6.8061504246e-08, -0.10899853614,
      -0.10375544679, -0.10526537461, -8.9385307499e-08, -0.11026675547,
      -0.10496269404, -0.10649013264, -0.13176416369, -0.11578580299,
      0.041441929548, 0.036416383001, 0.044535456481, 0.039134772786,
      0.045786627056, 0.040234212958, -3.5804367745e-09, -1.0802763365e-08,
      -0.026285674945, 0.14231625805, -0.11305692938, -1.1539101856e-08,
      -0.026591503739, 0.14397210606, -0.11437228963, 0.0012320642569,
      0.0010826601921, -0.023013802606, -0.020223043844, -0.094917168712,
      -0.083407008577, 0.11669893345, 0.10254735717, -4.332712982e-09,
      -2.2018043355e-08, 0.14544781729, -0.052034639861, -0.099317818251,
      -2.7857016712e-08, 0.14714001983, -0.052640069268, -0.10047339869,
      -0.0024464893656, -0.00214984874, 0.12297233119, 0.10806004661,
      -0.079853605298, -0.070170183026, -0.040672265127, -0.035740141775,
      0.047253781938, 0.19221821341, 3.2613352995e-11, 3.2622040307e-11,
      3.2385857572e-11, 0.10912151956, 4.1659397534e-11, 4.1630989674e-11,
      4.1355167053e-11, -0.0031472116733, 0.010847236182, -0.0031472117132,
      0.010847236135, -0.0031472117134, 0.010847236135, -0.0031472117132,
      0.010847236135, -9.2851961e-08, -5.002949892e-07, -0.078475753537,
      -0.077570437009, -0.077739721498, -6.4118466292e-07, -0.048412585503,
      -0.04785405442, -0.047958490892, 0.055883809681, 0.042287484666,
      -0.018366676723, -0.013898932902, -0.018718534244, -0.014165177912,
      -0.018799404164, -0.014226362882, 4.9818162148e-08, 2.7093703938e-07,
      -0.061619242479, 0.11019108878, -0.047748960593, 3.4790364457e-07,
      -0.038013517344, 0.067977673132, -0.02945684192, -0.00019684351386,
      -0.00014869611569, 0.029655944654, 0.022441493158, 0.023024865034,
      0.01742364516, -0.052483517022, -0.039714814863, 6.7265638491e-09,
      5.6985829797e-08, -0.09090463767, -0.0077281025645, 0.099476617254,
      7.8663430324e-08, -0.056079775674, -0.0047675621011, 0.061367908295,
      -0.0002016891448, -0.00015254552404, 0.043661903121, 0.033039797,
      -0.047356375446, -0.035835348615, 0.0038963601843, 0.0029485096143;

  bool check0_before = ref0b.isApprox(tc[0], 1e-5);
  if (!check0_before) {
    cout << "tc0" << endl;
    cout << tc[0] << endl;
    cout << "tc0_ref" << endl;
    cout << ref0b << endl;
  }
  BOOST_CHECK_EQUAL(check0_before, true);

  Eigen::MatrixXd ref2b = Eigen::MatrixXd::Zero(8, 17);
  ref2b << -3.58044e-09, -1.08028e-08, -0.0262857, 0.142316, -0.113057,
      -1.15391e-08, -0.0265915, 0.143972, -0.114372, 0.00123206, 0.00108266,
      -0.0230138, -0.020223, -0.0949172, -0.083407, 0.116699, 0.102547,
      9.54021e-09, -2.23538e-08, -0.00217721, 0.0101068, -0.00856299,
      -9.35387e-08, -0.00341331, 0.015845, -0.0134246, -0.00253892, -0.00159428,
      -0.0149157, -0.00936558, -0.0661098, -0.04151, 0.0835643, 0.0524694,
      0.0436978, 0.201784, -0.0215531, 0.00398082, -0.00501107, 0.232683,
      -0.0337897, 0.00624092, -0.00785608, 0.00994497, 0.228298, 0.0182043,
      0.233484, 0.150819, 0.316752, 0.222906, 0.362015, 1.16479e-09,
      -1.1638e-08, -0.00552672, -0.00926513, 0.0147801, -2.46699e-08,
      -0.00866447, -0.0145254, 0.0231714, -4.71412e-05, -2.96221e-05, -0.04426,
      -0.0277906, 0.118537, 0.0744287, -0.0742301, -0.0466087, -3.83792e-09,
      -1.12535e-08, -0.017741, 0.0960538, -0.0763057, -8.3889e-09, 0.00193927,
      -0.0104997, 0.00834096, -0.000237732, -0.000691164, 0.00444057, 0.01291,
      0.0183145, 0.0532455, -0.0225174, -0.0654643, -0.000267293, -0.00109263,
      0.00136471, -0.0064945, 0.00543171, -0.000943013, 0.00834688, -0.0397218,
      0.0332215, 0.000750763, 0.000994679, 0.00475257, 0.00856368, 0.0200663,
      0.0375282, -0.0248402, -0.0474083, 0.0323217, 0.132119, 0.0114547,
      -0.00489126, 0.00694045, 0.114024, 0.0700617, -0.0299176, 0.0424512,
      0.00337714, 0.0578195, -0.00433905, 0.0432251, -0.0213373, 0.0110747,
      -0.0658904, -0.0731935, -0.0141772, -0.057951, -0.00894259, -0.00455883,
      0.00757611, -0.0500143, -0.0546965, -0.0278838, 0.0463387, -0.00148532,
      -0.0253688, -0.012847, -0.0468582, 0.0493546, 0.0707899, 0.00366008,
      -0.0156367;

  bool check2_before = ref2b.isApprox(tc[2], 1e-5);
  if (!check2_before) {
    cout << "tc2" << endl;
    cout << tc[2] << endl;
    cout << "tc2_ref" << endl;
    cout << ref2b << endl;
  }

  BOOST_CHECK_EQUAL(check2_before, true);

  Eigen::MatrixXd ref4b = Eigen::MatrixXd::Zero(8, 17);
  ref4b << 0.0472538, 0.192218, 3.2613e-11, 3.26223e-11, 3.23861e-11, 0.109122,
      4.16604e-11, 4.16302e-11, 4.13551e-11, -0.00314721, 0.0108472,
      -0.00314721, 0.0108472, -0.00314721, 0.0108472, -0.00314721, 0.0108472,
      -9.69614e-09, -4.99998e-08, -0.0735666, -0.0700279, -0.071047,
      -2.88054e-08, 0.0080416, 0.0076548, 0.00776616, 0.0254243, 0.0739155,
      -0.00799633, -0.0232476, -0.00859324, -0.024983, -0.00883465, -0.0256848,
      -3.83792e-09, -1.12535e-08, -0.017741, 0.0960538, -0.0763057, -8.3889e-09,
      0.00193927, -0.0104997, 0.00834096, -0.000237732, -0.000691164,
      0.00444057, 0.01291, 0.0183145, 0.0532455, -0.0225174, -0.0654643,
      -4.05403e-09, -1.74883e-08, 0.0981674, -0.0351198, -0.0670328,
      -1.08265e-08, -0.0107307, 0.00383896, 0.00732739, 0.000472061, 0.00137243,
      -0.0237278, -0.0689835, 0.0154079, 0.0447953, 0.00784782, 0.0228158,
      0.0455444, 0.117954, 4.4569e-11, 4.45493e-11, 4.42396e-11, 0.0478717,
      -1.61634e-11, -1.61317e-11, -1.60405e-11, 0.0387858, 0.261043, 0.0387858,
      0.261043, 0.0387858, 0.261043, 0.0387858, 0.261043, -8.50072e-08,
      -3.87116e-07, -0.0459515, -0.0454214, -0.0455205, -2.34191e-07,
      -0.0225363, -0.0222763, -0.0223249, -0.0485099, -0.161219, 0.015943,
      0.0529854, 0.0162484, 0.0540005, 0.0163186, 0.0542338, 4.54389e-08,
      2.08758e-07, -0.0360811, 0.0645223, -0.0279594, 1.25806e-07, -0.0176956,
      0.0316445, -0.0137123, 0.000170843, 0.000567771, -0.0257425, -0.0855535,
      -0.0199865, -0.0664236, 0.0455581, 0.151409, 4.78832e-09, 3.70108e-08,
      -0.0532291, -0.00452519, 0.0582485, 1.85207e-08, -0.0261058, -0.00221932,
      0.0285675, 0.000175092, 0.000581897, -0.0379003, -0.125959, 0.0411074,
      0.136617, -0.00338222, -0.0112405;

  bool check4_before = ref4b.isApprox(tc[4], 1e-5);
  if (!check4_before) {
    cout << "tc4" << endl;
    cout << tc[4] << endl;
    cout << "tc4_ref" << endl;
    cout << ref4b << endl;
  }

  BOOST_CHECK_EQUAL(check4_before, true);
}
BOOST_AUTO_TEST_SUITE_END()
