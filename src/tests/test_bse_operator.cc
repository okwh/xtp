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

#define BOOST_TEST_MODULE bse_test
#include <boost/test/unit_test.hpp>
#include <fstream>
#include <votca/xtp/bse_operator.h>

using namespace votca::xtp;
using namespace std;

BOOST_AUTO_TEST_SUITE(bse_test)

BOOST_AUTO_TEST_CASE(bse_operator) {

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
  orbitals.LoadFromXYZ("molecule.xyz");
  BasisSet basis;
  basis.LoadBasisSet("3-21G.xml");
  orbitals.setDFTbasisName("3-21G.xml");
  AOBasis aobasis;
  aobasis.AOBasisFill(basis, orbitals.QMAtoms());

  orbitals.setBasisSetSize(17);
  orbitals.setNumberOfOccupiedLevels(4);
  Eigen::MatrixXd& MOs = orbitals.MOCoefficients();
  MOs = Eigen::MatrixXd::Zero(17, 17);
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

  Eigen::MatrixXd Hqp = Eigen::MatrixXd::Zero(17, 17);
  Hqp << -0.934164, 4.16082e-07, -1.3401e-07, 1.36475e-07, 0.031166,
      1.20677e-06, -6.81123e-07, -1.22621e-07, -1.83709e-07, 1.76372e-08,
      -1.7807e-07, -0.0220743, -1.4977e-06, -1.75301e-06, -5.29037e-08,
      0.00737784, -0.00775225, 4.16082e-07, -0.461602, 1.12979e-07,
      -1.47246e-07, -1.3086e-07, 0.0443459, 0.000553929, 0.000427421,
      8.38616e-05, 0.000289144, -0.0101872, -1.28339e-07, 0.0141886,
      -0.000147938, -0.000241557, 5.71202e-07, 2.1119e-09, -1.3401e-07,
      1.12979e-07, -0.461602, 1.72197e-07, 2.8006e-08, -0.000335948, 0.0406153,
      -0.0178151, 0.0101352, 0.00106636, 0.000113704, 1.22667e-07, -0.000128128,
      -0.0141459, 0.00111572, -4.57761e-07, 5.12848e-09, 1.36475e-07,
      -1.47246e-07, 1.72197e-07, -0.461601, -4.34283e-08, 0.000614026,
      -0.0178095, -0.0406149, 0.00106915, -0.0101316, -0.00027881, 4.86348e-08,
      0.000252415, 0.00111443, 0.0141441, 1.01087e-07, 1.3741e-09, 0.031166,
      -1.3086e-07, 2.8006e-08, -4.34283e-08, 0.00815998, -1.70198e-07,
      1.14219e-07, 1.10593e-09, -4.81365e-08, 2.75431e-09, -2.95191e-08,
      -0.0166337, 5.78666e-08, 8.52843e-08, -1.74815e-08, -0.00112475,
      -0.0204625, 1.20677e-06, 0.0443459, -0.000335948, 0.000614026,
      -1.70198e-07, 0.323811, 1.65813e-07, -1.51122e-08, -2.98465e-05,
      -0.000191357, 0.0138568, 2.86823e-07, -0.0372319, 6.58278e-05,
      0.000142268, -2.94575e-07, 3.11298e-08, -6.81123e-07, 0.000553929,
      0.0406153, -0.0178095, 1.14219e-07, 1.65813e-07, 0.323811, -6.98568e-09,
      -0.0120376, -0.00686446, -0.000120523, -1.7727e-07, 0.000108686,
      0.0351664, 0.0122284, 1.86591e-07, -1.95807e-08, -1.22621e-07,
      0.000427421, -0.0178151, -0.0406149, 1.10593e-09, -1.51122e-08,
      -6.98568e-09, 0.323811, 0.00686538, -0.0120366, -0.00015138, 1.6913e-07,
      0.000112864, -0.0122286, 0.0351659, -2.32341e-08, 2.57386e-09,
      -1.83709e-07, 8.38616e-05, 0.0101352, 0.00106915, -4.81365e-08,
      -2.98465e-05, -0.0120376, 0.00686538, 0.901732, 6.12076e-08, -9.96554e-08,
      2.57089e-07, -1.03264e-05, 0.00917151, -0.00170387, -3.30584e-07,
      -9.14928e-09, 1.76372e-08, 0.000289144, 0.00106636, -0.0101316,
      2.75431e-09, -0.000191357, -0.00686446, -0.0120366, 6.12076e-08, 0.901732,
      -2.4407e-08, -1.19304e-08, -9.06429e-05, 0.00170305, 0.00917133,
      -1.11726e-07, -6.52056e-09, -1.7807e-07, -0.0101872, 0.000113704,
      -0.00027881, -2.95191e-08, 0.0138568, -0.000120523, -0.00015138,
      -9.96554e-08, -2.4407e-08, 0.901732, 3.23124e-07, 0.00932737, 2.69633e-05,
      8.74181e-05, -4.83481e-07, -1.90439e-08, -0.0220743, -1.28339e-07,
      1.22667e-07, 4.86348e-08, -0.0166337, 2.86823e-07, -1.7727e-07,
      1.6913e-07, 2.57089e-07, -1.19304e-08, 3.23124e-07, 1.2237, -7.31155e-07,
      -6.14518e-07, 2.79634e-08, -0.042011, 0.0229724, -1.4977e-06, 0.0141886,
      -0.000128128, 0.000252415, 5.78666e-08, -0.0372319, 0.000108686,
      0.000112864, -1.03264e-05, -9.06429e-05, 0.00932737, -7.31155e-07,
      1.21009, -2.99286e-07, -4.29557e-08, 6.13566e-07, -7.73601e-08,
      -1.75301e-06, -0.000147938, -0.0141459, 0.00111443, 8.52843e-08,
      6.58278e-05, 0.0351664, -0.0122286, 0.00917151, 0.00170305, 2.69633e-05,
      -6.14518e-07, -2.99286e-07, 1.21009, 2.02234e-07, 7.00978e-07,
      -7.18964e-08, -5.29037e-08, -0.000241557, 0.00111572, 0.0141441,
      -1.74815e-08, 0.000142268, 0.0122284, 0.0351659, -0.00170387, 0.00917133,
      8.74181e-05, 2.79634e-08, -4.29557e-08, 2.02234e-07, 1.21009, 3.77938e-08,
      -4.85316e-09, 0.00737784, 5.71202e-07, -4.57761e-07, 1.01087e-07,
      -0.00112475, -2.94575e-07, 1.86591e-07, -2.32341e-08, -3.30584e-07,
      -1.11726e-07, -4.83481e-07, -0.042011, 6.13566e-07, 7.00978e-07,
      3.77938e-08, 1.93666, 0.0330278, -0.00775225, 2.1119e-09, 5.12848e-09,
      1.3741e-09, -0.0204625, 3.11298e-08, -1.95807e-08, 2.57386e-09,
      -9.14928e-09, -6.52056e-09, -1.90439e-08, 0.0229724, -7.73601e-08,
      -7.18964e-08, -4.85316e-09, 0.0330278, 19.4256;

  Eigen::VectorXd& mo_energy = orbitals.MOEnergies();
  mo_energy = Eigen::VectorXd::Zero(17);
  mo_energy << -0.612601, -0.341755, -0.341755, -0.341755, 0.137304, 0.16678,
      0.16678, 0.16678, 0.671592, 0.671592, 0.671592, 0.974255, 1.01205,
      1.01205, 1.01205, 1.64823, 19.4429;
  TCMatrix_gwbse Mmn;
  Mmn.Initialize(aobasis.AOBasisSize(), 0, 16, 0, 16);
  Mmn.Fill(aobasis, aobasis, MOs);

  Eigen::MatrixXd rpa_op =
      Eigen::MatrixXd::Zero(aobasis.AOBasisSize(), aobasis.AOBasisSize());
  rpa_op << -0.0819076, 0.873256, -0.378469, 1.68023e-06, -2.36934e-06,
      3.41943e-06, 6.54367e-07, -1.99125e-07, -7.11838e-07, 3.2758e-06,
      -2.64521e-06, -3.24863e-06, 0.0686844, 2.15553e-06, 2.93749e-06,
      -4.11309e-07, 0.287679, 0.0240972, -0.426346, -0.333171, 1.65383e-06,
      -2.28096e-06, 3.29356e-06, 9.00551e-07, -1.53341e-07, -9.39699e-07,
      6.0863e-06, -5.047e-06, -6.01352e-06, 0.146325, 6.04035e-06, 8.27975e-06,
      -8.64375e-07, 0.827789, -8.39873e-07, -8.90182e-08, -3.00985e-06,
      -0.289232, -0.329216, -0.368304, -0.175029, -0.124768, -0.260247,
      -0.167733, -0.432036, 0.489164, 9.22329e-06, -0.221579, 0.0579262,
      -0.227726, 1.09716e-06, -3.57458e-06, -3.57667e-07, -8.10118e-06,
      -0.245541, 0.466143, -0.223846, 0.119718, -0.308333, 0.0673053, -0.42415,
      0.456019, 0.257321, 3.32182e-05, -0.0937161, -0.308819, 0.0126313,
      4.8962e-06, -1.16485e-06, -1.28266e-07, -2.65886e-06, 0.428652, 0.0448793,
      -0.376736, -0.262609, -0.0574013, 0.204137, 0.496016, 0.24385, 0.385456,
      3.50738e-06, -0.215483, 0.0747438, 0.228675, 1.25422e-06, -0.204121,
      0.199358, 0.842556, -3.45133e-06, 4.58146e-06, -6.81122e-06, -1.26095e-06,
      7.92269e-07, 1.35382e-06, -2.86268e-06, 2.12005e-06, 2.865e-06,
      -0.0410056, 3.57738e-06, 4.80023e-06, -1.85119e-07, 0.454982, 8.28886e-07,
      1.3967e-07, 3.57945e-06, 0.389537, 0.44339, 0.496036, -0.233035,
      -0.166118, -0.346501, -0.0621401, -0.160059, 0.181223, 3.07702e-06,
      -0.257915, 0.0674255, -0.265072, 9.72181e-07, 2.53048e-06, 3.30071e-07,
      7.41356e-06, 0.330695, -0.627804, 0.301479, 0.159393, -0.41052, 0.0896121,
      -0.157137, 0.168943, 0.0953314, 1.06852e-05, -0.109084, -0.359464,
      0.0147028, 4.73825e-06, -5.68061e-07, -1.32258e-07, 1.03008e-06,
      -0.577308, -0.0604421, 0.507392, -0.349645, -0.0764239, 0.271796,
      0.183761, 0.09034, 0.142801, 1.52225e-06, -0.250819, 0.0870013, 0.266177,
      1.2118e-06, -0.0325802, 0.00156618, 0.0506931, -0.0258426, 0.0442732,
      -0.235944, -0.24355, -0.37576, 0.00857695, 0.0494365, -0.138083,
      -0.583542, 0.492079, -0.340925, -0.113144, 0.00872256, -0.0648254,
      0.48652, 0.0630491, 0.0799668, 0.00130198, -0.00223089, 0.0118918,
      0.30399, 0.469011, -0.0107047, 0.00216364, -0.00604935, -0.0255728,
      0.0297364, -0.62698, -0.208076, 0.0160413, 0.0452451, -0.0325854,
      0.00156536, 0.0506936, -0.115027, -0.204617, 0.0565651, -0.0246178,
      0.184595, -0.407309, 0.123532, 0.583497, 0.0791735, 0.492113, 0.0562797,
      0.187558, -0.301269, -0.0648315, 0.486516, 0.0630483, 0.0799646,
      0.00579876, 0.0103157, -0.00285252, 0.0307262, -0.230407, 0.508395,
      0.00541004, 0.0255707, 0.00347233, 0.0297353, 0.103503, 0.344932,
      -0.55405, 0.0452339, -0.0325877, 0.00156503, 0.0506913, 0.234616,
      -0.022415, 0.0524573, -0.158801, 0.287808, 0.304185, -0.560812, -0.113356,
      0.186099, 0.492121, 0.0641106, 0.209163, 0.285043, -0.0648311, 0.486518,
      0.0630486, 0.0799664, -0.0118291, 0.00113049, -0.00264498, 0.198213,
      -0.359236, -0.379678, -0.0245742, -0.00496569, 0.00815754, 0.0297341,
      0.117904, 0.384664, 0.524209, 0.0452344, -0.0325816, 0.00156597,
      0.0506945, -0.0937463, 0.182757, 0.126922, 0.426971, -0.096647, 0.094546,
      0.387903, -0.332107, 0.318209, 0.49209, 0.220536, -0.283577, 0.0075039,
      -0.0648279, 0.486517, 0.0630486, 0.0799651, 0.00472491, -0.00921029,
      -0.00640003, -0.532934, 0.120634, -0.118008, 0.0169963, -0.0145519,
      0.0139477, 0.0297365, 0.40558, -0.521511, 0.0138003, 0.0452407;
  Eigen::VectorXd epsilon_inv = Eigen::VectorXd::Zero(aobasis.AOBasisSize());
  Mmn.MultiplyRightWithAuxMatrix(rpa_op);
  epsilon_inv << 0.999807798016267, 0.994206065211371, 0.917916768047073,
      0.902913813951883, 0.902913745974602, 0.902913584797742,
      0.853352878674581, 0.853352727016914, 0.853352541699637, 0.79703468058566,
      0.797034577207669, 0.797034400395582, 0.787701833916331,
      0.518976361745313, 0.518975064844033, 0.518973712898761,
      0.459286057710524;

  BSEOperator_Options opt;
  opt.cmax = 8;
  opt.homo = 4;
  opt.qpmin = 0;
  opt.rpamin = 0;
  opt.vmin = 0;

  orbitals.setBSEindices(0, 16);

  HqpOperator Hqp_op(epsilon_inv, Mmn, Hqp);
  Hqp_op.configure(opt);
  Eigen::MatrixXd hqp_mat = Hqp_op.get_full_matrix();

  Eigen::MatrixXd hqp_ref = Eigen::MatrixXd::Zero(20, 20);
  hqp_ref << 1.25798, 1.65813e-07, -1.51122e-08, -2.98465e-05, -4.16082e-07, 0,
      0, 0, 1.3401e-07, 0, 0, 0, -1.36475e-07, 0, 0, 0, -0.031166, 0, 0, 0,
      1.65813e-07, 1.25798, -6.98568e-09, -0.0120376, 0, -4.16082e-07, 0, 0, 0,
      1.3401e-07, 0, 0, 0, -1.36475e-07, 0, 0, 0, -0.031166, 0, 0, -1.51122e-08,
      -6.98568e-09, 1.25798, 0.00686538, 0, 0, -4.16082e-07, 0, 0, 0,
      1.3401e-07, 0, 0, 0, -1.36475e-07, 0, 0, 0, -0.031166, 0, -2.98465e-05,
      -0.0120376, 0.00686538, 1.8359, 0, 0, 0, -4.16082e-07, 0, 0, 0,
      1.3401e-07, 0, 0, 0, -1.36475e-07, 0, 0, 0, -0.031166, -4.16082e-07, 0, 0,
      0, 0.785413, 1.65813e-07, -1.51122e-08, -2.98465e-05, -1.12979e-07, 0, 0,
      0, 1.47246e-07, 0, 0, 0, 1.3086e-07, 0, 0, 0, 0, -4.16082e-07, 0, 0,
      1.65813e-07, 0.785413, -6.98568e-09, -0.0120376, 0, -1.12979e-07, 0, 0, 0,
      1.47246e-07, 0, 0, 0, 1.3086e-07, 0, 0, 0, 0, -4.16082e-07, 0,
      -1.51122e-08, -6.98568e-09, 0.785413, 0.00686538, 0, 0, -1.12979e-07, 0,
      0, 0, 1.47246e-07, 0, 0, 0, 1.3086e-07, 0, 0, 0, 0, -4.16082e-07,
      -2.98465e-05, -0.0120376, 0.00686538, 1.36333, 0, 0, 0, -1.12979e-07, 0,
      0, 0, 1.47246e-07, 0, 0, 0, 1.3086e-07, 1.3401e-07, 0, 0, 0, -1.12979e-07,
      0, 0, 0, 0.785413, 1.65813e-07, -1.51122e-08, -2.98465e-05, -1.72197e-07,
      0, 0, 0, -2.8006e-08, 0, 0, 0, 0, 1.3401e-07, 0, 0, 0, -1.12979e-07, 0, 0,
      1.65813e-07, 0.785413, -6.98568e-09, -0.0120376, 0, -1.72197e-07, 0, 0, 0,
      -2.8006e-08, 0, 0, 0, 0, 1.3401e-07, 0, 0, 0, -1.12979e-07, 0,
      -1.51122e-08, -6.98568e-09, 0.785413, 0.00686538, 0, 0, -1.72197e-07, 0,
      0, 0, -2.8006e-08, 0, 0, 0, 0, 1.3401e-07, 0, 0, 0, -1.12979e-07,
      -2.98465e-05, -0.0120376, 0.00686538, 1.36333, 0, 0, 0, -1.72197e-07, 0,
      0, 0, -2.8006e-08, -1.36475e-07, 0, 0, 0, 1.47246e-07, 0, 0, 0,
      -1.72197e-07, 0, 0, 0, 0.785412, 1.65813e-07, -1.51122e-08, -2.98465e-05,
      4.34283e-08, 0, 0, 0, 0, -1.36475e-07, 0, 0, 0, 1.47246e-07, 0, 0, 0,
      -1.72197e-07, 0, 0, 1.65813e-07, 0.785412, -6.98568e-09, -0.0120376, 0,
      4.34283e-08, 0, 0, 0, 0, -1.36475e-07, 0, 0, 0, 1.47246e-07, 0, 0, 0,
      -1.72197e-07, 0, -1.51122e-08, -6.98568e-09, 0.785412, 0.00686538, 0, 0,
      4.34283e-08, 0, 0, 0, 0, -1.36475e-07, 0, 0, 0, 1.47246e-07, 0, 0, 0,
      -1.72197e-07, -2.98465e-05, -0.0120376, 0.00686538, 1.36333, 0, 0, 0,
      4.34283e-08, -0.031166, 0, 0, 0, 1.3086e-07, 0, 0, 0, -2.8006e-08, 0, 0,
      0, 4.34283e-08, 0, 0, 0, 0.315651, 1.65813e-07, -1.51122e-08,
      -2.98465e-05, 0, -0.031166, 0, 0, 0, 1.3086e-07, 0, 0, 0, -2.8006e-08, 0,
      0, 0, 4.34283e-08, 0, 0, 1.65813e-07, 0.315651, -6.98568e-09, -0.0120376,
      0, 0, -0.031166, 0, 0, 0, 1.3086e-07, 0, 0, 0, -2.8006e-08, 0, 0, 0,
      4.34283e-08, 0, -1.51122e-08, -6.98568e-09, 0.315651, 0.00686538, 0, 0, 0,
      -0.031166, 0, 0, 0, 1.3086e-07, 0, 0, 0, -2.8006e-08, 0, 0, 0,
      4.34283e-08, -2.98465e-05, -0.0120376, 0.00686538, 0.893572;

  bool check_hqp = hqp_mat.isApprox(hqp_ref, 0.001);
  BOOST_CHECK_EQUAL(check_hqp, true);

  HxOperator Hx(epsilon_inv, Mmn, Hqp);
  Hx.configure(opt);
  Eigen::MatrixXd hx_mat = Hx.get_full_matrix();
  Eigen::MatrixXd hx_ref = Eigen::MatrixXd::Zero(20, 20);
  hx_ref << 0.0317015, 4.06992e-08, -4.14577e-09, 9.36138e-06, 0.00091091,
      -1.20607e-05, -1.09028e-05, 2.24371e-05, -1.01492e-05, -0.000420091,
      0.000185019, 0.00325098, 2.12032e-05, 0.000181087, 0.000414638,
      0.000342333, 0.00119078, 4.10862e-08, 2.30707e-09, 3.31899e-05,
      4.06992e-08, 0.0317013, 7.76545e-09, 0.0037715, -1.19375e-05,
      -0.000463708, 1.06306e-05, 0.00290727, -0.000419931, -0.000624839,
      -0.000124124, 0.00343563, 0.000181026, 0.000115076, -0.000634214,
      -0.00295338, 8.78143e-08, 0.00119059, -6.26439e-10, 0.013381,
      -4.14577e-09, 7.76545e-09, 0.0317014, -0.00215103, -1.08803e-05,
      1.06413e-05, -0.000447228, -0.00164418, 0.000184951, -0.000124081,
      0.000634989, 0.00300657, 0.000414472, -0.000634159, -0.000136301,
      0.00343701, 2.35921e-09, -1.97262e-08, 0.0011905, -0.00763163,
      9.36138e-06, 0.0037715, -0.00215103, 0.0230454, -5.71537e-06,
      -0.000742875, 0.000420121, 0.00318535, -0.000830743, -0.000877814,
      -0.000768248, 0.00143223, -8.74755e-05, 0.000754659, -0.000878245,
      -0.00406991, 2.53494e-05, 0.0102192, -0.00582836, 0.000255093, 0.00091091,
      -1.19375e-05, -1.08803e-05, -5.71537e-06, 0.0702735, 0.000304627,
      0.000201634, 0.00023483, -0.00052808, 0.0234547, -0.0103174, 0.0357697,
      0.00101305, -0.0100469, -0.0229803, 0.00378956, 0.0315423, -0.000418981,
      -0.000377433, -7.25758e-05, -1.20607e-05, -0.000463708, 1.06413e-05,
      -0.000742875, 0.000304627, 0.00777325, -0.000339791, -0.00404126,
      0.00709574, 0.0110517, 0.00203678, -0.00439831, -0.00287127, -0.00245045,
      0.00991467, 0.00418666, -0.00041897, -0.0160561, 0.00036812, -0.00940634,
      -1.09028e-05, 1.06306e-05, -0.000447228, 0.000420121, 0.000201634,
      -0.000339791, 0.00722912, 0.00227024, -0.00314397, 0.00228412, -0.0105236,
      -0.00371587, -0.00662098, 0.0100501, 0.00143746, -0.00462777,
      -0.000377421, 0.000368108, -0.0154866, 0.00531967, 2.24371e-05,
      0.00290727, -0.00164418, 0.00318535, 0.00023483, -0.00404126, 0.00227024,
      0.00439426, -0.00451679, -0.00459631, -0.0042381, 0.00235892,
      -0.000455018, 0.00399959, -0.00497422, -0.00556497, 5.05012e-05,
      0.00654367, -0.0037007, 0.00656581, -1.01492e-05, -0.000419931,
      0.000184951, -0.000830743, -0.00052808, 0.00709574, -0.00314397,
      -0.00451679, 0.00759994, 0.0085118, 0.00617201, -0.00292845, 1.1332e-05,
      -0.0059183, 0.00882517, 0.00552135, -0.000351062, -0.0145382, 0.00640315,
      -0.0105189, -0.000420091, -0.000624839, -0.000124081, -0.000877814,
      0.0234547, 0.0110517, 0.00228412, -0.00459631, 0.0085118, 0.0548293,
      -0.0176831, 0.0134278, -0.00574617, -0.0173291, -0.0251939, 0.00640551,
      -0.0145382, -0.0216376, -0.00429764, -0.0111153, 0.000185019,
      -0.000124124, 0.000634989, -0.000768248, -0.0103174, 0.00203678,
      -0.0105236, -0.0042381, 0.00617201, -0.0176831, 0.0228469, -0.00476467,
      0.00860197, -0.00733196, 0.0173178, 0.00658172, 0.00640313, -0.00429762,
      0.0219889, -0.00972763, 0.00325098, 0.00343563, 0.00300657, 0.00143223,
      0.0357697, -0.00439831, -0.00371587, 0.00235892, -0.00292845, 0.0134278,
      -0.00476467, 0.0539635, 0.00611128, -0.0136642, -0.0286051, 0.00489966,
      0.00731748, 0.00773268, 0.00676718, 0.00295214, 2.12032e-05, 0.000181026,
      0.000414472, -8.74755e-05, 0.00101305, -0.00287127, -0.00662098,
      -0.000455018, 1.1332e-05, -0.00574617, 0.00860197, 0.00611128, 0.00740256,
      -0.0088164, -0.00637366, 0.00269601, 0.000734048, 0.00626705, 0.0143503,
      -0.00110766, 0.000181087, 0.000115076, -0.000634159, 0.000754659,
      -0.0100469, -0.00245045, 0.0100501, 0.00399959, -0.0059183, -0.0173291,
      -0.00733196, -0.0136642, -0.0088164, 0.0226735, 0.018023, -0.00842983,
      0.00626703, 0.00398579, -0.0219588, 0.00955559, 0.000414638, -0.000634214,
      -0.000136301, -0.000878245, -0.0229803, 0.00991467, 0.00143746,
      -0.00497422, 0.00882517, -0.0251939, 0.0173178, -0.0286051, -0.00637366,
      0.018023, 0.0552003, 0.00194874, 0.0143502, -0.0219588, -0.00472001,
      -0.0111203, 0.000342333, -0.00295338, 0.00343701, -0.00406991, 0.00378956,
      0.00418666, -0.00462777, -0.00556497, 0.00552135, 0.00640551, 0.00658172,
      0.00489966, 0.00269601, -0.00842983, 0.00194874, 0.0085925, 0.000770536,
      -0.0066475, 0.00773602, -0.00838907, 0.00119078, 8.78143e-08, 2.35921e-09,
      2.53494e-05, 0.0315423, -0.00041897, -0.000377421, 5.05012e-05,
      -0.000351062, -0.0145382, 0.00640313, 0.00731748, 0.000734048, 0.00626703,
      0.0143502, 0.000770536, 0.0455433, -3.75552e-08, -6.77755e-09,
      5.00633e-05, 4.10862e-08, 0.00119059, -1.97262e-08, 0.0102192,
      -0.000418981, -0.0160561, 0.000368108, 0.00654367, -0.0145382, -0.0216376,
      -0.00429762, 0.00773268, 0.00626705, 0.00398579, -0.0219588, -0.0066475,
      -3.75552e-08, 0.0455434, 2.00663e-08, 0.0201905, 2.30707e-09,
      -6.26439e-10, 0.0011905, -0.00582836, -0.000377433, 0.00036812,
      -0.0154866, -0.0037007, 0.00640315, -0.00429764, 0.0219889, 0.00676718,
      0.0143503, -0.0219588, -0.00472001, 0.00773602, -6.77755e-09, 2.00663e-08,
      0.0455436, -0.0115154, 3.31899e-05, 0.013381, -0.00763163, 0.000255093,
      -7.25758e-05, -0.00940634, 0.00531967, 0.00656581, -0.0105189, -0.0111153,
      -0.00972763, 0.00295214, -0.00110766, 0.00955559, -0.0111203, -0.00838907,
      5.00633e-05, 0.0201905, -0.0115154, 0.0226683;

  bool check_hx = hx_mat.isApprox(hx_ref, 0.001);
  BOOST_CHECK_EQUAL(check_hx, true);
  HdOperator Hd(epsilon_inv, Mmn, Hqp);
  Hd.configure(opt);
  Eigen::MatrixXd hd_mat = Hd.get_full_matrix();
  Eigen::MatrixXd hd_ref = Eigen::MatrixXd::Zero(20, 20);
  hd_ref << -0.335802, -8.4393e-08, 4.23805e-08, -1.06274e-05, 0.00516896,
      -6.87255e-05, -6.18472e-05, -3.07981e-05, -5.76426e-05, -0.00238218,
      0.00104919, -0.00446274, 0.000120325, 0.0010269, 0.00235137, -0.000469934,
      -0.0211312, -2.95095e-08, 4.2425e-09, -2.09562e-05, -8.4393e-08,
      -0.335802, -1.06929e-07, -0.00426822, -6.87255e-05, -0.00263072,
      6.03207e-05, -0.00399078, -0.00238218, -0.00354577, -0.000704212,
      -0.00471584, 0.0010269, 0.000653196, -0.0035982, 0.00405406, -2.95095e-08,
      -0.0211311, -6.55583e-09, -0.00844905, 4.23805e-08, -1.06929e-07,
      -0.335803, 0.0024344, -6.18472e-05, 6.03207e-05, -0.0025374, 0.00225695,
      0.00104919, -0.000704212, 0.00360305, -0.00412706, 0.00235137, -0.0035982,
      -0.000773406, -0.0047179, 4.2425e-09, -6.55583e-09, -0.0211311, 0.0048188,
      -1.06274e-05, -0.00426822, 0.0024344, -0.367554, -3.07981e-05,
      -0.00399078, 0.00225695, -0.00137419, -0.00446274, -0.00471584,
      -0.00412706, -0.000618165, -0.000469934, 0.00405406, -0.0047179,
      0.00175624, -2.09562e-05, -0.00844905, 0.0048188, -0.030619, 0.00516896,
      -6.87255e-05, -6.18472e-05, -3.07981e-05, -0.349332, -5.34818e-05,
      -1.93503e-05, -9.25475e-06, 0.000258142, -0.00466293, 0.00206469,
      -0.000329327, -0.000503232, 0.00188842, 0.00435145, -3.35309e-05,
      -0.0176993, 0.000235101, 0.000211784, 1.19226e-05, -6.87255e-05,
      -0.00263072, 6.03207e-05, -0.00399078, -5.34818e-05, -0.320001,
      0.000224556, -0.0011046, -0.00466293, -0.00712644, -0.00139707,
      -0.000357616, 0.00188842, 0.00155091, -0.00664818, 0.000301561,
      0.000235101, 0.00900931, -0.000206559, 0.00154545, -6.18472e-05,
      6.03207e-05, -0.0025374, 0.00225695, -1.93503e-05, 0.000224556, -0.319646,
      0.000627135, 0.00206469, -0.00139707, 0.00686856, -0.000299414,
      0.00435145, -0.00664818, -0.00104806, -0.000340601, 0.000211784,
      -0.000206559, 0.00868974, -0.000874015, -3.07981e-05, -0.00399078,
      0.00225695, -0.00137419, -9.25475e-06, -0.0011046, 0.000627135, -0.350415,
      -0.000329327, -0.000357616, -0.000299414, -0.00188879, -3.35309e-05,
      0.000301561, -0.000340601, 0.00498952, 1.19226e-05, 0.00154545,
      -0.000874015, -0.000559131, -5.76426e-05, -0.00238218, 0.00104919,
      -0.00446274, 0.000258142, -0.00466293, 0.00206469, -0.000329327,
      -0.319889, -0.00567365, -0.00401919, -0.000193428, -8.5491e-06,
      0.00392394, -0.00571663, 0.000405147, 0.000197048, 0.00815769,
      -0.00359294, 0.00172823, -0.00238218, -0.00354577, -0.000704212,
      -0.00471584, -0.00466293, -0.00712644, -0.00139707, -0.000357616,
      -0.00567365, -0.341076, 0.00730043, -0.00252153, 0.00392394, 0.00706939,
      0.00671003, 0.000183403, 0.00815769, 0.0121414, 0.00241149, 0.00182621,
      0.00104919, -0.000704212, 0.00360305, -0.00412706, 0.00206469,
      -0.00139707, 0.00686856, -0.000299414, -0.00401919, 0.00730043, -0.328014,
      0.00129479, -0.00571663, 0.00671003, -0.00706049, 0.000604704,
      -0.00359294, 0.00241149, -0.0123384, 0.00159823, -0.00446274, -0.00471584,
      -0.00412706, -0.000618165, -0.000329327, -0.000357616, -0.000299414,
      -0.00188879, -0.000193428, -0.00252153, 0.00129479, -0.368513,
      0.000405147, 0.000183403, 0.000604704, -0.00161508, 0.00172823,
      0.00182621, 0.00159823, -0.00025126, 0.000120325, 0.0010269, 0.00235137,
      -0.000469934, -0.000503232, 0.00188842, 0.00435145, -3.35309e-05,
      -8.5491e-06, 0.00392394, -0.00571663, 0.000405147, -0.319757, 0.00572691,
      0.00403866, 0.00019034, -0.000411927, -0.00351657, -0.00805223,
      0.000181988, 0.0010269, 0.000653196, -0.0035982, 0.00405406, 0.00188842,
      0.00155091, -0.00664818, 0.000301561, 0.00392394, 0.00706939, 0.00671003,
      0.000183403, 0.00572691, -0.327899, -0.00752529, -0.00130003, -0.00351657,
      -0.00223657, 0.0123216, -0.00156996, 0.00235137, -0.0035982, -0.000773406,
      -0.0047179, 0.00435145, -0.00664818, -0.00104806, -0.000340601,
      -0.00571663, 0.00671003, -0.00706049, 0.000604704, 0.00403866,
      -0.00752529, -0.341321, 0.000887874, -0.00805223, 0.0123216, 0.00264845,
      0.00182704, -0.000469934, 0.00405406, -0.0047179, 0.00175624,
      -3.35309e-05, 0.000301561, -0.000340601, 0.00498952, 0.000405147,
      0.000183403, 0.000604704, -0.00161508, 0.00019034, -0.00130003,
      0.000887874, -0.353872, 0.000181988, -0.00156996, 0.00182704, 0.000714173,
      -0.0211312, -2.95095e-08, 4.2425e-09, -2.09562e-05, -0.0176993,
      0.000235101, 0.000211784, 1.19226e-05, 0.000197048, 0.00815769,
      -0.00359294, 0.00172823, -0.000411927, -0.00351657, -0.00805223,
      0.000181988, -0.284456, -3.66628e-08, 3.20055e-08, -8.81468e-06,
      -2.95095e-08, -0.0211311, -6.55583e-09, -0.00844905, 0.000235101,
      0.00900931, -0.000206559, 0.00154545, 0.00815769, 0.0121414, 0.00241149,
      0.00182621, -0.00351657, -0.00223657, 0.0123216, -0.00156996,
      -3.66628e-08, -0.284456, -8.87303e-08, -0.00354179, 4.2425e-09,
      -6.55583e-09, -0.0211311, 0.0048188, 0.000211784, -0.000206559,
      0.00868974, -0.000874015, -0.00359294, 0.00241149, -0.0123384, 0.00159823,
      -0.00805223, 0.0123216, 0.00264845, 0.00182704, 3.20055e-08, -8.87303e-08,
      -0.284457, 0.00202007, -2.09562e-05, -0.00844905, 0.0048188, -0.030619,
      1.19226e-05, 0.00154545, -0.000874015, -0.000559131, 0.00172823,
      0.00182621, 0.00159823, -0.00025126, 0.000181988, -0.00156996, 0.00182704,
      0.000714173, -8.81468e-06, -0.00354179, 0.00202007, -0.297356;

  bool check_hd = hd_mat.isApprox(hd_ref, 0.001);

  if (!check_hd) {
    cout << "hd ref" << endl;
    cout << hd_ref << endl;
    cout << "hd result" << endl;
    cout << hd_mat << endl;
  }
  BOOST_CHECK_EQUAL(check_hd, true);

  Hd2Operator Hd2(epsilon_inv, Mmn, Hqp);
  Hd2.configure(opt);
  Eigen::MatrixXd hd2_mat = Hd2.get_full_matrix();
  Eigen::MatrixXd hd2_ref = Eigen::MatrixXd::Zero(20, 20);
  hd2_ref << -0.0256069, -2.91928e-08, 2.83015e-09, -7.52901e-06, -0.000289686,
      3.7802e-06, 3.45517e-06, -6.09283e-06, 3.22908e-06, 0.00013358,
      -5.88338e-05, -0.000882185, -6.74345e-06, -5.7584e-05, -0.000131839,
      -9.29009e-05, -0.00010064, -4.06155e-08, -1.75622e-09, -1.44151e-05,
      -2.91928e-08, -0.0256068, -2.15255e-09, -0.00303458, 3.81107e-06,
      0.000147473, -3.38629e-06, -0.000788889, 0.000133623, 0.000198709,
      3.94539e-05, -0.000932238, -5.75989e-05, -3.65925e-05, 0.000201681,
      0.000801408, -2.28014e-08, -0.000100546, 9.39613e-09, -0.00580974,
      2.83015e-09, -2.15255e-09, -0.0256069, 0.00173073, 3.46778e-06,
      -3.38144e-06, 0.000142225, 0.000446155, -5.88529e-05, 3.9469e-05,
      -0.000201929, -0.000815844, -0.000131887, 0.000201689, 4.33442e-05,
      -0.00093264, -2.52569e-09, 2.71301e-09, -0.000100494, 0.00331349,
      -7.52901e-06, -0.00303458, 0.00173073, -0.0196439, -1.82533e-05,
      -0.00236535, 0.0013377, -0.00179916, -0.00264504, -0.00279517,
      -0.00244615, -0.000808958, -0.00027853, 0.00240287, -0.00279635,
      0.00229878, -2.58894e-05, -0.01044, 0.00595433, 0.00210891, -0.000289686,
      3.81107e-06, 3.46778e-06, -1.82533e-05, -0.0363162, -0.000133456,
      -8.43293e-05, -9.45602e-05, 0.000272729, -0.0039637, 0.00175589,
      0.0025352, -0.000525204, 0.00160429, 0.00369862, 0.000255936, -0.0156059,
      0.000207297, 0.000186733, -2.05887e-05, 3.7802e-06, 0.000147473,
      -3.38144e-06, -0.00236535, -0.000133456, -0.00434139, 0.000190149,
      0.00226901, -0.0104119, -0.00613945, -0.00124991, 0.00261425, 0.00443273,
      0.001354, -0.00562495, -0.00226011, 0.000207304, 0.00794392, -0.000182128,
      -0.00266718, 3.45517e-06, -3.38629e-06, 0.000142225, 0.0013377,
      -8.43293e-05, 0.000190149, -0.00403763, -0.00127403, 0.00458345,
      -0.00115241, 0.00586673, 0.00236406, 0.0101469, -0.00557157, -0.00082882,
      0.00275813, 0.000186738, -0.000182133, 0.00766214, 0.00150839,
      -6.09283e-06, -0.000788889, 0.000446155, -0.00179916, -9.45602e-05,
      0.00226901, -0.00127403, -0.00296554, -0.0155794, 0.00252521, 0.00212925,
      -0.00155907, -0.00165261, -0.00234423, 0.00260235, 0.00375944,
      4.05481e-05, 0.00525425, -0.00297149, -0.00376323, 3.22908e-06,
      0.000133623, -5.88529e-05, -0.00264504, 0.000272729, -0.0104119,
      0.00458345, -0.0155794, -0.00424488, -0.00477475, -0.00343851, 0.00160979,
      -6.63416e-06, 0.00324683, -0.00482106, -0.00336823, 0.000173697,
      0.0071929, -0.00316801, -0.0029826, 0.00013358, 0.000198709, 3.9469e-05,
      -0.00279517, -0.0039637, -0.00613945, -0.00115241, 0.00252521,
      -0.00477475, -0.0281628, 0.0087969, -0.00341647, 0.00331468, 0.00859948,
      0.00456902, 0.00586321, 0.00719292, 0.0107054, 0.00212629, -0.00315181,
      -5.88338e-05, 3.94539e-05, -0.000201929, -0.00244615, 0.00175589,
      -0.00124991, 0.00586673, 0.00212925, -0.00343851, 0.0087969, -0.0122876,
      0.000866965, -0.00490904, 0.0116096, -0.00859285, 0.0119362, -0.00316802,
      0.0021263, -0.0108792, -0.00275826, -0.000882185, -0.000932238,
      -0.000815844, -0.000808958, 0.0025352, 0.00261425, 0.00236406,
      -0.00155907, 0.00160979, -0.00341647, 0.000866965, -0.0324321,
      -0.00310298, -0.003161, -0.00388528, -0.00288611, 0.0058757, 0.0062089,
      0.00543371, -0.00169203, -6.74345e-06, -5.75989e-05, -0.000131887,
      -0.00027853, -0.000525204, 0.00443273, 0.0101469, -0.00165261,
      -6.63416e-06, 0.00331468, -0.00490904, -0.00310298, -0.00413423,
      0.00490817, 0.00352285, -0.00150682, -0.000363181, -0.00310067,
      -0.0070999, -0.000314065, -5.7584e-05, -3.65925e-05, 0.000201689,
      0.00240287, 0.00160429, 0.001354, -0.00557157, -0.00234423, 0.00324683,
      0.00859948, 0.0116096, -0.003161, 0.00490817, -0.012191, -0.00898706,
      0.00454143, -0.00310068, -0.00197203, 0.0108643, 0.0027095, -0.000131839,
      0.000201681, 4.33442e-05, -0.00279635, 0.00369862, -0.00562495,
      -0.00082882, 0.00260235, -0.00482106, 0.00456902, -0.00859285,
      -0.00388528, 0.00352285, -0.00898706, -0.0283701, -0.00152861,
      -0.00709991, 0.0108643, 0.00233526, -0.00315316, -9.29009e-05,
      0.000801408, -0.00093264, 0.00229878, 0.000255936, -0.00226011,
      0.00275813, 0.00375944, -0.00336823, 0.00586321, 0.0119362, -0.00288611,
      -0.00150682, 0.00454143, -0.00152861, -0.00575493, 0.000618722,
      -0.00533762, 0.00621165, 0.00480824, -0.00010064, -2.28014e-08,
      -2.52569e-09, -2.58894e-05, -0.0156059, 0.000207304, 0.000186738,
      4.05481e-05, 0.000173697, 0.00719292, -0.00316802, 0.0058757,
      -0.000363181, -0.00310068, -0.00709991, 0.000618722, -0.0247801,
      5.57194e-09, 2.50858e-09, -2.51858e-05, -4.06155e-08, -0.000100546,
      2.71301e-09, -0.01044, 0.000207297, 0.00794392, -0.000182133, 0.00525425,
      0.0071929, 0.0107054, 0.0021263, 0.0062089, -0.00310067, -0.00197203,
      0.0108643, -0.00533762, 5.57194e-09, -0.0247801, -3.83195e-09, -0.0101552,
      -1.75622e-09, 9.39613e-09, -0.000100494, 0.00595433, 0.000186733,
      -0.000182128, 0.00766214, -0.00297149, -0.00316801, 0.00212629,
      -0.0108792, 0.00543371, -0.0070999, 0.0108643, 0.00233526, 0.00621165,
      2.50858e-09, -3.83195e-09, -0.0247802, 0.00579189, -1.44151e-05,
      -0.00580974, 0.00331349, 0.00210891, -2.05887e-05, -0.00266718,
      0.00150839, -0.00376323, -0.0029826, -0.00315181, -0.00275826,
      -0.00169203, -0.000314065, 0.0027095, -0.00315316, 0.00480824,
      -2.51858e-05, -0.0101552, 0.00579189, -0.0145753;

  bool check_hd2 = hd2_mat.isApprox(hd2_ref, 0.001);
  if (!check_hd2) {
    cout << "hd2 ref" << endl;
    cout << hd2_ref << endl;
    cout << "hd2 result" << endl;
    cout << hd2_mat << endl;
  }
  BOOST_CHECK_EQUAL(check_hd2, true);
}

BOOST_AUTO_TEST_SUITE_END()
