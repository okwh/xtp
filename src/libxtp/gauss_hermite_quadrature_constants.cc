/*
 *            Copyright 2009-2018 The VOTCA Development Team
 *                       (http://www.votca.org)
 *
 *      Licensed under the Apache License, Version 2.0 (the "License")
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *              http://www.apache.org/licenses/LICENSe-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <votca/xtp/eigen.h>
#include <votca/xtp/gauss_hermite_quadrature_constants.h>

using namespace std;

/*******************
 * Public Methods *
 *******************/
namespace votca {
    namespace xtp {


const Eigen::VectorXd & Gauss_Hermite_Quadrature_Constants::getPoints(int order){
    if (!this->_filled_Points) {
        this->FillPoints();
        _filled_Points = true;
    }
    if (_map_points.count(order) == 0)
        throw invalid_argument("Order " + std::to_string(order)+  " not in range {8,10,12,...,20}.");
    return _map_points.at(order);
}

const Eigen::VectorXd & Gauss_Hermite_Quadrature_Constants::getWeights(int order){
    if (!this->_filled_Weights) {
        this->FillWeights();
        _filled_Weights = true;
    }
    if (_map_weights.count(order) == 0)
        throw invalid_argument("Order " + std::to_string(order)+ " not in range {8,10,12,...,20}.");
    return _map_weights.at(order);
}


/*******************
 * Private Methods *
 *******************/


void Gauss_Hermite_Quadrature_Constants::FillPoints() {
    Eigen::VectorXd points_8(8);
    points_8<< -2.930637420257244019224, -1.981656756695842925855, 
            -1.157193712446780194721, -0.3811869902073221168547,
            0.3811869902073221168547, 1.157193712446780194721, 
            1.981656756695842925855, 2.930637420257244019224;
    _map_points[8]=points_8;
    Eigen::VectorXd points_10(10);
    points_10<< -3.436159118837737603327, -2.532731674232789796409,
            -1.756683649299881773451, -1.036610829789513654178,
            -0.3429013272237046087892, 0.3429013272237046087892,
            1.036610829789513654178, 1.756683649299881773451,
            2.532731674232789796409, 3.436159118837737603327;
    _map_points[10]=points_10;
    Eigen::VectorXd points_12(12);
    points_12<< -3.889724897869781919272, -3.020637025120889771711,
            -2.279507080501059900188, -1.59768263515260479671,
            -0.9477883912401637437046, -0.314240376254359111277,
            0.3142403762543591112766, 0.947788391240163743705,
            1.59768263515260479671, 2.279507080501059900188,
            3.020637025120889771711, 3.889724897869781919272;
    _map_points[12]=points_12;
    Eigen::VectorXd points_14(14);
    points_14<< -4.304448570473631812621, -3.462656933602270550209,
            -2.748470724985402568625, -2.095183258507716815735,
            -1.476682731141140870584, -0.8787137873293994161147,
            -0.2917455106725620784461, 0.2917455106725620784461,
            0.878713787329399416115, 1.476682731141140870584,
            2.095183258507716815735, 2.748470724985402568625,
            3.462656933602270550209, 4.304448570473631812621;
    _map_points[14]=points_14;
    Eigen::VectorXd points_16(16);
    points_16<< -4.688738939305818364688, -3.869447904860122698719,
            -3.176999161979956026814, -2.546202157847481362159,
            -1.951787990916253977435, -1.380258539198880796372,
            -0.8229514491446558925825, -0.2734810461381524521583,
            0.2734810461381524521583, 0.8229514491446558925825,
            1.380258539198880796372, 1.951787990916253977435,
            2.546202157847481362159, 3.176999161979956026814,
            3.869447904860122698719, 4.688738939305818364688;
    _map_points[16]=points_16;
    Eigen::VectorXd points_18(18);
    points_18<< -5.048364008874466768372, -4.248117873568126463023,
            -3.573769068486266079501, -2.961377505531606844779,
            -2.386299089166686000265, -1.835531604261628892254,
            -1.300920858389617365666, -0.7766829192674116613167,
            -0.2582677505190967592581, 0.2582677505190967592581,
            0.7766829192674116613167, 1.300920858389617365666,
            1.835531604261628892254, 2.386299089166686000265,
            2.961377505531606844779, 3.573769068486266079501,
            4.248117873568126463023, 5.048364008874466768372;
    _map_points[18]=points_18;
    Eigen::VectorXd points_20(20);
    points_20<< -5.387480890011232862017, -4.603682449550744273078,
            -3.944764040115625210376, -3.347854567383216326915,
            -2.78880605842813048053, -2.254974002089275523082,
            -1.738537712116586206781, -1.234076215395323007886,
            -0.7374737285453943587056, -0.2453407083009012499038,
            0.2453407083009012499038, 0.7374737285453943587056,
            1.234076215395323007886, 1.738537712116586206781,
            2.254974002089275523082, 2.78880605842813048053,
            3.347854567383216326915, 3.944764040115625210376,
            4.603682449550744273078, 5.387480890011232862017;
    _map_points[20]=points_20;
};

void Gauss_Hermite_Quadrature_Constants::FillWeights() {
    Eigen::VectorXd weights_8(8);
    weights_8<< 1.99604072211367619206e-4, 0.0170779830074134754562,
            0.2078023258148918795433, 0.6611470125582412910304,
            0.6611470125582412910304, 0.2078023258148918795433,
            0.0170779830074134754562, 1.996040722113676192061e-4;
    _map_weights[8]=weights_8;
    Eigen::VectorXd weights_10(10);
    weights_10<< 7.64043285523262062916e-6, 0.001343645746781232692202,
            0.0338743944554810631362, 0.2401386110823146864165,
            0.6108626337353257987836, 0.6108626337353257987836,
            0.2401386110823146864165, 0.03387439445548106313617,
            0.001343645746781232692202, 7.64043285523262062916e-6;
    _map_weights[10]=weights_10;
    Eigen::VectorXd weights_12(12);
    weights_12<< 2.65855168435630160602e-7, 8.5736870435878586546e-5,
            0.00390539058462906185999, 0.05160798561588392999187,
            0.2604923102641611292334, 0.5701352362624795783471,
            0.5701352362624795783471, 0.2604923102641611292334,
            0.05160798561588392999187, 0.00390539058462906185999,
            8.57368704358785865457e-5, 2.65855168435630160602e-7;
    _map_weights[12]=weights_12;
    Eigen::VectorXd weights_14(14);
    weights_14<< 8.62859116812515794532e-9, 4.71648435501891674888e-6,
            3.55092613551923610484e-4, 0.00785005472645794431049,
            0.0685055342234652055387, 0.2731056090642466033526,
            0.5364059097120901497949, 0.5364059097120901497949,
            0.2731056090642466033526, 0.0685055342234652055387,
            0.00785005472645794431049, 3.55092613551923610484e-4,
            4.71648435501891674888e-6, 8.62859116812515794532e-9;
    _map_weights[14]=weights_14;
    Eigen::VectorXd weights_16(16);
    weights_16<< 2.65480747401118224471e-10, 2.32098084486521065339e-7,
            2.71186009253788151202e-5, 9.32284008624180529914e-4,
            0.0128803115355099736835, 0.0838100413989858294154,
            0.2806474585285336753695, 0.5079294790166137419135,
            0.5079294790166137419135, 0.2806474585285336753695,
            0.0838100413989858294154, 0.01288031153550997368346,
            9.322840086241805299143e-4, 2.71186009253788151202e-5,
            2.32098084486521065339e-7, 2.65480747401118224471e-10;
    _map_weights[16]=weights_16;
    Eigen::VectorXd weights_18(18);
    weights_18<< 7.82819977211589102925e-12, 1.04672057957920824444e-8,
            1.8106544810934304096e-6, 9.1811268679294035291e-5,
            0.001888522630268417894382, 0.01864004238754465192193,
            0.0973017476413154293309, 0.2848072856699795785956,
            0.483495694725455552876, 0.4834956947254555528764,
            0.284807285669979578596, 0.0973017476413154293309,
            0.01864004238754465192193, 0.00188852263026841789438,
            9.18112686792940352915e-5, 1.810654481093430409597e-6,
            1.046720579579208244436e-8, 7.8281997721158910293e-12;
    _map_weights[18]=weights_18;
    Eigen::VectorXd weights_20(20);
    weights_20<< 2.22939364553415129252e-13, 4.3993409922731805536e-10,
            1.086069370769281694e-7, 7.80255647853206369415e-6,
            2.28338636016353967257e-4, 0.00324377334223786183218,
            0.0248105208874636108822, 0.1090172060200233200138,
            0.2866755053628341297197, 0.46224366960061008965,
            0.46224366960061008965, 0.28667550536283412972,
            0.1090172060200233200138, 0.0248105208874636108822,
            0.00324377334223786183218, 2.283386360163539672572e-4,
            7.8025564785320636941e-6, 1.086069370769281694e-7,
            4.39934099227318055363e-10, 2.22939364553415129252e-13;
    _map_weights[20]=weights_20;
};
}}