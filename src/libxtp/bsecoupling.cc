/*
 *            Copyright 2009-2019 The VOTCA Development Team
 *                       (http://www.votca.org)
 *
 *      Licensed under the Apache License, Version 2.0 (the "License")
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *              http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "votca/xtp/bse.h"
#include "votca/xtp/bse_operator.h"
#include <boost/format.hpp>
#include <votca/tools/constants.h>
#include <votca/xtp/aomatrix.h>
#include <votca/xtp/bsecoupling.h>

namespace votca {
namespace xtp {
using namespace std;
using boost::format;
using namespace tools;

void BSECoupling::Initialize(Property& options) {

  std::string key = Identify();

  string spintype =
      options.ifExistsReturnElseThrowRuntimeError<std::string>(key + ".spin");
  if (spintype == "all") {
    _doSinglets = true;
    _doTriplets = true;
  } else if (spintype == "triplet") {
    _doTriplets = true;
  } else if (spintype == "singlet") {
    _doSinglets = true;
  } else {
    throw std::runtime_error(
        (boost::format(
             "Choice % for type not known. Available singlet,triplet,all") %
         spintype)
            .str());
  }

  _output_perturbation = options.ifExistsReturnElseReturnDefault<bool>(
      key + ".use_perturbation", _output_perturbation);

  _levA =
      options.ifExistsReturnElseReturnDefault(key + ".moleculeA.states", _levA);
  _levB =
      options.ifExistsReturnElseReturnDefault(key + ".moleculeB.states", _levB);
  _occA = options.ifExistsReturnElseReturnDefault(key + ".moleculeA.occLevels",
                                                  _occA);
  _occB = options.ifExistsReturnElseReturnDefault(key + ".moleculeB.occLevels",
                                                  _occB);
  _unoccA = options.ifExistsReturnElseReturnDefault(
      key + ".moleculeA.unoccLevels", _unoccA);
  _unoccB = options.ifExistsReturnElseReturnDefault(
      key + ".moleculeB.unoccLevels", _unoccB);
}

void BSECoupling::WriteToProperty(Property& summary, const QMState& stateA,
                                  const QMState& stateB) const {
  Property& coupling_summary = summary.add("coupling", "");
  double JAB_pert = 0;
  double JAB_diag = 0;
  if (stateA.Type() == QMStateType::Singlet) {
    JAB_pert =
        getSingletCouplingElement(stateA.StateIdx(), stateB.StateIdx(), 0);
    JAB_diag =
        getSingletCouplingElement(stateA.StateIdx(), stateB.StateIdx(), 1);
  } else if (stateA.Type() == QMStateType::Triplet) {
    JAB_pert =
        getTripletCouplingElement(stateA.StateIdx(), stateB.StateIdx(), 0);
    JAB_diag =
        getTripletCouplingElement(stateA.StateIdx(), stateB.StateIdx(), 1);
  }
  coupling_summary.setAttribute("stateA", stateA.ToString());
  coupling_summary.setAttribute("stateB", stateB.ToString());
  coupling_summary.setAttribute("j_pert", (format("%1$1.6e") % JAB_pert).str());
  coupling_summary.setAttribute("j_diag", (format("%1$1.6e") % JAB_diag).str());
}

void BSECoupling::Addoutput(Property& type_summary, const Orbitals&,
                            const Orbitals&) const {
  tools::Property& bsecoupling = type_summary.add(Identify(), "");
  string algorithm = "j_diag";
  if (_output_perturbation) {
    algorithm = "j_pert";
  }
  if (_doSinglets) {
    QMStateType singlet = QMStateType(QMStateType::Singlet);
    Property& singlet_summary = bsecoupling.add(singlet.ToLongString(), "");
    singlet_summary.setAttribute("algorithm", algorithm);
    for (Index stateA = 0; stateA < _levA; ++stateA) {
      QMState qmstateA = QMState(singlet, stateA, false);
      for (Index stateB = 0; stateB < _levB; ++stateB) {
        QMState qmstateB = QMState(singlet, stateB, false);
        WriteToProperty(singlet_summary, qmstateA, qmstateB);
      }
    }
  }

  if (_doTriplets) {
    QMStateType triplet = QMStateType(QMStateType::Triplet);
    Property& triplet_summary = bsecoupling.add(triplet.ToLongString(), "");
    triplet_summary.setAttribute("algorithm", algorithm);
    for (Index stateA = 0; stateA < _levA; ++stateA) {
      QMState qmstateA = QMState(triplet, stateA, false);
      for (Index stateB = 0; stateB < _levB; ++stateB) {
        QMState qmstateB = QMState(triplet, stateB, false);
        WriteToProperty(triplet_summary, qmstateA, qmstateB);
      }
    }
  }
}

double BSECoupling::getSingletCouplingElement(Index levelA, Index levelB,
                                              Index methodindex) const {
  return JAB_singlet[methodindex](levelA, levelB + _levA) *
         votca::tools::conv::hrt2ev;
}

double BSECoupling::getTripletCouplingElement(Index levelA, Index levelB,
                                              Index methodindex) const {
  return JAB_triplet[methodindex](levelA, levelB + _levA) *
         votca::tools::conv::hrt2ev;
}

Eigen::MatrixXd BSECoupling::SetupCTStates(Index bseA_vtotal, Index bseB_vtotal,
                                           Index bseAB_vtotal,
                                           Index bseAB_ctotal,
                                           const Eigen::MatrixXd& A_AB,
                                           const Eigen::MatrixXd& B_AB) const {

  Index noAB = _occA * _unoccB;
  Index noBA = _unoccA * _occB;
  Index bseAB_total = bseAB_vtotal + bseAB_ctotal;
  Index bseAB_size = bseAB_vtotal * bseAB_ctotal;
  Eigen::MatrixXd CTstates = Eigen::MatrixXd::Zero(bseAB_size, noAB + noBA);

  auto A_occ = A_AB.block(0, bseA_vtotal - _occA, bseAB_total, _occA);
  auto A_unocc = A_AB.block(0, bseA_vtotal, bseAB_total, _unoccA);
  auto B_occ = B_AB.block(0, bseB_vtotal - _occB, bseAB_total, _occB);
  auto B_unocc = B_AB.block(0, bseB_vtotal, bseAB_total, _unoccB);

  const Eigen::MatrixXd A_occ_occ = A_occ.topRows(bseAB_vtotal);
  const Eigen::MatrixXd B_unocc_unocc = B_unocc.bottomRows(bseAB_ctotal);

  // notation AB is CT states with A+B-, BA is the counterpart
  // Setting up CT-states:
  XTP_LOG(Log::error, *_pLog)
      << TimeStamp() << "   Setting up CT-states" << flush;

  // Number of A+B- states
#pragma omp parallel for
  for (Index a_occ = 0; a_occ < _occA; a_occ++) {
    for (Index b_unocc = 0; b_unocc < _unoccB; b_unocc++) {
      Index index = a_occ * _unoccB + b_unocc;
      Eigen::MatrixXd Coeff =
          B_unocc_unocc.col(b_unocc) * A_occ_occ.col(a_occ).transpose();
      CTstates.col(index) =
          Eigen::Map<Eigen::VectorXd>(Coeff.data(), bseAB_size);
    }
  }
  XTP_LOG(Log::error, *_pLog)
      << TimeStamp() << "  " << noBA << " CT states A+B- created" << flush;

  const Eigen::MatrixXd A_unocc_unocc = A_unocc.bottomRows(bseAB_ctotal);
  const Eigen::MatrixXd B_occ_occ = B_occ.topRows(bseAB_vtotal);

#pragma omp parallel for
  for (Index b_occ = 0; b_occ < _occB; b_occ++) {
    for (Index a_unocc = 0; a_unocc < _unoccA; a_unocc++) {
      Index index = b_occ * _unoccA + a_unocc + noAB;
      Eigen::MatrixXd Coeff =
          A_unocc_unocc.col(a_unocc) * B_occ_occ.col(b_occ).transpose();
      CTstates.col(index) =
          Eigen::Map<Eigen::VectorXd>(Coeff.data(), bseAB_size);
    }
  }
  XTP_LOG(Log::error, *_pLog)
      << TimeStamp() << "  " << noBA << " CT states A-B+ created" << flush;
  return CTstates;
}

Eigen::MatrixXd BSECoupling::ProjectFrenkelExcitons(
    const Eigen::MatrixXd& BSE_Coeffs, const Eigen::MatrixXd& X_AB,
    Index bseX_vtotal, Index bseX_ctotal, Index bseAB_vtotal,
    Index bseAB_ctotal) const {
  Index bseAB_size = bseAB_vtotal * bseAB_ctotal;
  auto X_occ = X_AB.leftCols(bseX_vtotal);
  auto X_unocc = X_AB.rightCols(bseX_ctotal);
  const Eigen::MatrixXd X_occ_occ = X_occ.topRows(bseAB_vtotal);
  const Eigen::MatrixXd X_unocc_unocc = X_unocc.bottomRows(bseAB_ctotal);
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(bseAB_size, BSE_Coeffs.cols());
  // no pragma here because often we will only have one Coeff
  for (Index i = 0; i < BSE_Coeffs.cols(); i++) {
    Eigen::VectorXd coeff = BSE_Coeffs.col(i);
    Eigen::Map<Eigen::MatrixXd> coeffmatrix =
        Eigen::Map<Eigen::MatrixXd>(coeff.data(), bseX_ctotal, bseX_vtotal);
    Eigen::MatrixXd proj = X_unocc_unocc * coeffmatrix * X_occ_occ.transpose();
    result.col(i) = Eigen::Map<Eigen::VectorXd>(proj.data(), proj.size());
  }
  return result;
}

int GetSign(double value) {
  if (value < 0) {
    return -1;
  } else if (value > 0) {
    return 1;
  }
  return 0;
}

void BSECoupling::CalculateCouplings(const Orbitals& orbitalsA,
                                     const Orbitals& orbitalsB,
                                     const Orbitals& orbitalsAB) {
  XTP_LOG(Log::error, *_pLog)
      << TimeStamp() << "  Calculating exciton couplings" << flush;
  // set the parallelization
  XTP_LOG(Log::error, *_pLog) << TimeStamp() << " Using "
                              << OPENMP::getMaxThreads() << " threads" << flush;

  CheckAtomCoordinates(orbitalsA, orbitalsB, orbitalsAB);

  // constructing the direct product orbA x orbB
  Index basisB = orbitalsB.getBasisSetSize();
  Index basisA = orbitalsA.getBasisSetSize();
  if ((basisA == 0) || (basisB == 0)) {
    throw std::runtime_error("Basis set size is not stored in monomers");
  }

  // get exciton information of molecule A
  Index bseA_cmax = orbitalsA.getBSEcmax();
  Index bseA_cmin = orbitalsA.getBSEcmin();
  Index bseA_vmax = orbitalsA.getBSEvmax();
  Index bseA_vmin = orbitalsA.getBSEvmin();
  Index bseA_vtotal = bseA_vmax - bseA_vmin + 1;
  Index bseA_ctotal = bseA_cmax - bseA_cmin + 1;
  Index bseA_total = bseA_vtotal + bseA_ctotal;
  Index bseA_size = bseA_vtotal * bseA_ctotal;
  Index bseA_singlet_exc = orbitalsA.BSESinglets().eigenvectors().cols();
  Index bseA_triplet_exc = orbitalsA.BSETriplets().eigenvectors().cols();

  XTP_LOG(Log::error, *_pLog)
      << TimeStamp() << "   molecule A has " << bseA_singlet_exc
      << " singlet excitons with dimension " << bseA_size << flush;
  XTP_LOG(Log::error, *_pLog)
      << TimeStamp() << "   molecule A has " << bseA_triplet_exc
      << " triplet excitons with dimension " << bseA_size << flush;

  // get exciton information of molecule B
  Index bseB_cmax = orbitalsB.getBSEcmax();
  Index bseB_cmin = orbitalsB.getBSEcmin();
  Index bseB_vmax = orbitalsB.getBSEvmax();
  Index bseB_vmin = orbitalsB.getBSEvmin();
  Index bseB_vtotal = bseB_vmax - bseB_vmin + 1;
  Index bseB_ctotal = bseB_cmax - bseB_cmin + 1;
  Index bseB_total = bseB_vtotal + bseB_ctotal;
  Index bseB_size = bseB_vtotal * bseB_ctotal;
  Index bseB_singlet_exc = orbitalsB.BSESinglets().eigenvectors().cols();
  Index bseB_triplet_exc = orbitalsB.BSETriplets().eigenvectors().cols();

  XTP_LOG(Log::error, *_pLog)
      << TimeStamp() << "   molecule B has " << bseB_singlet_exc
      << " singlet excitons with dimension " << bseB_size << flush;
  XTP_LOG(Log::error, *_pLog)
      << TimeStamp() << "   molecule B has " << bseB_triplet_exc
      << " triplet excitons with dimension " << bseB_size << flush;

  if (_levA > bseA_singlet_exc) {
    XTP_LOG(Log::error, *_pLog)
        << TimeStamp()
        << "  Number of excitons you want is greater than stored for molecule "
           "A. Setting to max number available"
        << flush;
    _levA = bseA_singlet_exc;
  }
  if (_levB > bseB_singlet_exc) {
    XTP_LOG(Log::error, *_pLog)
        << TimeStamp()
        << "  Number of excitons you want is greater than stored for molecule "
           "B. Setting to max number available"
        << flush;
    _levB = bseB_singlet_exc;
  }

  if (_levA > bseA_singlet_exc) {
    XTP_LOG(Log::error, *_pLog)
        << TimeStamp()
        << "  Number of Frenkel states you want is greater than stored for "
           "molecule A. Setting to max number available"
        << flush;
    _levA = bseA_singlet_exc;
  }
  if (_levB > bseB_singlet_exc) {
    XTP_LOG(Log::error, *_pLog)
        << TimeStamp()
        << "  Number of Frenkel states you want is greater than stored for "
           "molecule B. Setting to max number available"
        << flush;
    _levB = bseB_singlet_exc;
  }

  if (_unoccA > bseA_ctotal) {
    XTP_LOG(Log::error, *_pLog)
        << TimeStamp()
        << "  Number of occupied orbitals in molecule A for CT creation "
           "exceeds number of KS-orbitals in BSE"
        << flush;
    _unoccA = bseA_ctotal;
  } else if (_unoccA < 0) {
    _unoccA = bseA_ctotal;
    XTP_LOG(Log::error, *_pLog)
        << TimeStamp()
        << "  Number of occupied orbitals in molecule B for CT creation "
           "exceeds number of KS-orbitals in BSE"
        << flush;
  }
  if (_unoccB > bseB_ctotal) {
    XTP_LOG(Log::error, *_pLog)
        << TimeStamp()
        << "  Number of occupied orbitals in molecule B for CT creation "
           "exceeds number of KS-orbitals in BSE"
        << flush;
    _unoccB = bseB_ctotal;
  } else if (_unoccB < 0) {
    _unoccB = bseB_ctotal;
  }

  if (_occA > bseA_vtotal) {
    XTP_LOG(Log::error, *_pLog)
        << TimeStamp()
        << "  Number of unoccupied orbitals in molecule A for CT creation "
           "exceeds number of KS-orbitals in BSE"
        << flush;
    _occA = bseA_vtotal;
  } else if (_occA < 0) {
    _occA = bseA_vtotal;
  }
  if (_occB > bseB_vtotal) {
    XTP_LOG(Log::error, *_pLog)
        << TimeStamp()
        << "  Number of unoccupied orbitals in molecule B for CT creation "
           "exceeds number of KS-orbitals in BSE"
        << flush;
    _occB = bseB_vtotal;
  } else if (_occB < 0) {
    _occB = bseB_vtotal;
  }

  // get exciton information of pair AB
  Index bseAB_cmax = orbitalsAB.getBSEcmax();
  Index bseAB_cmin = orbitalsAB.getBSEcmin();
  Index bseAB_vmax = orbitalsAB.getBSEvmax();
  Index bseAB_vmin = orbitalsAB.getBSEvmin();
  Index basisAB = orbitalsAB.getBasisSetSize();
  Index bseAB_vtotal = bseAB_vmax - bseAB_vmin + 1;
  Index bseAB_ctotal = bseAB_cmax - bseAB_cmin + 1;
  Index bseAB_total = bseAB_vtotal + bseAB_ctotal;
  Index bseAB_size = bseAB_vtotal * bseAB_ctotal;

  // DFT levels of monomers can be reduced to those used in BSE
  XTP_LOG(Log::error, *_pLog)
      << TimeStamp() << "   levels used for BSE of molA: " << bseA_vmin
      << " to " << bseA_cmax << " total: " << bseA_total << flush;
  XTP_LOG(Log::error, *_pLog)
      << TimeStamp() << "   levels used for BSE of molB: " << bseB_vmin
      << " to " << bseB_cmax << " total: " << bseB_total << flush;
  XTP_LOG(Log::error, *_pLog)
      << TimeStamp() << "   levels used for BSE of dimer AB: " << bseAB_vmin
      << " to " << bseAB_cmax << " total: " << bseAB_total << flush;

  Eigen::MatrixXd MOsA =
      orbitalsA.MOs().eigenvectors().block(0, bseA_vmin, basisA, bseA_total);
  Eigen::MatrixXd MOsB =
      orbitalsB.MOs().eigenvectors().block(0, bseB_vmin, basisB, bseB_total);
  Eigen::MatrixXd MOsAB = orbitalsAB.MOs().eigenvectors().block(
      0, bseAB_vmin, basisAB, bseAB_total);

  XTP_LOG(Log::info, *_pLog)
      << TimeStamp() << " Calculating overlap matrix for basisset: "
      << orbitalsAB.getDFTbasisName() << flush;

  Eigen::MatrixXd overlap = CalculateOverlapMatrix(orbitalsAB) * MOsAB;
  XTP_LOG(Log::info, *_pLog)
      << TimeStamp() << " Projecting monomers onto dimer orbitals" << flush;

  Eigen::MatrixXd A_AB = overlap.topRows(basisA).transpose() * MOsA;
  Eigen::MatrixXd B_AB = overlap.bottomRows(basisB).transpose() * MOsB;
  Eigen::VectorXd mag_A = A_AB.colwise().squaredNorm();
  if (mag_A.any() < 0.95) {
    XTP_LOG(Log::error, *_pLog)
        << "\nWarning: "
        << "Projection of orbitals of monomer A on dimer is insufficient,mag="
        << mag_A.minCoeff() << flush;
  }
  Eigen::VectorXd mag_B = B_AB.colwise().squaredNorm();
  if (mag_B.any() < 0.95) {
    XTP_LOG(Log::error, *_pLog)
        << "\nWarning: "
        << "Projection of orbitals of monomer B on dimer is insufficient,mag="
        << mag_B.minCoeff() << flush;
  }

  AOBasis dftbasis = orbitalsAB.SetupDftBasis();
  AOBasis auxbasis = orbitalsAB.SetupAuxBasis();
  TCMatrix_gwbse Mmn(*_pLog);
  // rpamin here, because RPA needs till rpamin
  Mmn.Initialize(auxbasis.AOBasisSize(), orbitalsAB.getRPAmin(),
                 orbitalsAB.getGWAmax(), orbitalsAB.getRPAmin(),
                 orbitalsAB.getRPAmax());
  Mmn.Fill(auxbasis, dftbasis, orbitalsAB.MOs().eigenvectors());

  const Eigen::MatrixXd& qpcoeff = orbitalsAB.QPdiag().eigenvectors();
  Eigen::MatrixXd Hqp = qpcoeff *
                        orbitalsAB.QPdiag().eigenvalues().asDiagonal() *
                        qpcoeff.transpose();
  BSE::options opt;
  opt.cmax = orbitalsAB.getBSEcmax();
  opt.homo = orbitalsAB.getHomo();
  opt.qpmin = orbitalsAB.getGWAmin();
  opt.rpamax = orbitalsAB.getRPAmax();
  opt.rpamin = orbitalsAB.getRPAmin();
  opt.useTDA = true;
  opt.vmin = orbitalsAB.getBSEvmin();
  BSE bse(*_pLog, Mmn, Hqp);
  bse.configure(opt, orbitalsAB.MOs().eigenvalues());
  XTP_LOG(Log::error, *_pLog) << TimeStamp() << " Setup BSE operator" << flush;

  // now the different spin types
  if (_doSinglets) {
    XTP_LOG(Log::error, *_pLog)
        << TimeStamp() << "   Evaluating singlets" << flush;
    XTP_LOG(Log::error, *_pLog)
        << TimeStamp() << "   Setup Hamiltonian" << flush;
    Eigen::MatrixXd FE_AB = Eigen::MatrixXd::Zero(bseAB_size, _levA + _levB);
    const Eigen::MatrixXd bseA =
        orbitalsA.BSESinglets().eigenvectors().leftCols(_levA);
    FE_AB.leftCols(_levA) = ProjectFrenkelExcitons(
        bseA, A_AB, bseA_vtotal, bseA_ctotal, bseAB_vtotal, bseAB_ctotal);
    const Eigen::MatrixXd bseB =
        orbitalsB.BSESinglets().eigenvectors().leftCols(_levB);
    FE_AB.rightCols(_levB) = ProjectFrenkelExcitons(
        bseB, B_AB, bseB_vtotal, bseB_ctotal, bseAB_vtotal, bseAB_ctotal);
    Eigen::MatrixXd CTStates = SetupCTStates(
        bseA_vtotal, bseB_vtotal, bseAB_vtotal, bseAB_ctotal, A_AB, B_AB);
    JAB_singlet =
        ProjectExcitons(FE_AB, CTStates, bse.getSingletOperator_TDA());
    XTP_LOG(Log::error, *_pLog)
        << TimeStamp() << "   calculated singlet couplings " << flush;
  }

  if (_doTriplets) {
    XTP_LOG(Log::error, *_pLog)
        << TimeStamp() << "   Evaluating triplets" << flush;
    Eigen::MatrixXd FE_AB = Eigen::MatrixXd::Zero(bseAB_size, _levA + _levB);
    const Eigen::MatrixXd bseA =
        orbitalsA.BSETriplets().eigenvectors().leftCols(_levA);
    FE_AB.leftCols(_levA) = ProjectFrenkelExcitons(
        bseA, A_AB, bseA_vtotal, bseA_ctotal, bseAB_vtotal, bseAB_ctotal);
    const Eigen::MatrixXd bseB =
        orbitalsB.BSETriplets().eigenvectors().leftCols(_levB);
    FE_AB.rightCols(_levB) = ProjectFrenkelExcitons(
        bseB, B_AB, bseB_vtotal, bseB_ctotal, bseAB_vtotal, bseAB_ctotal);
    Eigen::MatrixXd CTStates = SetupCTStates(
        bseA_vtotal, bseB_vtotal, bseAB_vtotal, bseAB_ctotal, A_AB, B_AB);
    JAB_triplet =
        ProjectExcitons(FE_AB, CTStates, bse.getTripletOperator_TDA());
    XTP_LOG(Log::error, *_pLog)
        << TimeStamp() << "   calculated triplet couplings " << flush;
  }

  XTP_LOG(Log::error, *_pLog)
      << TimeStamp() << "  Done with exciton couplings" << flush;
  return;
}

Eigen::MatrixXd BSECoupling::OrthogonalizeCTs(Eigen::MatrixXd& FE_AB,
                                              Eigen::MatrixXd& CTStates) const {
  Index ct = CTStates.cols();

  if (ct > 0) {
    XTP_LOG(Log::error, *_pLog)
        << TimeStamp() << " Orthogonalizing CT-states with respect to FE-states"
        << flush;
    Eigen::MatrixXd correction = FE_AB * (FE_AB.transpose() * CTStates);
    CTStates -= correction;

    // normalize
    Eigen::VectorXd norm = CTStates.colwise().norm();
    for (Index i = 0; i < CTStates.cols(); i++) {
      CTStates.col(i) /= norm(i);
    }
    Index minstateindex = 0;
    double minnorm = norm.minCoeff(&minstateindex);
    if (minnorm < 0.95) {
      XTP_LOG(Log::error, *_pLog)
          << TimeStamp() << " WARNING: CT-state " << minstateindex
          << " norm is only " << minnorm << flush;
    }
  }
  Index bse_exc = _levA + _levB;

  Index bseAB_size = CTStates.rows();
  Eigen::MatrixXd projection(bseAB_size, bse_exc + ct);
  XTP_LOG(Log::info, *_pLog)
      << TimeStamp() << " merging projections into one vector  " << flush;
  projection.leftCols(bse_exc) = FE_AB;
  FE_AB.resize(0, 0);
  if (ct > 0) {
    projection.rightCols(ct) = CTStates;
  }
  CTStates.resize(0, 0);
  return projection;
}

template <class BSE_OPERATOR>
Eigen::MatrixXd BSECoupling::CalcJ_dimer(BSE_OPERATOR& H,
                                         Eigen::MatrixXd& projection) const {

  XTP_LOG(Log::info, *_pLog)
      << TimeStamp() << "   Setting up coupling matrix size "
      << projection.cols() << flush;
  // matrix _J
  //  E_A         J_AB        J_A_ABCT        J_A_BACT
  //  J_BA        E_B         J_B_ABCT        J_B_BACT
  //  J_ABCT_A    J_ABCT_B    E_ABCT          J_ABCT_BACT
  //  J_BACT_A   J_BACT_B    J_BACT_ABCT     E_BACT

  // this only works for hermitian/symmetric H so only in TDA

  Eigen::MatrixXd temp = H * projection;
  Eigen::MatrixXd J_dimer = projection.transpose() * temp;

  XTP_LOG(Log::info, *_pLog)
      << TimeStamp() << "   Setting up overlap matrix size "
      << projection.cols() << flush;
  Eigen::MatrixXd S_dimer = projection.transpose() * projection;

  projection.resize(0, 0);
  if (projection.cols()) {
    XTP_LOG(Log::debug, *_pLog)
        << "---------------------------------------" << flush;
    XTP_LOG(Log::debug, *_pLog) << "_J_dimer[Ryd]" << flush;

    XTP_LOG(Log::debug, *_pLog) << J_dimer << flush;
    XTP_LOG(Log::debug, *_pLog) << "_S_dimer" << flush;

    XTP_LOG(Log::debug, *_pLog) << S_dimer << flush;
    XTP_LOG(Log::debug, *_pLog)
        << "---------------------------------------" << flush;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S_dimer);
  Eigen::MatrixXd Sm1 = es.operatorInverseSqrt();
  Eigen::MatrixXd J_ortho = Sm1 * J_dimer * Sm1;

  if (projection.cols()) {
    XTP_LOG(Log::debug, *_pLog)
        << "---------------------------------------" << flush;
    XTP_LOG(Log::debug, *_pLog) << "_J_ortho[Ryd]" << flush;
    XTP_LOG(Log::debug, *_pLog) << J_ortho << flush;
    XTP_LOG(Log::debug, *_pLog) << "_S-1/2" << flush;
    XTP_LOG(Log::debug, *_pLog) << Sm1 << flush;
    XTP_LOG(Log::debug, *_pLog)
        << "---------------------------------------" << flush;
  }
  XTP_LOG(Log::debug, *_pLog)
      << TimeStamp() << "   Smallest value of dimer overlapmatrix is "
      << es.eigenvalues()(0) << flush;
  return J_ortho;
}

template <class BSE_OPERATOR>
std::array<Eigen::MatrixXd, 2> BSECoupling::ProjectExcitons(
    Eigen::MatrixXd& FE_AB, Eigen::MatrixXd& CTStates, BSE_OPERATOR H) const {

  Eigen::MatrixXd projection = OrthogonalizeCTs(FE_AB, CTStates);
  Eigen::MatrixXd J_ortho = CalcJ_dimer(H, projection);

  std::array<Eigen::MatrixXd, 2> J;

  XTP_LOG(Log::info, *_pLog)
      << TimeStamp() << "   Running Perturbation algorithm" << flush;
  J[0] = Perturbation(J_ortho);
  XTP_LOG(Log::info, *_pLog)
      << TimeStamp() << "    Running Projection algorithm" << flush;
  J[1] = Fulldiag(J_ortho);

  XTP_LOG(Log::debug, *_pLog)
      << "---------------------------------------" << flush;
  XTP_LOG(Log::debug, *_pLog) << "Jeff_pert[Hrt]" << flush;
  XTP_LOG(Log::debug, *_pLog) << J[0] << flush;
  XTP_LOG(Log::debug, *_pLog) << "Jeff_diag[Hrt]" << flush;
  XTP_LOG(Log::debug, *_pLog) << J[1] << flush;
  XTP_LOG(Log::debug, *_pLog)
      << "---------------------------------------" << flush;

  return J;
}

Eigen::MatrixXd BSECoupling::Perturbation(
    const Eigen::MatrixXd& J_dimer) const {
  Index bse_exc = _levA + _levB;
  Index ct = J_dimer.rows() - bse_exc;
  Eigen::MatrixXd J_result = J_dimer;
  if (ct > 0) {
    Eigen::MatrixXd transformation =
        Eigen::MatrixXd::Identity(J_dimer.rows(), J_dimer.cols());
    Eigen::MatrixXd Ct = J_dimer.bottomRightCorner(ct, ct);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Ct);
    transformation.bottomRightCorner(ct, ct) = es.eigenvectors();
    Ct.resize(0, 0);

    XTP_LOG(Log::debug, *_pLog) << "FE state hamiltonian" << flush;
    XTP_LOG(Log::debug, *_pLog)
        << J_dimer.topLeftCorner(bse_exc, bse_exc) << flush;
    if (ct > 0) {
      XTP_LOG(Log::debug, *_pLog) << "eigenvalues of CT states" << flush;
      XTP_LOG(Log::debug, *_pLog) << es.eigenvalues().transpose() << flush;
    }

    J_result = transformation.transpose() * J_dimer * transformation;
    XTP_LOG(Log::debug, *_pLog)
        << "---------------------------------------" << flush;
    XTP_LOG(Log::debug, *_pLog) << "_J_ortho[Hrt] CT-state diag" << flush;
    XTP_LOG(Log::debug, *_pLog) << J_result << flush;
    XTP_LOG(Log::debug, *_pLog)
        << "---------------------------------------" << flush;
  }

  Eigen::MatrixXd Jmatrix = Eigen::MatrixXd::Zero(bse_exc, bse_exc);
  for (Index stateA = 0; stateA < _levA; stateA++) {
    double Ea = J_result(stateA, stateA);
    for (Index stateB = 0; stateB < _levB; stateB++) {
      Index stateBd = stateB + _levA;
      XTP_LOG(Log::info, *_pLog)
          << TimeStamp() << "   Calculating coupling between exciton A"
          << stateA + 1 << " and exciton B" << stateB + 1 << flush;
      double J = J_result(stateA, stateBd);

      double Eb = J_result(stateBd, stateBd);
      for (Index k = bse_exc; k < (bse_exc + ct); k++) {
        double Eab = J_result(k, k);
        if (std::abs(Eab - Ea) < 0.001) {
          XTP_LOG(Log::error, *_pLog)
              << TimeStamp() << "Energydifference between state A "
              << stateA + 1 << "and CT state " << k + 1 << " is " << Eab - Ea
              << "[Hrt]" << flush;
        }
        if (std::abs(Eab - Eb) < 0.001) {
          XTP_LOG(Log::error, *_pLog)
              << TimeStamp() << "Energydifference between state B "
              << stateB + 1 << "and CT state " << k + 1 << " is " << Eab - Eb
              << "[Hrt]" << flush;
        }
        J += 0.5 * J_result(k, stateA) * J_result(k, stateBd) *
             (1 / (Ea - Eab) + 1 / (Eb - Eab));  // Have no clue why 0.5
      }
      Jmatrix(stateA, stateBd) = J;
      Jmatrix(stateBd, stateA) = J;
    }
  }
  return Jmatrix;
}

Eigen::MatrixXd BSECoupling::Fulldiag(const Eigen::MatrixXd& J_dimer) const {
  Index bse_exc = _levA + _levB;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J_dimer);
  XTP_LOG(Log::debug, *_pLog)
      << "---------------------------------------" << flush;
  XTP_LOG(Log::debug, *_pLog) << "Eigenvectors of J" << flush;
  XTP_LOG(Log::debug, *_pLog) << es.eigenvectors() << flush;
  XTP_LOG(Log::debug, *_pLog) << "J_eigenvalues[Hrt]" << flush;
  XTP_LOG(Log::debug, *_pLog) << es.eigenvalues() << flush;
  XTP_LOG(Log::debug, *_pLog)
      << "---------------------------------------" << flush;
  Eigen::MatrixXd Jmat = Eigen::MatrixXd::Zero(bse_exc, bse_exc);
  // Calculate projection on subspace for every pair of excitons separately
  for (Index stateA = 0; stateA < _levA; stateA++) {
    for (Index stateB = 0; stateB < _levB; stateB++) {

      Index stateBd = stateB + _levA;
      XTP_LOG(Log::info, *_pLog)
          << TimeStamp() << "   Calculating coupling between exciton A"
          << stateA + 1 << " and exciton B" << stateB + 1 << flush;

      std::array<int, 2> indexes;
      std::array<int, 2> signs;

      // Find the eigenstate state, which in an L2 is closed to A or B
      // respectively
      es.eigenvectors().row(stateA).cwiseAbs().maxCoeff(&indexes[0]);
      es.eigenvectors().row(stateBd).cwiseAbs().maxCoeff(&indexes[1]);
      if (indexes[0] == indexes[1]) {
        Eigen::RowVectorXd stateamplitudes =
            es.eigenvectors().row(stateBd).cwiseAbs();
        stateamplitudes[indexes[1]] = 0.0;
        stateamplitudes.maxCoeff(&indexes[1]);
      }

      signs[0] = GetSign(es.eigenvectors()(stateA, indexes[0]));
      signs[1] = GetSign(es.eigenvectors()(stateBd, indexes[1]));

      XTP_LOG(Log::info, *_pLog)
          << TimeStamp() << "   Order is: [Initial state n->nth eigenvalue]"
          << flush;
      XTP_LOG(Log::info, *_pLog) << "    A" << stateA + 1 << ":" << stateA + 1
                                 << "->" << indexes[0] + 1 << " ";
      XTP_LOG(Log::info, *_pLog) << "    B" << stateB + 1 << ":" << stateBd + 1
                                 << "->" << indexes[1] + 1 << " " << flush;

      // setting up transformation matrix Tmat and diagonal matrix Emat for the
      // eigenvalues;
      Eigen::Matrix2d Emat = Eigen::Matrix2d::Zero();
      Eigen::Matrix2d Tmat = Eigen::Matrix2d::Zero();
      // find the eigenvectors which are most similar to the initial states
      // row
      for (Index i = 0; i < 2; i++) {
        Index k = indexes[i];
        double sign = signs[i];
        Tmat(0, i) = sign * es.eigenvectors()(stateA, k);
        Tmat(1, i) = sign * es.eigenvectors()(stateBd, k);
        Emat(i, i) = es.eigenvalues()(k);
      }
      Tmat.colwise().normalize();

      if (Tmat.determinant() < 0) {
        XTP_LOG(Log::info, *_pLog)
            << " Reduced state matrix is not in a right handed basis, "
               "multiplying second eigenvector by -1 "
            << flush;
        Tmat.col(1) *= -1;
      }

      XTP_LOG(Log::debug, *_pLog)
          << "---------------------------------------" << flush;
      XTP_LOG(Log::debug, *_pLog) << "_T" << flush;
      XTP_LOG(Log::debug, *_pLog) << Tmat << flush;

      Eigen::Matrix2d S_small = Tmat * Tmat.transpose();

      XTP_LOG(Log::debug, *_pLog) << "S_small" << flush;
      XTP_LOG(Log::debug, *_pLog) << S_small << flush;
      // orthogonalize that matrix

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> ss(S_small);
      Eigen::Matrix2d sm1 = ss.operatorInverseSqrt();
      Emat = sm1 * Emat * sm1;

      XTP_LOG(Log::info, *_pLog)
          << TimeStamp() << "   Smallest value of dimer overlapmatrix is "
          << ss.eigenvalues()(0) << flush;

      XTP_LOG(Log::debug, *_pLog) << "S-1/2" << flush;
      XTP_LOG(Log::debug, *_pLog) << sm1 << flush;
      XTP_LOG(Log::debug, *_pLog) << "E_ortho" << flush;
      XTP_LOG(Log::debug, *_pLog) << Emat << flush;

      Tmat = Tmat * sm1;

      XTP_LOG(Log::debug, *_pLog) << "T_ortho" << flush;
      XTP_LOG(Log::debug, *_pLog) << Tmat << flush;
      XTP_LOG(Log::debug, *_pLog)
          << "---------------------------------------" << flush;

      Eigen::Matrix2d J_small = Tmat * Emat * Tmat.transpose();
      XTP_LOG(Log::debug, *_pLog) << "T_ortho*E_ortho*T_ortho^T" << flush;
      XTP_LOG(Log::debug, *_pLog) << J_small << flush;

      Jmat(stateA, stateBd) = J_small(0, 1);
      Jmat(stateBd, stateA) = J_small(1, 0);
    }
  }

  return Jmat;
}

}  // namespace xtp
}  // namespace votca
