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

#pragma once
#ifndef VOTCA_XTP_STATEFILTER_H
#define VOTCA_XTP_STATEFILTER_H

#include <votca/xtp/logger.h>
#include <votca/xtp/orbitals.h>
#include <votca/xtp/populationanalysis.h>
#include <votca/xtp/qmfragment.h>
#include <votca/xtp/qmstate.h>

namespace votca {
namespace xtp {
/**
 *  \brief  Filters from a spectrum of states the state, which fullfills certain
 * criteria
 *
 *
 */

class Statefilter {

 public:
  Statefilter()
      : _use_oscfilter(false),
        _use_overlapfilter(false),
        _use_localisationfilter(false),
        _use_dQfilter(false) {
    ;
  }
  void Initialize(tools::Property& options);
  void setLogger(Logger* log) { _log = log; }
  void setInitialState(const QMState& state) { _statehist.push_back(state); }
  void PrintInfo() const;
  QMState CalcStateAndUpdate(Orbitals& orbitals);
  QMState CalcState(Orbitals& orbitals) const;

 private:
  std::vector<int> OscFilter(const Orbitals& orbitals) const;
  std::vector<int> LocFilter(const Orbitals& orbitals) const;
  std::vector<int> DeltaQFilter(const Orbitals& orbitals) const;
  std::vector<int> OverlapFilter(Orbitals& orbitals) const;

  Eigen::VectorXd CalculateOverlap(Orbitals& orbitals) const;

  void UpdateLastCoeff(Orbitals& orbitals);
  Eigen::MatrixXd CalcOrthoCoeffs(Orbitals& orbitals) const;

  std::vector<int> CollapseResults(
      std::vector<std::vector<int> >& results) const;
  std::vector<int> ComparePairofVectors(std::vector<int>& vec1,
                                        std::vector<int>& vec2) const;

  Logger* _log;

  std::vector<QMState> _statehist;

  Eigen::VectorXd _laststatecoeff;
  mutable Eigen::MatrixXd _S_onehalf;  // only used in one iteration, do not
                                       // want to introduce more function
                                       // arguments

  bool _use_oscfilter;
  double _oscthreshold;

  bool _use_overlapfilter;
  double _overlapthreshold;

  bool _use_localisationfilter;

  mutable std::vector<QMFragment<BSE_Population> >
      _fragment_loc;  // contain value and definition but between iterations it
                      // does not change so mutalbe now
  double _loc_threshold;

  bool _use_dQfilter;
  mutable std::vector<QMFragment<BSE_Population> > _fragment_dQ;  // contain
                                                                  // value and
                                                                  // definition
  double _dQ_threshold;
};
}  // namespace xtp
}  // namespace votca

#endif  // VOTCA_XTP_STATEFILTER_H
