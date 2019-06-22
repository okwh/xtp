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
#ifndef VOTCA_XTP_POLARREGION_H
#define VOTCA_XTP_POLARREGION_H

#include <votca/xtp/mmregion.h>

/**
 * \brief defines a polar region and of interacting electrostatic and induction
 * segments
 *
 *
 *
 */

namespace votca {
namespace xtp {
class QMRegion;
class PolarRegion;
class StaticRegion;

class PolarRegion : public MMRegion<PolarSegment> {
 public:
  PolarRegion(int id, Logger& log) : MMRegion<PolarSegment>(id, log){};

  std::string identify() const { return "polarregion"; }

  void Initialize(const tools::Property& prop);

  bool Converged() const;

  void Evaluate(std::vector<std::unique_ptr<Region> >& regions);

  void Reset();

 protected:
  void InteractwithQMRegion(const QMRegion& region);
  void InteractwithPolarRegion(const PolarRegion& region);
  void InteractwithStaticRegion(const StaticRegion& region);

 private:
  void CalcInducedDipoles();
  double StaticInteraction();
  void PolarInteraction_scf();
  double PolarInteraction_energy();

  hist<double> _E_hist;
  double _deltaE = 1e-5;
  double _deltaD = 1e-5;
  int _max_iter = 100;
  double _exp_damp = 0.39;

  int _openmp_threads = 1;
};

}  // namespace xtp
}  // namespace votca

#endif /* VOTCA_XTP_MMREGION_H */
