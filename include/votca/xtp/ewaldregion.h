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
#ifndef VOTCA_XTP_EWALDREGION_H
#define VOTCA_XTP_EWALDREGION_H

#include <votca/xtp/classicalsegment.h>
#include <votca/xtp/region.h>
/**
 * \brief defines an ewald region
 *
 *
 *
 */

namespace votca {
namespace xtp {
class QMRegion;
class PolarRegion;
class StaticRegion;
class EwaldRegion : public Region {

 public:
  EwaldRegion(Index id, Logger& log,
              const std::unique_ptr<csg::BoundaryCondition>& bc)
      : Region(id, log), _bc(*(bc.get())){};
  ~EwaldRegion() final = default;

  void Initialize(const tools::Property& prop) final;

  bool Converged() const final { return true; };

  void Evaluate(std::vector<std::unique_ptr<Region> >& regions) final;

  void WriteToCpt(CheckpointWriter& w) const final;

  void ReadFromCpt(CheckpointReader& r) final;

  Index size() const final { return _size; }

  void WritePDB(csg::PDBWriter& writer) const final;

  std::string identify() const final { return "ewald"; }

  void push_back(const StaticSegment& mol);

  void Reset() final;

  double charge() const final;
  double Etotal() const final { return 0.0; }

 protected:
  void AppendResult(tools::Property& prop) const final;
  double InteractwithQMRegion(const QMRegion& region) final;
  double InteractwithPolarRegion(const PolarRegion& region) final;
  double InteractwithStaticRegion(const StaticRegion& region) final;
  double InteractwithEwaldRegion(const EwaldRegion& region) final;

 private:
  Index _size = 0;
  const csg::BoundaryCondition& _bc;
};

}  // namespace xtp
}  // namespace votca

#endif  // VOTCA_XTP_EWALDREGION_H
