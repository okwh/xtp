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
#ifndef VOTCA_XTP_REGION_H
#define VOTCA_XTP_REGION_H

#include <iostream>
#include <votca/csg/io/pdbwriter.h>
#include <votca/csg/topology.h>
#include <votca/xtp/checkpoint.h>
#include <votca/xtp/logger.h>

/**
 * \brief base class to derive regions from
 *
 *
 *
 */

namespace votca {
namespace xtp {

class QMRegion;
class PolarRegion;
class StaticRegion;

class Region {

 public:
  Region(int id, Logger& log) : _id(id), _log(log){};
  virtual ~Region(){};

  virtual void WriteToCpt(CheckpointWriter& w) const = 0;

  virtual void ReadFromCpt(CheckpointReader& r) = 0;

  virtual void Initialize(const tools::Property& prop) = 0;

  virtual bool Converged() const = 0;

  virtual void Evaluate(std::vector<std::unique_ptr<Region> >& regions) = 0;

  virtual int size() const = 0;

  virtual std::string identify() const = 0;

  virtual void WritePDB(csg::PDBWriter<csg::Topology>& writer) const = 0;

  int getId() const { return _id; }

  friend std::ostream& operator<<(std::ostream& out, const Region& region) {
    out << "Id: " << region.getId() << " type: " << region.identify()
        << " size: " << region.size();
    return out;
  }

 protected:
  void ApplyInfluenceOfOtherRegions(
      std::vector<std::unique_ptr<Region> >& regions);

  virtual void ResetRegion() = 0;
  virtual void InteractwithQMRegion(const QMRegion& region) = 0;
  virtual void InteractwithPolarRegion(const PolarRegion& region) = 0;
  virtual void InteractwithStaticRegion(const StaticRegion& region) = 0;

  int _id = -1;
  Logger& _log;
};

}  // namespace xtp
}  // namespace votca

#endif  // VOTCA_XTP_REGION_H
