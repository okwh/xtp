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
#include <iostream>
#include <votca/csg/pdbwriter.h>
#include <votca/xtp/checkpoint.h>
#pragma once
#ifndef VOTCA_XTP_REGION_H
#define VOTCA_XTP_REGION_H

/**
 * \brief base class to derive regions from
 *
 *
 *
 */

namespace votca {
namespace xtp {

class Region {

 public:
  Region(int id) : _id(id){};
  virtual ~Region(){};

  virtual void WriteToCpt(CheckpointWriter& w) const = 0;

  virtual void ReadFromCpt(CheckpointReader& r) = 0;

  virtual int size() const = 0;

  virtual std::string identify() const = 0;

  virtual void WritePDB(csg::PDBWriter& writer) const = 0;

  int getId() const { return _id; }

  friend std::ostream& operator<<(std::ostream& out, const Region& region) {
    out << "Id: " << region.getId() << " type: " << region.identify()
        << " size: " << region.size();
    return out;
  }

 protected:
  int _id = -1;
};

}  // namespace xtp
}  // namespace votca

#endif  // VOTCA_XTP_REGION_H