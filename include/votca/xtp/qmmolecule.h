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
#ifndef VOTCA_XTP_QMMOLECULE_H
#define VOTCA_XTP_QMMOLECULE_H

#include <votca/xtp/atomcontainer.h>
#include <votca/xtp/qmatom.h>
#include <votca/tools/structureparameters.h>
//#include <votca/tools/unitconverter.h>
#include "units.h"
namespace votca {
namespace xtp {

class QMMolecule : public AtomContainer<QMAtom> {
 public:
  typedef QMAtom bead_t;

  typedef const Units units;
 
  QMMolecule(const tools::StructureParameters & params) : 
    AtomContainer<QMAtom>(
        params.get<std::string>(tools::StructureParameter::AtomContainerType),
        params.get<int>(tools::StructureParameter::AtomContainerId)) {};

  QMMolecule(std::string name, int id) : AtomContainer<QMAtom>(name, id){};

  tools::StructureParameters getParameters() const;
  void LoadFromFile(std::string filename);

  void WriteXYZ(std::string filename, std::string header) const;

  friend std::ostream& operator<<(std::ostream& out,
                                  const QMMolecule& container) {
    out << container.getId() << " " << container.getName() << "\n";
    for (const QMAtom& atom : container) {
      out << atom;
    }
    out << std::endl;
    return out;
  }
};

}  // namespace xtp
}  // namespace votca

#endif /* VOTCA_XTP_QMMOLECULE_H */
