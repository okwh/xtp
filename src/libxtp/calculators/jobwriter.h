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
#ifndef _VOTCA_XTP_JOBWRITER_H
#define _VOTCA_XTP_JOBWRITER_H

#include <votca/xtp/qmcalculator.h>
#include <votca/xtp/topology.h>

namespace votca {
namespace xtp {

class JobWriter : public QMCalculator {

 public:
  typedef void (JobWriter::*WriteFunct)(Topology &);

  std::string Identify() { return "jobwriter"; }
  void Initialize(tools::Property &options);
  bool EvaluateFrame(Topology &top);

  // NEED TO REGISTER ALL WRITE MEMBERS IN ::Initialize
  void mps_dimer(Topology &top);
  void mps_monomer(Topology &top);

 private:
  std::vector<std::string> _keys;
  tools::Property *_options;
  std::map<std::string, WriteFunct> _key_funct;
};

}  // namespace xtp
}  // namespace votca

#endif
