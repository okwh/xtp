/*
 *            Copyright 2016 The MUSCET Development Team
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

#include <votca/csg/io/pdbwriter.h>
#include <votca/csg/io/xyzreader.h>
#include <votca/csg/io/xyzwriter.h>
#include <votca/csg/csgtopology.h>
#include <votca/tools/elements.h>
#include <votca/xtp/checkpointreader.h>
#include <votca/xtp/checkpointwriter.h>
#include <votca/xtp/qmmolecule.h>

using namespace std;
using namespace votca::tools;

namespace votca {
namespace xtp {

void QMMolecule::WriteXYZ(std::string filename, std::string header) const {
  csg::XYZWriter<csg::CSG_Topology> writer;
  writer.Open(filename, false);
  writer.WriteHeader(header,size());
  writer.Write(*this);
  writer.Close();
  return;
}

void QMMolecule::LoadFromFile(std::string filename) {
  csg::XYZReader<csg::CSG_Topology> reader;
  reader.Open(filename);
  reader.ReadFile<QMMolecule>(*this);
  reader.Close();
}
}  // namespace xtp
}  // namespace votca
