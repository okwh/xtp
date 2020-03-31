/*
 *            Copyright 2009-2020 The VOTCA Development Team
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
#include "votca/xtp/basisset.h"
#include <votca/tools/property.h>

namespace votca {
namespace xtp {

Index FindLmax(const std::string& type) {

  assert(!type.empty() && "Shelltype must be non empty!");
  const char t = type.back();
  Index lmax = 0;
  if (t == 'S') {
    lmax = 0;
  } else if (t == 'P') {
    lmax = 1;
  } else if (t == 'D') {
    lmax = 2;
  } else if (t == 'F') {
    lmax = 3;
  } else if (t == 'G') {
    lmax = 4;
  } else if (t == 'H') {
    lmax = 5;
  } else if (t == 'I') {
    lmax = 6;
  } else {
    throw std::runtime_error("FindLmax: Shelltype not known");
  }
  return lmax;
}

Index FindLmin(const std::string& type) {
  assert(!type.empty() && "Shelltype must be non empty!");
  Index lmin = std::numeric_limits<Index>::max();
  const char t = type[0];
  if (t == 'S') {
    lmin = 0;
  } else if (t == 'P') {
    lmin = 1;
  } else if (t == 'D') {
    lmin = 2;
  } else if (t == 'F') {
    lmin = 3;
  } else if (t == 'G') {
    lmin = 4;
  } else if (t == 'H') {
    lmin = 5;
  } else if (t == 'I') {
    lmin = 6;
  } else {
    throw std::runtime_error("FindLmin: Shelltype not known");
  }
  return lmin;
}

Index OffsetFuncShell(const std::string& shell_type) {
  Index nbf = std::numeric_limits<Index>::max();
  assert(!shell_type.empty() && "Shelltype must be non empty!");
  const char t = shell_type[0];
  if (t == 'S') {
    nbf = 0;
  } else if (t == 'P') {
    nbf = 1;
  } else if (t == 'D') {
    nbf = 4;
  } else if (t == 'F') {
    nbf = 9;
  } else if (t == 'G') {
    nbf = 16;
  } else if (t == 'H') {
    nbf = 25;
  } else if (t == 'I') {
    nbf = 36;
  } else {
    throw std::runtime_error("OffsetFuncShell: Shelltype not known");
  }
  return nbf;
}

Index NumFuncShell(const std::string& shell_type) {
  Index nbf = 0;
  assert(!shell_type.empty() && "Shelltype must be non empty!");
  // single type shells
  for (const char& t : shell_type) {
    if (t == 'S') {
      nbf += 1;
    } else if (t == 'P') {
      nbf += 3;
    } else if (t == 'D') {
      nbf += 5;
    } else if (t == 'F') {
      nbf += 7;
    } else if (t == 'G') {
      nbf += 9;
    } else if (t == 'H') {
      nbf += 11;
    } else if (t == 'I') {
      nbf += 13;
    } else {
      throw std::runtime_error("FindnumofFunc: Shelltype not known");
    }
  }
  return nbf;
}

std::vector<Index> NumFuncSubShell(const std::string& shell_type) {
  std::vector<Index> subshells;
  // single type shells
  if (shell_type.length() == 1) {
    subshells.push_back(NumFuncShell(shell_type));
    // for combined shells, go over all contributions and add functions
  } else {
    for (Index i = 0; i < Index(shell_type.length()); ++i) {
      std::string local_shell = std::string(shell_type, i, 1);
      subshells.push_back(NumFuncShell(local_shell));
    }
  }
  return subshells;
}

Index NumFuncShell_cartesian(const std::string& shell_type) {
  Index nbf = 0;
  assert(!shell_type.empty() && "Shelltype must be non empty!");
  // single type shells defined here
  for (const char& t : shell_type) {
    if (t == 'S') {
      nbf += 1;
    } else if (t == 'P') {
      nbf += 3;
    } else if (t == 'D') {
      nbf += 6;
    } else if (t == 'F') {
      nbf += 10;
    } else if (t == 'G') {
      nbf += 15;
    } else if (t == 'H') {
      nbf += 21;
    } else if (t == 'I') {
      nbf += 28;
    } else {
      throw std::runtime_error("NumFuncShell_cartesian shell_type not known");
    }
  }
  return nbf;
}

Index OffsetFuncShell_cartesian(const std::string& shell_type) {
  Index nbf = std::numeric_limits<Index>::max();
  assert(!shell_type.empty() && "Shelltype must be non empty!");
  const char t = shell_type[0];
  if (t == 'S') {
    nbf = 0;
  } else if (t == 'P') {
    nbf = 1;
  } else if (t == 'D') {
    nbf = 4;
  } else if (t == 'F') {
    nbf = 10;
  } else if (t == 'G') {
    nbf = 20;
  } else if (t == 'H') {
    nbf = 35;
  } else if (t == 'I') {
    nbf = 56;
  } else {
    throw std::runtime_error("OffsetFuncShell_cartesian shell_type not known");
  }

  return nbf;
}

bool CheckShellType(const std::string& shelltype) {
  if (shelltype.empty()) {
    return false;
  }
  std::vector<char> allowed_shells = {'S', 'P', 'D', 'F', 'G', 'H', 'I'};
  std::vector<char>::iterator it =
      std::find(allowed_shells.begin(), allowed_shells.end(), shelltype[0]);
  if (it == allowed_shells.end()) {
    return false;
  } else {
    Index index = std::distance(allowed_shells.begin(), it);
    for (Index i = 1; i < Index(shelltype.size()); i++) {
      if (index + i > Index(allowed_shells.size()) ||
          shelltype[i] != allowed_shells[index + i]) {
        return false;
      }
    }
  }

  return true;
}

void BasisSet::Load(const std::string& name) {

  _name = name;
  // if name contains .xml, assume a basisset .xml file is located in the
  // working directory
  std::size_t found_xml = name.find(".xml");
  std::string xmlFile;
  if (found_xml != std::string::npos) {
    xmlFile = name;
  } else {
    // get the path to the shared folders with xml files
    char* votca_share = getenv("VOTCASHARE");
    if (votca_share == nullptr) {
      throw std::runtime_error("VOTCASHARE not set, cannot open help files.");
    }
    xmlFile = std::string(getenv("VOTCASHARE")) +
              std::string("/xtp/basis_sets/") + name + std::string(".xml");
  }
  tools::Property basis_property;
  basis_property.LoadFromXML(xmlFile);
  std::vector<tools::Property*> elementProps =
      basis_property.Select("basis.element");

  for (tools::Property* elementProp : elementProps) {
    std::string elementName = elementProp->getAttribute<std::string>("name");
    Element& element = addElement(elementName);
    std::vector<tools::Property*> shellProps = elementProp->Select("shell");
    for (tools::Property* shellProp : shellProps) {
      std::string shellType = shellProp->getAttribute<std::string>("type");
      if (!CheckShellType(shellType)) {
        throw std::runtime_error("Shelltype: '" + shellType +
                                 "' is not a valid shelltype!");
      }
      double shellScale = shellProp->getAttribute<double>("scale");

      Shell& shell = element.addShell(shellType, shellScale);
      std::vector<tools::Property*> constProps = shellProp->Select("constant");
      for (tools::Property* constProp : constProps) {
        double decay = constProp->getAttribute<double>("decay");
        std::vector<double> contraction =
            std::vector<double>(shell.getLmax() + 1, 0.0);
        std::vector<tools::Property*> contrProps =
            constProp->Select("contractions");
        for (tools::Property* contrProp : contrProps) {
          std::string contrType = contrProp->getAttribute<std::string>("type");
          double contrFactor = contrProp->getAttribute<double>("factor");
          if (contrType == "S") {
            contraction[0] = contrFactor;
          } else if (contrType == "P") {
            contraction[1] = contrFactor;
          } else if (contrType == "D") {
            contraction[2] = contrFactor;
          } else if (contrType == "F") {
            contraction[3] = contrFactor;
          } else if (contrType == "G") {
            contraction[4] = contrFactor;
          } else if (contrType == "H") {
            contraction[5] = contrFactor;
          } else if (contrType == "I") {
            contraction[6] = contrFactor;
          } else {
            throw std::runtime_error("LoadBasiset:Contractiontype not known");
          }
        }
        shell.addGaussian(decay, contraction);
      }
    }
  }
  return;
}

// adding an Element to a Basis Set
Element& BasisSet::addElement(std::string elementType) {
  auto e = _elements.insert({elementType, Element(elementType)});
  if (!e.second) {
    throw std::runtime_error("Inserting element into basisset failed!");
  }
  return e.first->second;
}

const Element& BasisSet::getElement(std::string element_type) const {
  std::map<std::string, Element>::const_iterator itm =
      _elements.find(element_type);
  if (itm == _elements.end()) {
    throw std::runtime_error("Basis set " + _name +
                             " does not have element of type " + element_type);
  }
  return itm->second;
}

std::ostream& operator<<(std::ostream& out, const Shell& shell) {

  out << "Type:" << shell.getType() << " Scale:" << shell.getScale()
      << " Func: " << shell.getnumofFunc() << "\n";
  for (const auto& gaussian : shell._gaussians) {
    out << " Gaussian Decay: " << gaussian.decay();
    out << " Contractions:";
    for (const double& contraction : gaussian.Contractions()) {
      out << " " << contraction;
    }
    out << "\n";
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const Element& element) {
  out << "Element:" << element.getType() << "\n";
  for (const auto& shell : element) {
    out << shell;
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const BasisSet& basis) {
  out << "BasisSet:" << basis._name << "\n";
  for (const auto& element : basis) {
    out << element.second;
  }
  out << std::flush;
  return out;
}

GaussianPrimitive& Shell::addGaussian(double decay,
                                      std::vector<double> contraction) {
  _gaussians.push_back(GaussianPrimitive(decay, contraction));
  return _gaussians.back();
}

}  // namespace xtp
}  // namespace votca
