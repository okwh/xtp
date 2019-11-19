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
#ifndef _VOTCA_XTP_TOOLS_COUPLINGH_H
#define _VOTCA_XTP_TOOLS_COUPLINGH_H

#include <stdio.h>

#include <votca/xtp/logger.h>
#include <votca/xtp/qmpackagefactory.h>

namespace votca {
namespace xtp {

class Coupling : public QMTool {
 public:
  Coupling() = default;
  ~Coupling() override = default;

  std::string Identify() final { return "coupling"; }

  void Initialize(tools::Property &options) final;
  bool Run() final;

 private:
  std::string _MOsA, _MOsB, _MOsAB;
  std::string _logA, _logB, _logAB;

  std::string _package;
  tools::Property _package_options;
  tools::Property _dftcoupling_options;

  std::string _output_file;

  Logger _log;
};

}  // namespace xtp
}  // namespace votca

#endif
