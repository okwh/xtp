/*
 *            Copyright 2009-2018 The VOTCA Development Team
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
/// For an earlier history see ctp repo commit 77795ea591b29e664153f9404c8655ba28dc14e9

#ifndef __VOTCA_XTP_JOBCALCULATOR_H
#define __VOTCA_XTP_JOBCALCULATOR_H


#include <votca/xtp/qmcalculator.h>
#include <votca/xtp/topology.h>
#include <votca/xtp/progressobserver.h>

namespace votca { namespace xtp {

class JobCalculator : public QMCalculator
{
public:

                    JobCalculator() {}
    virtual        ~JobCalculator() {}

    virtual std::string  Identify() { return "Generic Job calculator"; }

    virtual bool    EvaluateFrame(XTP::Topology *top) { return true; }
    virtual void    EndEvaluate(XTP::Topology *top) { }

    virtual void    WriteJobFile(XTP::Topology *top)  { ; }
    virtual void    ReadJobFile(XTP::Topology *top) { ; }

    void            setProgObserver(ProgObserver< std::vector<Job*>, Job*, Job::JobResult > *obs) { _progObs = obs; }

protected:
    
    ProgObserver< std::vector<Job*>, Job*, Job::JobResult > *_progObs;

};

}}

#endif // __VOTCA_XTP_JOBCALCULATOR_H

