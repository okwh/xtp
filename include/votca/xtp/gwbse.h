/*
 *            Copyright 2009-2017 The VOTCA Development Team
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

#ifndef _VOTCA_XTP_GWBSE_H
#define _VOTCA_XTP_GWBSE_H
#include <votca/xtp/votca_config.h>
#include <unistd.h>
#include <votca/ctp/parallelxjobcalc.h>
#include <votca/ctp/segment.h>
#include <votca/xtp/orbitals.h>
#include <votca/xtp/qmpackagefactory.h>
#include <votca/xtp/threecenter.h>

#include <fstream>
#include <sys/stat.h>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/filesystem.hpp>



namespace votca {
namespace xtp {


/**
* \brief Electronic excitations from GW-BSE
*
* Evaluates electronic excitations in molecular systems based on
* many-body Green's functions theory within the GW approximation and
* the Bethe-Salpeter equation. Requires molecular orbitals 
*
*  B. Baumeier, Y. Ma, D. Andrienko, M. Rohlfing
*  J. Chem. Theory Comput. 8, 997-1002 (2012)
*
*  B. Baumeier, D. Andrienko, M. Rohlfing
*  J. Chem. Theory Comput. 8, 2790-2795 (2012)
*
*/

class GWBSE {
 public:
  GWBSE(Orbitals* orbitals)
      : _orbitals(orbitals),
        _dft_orbitals(orbitals->MOCoefficients()),
        _qp_diag_energies(orbitals->QPdiagEnergies()),
        _qp_diag_coefficients(orbitals->QPdiagCoefficients()),
        _eh_x(orbitals->eh_x()),
        _eh_d(orbitals->eh_d()),
        _bse_singlet_energies(orbitals->BSESingletEnergies()),
        _bse_singlet_coefficients(orbitals->BSESingletCoefficients()),
        _bse_singlet_coefficients_AR(orbitals->BSESingletCoefficientsAR()),
        _bse_triplet_energies(orbitals->BSETripletEnergies()),
        _bse_triplet_coefficients(orbitals->BSETripletCoefficients()){};


  void Initialize(Property* options);

  std::string Identify() { return "gwbse"; }

  void CleanUp();


  void setLogger(ctp::Logger* pLog) { _pLog = pLog; }

  bool Evaluate();
 


  // interfaces for options getting/setting

  void addoutput(Property* _summary);

 private:
  ctp::Logger* _pLog;

  // bool   _maverick;

  // program tasks
  bool _do_qp_diag;
  bool _do_bse_diag;
  bool _do_bse_singlets;
  bool _do_bse_triplets;

  // storage tasks
  bool _store_qp_pert;
  bool _store_qp_diag;
  bool _store_bse_singlets;
  bool _store_bse_triplets;
  bool _store_eh_interaction;

  // iterate G and W and not only G
  bool _iterate_gw;

  // options for own Vxc calculation
  bool _doVxc;
  std::string _functional;
  std::string _grid;

  int _openmp_threads;

  // fragment definitions
  int _fragA;
  int _fragB;

  // BSE variant
  bool _do_full_BSE;
  bool _ignore_corelevels;


  // basis sets
  std::string _auxbasis_name;
  std::string _dftbasis_name;

  std::string _ranges;  // range types
  unsigned int _homo;   // HOMO index
  unsigned int _rpamin;
  unsigned int _rpamax;
  double _rpamaxfactor;  // RPA level range
  unsigned int _qpmin;
  unsigned int _qpmax;
  unsigned int _qptotal;
  double _qpminfactor;
  double _qpmaxfactor;  // QP level range
  double _bseminfactor;
  double _bsemaxfactor;
  double _ScaHFX;

  double _g_sc_limit;  // convergence criteria for g iteration [Hartree]]
  unsigned int _g_sc_max_iterations;
  unsigned int _gw_sc_max_iterations;
  double _gw_sc_limit;  // convergence criteria for gw iteration [Hartree]]
  unsigned int _bse_vmin;
  unsigned int _bse_vmax;
  unsigned int _bse_cmin;
  unsigned int _bse_cmax;
  unsigned int _bse_size;
  unsigned int _bse_vtotal;
  unsigned int _bse_ctotal;
  int _bse_nmax;
  int _bse_nprint;
  double _min_print_weight;

  double _shift;  // pre-shift of DFT energies
  AOBasis _dftbasis;

  
   Eigen::MatrixXd CalculateVXC();
  
  Orbitals* _orbitals;
  Eigen::MatrixXd& _dft_orbitals;
  // RPA related variables and functions
  // container for the epsilon matrix
  std::vector<Eigen::MatrixXd > _epsilon;
  // container for frequencies in screening (index 0: real part, index 1:
  // imaginary part)
  Eigen::MatrixXd _screening_freq;
  
  void RPA_calculate_epsilon(const TCMatrix_gwbse& _Mmn_RPA);

  Eigen::MatrixXd RPA_real(const TCMatrix_gwbse& _Mmn_RPA,
                              const double screening_freq);

  Eigen::MatrixXd RPA_imaginary(const TCMatrix_gwbse& _Mmn_RPA,
                                   const double screening_freq);

  void RPA_prepare_threecenters(TCMatrix_gwbse& _Mmn_RPA, const TCMatrix_gwbse& _Mmn_full);

  // PPM related variables and functions
  Eigen::MatrixXd _ppm_phi_T;
  Eigen::VectorXd _ppm_freq;
  Eigen::VectorXd _ppm_weight;
 

  void PPM_construct_parameters(
      const Eigen::MatrixXd& _overlap_cholesky_inverse);

  // Sigma related variables and functions
  Eigen::MatrixXd _sigma_x;  // exchange term
  Eigen::MatrixXd _sigma_c;  // correlation term

  void sigma_prepare_threecenters(TCMatrix_gwbse& _Mmn);

  void sigma_diag(const TCMatrix_gwbse& _Mmn);
  void sigma_offdiag(const TCMatrix_gwbse& _Mmn);

  // QP variables and functions
  Eigen::VectorXd _qp_energies;
  Eigen::MatrixXd _vxc;
  Eigen::VectorXd& _qp_diag_energies;      // stored in orbitals object
  Eigen::MatrixXd& _qp_diag_coefficients;  // dito
  void FullQPHamiltonian();

  // BSE variables and functions
  MatrixXfd& _eh_x;  // stored in orbitals object
  MatrixXfd& _eh_d;  // stored in orbitals object
  MatrixXfd _eh_d2;  // because it is not stored in orbitals object
  MatrixXfd _eh_qp;

  VectorXfd& _bse_singlet_energies;  // stored in orbitals object
  MatrixXfd& _bse_singlet_coefficients;  // stored in orbitals
                                                      // object
  MatrixXfd& _bse_singlet_coefficients_AR;  // stored in orbitals
                                                         // object
  VectorXfd& _bse_triplet_energies;  // stored in orbitals object
  MatrixXfd& _bse_triplet_coefficients;  // stored in orbitals
                                                      // object

  std::vector<Eigen::MatrixXd > _interlevel_dipoles;
  std::vector<Eigen::MatrixXd > _interlevel_dipoles_electrical;
  void BSE_x_setup(TCMatrix_gwbse& _Mmn);
  void BSE_d_setup(TCMatrix_gwbse& _Mmn);
  void BSE_d2_setup(TCMatrix_gwbse& _Mmn);
  void BSE_qp_setup();
  void BSE_Add_qp2H(MatrixXfd& qp);
  void BSE_solve_triplets();
  void BSE_solve_singlets();
  void BSE_solve_singlets_BTDA();

  void Solve_nonhermitian(Eigen::MatrixXd& H, Eigen::MatrixXd& L);
  std::vector<int> _index2v;
  std::vector<int> _index2c;

  // some cleaner analysis
  void BSE_analyze_triplets();
  void BSE_analyze_singlets();
 

  void BSE_analyze_eh_interaction_Triplet(std::vector<real_gwbse>& _c_d,
                                          std::vector<real_gwbse>& _c_qp);
  
  void BSE_analyze_eh_interaction_Singlet(std::vector<real_gwbse>& _c_x,
                                               std::vector<real_gwbse>& _c_d,
                                               std::vector<real_gwbse>& _c_qp);

  void BSE_FragmentPopulations(const string& spin,
                               std::vector<Eigen::VectorXd >& popH,
                               std::vector<Eigen::VectorXd >& popE,
                               std::vector<Eigen::VectorXd >& Crgs);

  void BSE_FreeTransition_Dipoles();

  void BSE_CoupledTransition_Dipoles();
};
}
}

#endif /* _VOTCA_XTP_GWBSE_H */
