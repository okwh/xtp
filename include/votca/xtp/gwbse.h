/*
 *            Copyright 2009-2016 The VOTCA Development Team
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

// UBLAS stops checking types and array bounds if this flag is defined
#define NDEBUG
#define BOOST_UBLAS_NDEBUG

#ifndef _VOTCA_XTP_GWBSE_H
#define	_VOTCA_XTP_GWBSE_H

#include <votca/xtp/segment.h>
#include <votca/xtp/orbitals.h>
#include <votca/xtp/threecenters.h>
#include <votca/xtp/qmpackagefactory.h>
#include <votca/xtp/parallelxjobcalc.h>
#include <unistd.h>

#include <fstream>
#include <sys/stat.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/filesystem.hpp>
#include <votca/tools/linalg.h>

#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
// #include <gsl/gsl_eigen.h>
// #include <gsl/gsl_linalg.h>
// #include <gsl/gsl_cblas.h>

namespace votca { namespace xtp {
    namespace ub = boost::numeric::ublas;

        /**
         * \brief Electronic excitations from GW-BSE
         *
         * Evaluates electronic excitations in molecular systems based on
         * many-body Green's functions theory within the GW approximation and the
         * Bethe-Salpeter equation. Requires molecular orbitals of the object
         * in GAUSSIAN, NWChem, or TURBOMOLE format.
         * 
         *  B. Baumeier, Y. Ma, D. Andrienko, M. Rohlfing
         *  J. Chem. Theory Comput. 8, 997-1002 (2012)
         * 
         *  B. Baumeier, D. Andrienko, M. Rohlfing
         *  J. Chem. Theory Comput. 8, 2790-2795 (2012) 
         * 
         */

class GWBSE 
{
public:

    GWBSE(Orbitals* orbitals):  _orbitals(orbitals),
            _qp_diag_energies(orbitals->QPdiagEnergies()), 
             _qp_diag_coefficients(orbitals->QPdiagCoefficients()),
              _eh_x(orbitals->eh_x()),
              _eh_d(orbitals->eh_d()),
              _bse_singlet_energies(orbitals->BSESingletEnergies()),
    _bse_singlet_coefficients(orbitals->BSESingletCoefficients()),
       _bse_triplet_energies(orbitals->BSETripletEnergies()),
    _bse_triplet_coefficients(orbitals->BSETripletCoefficients())
   
    {};
   ~GWBSE() { };

   
   
    void    Initialize( Property *options);
  std::string  Identify() { return "gwbse"; }
   
  /*  void    ParseOrbitalsXML(Topology *top, Property *options);
    Job::JobResult EvalJob(Topology *top, Job *job, QMThread *thread);
 */
    void    CleanUp();

    // int getMlower(){ return mmin -1; };
    // int getMupper(){ return mmax -1; };
    
    void setLogger( Logger* pLog ) { _pLog = pLog; }
    
    bool Evaluate();

    // interfaces for options getting/setting
    bool get_do_qp_diag(){ return _do_qp_diag ;}
    void set_do_qp_diag( bool inp ){ _do_qp_diag = inp;}
    
    bool get_do_bse_diag(){ return _do_bse_diag ;}
    void set_do_bse_diag( bool inp ){ _do_bse_diag = inp;}
    
    bool get_do_bse_singlets(){ return _do_bse_singlets ;}
    void set_do_bse_singlets( bool inp ){ _do_bse_singlets = inp;}

    bool get_do_bse_triplets(){ return _do_bse_triplets ;}
    void set_do_bse_triplets( bool inp ){ _do_bse_triplets = inp;}

    bool get_store_qp_pert(){ return _store_qp_pert ;}
    void set_store_qp_pert( bool inp ){ _store_qp_pert = inp;}
    
    bool get_store_qp_diag(){ return _store_qp_diag ;}
    void set_store_qp_diag( bool inp ){ _store_qp_diag = inp;}
    
    bool get_store_bse_singlets(){ return _store_bse_singlets ;}
    void set_store_bse_singlets( bool inp ){ _store_bse_singlets = inp;}

    bool get_store_eh_interaction(){ return _store_eh_interaction ;}
    void set_store_eh_interaction( bool inp ){ _store_eh_interaction = inp;}

    
    bool get_store_bse_triplets(){ return _store_bse_triplets ;}
    void set_store_bse_triplets( bool inp ){ _store_bse_triplets = inp;}
    
    std::string get_ranges(){ return _ranges ;}
    void set_ranges( std::string inp ){ _ranges = inp;}
    
    double get_rpamaxfactor() {return _rpamaxfactor ;}
    void set_rpamaxfactor( double inp ) { _rpamaxfactor = inp; }
  
    double get_qpmaxfactor() {return _qpmaxfactor ;}
    void set_qpmaxfactor( double inp ) { _qpmaxfactor = inp; }

    double get_qpminfactor() {return _qpminfactor ;}
    void set_qpminfactor( double inp ) { _qpminfactor = inp; }

    double get_bsemaxfactor() {return _bsemaxfactor ;}
    void set_bsemaxfactor( double inp ) { _bsemaxfactor = inp; }

    double get_bseminfactor() {return _bseminfactor ;}
    void set_bseminfactor( double inp ) { _bseminfactor = inp; }
    
    unsigned int get_bse_vmin() {return _bse_vmin ;}
    void set_bse_vmin( unsigned int inp ) { _bse_vmin = inp; }

    unsigned int get_bse_cmax() {return _bse_cmax ;}
    void set_bse_cmax( unsigned int inp ) { _bse_cmax = inp; }

    unsigned int get_rpamax() {return _rpamax ;}
    void set_rpamax( unsigned int inp ) { _rpamax = inp; }

    unsigned int get_qpmax() {return _qpmax ;}
    void set_qpmax( unsigned int inp ) { _qpmax = inp; }

    unsigned int get_qpmin() {return _qpmin ;}
    void set_qpmin( unsigned int inp ) { _qpmin = inp; }
    
    int get_bse_nmax(){return _bse_nmax;}
    void set_bse_nmax( int inp){ _bse_nmax = inp;}
    
    int get_bse_nprint(){return _bse_nprint;}
    void set_bse_nprint( int inp){ _bse_nprint = inp;}
    
    std::string get_gwbasis_name(){return _gwbasis_name;}
    void set_gwbasis_name(std::string inp){ _gwbasis_name = inp;}
    
    std::string get_dftbasis_name(){return _dftbasis_name;}
    void set_dftbasis_name(std::string inp){ _dftbasis_name = inp;}

    double get_shift() {return _shift ;}
    void set_shift( double inp ) { _shift = inp; }
    
    void set_fragA( int n ) { _fragA = n; }
    void set_fragB( int n ) { _fragB = n; }
    
    void addoutput(Property *_summary);
    
    private:

    Logger *_pLog;
    

    
    //bool   _maverick;
    
    // program tasks
    bool                                _do_qp_diag;
    bool                                _do_bse_diag;
    bool                                _do_bse_singlets;
    bool                                _do_bse_triplets;
    
    // storage tasks
    bool                                _store_qp_pert;
    bool                                _store_qp_diag;
    bool                                _store_bse_singlets;
    bool                                _store_bse_triplets;
    bool                                _store_eh_interaction;
    
    // automatic scissors iteration
    bool                                _iterate_shift;
    bool                                _shift_converged;
    
    // options for own Vxc calculation
    bool                                _doVxc;
    std::string                              _functional;
    std::string                              _grid;
    
    int                                 _openmp_threads;
    
    // fragment definitions
    int                                 _fragA;
    int                                 _fragB;

    // BSE variant
    bool                                _do_full_BSE;

    
    std::string _outParent;
    std::string _outMonDir;
    
    std::string _package;
    Property _package_options;   
    
    std::string _gwpackage;
    Property _gwpackage_options; 
    
    // basis sets
    std::string                              _gwbasis_name;
    std::string                              _dftbasis_name;

    std::string                              _ranges;          // range types
    unsigned int                        _homo;            // HOMO index
    unsigned int                        _rpamin;
    unsigned int                        _rpamax;
    double                              _rpamaxfactor;    // RPA level range
    unsigned int                        _qpmin;
    unsigned int                        _qpmax;
    unsigned int                        _qptotal;
    double                              _qpminfactor;
    double                              _qpmaxfactor;     // QP level range
    double                              _bseminfactor;
    double                              _bsemaxfactor;
    double                              _ScaHFX; 
    
    double                              _qp_limit;  //convergence criteria for qp iteration [Ryd]]
    double                              _shift_limit; //convergence criteria for shift iteration [Ryd]]
    unsigned int                        _bse_vmin;
    unsigned int                        _bse_vmax;
    unsigned int                        _bse_cmin;
    unsigned int                        _bse_cmax;
    unsigned int                        _bse_size;
    unsigned int                        _bse_vtotal;
    unsigned int                        _bse_ctotal;
    int                                 _bse_nmax;
    int                                 _bse_nprint;
         
    double                              _shift;  // pre-shift of DFT energies

    Orbitals* _orbitals;
    // RPA related variables and functions
    // container for the epsilon matrix
    std::vector< ub::matrix<double> > _epsilon;
    // container for frequencies in screening (index 0: real part, index 1: imaginary part)
    ub::matrix<double> _screening_freq;
    void symmetrize_threecenters(TCMatrix& _Mmn, ub::matrix<double>& _coulomb);
    void RPA_calculate_epsilon(const TCMatrix& _Mmn_RPA, const ub::matrix<double>& screening_freq,
                const double& _shift, const ub::vector<double>& _dft_energies);
    
    void RPA_real(ub::matrix<double>& result,const TCMatrix& _Mmn_RPA,const double& _shift,
        const ub::vector<double>& _dft_energies,const double& screening_freq);
    
    void RPA_imaginary(ub::matrix<double>& result,const TCMatrix& _Mmn_RPA, const double& _shift,
        const ub::vector<double>& _dft_energies,const double& screening_freq);
               
    void RPA_prepare_threecenters(TCMatrix& _Mmn_RPA,const TCMatrix& _Mmn_full,AOBasis& gwbasis,
            const AOMatrix& gwoverlap,const AOMatrix& gwoverlap_inverse );

    
    // PPM related variables and functions
    ub::matrix<double> _ppm_phi;
    ub::vector<double> _ppm_freq;
    ub::vector<double> _ppm_weight;
    
    void PPM_construct_parameters( ub::matrix<double>& _overlap_cholesky_inverse   );
    
    // Sigma related variables and functions
    ub::symmetric_matrix<double> _sigma_x; // exchange term
    ub::symmetric_matrix<double> _sigma_c; // correlation term
    
    void sigma_prepare_threecenters( TCMatrix& _Mmn );
    void sigma_x_setup(const TCMatrix& _Mmn );
    void sigma_c_setup(const TCMatrix& _Mmn , const ub::vector<double>& _edft );
    
    // QP variables and functions
    ub::vector<double> _qp_energies;
    ub::matrix<double> _vxc;
    ub::vector<double>& _qp_diag_energies;     // stored in orbitals object 
    ub::matrix<double>& _qp_diag_coefficients; // dito
    void FullQPHamiltonian();
    
    // BSE variables and functions
    //ub::matrix<double> _eh_x;
    //ub::matrix<double> _eh_d;
    //ub::matrix<double> _eh_qp;
    
    ub::matrix<real>& _eh_x;//stored in orbitals object
    ub::matrix<real>& _eh_d;//stored in orbitals object
    ub::matrix<real> _eh_d2;//because it is not stored in orbitals object
    ub::matrix<real> _eh_qp;//not used right now
    
    // ub::vector<double> _bse_singlet_energies;
    // ub::matrix<double> _bse_singlet_coefficients;
    //ub::vector<double> _bse_triplet_energies;
    //ub::matrix<double> _bse_triplet_coefficients;
    ub::vector<real>& _bse_singlet_energies;//stored in orbitals object
    ub::matrix<real>& _bse_singlet_coefficients;//stored in orbitals object
    ub::vector<real>& _bse_triplet_energies;//stored in orbitals object
    ub::matrix<real>& _bse_triplet_coefficients;//stored in orbitals object
    
    std::vector< ub::matrix<double> > _interlevel_dipoles;
    std::vector< ub::matrix<double> > _interlevel_dipoles_electrical;
    void BSE_x_setup( TCMatrix& _Mmn );
    void BSE_d_setup( TCMatrix& _Mmn );
    void BSE_d2_setup( TCMatrix& _Mmn );
    void BSE_qp_setup( );
    void BSE_Add_qp2H( ub::matrix<real>& qp );
    void BSE_solve_triplets();
    void BSE_solve_singlets();
    void BSE_solve_singlets_BTDA();
    std::vector<int> _index2v;
    std::vector<int> _index2c;
    
};


}}

#endif	/* _VOTCA_XTP_GWBSE_H */
