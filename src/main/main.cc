#include "Python.h"
#include "orbit_mpi.hh"

#include <iostream>

//modules headers
#include "wrap_orbit_mpi.hh"
#include "wrap_bunch.hh"
#include "wrap_utils.hh"
#include "wrap_teapotbase.hh"
#include "wrap_errorbase.hh"
#include "wrap_trackerrk4.hh"
#include "wrap_spacecharge.hh"
#include "wrap_linacmodule.hh"
#include "wrap_collimator.hh"
#include "wrap_foil.hh"
#include "wrap_rfcavities.hh"
#include "wrap_aperture.hh"
#include "wrap_fieldtracker.hh"
#include "wrap_impedances.hh"
#include "wrap_envelope.hh"

/**
 * The main function that will initialize the MPI and will
 * call the python interpreter: Py_Main(argc,argv).
 */

int main(int argc, char **argv)
{
  //  for(int i = 0; i < argc; i++)
  //  {
  //    std::cout << "before i = " << i << " arg = " << argv[i] << std::endl;
  //  }

  ORBIT_MPI_Init(&argc, &argv);

  //  for(int i = 0; i < argc; i++)
  //  {
  //    std::cout << "after i = " << i <<" arg = " << argv[i] << std::endl;
  //  }

  //  int rank = 0;
  //  int size = 0;
  //  ORBIT_MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //  ORBIT_MPI_Comm_size(MPI_COMM_WORLD, &size);
  //  std::cout << "rank = " << rank << " size = " << size << std::endl;

  // We need to initialize the extra ORBIT modules
  Py_Initialize();

  // ORBIT module initializations

  wrap_orbit_mpi::initorbit_mpi();
  wrap_orbit_bunch::initbunch();
  wrap_orbit_utils::initutils();
  wrap_teapotbase::initteapotbase();
  wrap_errorbase::initerrorbase();
  wrap_linac::initlinac();
  wrap_collimator::initcollimator();
  wrap_aperture::initaperture();
  wrap_foil::initfoil();
  wrap_rfcavities::initrfcavities();
  wrap_fieldtracker::initfieldtracker();
  wrap_impedances::initimpedances();

  // Runge-Kutta tracker package

  inittrackerrk4();

  // Space-charge package

  initspacecharge();
    
  // Danilov envelope trackers
    
  wrap_envelope::initenvelope();

  // The python interpreter
  // It will call Py_Initialize() again, but there is no harm.

  Py_Main(argc, argv);

  // std::cout << "MPI - stopped" << std::endl;
  ORBIT_MPI_Finalize();

  return 0;
}
