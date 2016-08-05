#include "config.hpp"
#include "elastic_wave.hpp"
#include "mfem.hpp"
#include "parameters.hpp"
#include "utilities.hpp"

using namespace std;
using namespace mfem;



int main(int argc, char *argv[])
{
  int myid = 0;
  int size = 1;
#ifdef MFEM_USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

  if (argc == 1) // no arguments
  {
    if (myid == 0)
      cout << "\nGet help with:\n" << argv[0] << " -h\n" << endl;
#ifdef MFEM_USE_MPI
    MPI_Finalize();
#endif
    return 0;
  }

#ifdef MFEM_DEBUG
  if (myid == 0)
  {
    cout << "****************************\n";
    cout << "*     DEBUG VERSION        *\n";
    cout << "****************************\n";
  }
#endif // MFEM_DEBUG

#ifdef MFEM_USE_MPI
  if (myid == 0)
  {
    cout << "****************************\n";
    cout << "PARALLEL: size " << size << "\n";
    cout << "****************************\n";
  }
#endif // MFEM_USE_MPI

  try
  {
    StopWatch chrono;
    chrono.Start();

    Parameters param;
    param.init(argc, argv);

//    show_SRM_damp_weights(param);

    ElasticWave elwave(param);
    elwave.run();

    if (myid == 0)
      cout << "\nTOTAL TIME " << chrono.RealTime() << " sec\n" << endl;
  }
  catch (int ierr)
  {
#ifdef MFEM_USE_MPI
    MPI_Finalize();
#endif
    return ierr;
  }
  catch (...)
  {
#ifdef MFEM_USE_MPI
    MPI_Finalize();
#endif
    cerr << "\nEXCEPTION\n";
  }

#ifdef MFEM_USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
