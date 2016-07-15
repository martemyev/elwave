#include "elastic_wave.hpp"
#include "parameters.hpp"
#include "receivers.hpp"

#include <float.h>
#include <fstream>
#include <vector>

using namespace std;
using namespace mfem;

//#define OUTPUT_MASS_MATRIX



void ElasticWave::run_DG()
{
#ifdef MFEM_USE_MPI
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size == 1)
    run_DG_serial();
  else
    run_DG_parallel();
#else
  run_DG_serial();
#endif
}


static void par_time_step(HypreParMatrix &M, HypreParMatrix &S,
                          HypreParVector &b, double timeval, double dt,
                          Solver &solver, HypreParVector &U_0,
                          HypreParVector &U_1, HypreParVector &U_2)
{
  HypreParVector y = U_1; y *= 2.0; y -= U_2;        // y = 2*u_1 - u_2

  HypreParVector z0 = U_0;                           // z0 = M * (2*u_1 - u_2)
  M.Mult(y, z0);

  HypreParVector z1 = U_0;                           // z1 = S * u_1
  S.Mult(U_1, z1);
  HypreParVector z2 = b; z2 *= timeval; // z2 = timeval*source

  // y = dt^2 * (S*u_1 - timeval*source), where it can be
  // y = dt^2 * (S*u_1 - ricker*pointforce) OR
  // y = dt^2 * (S*u_1 - gaussfirstderivative*momenttensor)
  y = z1; y -= z2; y *= dt*dt;

  // RHS = M*(2*u_1-u_2) - dt^2*(S*u_1-timeval*source)
  HypreParVector RHS = z0; RHS -= y;

  solver.Mult(RHS, U_0);

  U_2 = U_1;
  U_1 = U_0;
}



void ElasticWave::run_DG_serial()
{
  StopWatch chrono;

  chrono.Start();

  const int dim = param.mesh->Dimension();
//  const int n_elements = param.mesh->GetNE();

  cout << "FE space generation..." << flush;
  FiniteElementCollection *fec = new DG_FECollection(param.method.order, dim);
  FiniteElementSpace fespace(param.mesh, fec, dim); //, Ordering::byVDIM);
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  cout << "Number of unknowns: " << fespace.GetVSize() << endl;

  CWConstCoefficient rho_coef(param.media.rho_array, false);
  CWConstCoefficient lambda_coef(param.media.lambda_array, false);
  CWConstCoefficient mu_coef(param.media.mu_array, false);

  cout << "Stif matrix..." << flush;
  chrono.Clear();
  BilinearForm stif(&fespace);
  stif.AddDomainIntegrator(new ElasticityIntegrator(lambda_coef, mu_coef));
  stif.AddInteriorFaceIntegrator(
     new DGElasticityIntegrator(lambda_coef, mu_coef,
                                param.method.dg_sigma, param.method.dg_kappa));
  stif.AddBdrFaceIntegrator(
     new DGElasticityIntegrator(lambda_coef, mu_coef,
                                param.method.dg_sigma, param.method.dg_kappa));
  stif.Assemble();
  stif.Finalize();
  const SparseMatrix& S = stif.SpMat();
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  cout << "Mass matrix..." << flush;
  chrono.Clear();
  BilinearForm mass(&fespace);
  mass.AddDomainIntegrator(new VectorMassIntegrator(rho_coef));
  mass.Assemble();
  mass.Finalize();
  const SparseMatrix& M = mass.SpMat();
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  /*

#if defined(OUTPUT_MASS_MATRIX)
  {
    cout << "Output mass matrix..." << flush;
    ofstream mout("mass_mat.dat");
    mass.PrintMatlab(mout);
    cout << "M.nnz = " << M.NumNonZeroElems() << endl;
    cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
    chrono.Clear();
  }
#endif

//  cout << "Damp matrix..." << flush;
//  BilinearForm dampM(&fespace);
//  dampM.AddDomainIntegrator(new VectorMassIntegrator(rho_damp_coef));
//  dampM.Assemble();
//  dampM.Finalize();
//  SparseMatrix& D = dampM.SpMat();
//  double omega = 2.0*M_PI*param.source.frequency; // angular frequency
//  D *= 0.5*param.dt*omega;
//  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
//  chrono.Clear();

  cout << "System matrix..." << flush;
  SparseMatrix SysMat(M);
//  SysMat += D;
  GSSmoother Prec(SysMat);
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();

  cout << "RHS vector... " << flush;
  LinearForm b(&fespace);
  if (param.source.plane_wave)
  {
    PlaneWaveSource plane_wave_source(dim, param);
    b.AddDomainIntegrator(new VectorDomainLFIntegrator(plane_wave_source));
    b.Assemble();
  }
  else
  {
    if (!strcmp(param.source.type, "pointforce"))
    {
      VectorPointForce vector_point_force(dim, param);
      b.AddDomainIntegrator(new VectorDomainLFIntegrator(vector_point_force));
      b.Assemble();
    }
    else if (!strcmp(param.source.type, "momenttensor"))
    {
      MomentTensorSource momemt_tensor_source(dim, param);
      b.AddDomainIntegrator(new VectorDomainLFIntegrator(momemt_tensor_source));
      b.Assemble();
    }
    else MFEM_ABORT("Unknown source type: " + string(param.source.type));
  }
  cout << "||b||_L2 = " << b.Norml2() << endl;
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();

//  Vector diagM; M.GetDiag(diagM); // mass matrix is diagonal
//  Vector diagD; D.GetDiag(diagD); // damping matrix is diagonal
//  for (int i = 0; i < diagM.Size(); ++i)
//  {
//    MFEM_VERIFY(fabs(diagM[i]) > FLOAT_NUMBERS_EQUALITY_TOLERANCE,
//                "There is a small (" + d2s(diagM[i]) + ") number (row "
//                + d2s(i) + ") on the mass matrix diagonal");
//  }

  const string method_name = "DG_";

  cout << "Open seismograms files..." << flush;
  ofstream *seisU; // for displacement
  ofstream *seisV; // for velocity
  open_seismo_outs(seisU, seisV, param, method_name);
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();

  GridFunction u_0(&fespace); // displacement
  GridFunction u_1(&fespace);
  GridFunction u_2(&fespace);
  GridFunction v_1(&fespace); // velocity
  u_0 = 0.0;
  u_1 = 0.0;
  u_2 = 0.0;

  const int n_time_steps = param.T / param.dt + 0.5; // nearest integer
  const int tenth = 0.1 * n_time_steps;

  const int N = u_0.Size();

  cout << "N time steps = " << n_time_steps
       << "\nTime loop..." << endl;

  // the values of the time-dependent part of the source
  vector<double> time_values(n_time_steps);
  if (!strcmp(param.source.type, "pointforce")) {
    for (int time_step = 1; time_step <= n_time_steps; ++time_step) {
      const double cur_time = time_step * param.dt;
      time_values[time_step-1] = RickerWavelet(param.source,
                                               cur_time - param.dt);
    }
  } else if (!strcmp(param.source.type, "momenttensor")) {
    for (int time_step = 1; time_step <= n_time_steps; ++time_step) {
      const double cur_time = time_step * param.dt;
      time_values[time_step-1] = GaussFirstDerivative(param.source,
                                                      cur_time - param.dt);
    }
  } else MFEM_ABORT("Unknown source type: " + string(param.source.type));

  const string name = method_name + param.output.extra_string;
  const string pref_path = (string)param.output.directory + "/" + SNAPSHOTS_DIR;
  VisItDataCollection visit_dc(name.c_str(), param.mesh);
  visit_dc.SetPrefixPath(pref_path.c_str());
  visit_dc.RegisterField("displacement", &u_0);
  visit_dc.RegisterField("velocity", &v_1);

  StopWatch time_loop_timer;
  time_loop_timer.Start();
  double time_of_snapshots = 0.;
  double time_of_seismograms = 0.;
  for (int time_step = 1; time_step <= n_time_steps; ++time_step)
  {
    Vector y = u_1; y *= 2.0; y -= u_2;        // y = 2*u_1 - u_2

    Vector z0; z0.SetSize(N);                  // z0 = M * (2*u_1 - u_2)
    //for (int i = 0; i < N; ++i) z0[i] = diagM[i] * y[i];
    M.Mult(y, z0);

    Vector z1; z1.SetSize(N); S.Mult(u_1, z1);     // z1 = S * u_1
    Vector z2 = b; z2 *= time_values[time_step-1]; // z2 = timeval*source

    // y = dt^2 * (S*u_1 - timeval*source), where it can be
    // y = dt^2 * (S*u_1 - ricker*pointforce) OR
    // y = dt^2 * (S*u_1 - gaussfirstderivative*momenttensor)
    y = z1; y -= z2; y *= param.dt*param.dt;

    // RHS = M*(2*u_1-u_2) - dt^2*(S*u_1-timeval*source)
    Vector RHS = z0; RHS -= y;

    //for (int i = 0; i < N; ++i) y[i] = diagD[i] * u_2[i]; // y = D * u_2
//    D.Mult(u_2, y);
    // RHS = M*(2*u_1-u_2) - dt^2*(S*u_1-timeval*source) + D*u_2
//    RHS += y;

    // (M+D)*x_0 = M*(2*x_1-x_2) - dt^2*(S*x_1-r*b) + D*x_2
    //for (int i = 0; i < N; ++i) u_0[i] = RHS[i] / (diagM[i]+diagD[i]);
    PCG(SysMat, Prec, RHS, u_0, 0, 200, 1e-12, 0.0);

    u_2 = u_1;
    u_1 = u_0;

    // velocity: v = du/dt, we use the central difference here
    v_1  = u_0;
    v_1 -= u_2;
    v_1 /= 2.0*param.dt;

    // Compute and print the L^2 norm of the error
    if (time_step % tenth == 0) {
      cout << "step " << time_step << " / " << n_time_steps
           << " ||solution||_{L^2} = " << u_0.Norml2() << endl;
    }

    if (time_step % param.step_snap == 0) {
      StopWatch timer;
      timer.Start();
      visit_dc.SetCycle(time_step);
      visit_dc.SetTime(time_step*param.dt);
      visit_dc.Save();
      timer.Stop();
      time_of_snapshots += timer.UserTime();
    }

    if (time_step % param.step_seis == 0) {
      StopWatch timer;
      timer.Start();
      output_seismograms(param, *param.mesh, u_0, v_1, seisU, seisV);
      timer.Stop();
      time_of_seismograms += timer.UserTime();
    }
  }

  time_loop_timer.Stop();

  delete[] seisU;
  delete[] seisV;

  cout << "Time loop is over\n\tpure time = " << time_loop_timer.UserTime()
       << "\n\ttime of snapshots = " << time_of_snapshots
       << "\n\ttime of seismograms = " << time_of_seismograms << endl;

  */

  delete fec;
}



#ifdef MFEM_USE_MPI
void ElasticWave::run_DG_parallel()
{
  MFEM_VERIFY(param.mesh, "The serial mesh is not initialized");
  MFEM_VERIFY(param.par_mesh, "The parallel mesh is not initialized");

  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  StopWatch chrono;
  chrono.Start();

  const int dim = param.dimension;

  if (myid == 0)
    cout << "FE space generation..." << flush;
  DG_FECollection fec(param.method.order, dim);
  ParFiniteElementSpace fespace(param.par_mesh, &fec, dim);
  if (myid == 0)
    cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  const HYPRE_Int size = fespace.GlobalTrueVSize();
  if (myid == 0)
    cout << "Number of unknowns: " << size << endl;

//  CWConstCoefficient rho_coef(param.media.rho_array, false);
//  CWConstCoefficient lambda_coef(param.media.lambda_array, false);
//  CWConstCoefficient mu_coef(param.media.mu_array, false);

  ConstantCoefficient rho_coef(param.media.rho_array[0]);
  ConstantCoefficient lambda_coef(param.media.lambda_array[0]);
  ConstantCoefficient mu_coef(param.media.mu_array[0]);

  if (myid == 0)
    cout << "Fine scale stif matrix..." << flush;
  chrono.Clear();
  ParBilinearForm stif_fine(&fespace);
  stif_fine.AddDomainIntegrator(new ElasticityIntegrator(lambda_coef, mu_coef));
  stif_fine.AddInteriorFaceIntegrator(
     new DGElasticityIntegrator(lambda_coef, mu_coef,
                                param.method.dg_sigma, param.method.dg_kappa));
  stif_fine.AddBdrFaceIntegrator(
     new DGElasticityIntegrator(lambda_coef, mu_coef,
                                param.method.dg_sigma, param.method.dg_kappa));
  stif_fine.Assemble();
  stif_fine.Finalize();

  HypreParMatrix *S_fine = stif_fine.ParallelAssemble();
  if (myid == 0)
    cout << "done. Time = " << chrono.RealTime() << " sec" << endl;


  if (myid == 0)
    cout << "Fine scale mass matrix..." << flush;
  chrono.Clear();
  ParBilinearForm mass_fine(&fespace);
  mass_fine.AddDomainIntegrator(new VectorMassIntegrator(rho_coef));
  mass_fine.Assemble();
  mass_fine.Finalize();

  HypreParMatrix *M_fine = mass_fine.ParallelAssemble();
  if (myid == 0)
    cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  /*

  //HypreParMatrix SysFine((hypre_ParCSRMatrix*)(*M_fine));
  //SysFine = 0.0;
  //SysFine += D;
  //SysFine += M_fine;


  if (myid == 0)
    cout << "Fine scale RHS vector... " << flush;
  chrono.Clear();
  ParLinearForm b_fine(&fespace);
  if (param.source.plane_wave)
  {
    PlaneWaveSource plane_wave_source(dim, param);
    b_fine.AddDomainIntegrator(new VectorDomainLFIntegrator(plane_wave_source));
    b_fine.Assemble();
  }
  else
  {
    if (!strcmp(param.source.type, "pointforce"))
    {
      VectorPointForce vector_point_force(dim, param);
      b_fine.AddDomainIntegrator(new VectorDomainLFIntegrator(vector_point_force));
      b_fine.Assemble();
    }
    else if (!strcmp(param.source.type, "momenttensor"))
    {
      MomentTensorSource momemt_tensor_source(dim, param);
      b_fine.AddDomainIntegrator(new VectorDomainLFIntegrator(momemt_tensor_source));
      b_fine.Assemble();
    }
    else MFEM_ABORT("Unknown source type: " + string(param.source.type));
  }
  if (myid == 0)
    cout << "||b_h||_L2 = " << b_fine.Norml2() << endl
         << "done. Time = " << chrono.RealTime() << " sec" << endl;
  HypreParVector b(&fespace);
  b_fine.ParallelAssemble(b);


  HypreBoomerAMG amg(*M_fine);
  HyprePCG pcg(*M_fine);
  pcg.SetTol(1e-12);
  pcg.SetMaxIter(200);
  pcg.SetPrintLevel(2);
  pcg.SetPreconditioner(amg);


  const string method_name = "parGMsFEM_";

  if (myid == 0)
    cout << "Open seismograms files..." << flush;
  ofstream *seisU; // for displacement
  ofstream *seisV; // for velocity
  if (myid == 0)
    open_seismo_outs(seisU, seisV, param, method_name);
  if (myid == 0)
    cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();

  HypreParVector U_0(*M_fine);
  U_0 = 0.0;
  HypreParVector U_1 = U_0;
  HypreParVector U_2 = U_0;
  ParGridFunction u_0(&fespace, &U_0);

  const int n_time_steps = param.T / param.dt + 0.5; // nearest integer
  const int tenth = 0.1 * n_time_steps;

  cout << "N time steps = " << n_time_steps
       << "\nTime loop..." << endl;

  // the values of the time-dependent part of the source
  vector<double> time_values(n_time_steps);
  for (int time_step = 1; time_step <= n_time_steps; ++time_step)
  {
    const double cur_time = time_step * param.dt;
    time_values[time_step-1] = RickerWavelet(param.source,
                                             cur_time - param.dt);
  }

  const string name = method_name + param.output.extra_string;
  const string pref_path = string(param.output.directory) + "/" + SNAPSHOTS_DIR;
  VisItDataCollection visit_dc(name.c_str(), param.par_mesh);
  visit_dc.SetPrefixPath(pref_path.c_str());
//  visit_dc.RegisterField("fine_pressure", &u_fine_0);
  visit_dc.RegisterField("coarse_pressure", &u_0);
  {
    visit_dc.SetCycle(0);
    visit_dc.SetTime(0.0);
//    Vector u_tmp(u_fine_0.Size());
//    R_global_T->Mult(U_0, u_tmp);
//    u_0.MakeRef(&fespace, u_tmp, 0);
    visit_dc.Save();
  }

  StopWatch time_loop_timer;
  time_loop_timer.Start();
  double time_of_snapshots = 0.;
  double time_of_seismograms = 0.;
  for (int t_step = 1; t_step <= n_time_steps; ++t_step)
  {
    {
      par_time_step(*M_fine, *S_fine, b, time_values[t_step-1],
                    param.dt, pcg, U_0, U_1, U_2);
    }
    {
//      time_step(M_fine, S_fine, b_fine, time_values[t_step-1],
//                param.dt, SysFine, PrecFine, u_fine_0, u_fine_1, u_fine_2);
    }

    // Compute and print the L^2 norm of the error
    if (t_step % tenth == 0) {
      cout << "step " << t_step << " / " << n_time_steps
           << " ||U||_{L^2} = " << U_0.Norml2()
           //<< " ||u||_{L^2} = " << u_fine_0.Norml2()
           << endl;
    }

    if (t_step % param.step_snap == 0) {
      StopWatch timer;
      timer.Start();
      visit_dc.SetCycle(t_step);
      visit_dc.SetTime(t_step*param.dt);
//      Vector u_tmp(u_fine_0.Size());
//      R_global_T->Mult(U_0, u_tmp);
//      u_0.MakeRef(&fespace, u_tmp, 0);
      visit_dc.Save();
      timer.Stop();
      time_of_snapshots += timer.UserTime();
    }

//    if (t_step % param.step_seis == 0) {
//      StopWatch timer;
//      timer.Start();
//      R_global_T.Mult(U_0, u_0);
//      output_seismograms(param, *param.par_mesh, u_0, seisU);
//      timer.Stop();
//      time_of_seismograms += timer.UserTime();
//    }
  }

  time_loop_timer.Stop();

  delete[] seisU;
  delete[] seisV;

  if (myid == 0)
    cout << "Time loop is over\n\tpure time = " << time_loop_timer.UserTime()
         << "\n\ttime of snapshots = " << time_of_snapshots
         << "\n\ttime of seismograms = " << time_of_seismograms << endl;

  */
}
#endif // MFEM_USE_MPI


