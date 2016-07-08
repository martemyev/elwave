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

void ElasticWave::run_DG_serial()
{
  StopWatch chrono;

  chrono.Start();

  const int dim = param.mesh->Dimension();
  const int n_elements = param.mesh->GetNE();

  cout << "FE space generation..." << flush;
  FiniteElementCollection *fec = new DG_FECollection(param.method.order, dim);
  FiniteElementSpace fespace(param.mesh, fec, dim); //, Ordering::byVDIM);
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();

  cout << "Number of unknowns: " << fespace.GetVSize() << endl;

  double *lambda_array = new double[n_elements];
  double *mu_array     = new double[n_elements];

  double Rho[] = { DBL_MAX, DBL_MIN };
  double Vp[]  = { DBL_MAX, DBL_MIN };
  double Vs[]  = { DBL_MAX, DBL_MIN };
  double Lam[] = { DBL_MAX, DBL_MIN };
  double Mu[]  = { DBL_MAX, DBL_MIN };

  for (int i = 0; i < n_elements; ++i)
  {
    const double rho = param.media.rho_array[i];
    const double vp  = param.media.vp_array[i];
    const double vs  = param.media.vs_array[i];

    MFEM_VERIFY(rho > 1.0 && vp > 1.0 && vs > 1.0, "Incorrect media properties "
                "arrays");

    lambda_array[i]  = rho*(vp*vp - 2.*vs*vs);
    mu_array[i]      = rho*vs*vs;

    Rho[0] = std::min(Rho[0], rho);
    Rho[1] = std::max(Rho[1], rho);
    Vp[0]  = std::min(Vp[0], vp);
    Vp[1]  = std::max(Vp[1], vp);
    Vs[0]  = std::min(Vs[0], vs);
    Vs[1]  = std::max(Vs[1], vs);
    Lam[0] = std::min(Lam[0], lambda_array[i]);
    Lam[1] = std::max(Lam[1], lambda_array[i]);
    Mu[0]  = std::min(Mu[0], mu_array[i]);
    Mu[1]  = std::max(Mu[1], mu_array[i]);
  }

  std::cout << "Rho: min " << Rho[0] << " max " << Rho[1] << "\n";
  std::cout << "Vp: min "  << Vp[0]  << " max " << Vp[1] << "\n";
  std::cout << "Vs: min "  << Vs[0]  << " max " << Vs[1] << "\n";
  std::cout << "Lam: min " << Lam[0] << " max " << Lam[1] << "\n";
  std::cout << "Mu: min "  << Mu[0]  << " max " << Mu[1] << "\n";

  const bool own_array = false;
  CWConstCoefficient rho_coef(param.media.rho_array, own_array);
  CWConstCoefficient lambda_coef  (lambda_array, own_array);
  CWConstCoefficient mu_coef      (mu_array, own_array);
  CWConstCoefficient rho_damp_coef(param.media.rho_array, own_array);

  cout << "Stif matrix..." << flush;
  BilinearForm stif(&fespace);
  stif.AddDomainIntegrator(new ElasticityIntegrator(lambda_coef, mu_coef));
  stif.AddBdrFaceIntegrator(
     new DGElasticityIntegrator(lambda_coef, mu_coef,
                                param.method.dg_sigma, param.method.dg_kappa));
  stif.Assemble();
  stif.Finalize();
  const SparseMatrix& S = stif.SpMat();
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();

  cout << "Mass matrix..." << flush;
  BilinearForm mass(&fespace);
  mass.AddDomainIntegrator(new VectorMassIntegrator(rho_coef));
  mass.Assemble();
  mass.Finalize();
  const SparseMatrix& M = mass.SpMat();
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();

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

  cout << "Damp matrix..." << flush;
  BilinearForm dampM(&fespace);
  dampM.AddDomainIntegrator(new VectorMassIntegrator(rho_damp_coef));
  dampM.Assemble();
  dampM.Finalize();
  SparseMatrix& D = dampM.SpMat();
  double omega = 2.0*M_PI*param.source.frequency; // angular frequency
  D *= 0.5*param.dt*omega;
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();

  cout << "System matrix..." << flush;
  SparseMatrix SysMat(M);
  SysMat += D;
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

  const string name = method_name + param.extra_string;
  const string pref_path = (string)param.output_dir + "/" + SNAPSHOTS_DIR;
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
    D.Mult(u_2, y);

    // RHS = M*(2*u_1-u_2) - dt^2*(S*u_1-timeval*source) + D*u_2
    RHS += y;

    // (M+D)*x_0 = M*(2*x_1-x_2) - dt^2*(S*x_1-r*b) + D*x_2
    //for (int i = 0; i < N; ++i) u_0[i] = RHS[i] / (diagM[i]+diagD[i]);
    PCG(SysMat, Prec, RHS, u_0, 0, 200, 1e-12, 0.0);

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

    u_2 = u_1;
    u_1 = u_0;
  }

  time_loop_timer.Stop();

  delete[] seisU;
  delete[] seisV;

  cout << "Time loop is over\n\tpure time = " << time_loop_timer.UserTime()
       << "\n\ttime of snapshots = " << time_of_snapshots
       << "\n\ttime of seismograms = " << time_of_seismograms << endl;

  delete fec;
}

#ifdef MFEM_USE_MPI
void ElasticWave::run_DG_parallel()
{
  /*
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  const int dim = param.dimension;
  const int n_elements = param.mesh->GetNE();

  StopWatch chrono;

  chrono.Start();
  if (myid == 0)
    cout << "FE space generation..." << flush;
  FiniteElementCollection *fec = new H1_FECollection(param.order, dim);
  ParFiniteElementSpace fespace(param.pmesh, fec, dim); //, Ordering::byVDIM);
  if (myid == 0)
    cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  HYPRE_Int size = fespace.GlobalTrueVSize();
  if (myid == 0)
    cout << "Number of unknowns: " << size << endl;

  double *lambda_array = new double[n_elements];
  double *mu_array     = new double[n_elements];

  chrono.Clear();

  for (int i = 0; i < n_elements; ++i)
  {
    const double rho = param.media.rho_array[i];
    const double vp  = param.media.vp_array[i];
    const double vs  = param.media.vs_array[i];

    MFEM_VERIFY(rho > 1.0 && vp > 1.0 && vs > 1.0, "Incorrect media properties "
                "arrays");

    lambda_array[i]  = rho*(vp*vp - 2.*vs*vs);
    mu_array[i]      = rho*vs*vs;
  }

  const bool own_array = false;
  CWConstCoefficient rho_coef(param.media.rho_array, own_array);
  CWFunctionCoefficient lambda_coef  (stif_damp_weight, param, lambda_array);
  CWFunctionCoefficient mu_coef      (stif_damp_weight, param, mu_array);
  CWFunctionCoefficient rho_damp_coef(mass_damp_weight, param,
                                      param.media.rho_array, own_array);

  IntegrationRule segment_GLL;
  create_segment_GLL_rule(param.order, segment_GLL);
  IntegrationRule *GLL_rule = nullptr;
  if (dim == 2)
    GLL_rule = new IntegrationRule(segment_GLL, segment_GLL);
  else
    GLL_rule = new IntegrationRule(segment_GLL, segment_GLL, segment_GLL);

  if (myid == 0)
    cout << "Stif matrix..." << flush;
  chrono.Clear();
  ElasticityIntegrator *elast_int = new ElasticityIntegrator(lambda_coef, mu_coef);
  elast_int->SetIntRule(GLL_rule);
  ParBilinearForm stif(&fespace);
  stif.AddDomainIntegrator(elast_int);
  stif.Assemble();
  stif.Finalize();
  const HypreParMatrix *S = stif.ParallelAssemble();
  if (myid == 0)
    cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  if (myid == 0)
    cout << "Mass matrix..." << flush;
  chrono.Clear();
  VectorMassIntegrator *mass_int = new VectorMassIntegrator(rho_coef);
  mass_int->SetIntRule(GLL_rule);
  ParBilinearForm mass(&fespace);
  mass.AddDomainIntegrator(mass_int);
  mass.Assemble();
  mass.Finalize();
  const HypreParMatrix *M = mass.ParallelAssemble();
  if (myid == 0)
    cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  if (myid == 0)
    cout << "Damp matrix..." << flush;
  chrono.Clear();
  VectorMassIntegrator *damp_int = new VectorMassIntegrator(rho_damp_coef);
  damp_int->SetIntRule(GLL_rule);
  ParBilinearForm dampM(&fespace);
  dampM.AddDomainIntegrator(damp_int);
  dampM.Assemble();
  dampM.Finalize();
  HypreParMatrix *D = dampM.ParallelAssemble();
  double omega = 2.0*M_PI*param.source.frequency; // angular frequency
  (*D) *= 0.5*param.dt*omega;
  if (myid == 0)
    cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  if (myid == 0)
    cout << "RHS vector... " << flush;
  ParLinearForm b(&fespace);
  if (param.source.plane_wave)
  {
    PlaneWaveSource plane_wave_source(dim, param);
    VectorDomainLFIntegrator *plane_wave_int =
        new VectorDomainLFIntegrator(plane_wave_source);
    plane_wave_int->SetIntRule(GLL_rule);
    b.AddDomainIntegrator(plane_wave_int);
    b.Assemble();
  }
  else
  {
    if (!strcmp(param.source.type, "pointforce"))
    {
      VectorPointForce vector_point_force(dim, param);
      VectorDomainLFIntegrator *point_force_int =
          new VectorDomainLFIntegrator(vector_point_force);
      point_force_int->SetIntRule(GLL_rule);
      b.AddDomainIntegrator(point_force_int);
      b.Assemble();
    }
    else if (!strcmp(param.source.type, "momenttensor"))
    {
      MomentTensorSource momemt_tensor_source(dim, param);
      VectorDomainLFIntegrator *moment_tensor_int =
          new VectorDomainLFIntegrator(momemt_tensor_source);
      moment_tensor_int->SetIntRule(GLL_rule);
      b.AddDomainIntegrator(moment_tensor_int);
      b.Assemble();
    }
    else MFEM_ABORT("Unknown source type: " + string(param.source.type));
  }
  if (myid == 0)
  {
    cout << "||b||_L2 = " << b.Norml2() << endl;
    cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  }
  chrono.Clear();

  HypreParVector diagM; M->GetDiag(diagM); // mass matrix is diagonal
  HypreParVector diagD; D->GetDiag(diagD); // damping matrix is diagonal

  for (int i = 0; i < diagM.Size(); ++i)
  {
    MFEM_VERIFY(fabs(diagM[i]) > FLOAT_NUMBERS_EQUALITY_TOLERANCE,
                "There is a small (" + d2s(diagM[i]) + ") number (row "
                + d2s(i) + ") on the mass matrix diagonal");
  }

  const string method_name = "SEM_";

  if (myid == 0)
    cout << "Open seismograms files..." << flush;
  chrono.Clear();
  ofstream *seisU; // for displacement
  ofstream *seisV; // for velocity
  if (myid == 0)
  {
    open_seismo_outs(seisU, seisV, param, method_name);
    cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  }

  ParGridFunction u_0(&fespace); // displacement
  ParGridFunction u_1(&fespace);
  ParGridFunction u_2(&fespace);
  ParGridFunction v_1(&fespace); // velocity
  u_0 = 0.0;
  u_1 = 0.0;
  u_2 = 0.0;

  const int n_time_steps = param.T / param.dt + 0.5; // round to the nearest int
  const int tenth = 0.1 * n_time_steps;

  const string snapshot_filebase = (string)param.output_dir + "/" +
                                   SNAPSHOTS_DIR + method_name +
                                   param.extra_string;
  const int N = u_0.Size();

  if (myid == 0)
    cout << "N time steps = " << n_time_steps << "\nTime loop..." << endl;

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

  StopWatch time_loop_timer;
  time_loop_timer.Start();
  double time_of_snapshots = 0.;
  double time_of_seismograms = 0.;
  for (int time_step = 1; time_step <= n_time_steps; ++time_step)
  {
    HypreParVector y = u_1; y *= 2.0; y -= u_2;        // y = 2*u_1 - u_2

    HypreParVector z0; z0.SetSize(N);                  // z0 = M * (2*u_1 - u_2)
    for (int i = 0; i < N; ++i) z0[i] = diagM[i] * y[i];

    HypreParVector z1; z1.SetSize(N); S->Mult(u_1, z1);     // z1 = S * u_1
    HypreParVector z2 = b; z2 *= time_values[time_step-1]; // z2 = timeval*source

    // y = dt^2 * (S*u_1 - timeval*source), where it can be
    // y = dt^2 * (S*u_1 - ricker*pointforce) OR
    // y = dt^2 * (S*u_1 - gaussfirstderivative*momenttensor)
    y = z1; y -= z2; y *= param.dt*param.dt;

    // RHS = M*(2*u_1-u_2) - dt^2*(S*u_1-timeval*source)
    HypreParVector RHS = z0; RHS -= y;

    for (int i = 0; i < N; ++i) y[i] = diagD[i] * u_2[i]; // y = D * u_2

    // RHS = M*(2*u_1-u_2) - dt^2*(S*u_1-timeval*source) + D*u_2
    RHS += y;

    // (M+D)*x_0 = M*(2*x_1-x_2) - dt^2*(S*x_1-r*b) + D*x_2
    for (int i = 0; i < N; ++i) u_0[i] = RHS[i] / (diagM[i]+diagD[i]);

    // velocity: v = du/dt, we use the central difference here
    v_1  = u_0;
    v_1 -= u_2;
    v_1 /= 2.0*param.dt;

    // Compute and print the L^2 norm of the error
    if (time_step % tenth == 0) {
      cout << "step " << time_step << " / " << n_time_steps
           << " ||solution||_{L^2} = " << u_0.Norml2() << endl;
    }

//    if (time_step % param.step_snap == 0) {
//      StopWatch timer;
//      timer.Start();
//      if (myid == 0)
//        output_snapshots(time_step, snapshot_filebase, param, u_0, v_1, mesh);
//      timer.Stop();
//      time_of_snapshots += timer.UserTime();
//    }

//    if (time_step % param.step_seis == 0) {
//      StopWatch timer;
//      timer.Start();
//      if (myid == 0)
//        output_seismograms(param, mesh, u_0, v_1, seisU, seisV);
//      timer.Stop();
//      time_of_seismograms += timer.UserTime();
//    }

    u_2 = u_1;
    u_1 = u_0;
  }

  time_loop_timer.Stop();

  delete[] seisU;
  delete[] seisV;

  if (myid == 0)
  {
    cout << "Time loop is over\n\tpure time = " << time_loop_timer.UserTime()
         << "\n\ttime of snapshots = " << time_of_snapshots
         << "\n\ttime of seismograms = " << time_of_seismograms << endl;
  }

  delete M;
  delete S;
  delete pmesh;
  delete fec;
  */
}
#endif // MFEM_USE_MPI

