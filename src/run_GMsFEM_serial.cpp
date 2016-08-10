#include "elastic_wave.hpp"
#include "parameters.hpp"
#include "utilities.hpp"

#include <float.h>

#ifdef MFEM_USE_MPI

using namespace std;
using namespace mfem;



static void time_step(const SparseMatrix &M, const SparseMatrix &S,
                      const Vector &b, double timeval, double dt,
                      const SparseMatrix &SysMat, Solver &Prec,
                      Vector &U_0, Vector &U_1, Vector &U_2)
{
  Vector y = U_1; y *= 2.0; y -= U_2;        // y = 2*u_1 - u_2

  Vector z0; z0.SetSize(U_0.Size());         // z0 = M * (2*u_1 - u_2)
  M.Mult(y, z0);

  Vector z1; z1.SetSize(U_0.Size()); S.Mult(U_1, z1); // z1 = S * u_1
  Vector z2 = b; z2 *= timeval; // z2 = timeval*source

  // y = dt^2 * (S*u_1 - timeval*source), where it can be
  // y = dt^2 * (S*u_1 - ricker*pointforce) OR
  // y = dt^2 * (S*u_1 - gaussfirstderivative*momenttensor)
  y = z1; y -= z2; y *= dt*dt;

  // RHS = M*(2*u_1-u_2) - dt^2*(S*u_1-timeval*source)
  Vector RHS = z0; RHS -= y;

//    for (int i = 0; i < N; ++i) y[i] = diagD[i] * u_2[i]; // y = D * u_2

  // RHS = M*(2*u_1-u_2) - dt^2*(S*u_1-timeval*source) + D*u_2
//    RHS += y;

  // (M+D)*x_0 = M*(2*x_1-x_2) - dt^2*(S*x_1-r*b) + D*x_2
  PCG(SysMat, Prec, RHS, U_0, 0, 200, 1e-12, 0.0);

  U_2 = U_1;
  U_1 = U_0;
}



void ElasticWave::run_GMsFEM_serial() const
{
  MFEM_VERIFY(param.mesh, "The mesh is not initialized");

  StopWatch chrono;
  chrono.Start();

  const int dim = param.dimension;

  cout << "FE space generation..." << flush;
  FiniteElementCollection *fec = new DG_FECollection(param.method.order, dim);
  FiniteElementSpace fespace(param.mesh, fec, dim);
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  cout << "Number of unknowns: " << fespace.GetVSize() << endl;

  CWConstCoefficient rho_coef(param.media.rho_array, false);
  CWConstCoefficient lambda_coef(param.media.lambda_array, false);
  CWConstCoefficient mu_coef(param.media.mu_array, false);

  cout << "Fine scale stif matrix..." << flush;
  chrono.Clear();
  BilinearForm stif_fine(&fespace);
  stif_fine.AddDomainIntegrator(new ElasticityIntegrator(lambda_coef, mu_coef));
  stif_fine.AddInteriorFaceIntegrator(
     new DGElasticityIntegrator(lambda_coef, mu_coef,
                                param.method.dg_sigma, param.method.dg_kappa));
  stif_fine.AddBdrFaceIntegrator(
     new DGElasticityIntegrator(lambda_coef, mu_coef,
                                param.method.dg_sigma, param.method.dg_kappa));
  stif_fine.Assemble();
  stif_fine.Finalize();
  const SparseMatrix& S_fine = stif_fine.SpMat();
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  cout << "Fine scale mass matrix..." << flush;
  chrono.Clear();
  BilinearForm mass_fine(&fespace);
  mass_fine.AddDomainIntegrator(new VectorMassIntegrator(rho_coef));
  mass_fine.Assemble();
  mass_fine.Finalize();
  const SparseMatrix& M_fine = mass_fine.SpMat();
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

//  cout << "Damp matrix..." << flush;
//  VectorMassIntegrator *damp_int = new VectorMassIntegrator(rho_damp_coef);
//  damp_int->SetIntRule(GLL_rule);
//  BilinearForm dampM(&fespace);
//  dampM.AddDomainIntegrator(damp_int);
//  dampM.Assemble();
//  dampM.Finalize();
//  SparseMatrix& D = dampM.SpMat();
//  double omega = 2.0*M_PI*param.source.frequency; // angular frequency
//  D *= 0.5*param.dt*omega;
//  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
//  chrono.Clear();

  SparseMatrix SysFine(M_fine);
  //SysFine = 0.0;
  //SysFine += D;
  //SysFine += M_fine;
  GSSmoother PrecFine(SysFine);

  cout << "Fine scale RHS vector... " << flush;
  chrono.Clear();
  LinearForm b_fine(&fespace);
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
  cout << "||b_h||_L2 = " << b_fine.Norml2() << endl;
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;

  vector<vector<int> > map_cell_dofs(fespace.GetNE());
  for (int el = 0; el < fespace.GetNE(); ++el)
  {
    Array<int> dofs;
    fespace.GetElementVDofs(el, dofs);
    map_cell_dofs[el].resize(dofs.Size());
    for (int i = 0; i < dofs.Size(); ++i)
      map_cell_dofs[el][i] = dofs[i];
  }

  vector<DenseMatrix> R;
  vector<vector<int> > local2global;
  compute_R_matrices(cout, map_cell_dofs, local2global, R);

  // global sparse R matrix
  int n_rows = 0;
  int n_cols = 0;
  int n_non_zero = 0;
  for (size_t i = 0; i < R.size(); ++i)
  {
    const int h = R[i].Height();
    const int w = R[i].Width();
    n_rows += w; // transpose
    n_cols += h; // transpose
    n_non_zero += h * w;
  }
  MFEM_VERIFY(n_cols == S_fine.Height(), "Dimensions mismatch");
  int *Ri = new int[n_rows + 1];
  int *Rj = new int[n_non_zero];
  double *Rdata = new double[n_non_zero];

  Ri[0] = 0;
  int k = 0;
  int p = 0;
//  int offset = 0;
  for (size_t r = 0; r < R.size(); ++r)
  {
    const int h = R[r].Height();
    const int w = R[r].Width();
    for (int i = 0; i < w; ++i)
    {
      Ri[k+1] = Ri[k] + h;
      ++k;

      for (int j = 0; j < h; ++j)
      {
        Rj[p] = local2global[r][j];
        Rdata[p] = R[r](j, i);
        ++p;
      }
    }
//    offset += h;
  }

  SparseMatrix R_global(Ri, Rj, Rdata, n_rows, n_cols);

  SparseMatrix *R_global_T = Transpose(R_global);

  SparseMatrix *M_coarse = RAP(M_fine, R_global);
  SparseMatrix *S_coarse = RAP(S_fine, R_global);

  Vector b_coarse(M_coarse->Height());
  R_global.Mult(b_fine, b_coarse);

  SparseMatrix SysCoarse(*M_coarse);
  //SysCoarse = 0.0;
  //SysCoarse += D;
  //SysCoarse += *M_coarse;
  GSSmoother PrecCoarse(SysCoarse);

  if (param.output.print_matrices)
  {
    {
      chrono.Clear();
      cout << "Output R local matrices..." << flush;
      for (size_t r = 0; r < R.size(); ++r) {
        const string fname = string(param.output.directory) + "/r" + d2s(r) + "_local_ser.dat";
        ofstream mout(fname.c_str());
        MFEM_VERIFY(mout, "Cannot open file " + fname);
        R[r].PrintMatlab(mout);
      }
      cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
    }
    {
      chrono.Clear();
      cout << "Output R_global matrix..." << flush;
      const string fname = string(param.output.directory) + "/r_global_ser.dat";
      ofstream mout(fname.c_str());
      MFEM_VERIFY(mout, "Cannot open file " + fname);
      R_global.PrintMatlab(mout);
      cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
    }
    {
      chrono.Clear();
      cout << "Output R_global_T matrix..." << flush;
      const string fname = string(param.output.directory) + "/r_global_t_ser.dat";
      ofstream mout(fname.c_str());
      MFEM_VERIFY(mout, "Cannot open file " + fname);
      R_global_T->PrintMatlab(mout);
      cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
    }
    {
      chrono.Clear();
      cout << "Output M_fine matrix..." << flush;
      const string fname = string(param.output.directory) + "/m_fine_ser.dat";
      ofstream mout(fname.c_str());
      MFEM_VERIFY(mout, "Cannot open file " + fname);
      M_fine.PrintMatlab(mout);
      cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
    }
    {
      chrono.Clear();
      cout << "Output S_fine matrix..." << flush;
      const string fname = string(param.output.directory) + "/s_fine_ser.dat";
      ofstream mout(fname.c_str());
      MFEM_VERIFY(mout, "Cannot open file " + fname);
      S_fine.PrintMatlab(mout);
      cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
    }
    {
      chrono.Clear();
      cout << "Output M_coarse matrix..." << flush;
      const string fname = string(param.output.directory) + "/m_coarse_ser.dat";
      ofstream mout(fname.c_str());
      MFEM_VERIFY(mout, "Cannot open file " + fname);
      M_coarse->PrintMatlab(mout);
      cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
    }
    {
      chrono.Clear();
      cout << "Output S_coarse matrix..." << flush;
      const string fname = string(param.output.directory) + "/s_coarse_ser.dat";
      ofstream mout(fname.c_str());
      MFEM_VERIFY(mout, "Cannot open file " + fname);
      S_coarse->PrintMatlab(mout);
      cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
    }
  }

  const string method_name = "GMsFEM_";

  cout << "Open seismograms files..." << flush;
  ofstream *seisU; // for displacement
  ofstream *seisV; // for velocity
  open_seismo_outs(seisU, seisV, param, method_name);
  cout << "done. Time = " << chrono.RealTime() << " sec" << endl;
  chrono.Clear();

  Vector U_0(M_coarse->Height()); // coarse scale displacement
  U_0 = 0.0;
  Vector U_1 = U_0;
  Vector U_2 = U_0;
  GridFunction u_0(&fespace); // fine scale projected from coarse scale

  GridFunction u_fine_0(&fespace); // fine scale displacement
  u_fine_0 = 0.0;
  Vector u_fine_1 = u_fine_0;
  Vector u_fine_2 = u_fine_0;

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
  VisItDataCollection visit_dc(name.c_str(), param.mesh);
  visit_dc.SetPrefixPath(pref_path.c_str());
  visit_dc.RegisterField("fine_displacement", &u_fine_0);
  visit_dc.RegisterField("coarse_displacement", &u_0);
  {
    visit_dc.SetCycle(0);
    visit_dc.SetTime(0.0);
    Vector u_tmp(u_fine_0.Size());
    R_global_T->Mult(U_0, u_tmp); // USE MultTranspose and DON'T USE R_global_T !!!!
    u_0.MakeRef(&fespace, u_tmp, 0);
    visit_dc.Save();
  }

  StopWatch time_loop_timer;
  time_loop_timer.Start();
  double time_of_snapshots = 0.;
  double time_of_seismograms = 0.;
  for (int t_step = 1; t_step <= n_time_steps; ++t_step)
  {
    {
      time_step(*M_coarse, *S_coarse, b_coarse, time_values[t_step-1],
                param.dt, SysCoarse, PrecCoarse, U_0, U_1, U_2);
    }
    {
//      time_step(M_fine, S_fine, b_fine, time_values[t_step-1],
//                param.dt, SysFine, PrecFine, u_fine_0, u_fine_1, u_fine_2);
    }

    // Compute and print the L^2 norm of the error
    if (t_step % tenth == 0) {
      cout << "step " << t_step << " / " << n_time_steps
           << " ||U||_{L^2} = " << U_0.Norml2()
           /*<< " ||u||_{L^2} = " << u_fine_0.Norml2()*/ << endl;
    }

    if (t_step % param.step_snap == 0) {
      StopWatch timer;
      timer.Start();
      visit_dc.SetCycle(t_step);
      visit_dc.SetTime(t_step*param.dt);
      Vector u_tmp(u_fine_0.Size());
      R_global_T->Mult(U_0, u_tmp); // USE MultTranspose and DON'T USE R_global_T !!!!
      u_0.MakeRef(&fespace, u_tmp, 0);
      visit_dc.Save();
      timer.Stop();
      time_of_snapshots += timer.UserTime();
    }

//    if (t_step % param.step_seis == 0) {
//      StopWatch timer;
//      timer.Start();
//      R_global_T.Mult(U_0, u_0); // USE MultTranspose and DON'T USE R_global_T !!!!
//      output_seismograms(param, *param.mesh, u_0, seisU);
//      timer.Stop();
//      time_of_seismograms += timer.UserTime();
//    }

    if (param.output.serial_solution && t_step % param.step_seis == 0)
    {
      Vector u_tmp(u_fine_0.Size());
      R_global_T->Mult(U_0, u_tmp);

      const string fname = string(param.output.directory) + "/" +
                           param.output.extra_string + "_sol_t" + d2s(t_step) +
                           ".bin";
      ofstream bin_sol(fname.c_str(), std::ios::binary);
      MFEM_VERIFY(bin_sol, "Cannot open file " << fname);
      for (int i = 0; i < u_tmp.Size(); ++i) {
        float val = u_tmp(i);
        bin_sol.write(reinterpret_cast<char*>(&val), sizeof(val));
      }
    }
  }

  time_loop_timer.Stop();

  delete[] seisU;
  delete[] seisV;

  cout << "Time loop is over\n\tpure time = " << time_loop_timer.UserTime()
       << "\n\ttime of snapshots = " << time_of_snapshots
       << "\n\ttime of seismograms = " << time_of_seismograms << endl;

  delete S_coarse;
  delete M_coarse;
  delete R_global_T;

  delete fec;
}


#endif // MFEM_USE_MPI
