#include "elastic_wave.hpp"
#include "parameters.hpp"
#include "utilities.hpp"

using namespace std;
using namespace mfem;

//#define VIEW_SNAPSHOT_SPACE
//#define VIEW_BOUNDARY_BASIS
//#define VIEW_INTERIOR_BASIS
//#define VIEW_DG_BASIS


static
void compute_boundary_basis_CG(ostream &log, const Parameters &param, Mesh *fine_mesh,
                               int n_boundary_bf, int n_interior_bf,
                               Coefficient &rho_coef,
                               Coefficient &lambda_coef,
                               Coefficient &mu_coef,
                               DenseMatrix &R)
{
  StopWatch chrono;
  chrono.Start();

  log << "FE space generation..." << flush;
  H1_FECollection fec(param.method.order, param.dimension);
  FiniteElementSpace fespace(fine_mesh, &fec, param.dimension);
  log << "done. Time = " << chrono.RealTime() << " sec" << endl;

  chrono.Clear();
  log << "Stif matrix..." << flush;
  BilinearForm stif(&fespace);
  stif.AddDomainIntegrator(new ElasticityIntegrator(lambda_coef, mu_coef));
  stif.Assemble();
  log << "done. Time = " << chrono.RealTime() << " sec" << endl;

  Array<int> ess_bdr;
  Array<int> ess_tdof_list;
  if (fine_mesh->bdr_attributes.Size())
  {
    ess_bdr.SetSize(fine_mesh->bdr_attributes.Max());
    ess_bdr = 1;
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }

  chrono.Clear();
  log << "Snapshot matrix..." << flush;
  DenseMatrix W(fespace.GetVSize(), ess_tdof_list.Size());
  {

    Vector b(fespace.GetVSize()); // RHS (it's always 0 in the loop)

    const int maxiter = 1000;
    const double rtol = 1e-12;
    const double atol = 1e-24;
    for (int bd = 0; bd < ess_tdof_list.Size(); ++bd)
    {
      Vector x;
      W.GetColumnReference(bd, x);
      x = 0.;
      x(ess_tdof_list[bd]) = 1.;
      b = 0.;

      mfem::SparseMatrix A; // mat after eliminating b.c.
      mfem::Vector X, B;
      stif.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      mfem::GSSmoother precond(A);
      mfem::PCG(A, precond, B, X, 0, maxiter, rtol, atol);

      stif.RecoverFEMSolution(X, b, x);
    }

    if (param.output.view_snapshot_space)
    {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mode_sock(vishost, visport);
      mode_sock.precision(8);
      for (int bd = 0; bd < ess_tdof_list.Size(); ++bd) {
        Vector x;
        W.GetColumn(bd, x);
        GridFunction X;
        X.MakeRef(&fespace, x, 0);
        mode_sock << "solution\n" << *fine_mesh << X
                  << "window_title 'Snapshot " << bd+1 << '/' << ess_tdof_list.Size()
                  << "'" << endl;

        char c;
        cout << "press (q)uit or (c)ontinue --> " << flush;
        cin >> c;
        if (c != 'c')
          break;
      }
      mode_sock.close();
    }
  }
  log << "done. Time = " << chrono.RealTime() << " sec" << endl;

  chrono.Clear();
  log << "Sfull matrix..." << flush;
  const SparseMatrix &S = stif.SpMat();
  const SparseMatrix &Se= stif.SpMatElim();
  SparseMatrix *Sfull = Add(S, Se);
  log << "done. Time = " << chrono.RealTime() << " sec" << endl;

  chrono.Clear();
  log << "WTSW matrix..." << flush;
  const DenseMatrix *WTSW = RAP(*Sfull, W);
  delete Sfull;
  log << "done. Time = " << chrono.RealTime() << " sec" << endl;

  chrono.Clear();
  log << "Edge mass matrix..." << flush;
  BilinearForm edge_mass(&fespace);
  edge_mass.AddBoundaryIntegrator(new VectorMassIntegrator(rho_coef));
  edge_mass.Assemble();
  edge_mass.Finalize();
  log << "done. Time = " << chrono.RealTime() << " sec" << endl;

  chrono.Clear();
  log << "WTEMW matrix..." << flush;
  const SparseMatrix &EM = edge_mass.SpMat();
  const DenseMatrix *WTEMW = RAP(EM, W);
  log << "done. Time = " << chrono.RealTime() << " sec" << endl;

  chrono.Clear();
  log << "Solving the eigenproblem..." << endl;
  DenseMatrix eigenvectors(WTSW->Height(), WTSW->Width());
  solve_dsygvd(*WTSW, *WTEMW, eigenvectors);
  log << "done. Time = " << chrono.RealTime() << " sec" << endl;

  DenseMatrix selected_eigenvectors(eigenvectors.Height(), n_boundary_bf);
  selected_eigenvectors.CopyCols(eigenvectors, 0, n_boundary_bf-1);

  DenseMatrix boundary_basis(fespace.GetVSize(), n_boundary_bf);
  Mult(W, selected_eigenvectors, boundary_basis);

  R.SetSize(fespace.GetVSize(), n_boundary_bf + n_interior_bf);
  for (int col = 0; col < n_boundary_bf; ++col)
  {
    for (int row = 0; row < boundary_basis.Height(); ++row)
      R(row, col) = boundary_basis(row, col);
  }

  delete WTEMW;
  delete WTSW;

  if (param.output.view_boundary_basis)
  {
    char vishost[] = "localhost";
    int  visport   = 19916;
    socketstream mode_sock(vishost, visport);
    mode_sock.precision(8);
    for (int bf = 0; bf < n_boundary_bf; ++bf) {
      Vector x;
      R.GetColumn(bf, x);
      GridFunction X;
      X.MakeRef(&fespace, x, 0);
      mode_sock << "solution\n" << *fine_mesh << X
                << "window_title 'CG boundary basis " << bf+1 << '/'
                << n_boundary_bf << "'" << endl;

      char c;
      cout << "press (q)uit or (c)ontinue --> " << flush;
      cin >> c;
      if (c != 'c')
        break;
    }
    mode_sock.close();
  }
}



static
void compute_interior_basis_CG(ostream &log, const Parameters &param, Mesh *fine_mesh,
                               int n_boundary_bf, int n_interior_bf,
                               Coefficient &rho_coef,
                               Coefficient &lambda_coef,
                               Coefficient &mu_coef,
                               DenseMatrix &R)
{
  StopWatch chrono;
  chrono.Start();

  ParMesh par_fine_mesh(MPI_COMM_SELF, *fine_mesh);

  log << "Parallel FE space generation..." << flush;
  H1_FECollection fec(param.method.order, param.dimension);
  ParFiniteElementSpace par_fespace(&par_fine_mesh, &fec, param.dimension);
  log << "done. Time = " << chrono.RealTime() << " sec" << endl;

  Array<int> ess_bdr;
  if (fine_mesh->bdr_attributes.Size())
  {
    ess_bdr.SetSize(fine_mesh->bdr_attributes.Max());
    ess_bdr = 1;
  }

  chrono.Clear();
  log << "Par Stif matrix..." << flush;
  ParBilinearForm par_stif(&par_fespace);
  par_stif.AddDomainIntegrator(new ElasticityIntegrator(lambda_coef, mu_coef));
  par_stif.Assemble();
  par_stif.EliminateEssentialBCDiag(ess_bdr, 1.0);
  par_stif.Finalize();
  HypreParMatrix *par_S = par_stif.ParallelAssemble();
  log << "done. Time = " << chrono.RealTime() << " sec" << endl;

  chrono.Clear();
  log << "Par Mass matrix..." << flush;
  ParBilinearForm par_mass(&par_fespace);
  par_mass.AddDomainIntegrator(new VectorMassIntegrator(rho_coef));
  par_mass.Assemble();
  par_mass.EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
  par_mass.Finalize();
  HypreParMatrix *par_M = par_mass.ParallelAssemble();
  log << "done. Time = " << chrono.RealTime() << " sec" << endl;

  HypreBoomerAMG amg(*par_S);
  amg.SetPrintLevel(0);

  HypreLOBPCG lobpcg(MPI_COMM_SELF);
  lobpcg.SetNumModes(n_interior_bf);
  lobpcg.SetPreconditioner(amg);
  lobpcg.SetMaxIter(100);
  lobpcg.SetTol(1e-8);
  lobpcg.SetPrecondUsageMode(1);
  lobpcg.SetPrintLevel(0);
  lobpcg.SetMassMatrix(*par_M);
  lobpcg.SetOperator(*par_S);

  chrono.Clear();
  log << "Solving the eigenproblem..." << endl;
//  Array<double> eigenvalues;
  lobpcg.Solve();
//  lobpcg.GetEigenvalues(eigenvalues);
  log << "done. Time = " << chrono.RealTime() << " sec" << endl;

  for (int i = 0; i < n_interior_bf; ++i)
  {
    Vector x;
    R.GetColumnReference(n_boundary_bf + i, x);
    x = lobpcg.GetEigenvector(i);
  }

  delete par_M;
  delete par_S;

  if (param.output.view_interior_basis)
  {
    char vishost[] = "localhost";
    int  visport   = 19916;
    socketstream mode_sock(vishost, visport);
    mode_sock.precision(8);
    for (int bf = 0; bf < n_interior_bf; ++bf) {
      Vector x;
      R.GetColumn(n_boundary_bf + bf, x);
      GridFunction X;
      X.MakeRef(&par_fespace, x, 0);
      mode_sock << "solution\n" << par_fine_mesh << X
                << "window_title 'CG interior basis " << bf+1 << '/'
                << n_interior_bf << "'" << endl;

      char c;
      cout << "press (q)uit or (c)ontinue --> " << flush;
      cin >> c;
      if (c != 'c')
        break;
    }
    mode_sock.close();
  }
}



static
void project_to_DG_space(ostream &log, const Parameters &param, Mesh *fine_mesh,
                         const DenseMatrix &R_CG, DenseMatrix &R_DG)
{
  StopWatch chrono;
  chrono.Start();

  log << "FE space generation..." << flush;
  H1_FECollection CG_fec(param.method.order, param.dimension);
  DG_FECollection DG_fec(param.method.order, param.dimension);

  FiniteElementSpace CG_fespace(fine_mesh, &CG_fec, param.dimension);
  FiniteElementSpace DG_fespace(fine_mesh, &DG_fec, param.dimension);
  log << "done. Time = " << chrono.RealTime() << " sec" << endl;

  R_DG.SetSize(DG_fespace.GetVSize(), R_CG.NumCols());

  chrono.Clear();
  log << "Project to DG space..." << flush;
  Vector x, y;
  GridFunction vec_CG;
  for (int c = 0; c < R_CG.NumCols(); ++c)
  {
    R_CG.GetColumn(c, x);
    vec_CG.MakeRef(&CG_fespace, x, 0);
    VectorGridFunctionCoefficient grid_coef(&vec_CG);

    GridFunction vec_DG(&DG_fespace);
    vec_DG.ProjectCoefficient(grid_coef);

    R_DG.GetColumnReference(c, y);
    y = vec_DG;
  }
  log << "done. Time = " << chrono.RealTime() << " sec" << endl;

  if (param.output.view_dg_basis)
  {
    char vishost[] = "localhost";
    int  visport   = 19916;
    socketstream mode_sock(vishost, visport);
    mode_sock.precision(8);
    for (int col = 0; col < R_DG.NumCols(); ++col) {
      Vector x;
      R_DG.GetColumn(col, x);
      GridFunction X;
      X.MakeRef(&DG_fespace, x, 0);
      mode_sock << "solution\n" << *fine_mesh << X
                << "window_title 'R_DG column " << col+1 << '/'
                << R_DG.NumCols() << "'" << endl;

      char c;
      cout << "press (q)uit or (c)ontinue --> " << flush;
      cin >> c;
      if (c != 'c')
        break;
    }
    mode_sock.close();
  }
}



void ElasticWave::
compute_basis_CG(ostream &log, Mesh *fine_mesh, int n_boundary_bf, int n_interior_bf,
                 Coefficient &rho_coef, Coefficient &lambda_coef,
                 Coefficient &mu_coef, DenseMatrix &R) const
{
  DenseMatrix R_CG;
  compute_boundary_basis_CG(log, param, fine_mesh, n_boundary_bf, n_interior_bf,
                            rho_coef, lambda_coef, mu_coef, R_CG);
  compute_interior_basis_CG(log, param, fine_mesh, n_boundary_bf, n_interior_bf,
                            rho_coef, lambda_coef, mu_coef, R_CG);
  project_to_DG_space(log, param, fine_mesh, R_CG, R);
}


