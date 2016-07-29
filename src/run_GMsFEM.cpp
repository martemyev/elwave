#include "elastic_wave.hpp"
#include "parameters.hpp"
#include "utilities.hpp"

#include <float.h>

#ifdef MFEM_USE_MPI

using namespace std;
using namespace mfem;



static void fill_up_n_fine_cells_per_coarse(int n_fine, int n_coarse,
                                     std::vector<int> &result)
{
  const int k = n_fine / n_coarse;
  for (size_t i = 0; i < result.size(); ++i)
    result[i] = k;
  const int p = n_fine % n_coarse;
  for (int i = 0; i < p; ++i)
    ++result[i];
}



void ElasticWave::compute_R_matrices(ostream &out,
                                     const vector<vector<int> > &map_cell_dofs,
                                     vector<vector<int> > &local2global,
                                     vector<DenseMatrix> &R) const
{
  int myid, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  std::vector<int> n_fine_cell_per_coarse_x(param.method.gms_Nx);
  fill_up_n_fine_cells_per_coarse(param.grid.nx, param.method.gms_Nx,
                                  n_fine_cell_per_coarse_x);

  std::vector<int> n_fine_cell_per_coarse_y(param.method.gms_Ny);
  fill_up_n_fine_cells_per_coarse(param.grid.ny, param.method.gms_Ny,
                                  n_fine_cell_per_coarse_y);

  const double hx = param.grid.get_hx();
  const double hy = param.grid.get_hy();

  int n_coarse_cells = param.method.gms_Nx * param.method.gms_Ny;
  if (param.dimension == 3)
    n_coarse_cells *= param.method.gms_Nz;

  // number of coarse cells that all processes will have (at least)
  const int min_n_cells = n_coarse_cells / nproc;
  // number of coarse cells that should be distributed among some processes
  const int extra_cells = n_coarse_cells % nproc;
  // first and last (not including) indices of coarse element for the current
  // 'myid' process
  const int my_start_cell = min_n_cells * myid + (extra_cells < myid ? extra_cells : myid);
  const int my_end_cell   = my_start_cell + min_n_cells + (extra_cells > myid);

  out << "coarse cells: start " << my_start_cell << " end " << my_end_cell << endl;

  local2global.resize(my_end_cell - my_start_cell);
  R.resize(my_end_cell - my_start_cell);

  const int gen_edges = 1;

  if (param.dimension == 2)
  {
    int offset_x, offset_y = 0;

    for (int iy = 0; iy < param.method.gms_Ny; ++iy)
    {
      const int n_fine_y = n_fine_cell_per_coarse_y[iy];
      const double SY = n_fine_y * hy;

      offset_x = 0;
      for (int ix = 0; ix < param.method.gms_Nx; ++ix)
      {
        const int n_fine_x = n_fine_cell_per_coarse_x[ix];
        const double SX = n_fine_x * hx;

        const int global_coarse_cell = iy*param.method.gms_Nx + ix;
        if (global_coarse_cell < my_start_cell || global_coarse_cell >= my_end_cell)
        {
          offset_x += n_fine_x;
          continue;
        }
        const int my_coarse_cell = global_coarse_cell - my_start_cell;
        out << "\nglobal_coarse_cell " << global_coarse_cell
            << " my_coarse_cell " << my_coarse_cell << endl;

        Mesh *ccell_fine_mesh =
            new Mesh(n_fine_x, n_fine_y, Element::QUADRILATERAL, gen_edges, SX, SY);

        double *local_rho    = new double[n_fine_x * n_fine_y];
        double *local_lambda = new double[n_fine_x * n_fine_y];
        double *local_mu     = new double[n_fine_x * n_fine_y];
        for (int fiy = 0; fiy < n_fine_y; ++fiy)
        {
          for (int fix = 0; fix < n_fine_x; ++fix)
          {
            const int loc_cell = fiy*n_fine_x + fix;
            const int glob_cell = (offset_y + fiy) * param.grid.nx +
                                  (offset_x + fix);

            local_rho[loc_cell]    = param.media.rho_array[glob_cell];
            local_lambda[loc_cell] = param.media.lambda_array[glob_cell];
            local_mu[loc_cell]     = param.media.mu_array[glob_cell];
          }
        }

        CWConstCoefficient local_rho_coef(local_rho, true);
        CWConstCoefficient local_lambda_coef(local_lambda, true);
        CWConstCoefficient local_mu_coef(local_mu, true);

        compute_basis_CG(out, ccell_fine_mesh, param.method.gms_nb, param.method.gms_ni,
                         local_rho_coef, local_lambda_coef, local_mu_coef,
                         R[my_coarse_cell]);

        // initialize with all -1 to check that all values are defined later
        local2global[my_coarse_cell].resize(R[my_coarse_cell].Height(), -1);
        DG_FECollection DG_fec(param.method.order, param.dimension);
        FiniteElementSpace DG_fespace(ccell_fine_mesh, &DG_fec, param.dimension);
        Array<int> loc_dofs;
        for (int fiy = 0; fiy < n_fine_y; ++fiy)
        {
          for (int fix = 0; fix < n_fine_x; ++fix)
          {
            const int loc_cell = fiy*n_fine_x + fix;
            const int glob_cell = (offset_y + fiy) * param.grid.nx +
                                  (offset_x + fix);
            MFEM_ASSERT(glob_cell >= 0 && glob_cell < (int)map_cell_dofs.size(),
                        "glob_cell is out of range");

            DG_fespace.GetElementVDofs(loc_cell, loc_dofs);
            const vector<int> &glob_dofs = map_cell_dofs[glob_cell];
            MFEM_ASSERT(loc_dofs.Size() == (int)glob_dofs.size(), "Dimensions mismatch");

            for (int di = 0; di < loc_dofs.Size(); ++di)
              local2global[my_coarse_cell][loc_dofs[di]] = glob_dofs[di];
          }
        }

        // check that all values were defined
        for (size_t ii = 0; ii < local2global[my_coarse_cell].size(); ++ii) {
          MFEM_VERIFY(local2global[my_coarse_cell][ii] >= 0, "Some values of "
                      "local2global vector were not defined");
        }

        delete ccell_fine_mesh;

        offset_x += n_fine_x;
      }
      offset_y += n_fine_y;
    }
  }
  else // 3D
  {
    std::vector<int> n_fine_cell_per_coarse_z(param.method.gms_Nz);
    fill_up_n_fine_cells_per_coarse(param.grid.nz, param.method.gms_Nz,
                                    n_fine_cell_per_coarse_z);

    const double hz = param.grid.get_hz();

    int offset_x, offset_y, offset_z = 0;

    for (int iz = 0; iz < param.method.gms_Nz; ++iz)
    {
      const int n_fine_z = n_fine_cell_per_coarse_z[iz];
      const double SZ = n_fine_z * hz;

      offset_y = 0;
      for (int iy = 0; iy < param.method.gms_Ny; ++iy)
      {
        const int n_fine_y = n_fine_cell_per_coarse_y[iy];
        const double SY = n_fine_y * hy;

        offset_x = 0;
        for (int ix = 0; ix < param.method.gms_Nx; ++ix)
        {
          const int n_fine_x = n_fine_cell_per_coarse_x[ix];
          const double SX = n_fine_x * hx;

          const int global_coarse_cell = iz*param.method.gms_Nx*param.method.gms_Ny +
                                         iy*param.method.gms_Nx + ix;
          if (global_coarse_cell < my_start_cell || global_coarse_cell >= my_end_cell)
          {
            offset_x += n_fine_x;
            continue;
          }
          const int my_coarse_cell = global_coarse_cell - my_start_cell;
          out << "\nglobal_coarse_cell " << global_coarse_cell
              << " my_coarse_cell " << my_coarse_cell << endl;

          Mesh *ccell_fine_mesh =
              new Mesh(n_fine_cell_per_coarse_x[ix],
                       n_fine_cell_per_coarse_y[iy],
                       n_fine_cell_per_coarse_z[iz],
                       Element::HEXAHEDRON, gen_edges, SX, SY, SZ);

          double *local_rho    = new double[n_fine_x * n_fine_y * n_fine_z];
          double *local_lambda = new double[n_fine_x * n_fine_y * n_fine_z];
          double *local_mu     = new double[n_fine_x * n_fine_y * n_fine_z];
          for (int fiz = 0; fiz < n_fine_z; ++fiz)
          {
            for (int fiy = 0; fiy < n_fine_y; ++fiy)
            {
              for (int fix = 0; fix < n_fine_x; ++fix)
              {
                const int loc_cell = fiz*n_fine_x*n_fine_y + fiy*n_fine_x + fix;
                const int glob_cell = (offset_z + fiz) * param.grid.nx * param.grid.ny +
                                      (offset_y + fiy) * param.grid.nx +
                                      (offset_x + fix);

                local_rho[loc_cell]    = param.media.rho_array[glob_cell];
                local_lambda[loc_cell] = param.media.lambda_array[glob_cell];
                local_mu[loc_cell]     = param.media.mu_array[glob_cell];
              }
            }
          }

          CWConstCoefficient local_rho_coef(local_rho, true);
          CWConstCoefficient local_lambda_coef(local_lambda, true);
          CWConstCoefficient local_mu_coef(local_mu, true);

          compute_basis_CG(cout, ccell_fine_mesh, param.method.gms_nb, param.method.gms_ni,
                           local_rho_coef, local_lambda_coef, local_mu_coef,
                           R[my_coarse_cell]);

          // initialize with all -1 to check that all values are defined later
          local2global[my_coarse_cell].resize(R[my_coarse_cell].Height(), -1);
          DG_FECollection DG_fec(param.method.order, param.dimension);
          FiniteElementSpace DG_fespace(ccell_fine_mesh, &DG_fec, param.dimension);
          Array<int> loc_dofs;
          for (int fiz = 0; fiz < n_fine_z; ++fiz)
          {
            for (int fiy = 0; fiy < n_fine_y; ++fiy)
            {
              for (int fix = 0; fix < n_fine_x; ++fix)
              {
                const int loc_cell = fiz*n_fine_x*n_fine_y + fiy*n_fine_x + fix;
                const int glob_cell = (offset_z + fiz) * param.grid.nx * param.grid.ny +
                                      (offset_y + fiy) * param.grid.nx +
                                      (offset_x + fix);
                MFEM_ASSERT(glob_cell >= 0 && glob_cell < (int)map_cell_dofs.size(),
                            "glob_cell is out of range");

                DG_fespace.GetElementVDofs(loc_cell, loc_dofs);
                const vector<int> &glob_dofs = map_cell_dofs[glob_cell];
                MFEM_ASSERT(loc_dofs.Size() == (int)glob_dofs.size(), "Dimensions mismatch");

                for (int di = 0; di < loc_dofs.Size(); ++di)
                  local2global[my_coarse_cell][loc_dofs[di]] = glob_dofs[di];
              }
            }
          }

          // check that all values were defined
          for (size_t ii = 0; ii < local2global[my_coarse_cell].size(); ++ii) {
            MFEM_VERIFY(local2global[my_coarse_cell][ii] >= 0, "Some values of "
                        "local2global vector were not defined");
          }

          delete ccell_fine_mesh;

          offset_x += n_fine_x;
        }
        offset_y += n_fine_y;
      }
      offset_z += n_fine_z;
    }
  }
}

#endif // MFEM_USE_MPI


