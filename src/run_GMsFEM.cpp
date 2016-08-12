#include "elastic_wave.hpp"
#include "parameters.hpp"
#include "utilities.hpp"

#include <float.h>

#ifdef MFEM_USE_MPI

using namespace std;
using namespace mfem;



static void write_gmsh_coarse_cell(const Parameters &param,
                                   const string &mesh_fname,
                                   const vector<int> &fine_cells)
{
  const Mesh &mesh = *(param.mesh);

  map<int, int> glob2loc_vertices;

  int loc_vert_ID = 1; // Gmsh numerates vertices from 1
  for (size_t el = 0; el < fine_cells.size(); ++el) {
    Array<int> vert;
    mesh.GetElementVertices(fine_cells[el], vert);

    for (int v = 0; v < vert.Size(); ++v) {
      auto result = glob2loc_vertices.insert(make_pair(vert[v], loc_vert_ID));
      if (result.second)
        ++loc_vert_ID;
    }
  }

  ofstream out(mesh_fname.c_str());
  MFEM_VERIFY(out, "Cannot open file " << mesh_fname);
  out << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n";
  out << "$Nodes\n" << glob2loc_vertices.size() << "\n";
  for (auto iter = glob2loc_vertices.begin(); iter != glob2loc_vertices.end(); ++iter) {
    const double *vert = mesh.GetVertex(iter->first);
    out << iter->second << " ";
    for (int d = 0; d < mesh.Dimension(); ++d)
      out << vert[d] << " ";
    if (mesh.Dimension() < 3)
      out << "0.0";
    out << "\n";
  }
  out << "$EndNodes\n";

  map<int, int> GeoType_GmshType;
  GeoType_GmshType[Element::TRIANGLE]      = 2;
  GeoType_GmshType[Element::QUADRILATERAL] = 3;
  GeoType_GmshType[Element::TETRAHEDRON]   = 4;
  GeoType_GmshType[Element::HEXAHEDRON]    = 5;

  out << "$Elements\n" << fine_cells.size() << "\n";
  for (size_t el = 0; el < fine_cells.size(); ++el) {
    auto type = GeoType_GmshType.find(mesh.GetElementType(fine_cells[el]));
    MFEM_VERIFY(type != GeoType_GmshType.end(), "Mesh type was not found");
    out << el + 1 << " " << type->second << " 2 " << el + 1 << " " << 2*el + 1 << " ";
    Array<int> vert_glob;
    mesh.GetElementVertices(fine_cells[el], vert_glob);
    for (int v = 0; v < vert_glob.Size(); ++v) {
      auto vert = glob2loc_vertices.find(vert_glob[v]);
      MFEM_VERIFY(vert != glob2loc_vertices.end(), "Vertex was not found");
      out << vert->second << " ";
    }
    out << "\n";
  }
  out << "$EndElements\n";
}



void ElasticWave::compute_R_matrices(ostream &log,
                                     const vector<vector<int> > &map_coarse_cell_fine_cells,
                                     const vector<vector<int> > &map_cell_dofs,
                                     vector<vector<int> > &local2global,
                                     vector<DenseMatrix> &R) const
{
  int myid, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  const int n_coarse_cells = map_coarse_cell_fine_cells.size();

  // number of coarse cells that all processes will have (at least)
  const int min_n_cells = n_coarse_cells / nproc;
  // number of coarse cells that should be distributed among some processes
  const int extra_cells = n_coarse_cells % nproc;
  // first and last (not including) indices of coarse element for the current
  // 'myid' process
  const int my_start_cell = min_n_cells * myid + (extra_cells < myid ? extra_cells : myid);
  const int my_end_cell   = my_start_cell + min_n_cells + (extra_cells > myid);

  log << "coarse cells: start " << my_start_cell << " end " << my_end_cell << endl;

  local2global.resize(my_end_cell - my_start_cell);
  R.resize(my_end_cell - my_start_cell);

  const int generate_edges = 1;
  const int refine = 0;
  const bool fix_orientation = true;

  for (int global_coarse_cell = my_start_cell;
       global_coarse_cell < my_end_cell;
       ++global_coarse_cell) {

    const int my_coarse_cell = global_coarse_cell - my_start_cell;

    log << "\nglobal_coarse_cell " << global_coarse_cell
        << " my_coarse_cell " << my_coarse_cell << endl;

    const vector<int> &fine_cells = map_coarse_cell_fine_cells[global_coarse_cell];
    const string mesh_fname = string(param.output.directory) + "/" + MESHES_DIR +
                              "/coarse_cell_" + d2s(global_coarse_cell) + ".msh";
    write_gmsh_coarse_cell(param, mesh_fname, fine_cells);

    Mesh *ccell_fine_mesh =
        new Mesh(mesh_fname.c_str(), generate_edges, refine, fix_orientation);

    const int n_fine_cells = fine_cells.size();
    double *local_rho    = new double[n_fine_cells];
    double *local_lambda = new double[n_fine_cells];
    double *local_mu     = new double[n_fine_cells];
    for (int el = 0; el < n_fine_cells; ++el) {
      const int loc_cell  = el;
      const int glob_cell = fine_cells[el];

      local_rho[loc_cell]    = param.media.rho_array[glob_cell];
      local_lambda[loc_cell] = param.media.lambda_array[glob_cell];
      local_mu[loc_cell]     = param.media.mu_array[glob_cell];
    }

    CWConstCoefficient local_rho_coef(local_rho, true);
    CWConstCoefficient local_lambda_coef(local_lambda, true);
    CWConstCoefficient local_mu_coef(local_mu, true);

    compute_basis_CG(log, ccell_fine_mesh, param.method.gms_nb, param.method.gms_ni,
                     local_rho_coef, local_lambda_coef, local_mu_coef,
                     R[my_coarse_cell]);

    // initialize with all -1 to check that all values are defined later
    local2global[my_coarse_cell].resize(R[my_coarse_cell].Height(), -1);
    DG_FECollection DG_fec(param.method.order, param.dimension);
    FiniteElementSpace DG_fespace(ccell_fine_mesh, &DG_fec, param.dimension);
    Array<int> loc_dofs;
    for (int el = 0; el < n_fine_cells; ++el) {
      const int loc_cell  = el;
      const int glob_cell = fine_cells[el];
      MFEM_ASSERT(glob_cell >= 0 && glob_cell < (int)map_cell_dofs.size(),
                  "glob_cell is out of range");

      DG_fespace.GetElementVDofs(loc_cell, loc_dofs);
      const vector<int> &glob_dofs = map_cell_dofs[glob_cell];
      MFEM_ASSERT(loc_dofs.Size() == (int)glob_dofs.size(), "Dimensions mismatch");

      for (int di = 0; di < loc_dofs.Size(); ++di)
        local2global[my_coarse_cell][loc_dofs[di]] = glob_dofs[di];
    }

    // check that all values were defined
    for (size_t ii = 0; ii < local2global[my_coarse_cell].size(); ++ii) {
      MFEM_VERIFY(local2global[my_coarse_cell][ii] >= 0, "Some values of "
                  "local2global vector were not defined");
    }

    delete ccell_fine_mesh;
  }
}

#endif // MFEM_USE_MPI


