#include "elastic_wave.hpp"
#include "GLL_quadrature.hpp"
#include "parameters.hpp"
#include "receivers.hpp"
#include "utilities.hpp"

using namespace std;
using namespace mfem;



void ElasticWave::run()
{
  if (!strcmp(param.method.name, "fem") || !strcmp(param.method.name, "FEM"))
  {
    //run_FEM_ALID();
    MFEM_ABORT("FEM method is not implemented");
  }
  else if (!strcmp(param.method.name, "sem") || !strcmp(param.method.name, "SEM"))
  {
    run_SEM_SRM();
  }
  else if (!strcmp(param.method.name, "dg") || !strcmp(param.method.name, "DG"))
  {
    run_DG();
  }
  else
  {
    MFEM_ABORT("Unknown method to be used: " << param.method.name);
  }
}



//------------------------------------------------------------------------------
//
// Auxiliary useful functions
//
//------------------------------------------------------------------------------
Vector compute_function_at_point(const Mesh& mesh, const Vertex& point,
                                 int cell, const GridFunction& U)
{
  const int dim = mesh.Dimension();
  MFEM_VERIFY(dim == 2 || dim == 3, "Wrong dimension");

  const Element* element = mesh.GetElement(cell);

  Array<int> vert_indices;
  element->GetVertices(vert_indices);

  double x0, x1, y0, y1, z0, z1;

  const double *vert0 = mesh.GetVertex(vert_indices[0]);
  const double *vert2 = mesh.GetVertex(vert_indices[2]);
  x0 = vert0[0];
  y0 = vert0[1];
  x1 = vert2[0];
  y1 = vert2[1];

  if (dim == 2)
  {
    MFEM_VERIFY(dynamic_cast<const Quadrilateral*>(element), "The mesh "
                "element has to be a quadrilateral");
  }
  else // 3D
  {
    MFEM_VERIFY(dynamic_cast<const Hexahedron*>(element), "The mesh "
                "element has to be a hexahedron");
    const double *vert6 = mesh.GetVertex(vert_indices[6]);
    z0 = vert0[2];
    z1 = vert6[2];
  }

  IntegrationPoint ip;
  ip.x = (point(0) - x0) / (x1 - x0); // transfer to the reference space [0,1]^d
  ip.y = (point(1) - y0) / (y1 - y0);
  if (dim == 3)
    ip.z = (point(2) - z0) / (z1 - z0);

  Vector values;
  U.GetVectorValue(cell, ip, values);

  return values;
}



Vector compute_function_at_points(const Mesh& mesh, int n_points,
                                  const Vertex *points,
                                  const int *cells_containing_points,
                                  const GridFunction& U)
{
  const int dim = mesh.Dimension();
  Vector U_at_points(dim*n_points);

  for (int p = 0; p < n_points; ++p)
  {
    Vector values = compute_function_at_point(mesh, points[p],
                                              cells_containing_points[p], U);
    MFEM_ASSERT(values.Size() == dim, "Unexpected vector size");
    for (int c = 0; c < dim; ++c)
      U_at_points(p*dim + c) = values(c);
  }
  return U_at_points;
}



void open_seismo_outs(ofstream* &seisU, ofstream* &seisV,
                      const Parameters &param, const string &method_name)
{
  const int dim = param.dimension;

  const int n_rec_sets = param.sets_of_receivers.size();

  seisU = new ofstream[dim*n_rec_sets];
  seisV = new ofstream[dim*n_rec_sets];

  for (int r = 0; r < n_rec_sets; ++r)
  {
    const ReceiversSet *rec_set = param.sets_of_receivers[r];
    const string desc = rec_set->description();
    const string variable = rec_set->get_variable();

    if (variable.find("U") != string::npos) {
      for (int c = 0; c < dim; ++c) {
        string seismofile = (string)param.output_dir + "/" + SEISMOGRAMS_DIR +
                            method_name + param.extra_string + desc + "_u" +
                            d2s(c) + ".bin";
        seisU[r*dim + c].open(seismofile.c_str(), ios::binary);
        MFEM_VERIFY(seisU[r*dim + c], "File '" + seismofile +
                    "' can't be opened");
      }
    }

    if (variable.find("V") != string::npos) {
      for (int c = 0; c < dim; ++c) {
        string seismofile = (string)param.output_dir + "/" + SEISMOGRAMS_DIR +
                            method_name + param.extra_string + desc + "_v" +
                            d2s(c) + ".bin";
        seisV[r*dim + c].open(seismofile.c_str(), ios::binary);
        MFEM_VERIFY(seisV[r*dim + c], "File '" + seismofile +
                    "' can't be opened");
      }
    }
  } // loop for sets of receivers
}



void output_seismograms(const Parameters& param, const Mesh& mesh,
                        const GridFunction &U, const GridFunction &V,
                        ofstream* &seisU, ofstream* &seisV)
{
  const int dim = mesh.Dimension();

  // for each set of receivers
  for (size_t rec = 0; rec < param.sets_of_receivers.size(); ++rec)
  {
    const ReceiversSet *rec_set = param.sets_of_receivers[rec];
    const string variable = rec_set->get_variable();

    // Displacement
    if (variable.find("U") != string::npos) {
      for (int c = 0; c < dim; ++c) {
        MFEM_VERIFY(seisU[rec*dim+c].is_open(), "The stream for "
                    "writing displacement seismograms is not open");
      }
      // displacement at the receivers
      const Vector u =
        compute_function_at_points(mesh, rec_set->n_receivers(),
                                   &(rec_set->get_receivers()[0]),
                                   &(rec_set->get_cells_containing_receivers()[0]), U);
      MFEM_ASSERT(u.Size() == dim*rec_set->n_receivers(), "Sizes mismatch");
      for (int i = 0; i < u.Size(); i += dim) {
        for (int j = 0; j < dim; ++j) {
          float val = u(i+j); // displacement
          seisU[rec*dim + j].write(reinterpret_cast<char*>(&val), sizeof(val));
        }
      }
    }

    // Particle velocity
    if (variable.find("V") != string::npos) {
      for (int c = 0; c < dim; ++c) {
        MFEM_VERIFY(seisV[rec*dim+c].is_open(), "The stream for "
                    "writing velocity seismograms is not open");
      }
      // velocity at the receivers
      const Vector v =
        compute_function_at_points(mesh, rec_set->n_receivers(),
                                   &(rec_set->get_receivers()[0]),
                                   &(rec_set->get_cells_containing_receivers()[0]), V);
      MFEM_ASSERT(v.Size() == dim*rec_set->n_receivers(),
                  "Sizes mismatch");
      for (int i = 0; i < v.Size(); i += dim) {
        for (int j = 0; j < dim; ++j) {
          float val = v(i+j); // velocity
          seisV[rec*dim + j].write(reinterpret_cast<char*>(&val), sizeof(val));
        }
      }
    }
  } // loop over receiver sets
}



