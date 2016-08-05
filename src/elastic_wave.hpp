#ifndef ELASTIC_WAVE3D_HPP
#define ELASTIC_WAVE3D_HPP

#include "config.hpp"
#include "mfem.hpp"

#include <fstream>
#include <vector>

class Parameters;



/**
 * Elastic wave run by finite elements or spectral elements (both are
 * continuous Galerkin approaches).
 */
class ElasticWave
{
public:
  ElasticWave(const Parameters& p) : param(p) { }
  ~ElasticWave() { }

  void run();

private:
  const Parameters& param;

  /**
   * Finite Element Method (FEM) (non-diagonal mass matrix) with Absorbing
   * Layers by Increasing Damping (ALID) for implementation of absorbing
   * boundary condition.
   */
  void run_FEM_ALID();

  /**
   * Spectral Element Method (SEM) (diagonal mass matrix) with Stiffness
   * Reduction Method (SRM) for implementation of absorbing boundary condition.
   */
  void run_SEM_SRM();

  /**
   * Discontinuous Galerkin method
   */
  void run_DG();

  void run_SEM_SRM_serial();
  void run_DG_serial();

#if defined(MFEM_USE_MPI)
  void run_SEM_SRM_parallel();
  void run_DG_parallel();

  /**
   * Generalized multiscale finite element method. It uses an MFEM eigensolver
   * that works only in parallel setup.
   */
  void run_GMsFEM() const;

  /**
   * Still, GMsFEM can be run sequentially, even the MFEM parallel eigensolver
   * (with MPI_COMM_SELF).
   */
  void run_GMsFEM_serial() const;

  /**
   * Parallel execution of the GMsFEM method.
   */
  void run_GMsFEM_parallel() const;

  /**
   * Filling out the R matrices and the map between the local indices used for
   * local R matrices computations and global indices to assemble a global R.
   */
  void compute_R_matrices(std::ostream &out,
                          const std::vector<std::vector<int> > &map_cell_dofs,
                          std::vector<std::vector<int> > &local2global,
                          std::vector<mfem::DenseMatrix> &R) const;

  /**
   * Computation of the multiscale basis (each basis function is a column of the
   * R matrix, which is a projection from fine scale to coarse scale spaces).
   */
  void compute_basis_CG(std::ostream &out, mfem::Mesh *fine_mesh, int n_boundary_bf, int n_interior_bf,
                        mfem::Coefficient &rho_coef, mfem::Coefficient &lambda_coef,
                        mfem::Coefficient &mu_coef, mfem::DenseMatrix &R) const;
#endif
};



/**
 * Cell-wise constant coefficient.
 */
class CWConstCoefficient : public mfem::Coefficient
{
public:
  CWConstCoefficient(double *array, bool own = 1)
    : val_array(array), own_array(own)
  { }

  virtual ~CWConstCoefficient() { if (own_array) delete[] val_array; }

  virtual double Eval(mfem::ElementTransformation &T,
                      const mfem::IntegrationPoint &/*ip*/)
  {
    const int index = T.Attribute - 1; // use attribute as a cell number
    return val_array[index];
  }

protected:
  double *val_array;
  bool own_array;
};



/**
 * A coefficient obtained with multiplication of a cell-wise constant
 * coefficient and a function.
 */
class CWFunctionCoefficient : public CWConstCoefficient
{
public:
  CWFunctionCoefficient(double(*F)(const mfem::Vector&, const Parameters&),
                        const Parameters& p,
                        double *array, bool own = 1)
    : CWConstCoefficient(array, own)
    , Function(F)
    , param(p)
  { }

  virtual ~CWFunctionCoefficient() { }

  virtual double Eval(mfem::ElementTransformation &T,
                      const mfem::IntegrationPoint &ip)
  {
    const int index = T.Attribute - 1; // use attribute as a cell number
    const double cw_coef = val_array[index];
    mfem::Vector transip;
    T.Transform(ip, transip);
    const double func_val = (*Function)(transip, param);
    return cw_coef * func_val;
  }

protected:
  double(*Function)(const mfem::Vector&, const Parameters&);
  const Parameters& param;
};



void show_SRM_damp_weights(const Parameters& param);

mfem::Vector compute_function_at_point(const mfem::Mesh& mesh,
                                       const mfem::Vertex& point, int cell,
                                       const mfem::GridFunction& U);

mfem::Vector compute_function_at_points(const mfem::Mesh& mesh, int n_points,
                                        const mfem::Vertex *points,
                                        const int *cells_containing_points,
                                        const mfem::GridFunction& U);

void open_seismo_outs(std::ofstream* &seisU, std::ofstream* &seisV,
                      const Parameters &param, const std::string &method_name);

void output_snapshots(int time_step, const std::string& snapshot_filebase,
                      const Parameters& param, const mfem::GridFunction& U,
                      const mfem::GridFunction& V, const mfem::Mesh& mesh);

void output_seismograms(const Parameters& param, const mfem::Mesh& mesh,
                        const mfem::GridFunction &U, const mfem::GridFunction &V,
                        std::ofstream* &seisU, std::ofstream* &seisV);

#ifdef MFEM_USE_MPI
void par_output_seismograms(const Parameters& param,
                            mfem::ParFiniteElementSpace &fespace,
                            const mfem::HypreParMatrix &RT,
                            const mfem::Vector &U, std::ofstream* &seisU);
#endif // MFEM_USE_MPI

void solve_dsygvd(const mfem::DenseMatrix &A, const mfem::DenseMatrix &B,
                  mfem::DenseMatrix &eigenvectors);

extern "C" {
void dsygvd_(int *ITYPE, char *JOBZ, char *UPLO, int *N, double *A, int *LDA,
             double *B, int *LDB, double *W, double *WORK, int *LWORK,
             int *IWORK, int *LIWORK, int *INFO);
}

#endif // ELASTIC_WAVE3D_HPP
