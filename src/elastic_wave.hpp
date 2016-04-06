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

  void run_SEM_SRM_serial();
#if defined(MFEM_USE_MPI)
  void run_SEM_SRM_parallel();
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

#endif // ELASTIC_WAVE3D_HPP
