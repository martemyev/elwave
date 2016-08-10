#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include "config.hpp"
#include "mfem.hpp"
#include "source.hpp"

#include <string>
#include <vector>

static const char DEFAULT_FILE_NAME[] = "no-file";

class ReceiversSet;
class SnapshotsSet;



/**
 * Parameters describing the domain and the grid.
 */
class GridParameters
{
public:
  GridParameters();
  ~GridParameters() { }

  double sx, sy, sz; ///< size of the computational domain
  int nx, ny, nz; ///< number of cells in x-, y- and z-directions

  const char* meshfile; ///< name of file with mesh
  bool fix_orientation; ///< if we need to fix the mesh orientation

  double get_hx() const { return sx / nx; }
  double get_hy() const { return sy / ny; }
  double get_hz() const { return sz / nz; }

  void AddOptions(mfem::OptionsParser& args);
  void check_parameters() const;

private:
  GridParameters(const GridParameters&);
  GridParameters& operator=(const GridParameters&);
};



/**
 * Parameters describing the source.
 */
class SourceParameters
{
public:
  SourceParameters();
  ~SourceParameters() { }

  mfem::Vertex location;
  double frequency;
  int direction; ///< The direction of the point force component of the source.
                 ///< The values 1-3 mean the direction along one of the axis:
                 ///< 1 OX, 2 OY, 3 OZ.
  double scale;
  double Mxx, Mxy, Mxz, Myy, Myz, Mzz; ///< components of a moment tensor
  const char *type; ///< "pointforce", "momenttensor"
  const char *spatial_function; ///< "delta", "gauss"
  double gauss_support; ///< size of the support for the "gauss" spatial function
  bool plane_wave; ///< plane wave as a source at the depth of y-coordinate of
                   ///< the source location

  void AddOptions(mfem::OptionsParser& args);
  void check_parameters() const;

private:
  SourceParameters(const SourceParameters&);
  SourceParameters& operator=(const SourceParameters&);
};



/**
 * Parameters describing the media properties.
 */
class MediaPropertiesParameters
{
public:
  MediaPropertiesParameters();
  ~MediaPropertiesParameters();

  double rho, vp, vs; ///< homogeneous media properties

  const char *rhofile; ///< file names for heterogeneous media properties
  const char *vpfile;
  const char *vsfile;

  /**
   * Arrays of values describing media properties
   */
  double *rho_array, *vp_array, *vs_array, *lambda_array, *mu_array;

  double min_rho, max_rho, min_vp, max_vp, min_vs, max_vs;

  void AddOptions(mfem::OptionsParser& args);
  void check_parameters() const;
  void init(int n_elements);

private:
  MediaPropertiesParameters(const MediaPropertiesParameters&);
  MediaPropertiesParameters& operator=(const MediaPropertiesParameters&);
};



/**
 * Parameters describing the boundary conditions.
 */
class BoundaryConditionsParameters
{
public:
  BoundaryConditionsParameters();
  ~BoundaryConditionsParameters() { }

  const char* left;   ///< left surface   (X=0) : absorbing (abs) or free
  const char* right;  ///< right surface  (X=sx): absorbing (abs) or free
  const char* bottom; ///< bottom surface (Y=0) : absorbing (abs) or free
  const char* top;    ///< top surface    (Y=sy): absorbing (abs) or free
  const char* front;  ///< front surface  (Z=0) : absorbing (abs) or free
  const char* back;   ///< back surface   (Z=sz): absorbing (abs) or free
  double damp_layer; ///< thickness of a damping layer
  double damp_power; ///< power in damping coefficient functions

  void AddOptions(mfem::OptionsParser& args);
  void check_parameters() const;

private:
  BoundaryConditionsParameters(const BoundaryConditionsParameters&);
  BoundaryConditionsParameters& operator=(const BoundaryConditionsParameters&);
};



/**
 * Parameters describing the method and some specific parameters of the method
 * (if any).
 */
class MethodParameters
{
public:
  MethodParameters();
  ~MethodParameters() { }

  int order; ///< finite element order
  const char *name; ///< FEM, SEM, DG, GMsFEM

  /**
   * Parameters of the DG method.
   * sigma = -1, kappa >= kappa0: symm. interior penalty (IP or SIPG) method,
   * sigma = +1, kappa > 0: non-symmetric interior penalty (NIPG) method,
   * sigma = +1, kappa = 0: the method of Baumann and Oden
   */
  double dg_sigma, dg_kappa;

  /**
   * Parameters of the GMsFEM method
   */
  int gms_Nx, gms_Ny, gms_Nz; // number of coarse cells
  int gms_nb, gms_ni; // number of basis functions


  void AddOptions(mfem::OptionsParser& args);
  void check_parameters() const;

private:
  MethodParameters(const MethodParameters&);
  MethodParameters& operator=(const MethodParameters&);
};



/**
 * Handle output results and intermediate steps of computations
 */
class OutputParameters
{
public:
  OutputParameters();
  ~OutputParameters() { }

  const char *directory; ///< directory for saving results of computations
  const char *extra_string; ///< added to output files for distinguishing the
                            ///< results

  bool print_matrices; ///< output intermediate matrices (yes, no)
  bool view_snapshot_space; ///< view some solutions via GLVis
  bool view_boundary_basis;
  bool view_interior_basis;
  bool view_dg_basis;

  /**
   * When computed in parallel - whether to save the solution corresponding a
   * serial one. That allows to compare the solutions of parallel and serial
   * runs, and also parallel solutions obtained with different number of
   * processes.
   */
  bool serial_solution;

  /**
   * Print an array of receivers with coordinates and the corresponding cells
   * of a serial and a parallel grids where each receiver is.
   */
  bool cells_containing_receivers;

  /**
   * Format of output of the snapshots: vts, glvis, vtsglvis
   */
  const char *snap_format;

  void AddOptions(mfem::OptionsParser& args);
  void check_parameters() const;

  bool snap_format_VTS() const {
    return (strstr(snap_format, "vts") != NULL);
  }
  bool snap_format_GLVis() const {
    return (strstr(snap_format, "glvis") != NULL);
  }

private:
  OutputParameters(const OutputParameters&);
  OutputParameters& operator =(const OutputParameters&);
};



/**
 * Parameters of the problem to be solved.
 */
class Parameters
{
public:
  Parameters();
  ~Parameters();

  int dimension; ///< 2D or 3D simulation

  GridParameters grid;
  SourceParameters source;
  MediaPropertiesParameters media;
  BoundaryConditionsParameters bc;
  MethodParameters method;
  OutputParameters output;

  mfem::Mesh *mesh;
  mfem::ParMesh *par_mesh;

  double T; ///< simulation time
  double dt; ///< time step

  bool serial; ///< serial or parallel execution

  int step_snap; ///< time step for outputting snapshots (every *th time step)
  int step_seis; ///< time step for outputting seismograms (every *th time step)
  const char *receivers_file; ///< file describing the sets of receivers
  std::vector<ReceiversSet*> sets_of_receivers;

  void init(int argc, char **argv);
  void check_parameters() const;

private:
  Parameters(const Parameters&); // no copies
  Parameters& operator=(const Parameters&); // no copies
};

#endif // PARAMETERS_HPP
