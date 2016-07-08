#include "parameters.hpp"
#include "receivers.hpp"
#include "utilities.hpp"

#include <cfloat>

using namespace std;
using namespace mfem;



//------------------------------------------------------------------------------
//
// Grid parameters
//
//------------------------------------------------------------------------------
GridParameters::GridParameters()
  : sx(1000.0)
  , sy(1000.0)
  , sz(1000.0)
  , nx(10)
  , ny(10)
  , nz(10)
  , meshfile(DEFAULT_FILE_NAME)
  , fix_orientation(true)
{ }

void GridParameters::AddOptions(OptionsParser& args)
{
  args.AddOption(&sx, "-sx", "--sizex", "Size of domain in x-direction, m");
  args.AddOption(&sy, "-sy", "--sizey", "Size of domain in y-direction, m");
  args.AddOption(&sz, "-sz", "--sizez", "Size of domain in z-direction, m");
  args.AddOption(&nx, "-nx", "--numberx", "Number of elements in x-direction");
  args.AddOption(&ny, "-ny", "--numbery", "Number of elements in y-direction");
  args.AddOption(&nz, "-nz", "--numberz", "Number of elements in z-direction");
  args.AddOption(&meshfile, "-meshfile", "--mesh-file", "Name of file with mesh");
  args.AddOption(&fix_orientation, "-fix-orientation", "--fix-orientation",
                 "-no-fix-orientation", "--no-fix-orientation", "If we need to "
                 "fix mesh orientation");
}

void GridParameters::check_parameters() const
{
  MFEM_VERIFY(sx > 0 && sy > 0 && sz > 0, "Size of the domain (sx=" + d2s(sx) +
              " m, sy=" + d2s(sy) + " m, sz=" + d2s(sz) + " m) must be >0");
  MFEM_VERIFY(nx > 0 && ny > 0 && nz > 0, "Number of cells (nx=" + d2s(nx) +
              ", ny=" + d2s(ny) + ", nz=" + d2s(nz) + ") must be >0");
}



//------------------------------------------------------------------------------
//
// Source parameters
//
//------------------------------------------------------------------------------
SourceParameters::SourceParameters()
  : location(500.0, 500.0, 500.0)
  , frequency(10.0)
  , direction(2) // OY
  , scale(1e+6)
  , Mxx(1.0), Mxy(0.0), Mxz(0.0), Myy(1.0), Myz(0.0), Mzz(1.0) // explosive source
  , type("pointforce")
  , spatial_function("gauss")
  , gauss_support(10.0)
  , plane_wave(false)
{ }

void SourceParameters::AddOptions(OptionsParser& args)
{
  args.AddOption(&location(0), "-srcx", "--source-x", "x-coord of a source location");
  args.AddOption(&location(1), "-srcy", "--source-y", "y-coord of a source location");
  args.AddOption(&location(2), "-srcz", "--source-z", "z-coord of a source location");
  args.AddOption(&frequency, "-f0", "--source-frequency", "Central frequency of a source");
  args.AddOption(&direction, "-dir", "--source-direction", "Direction of point force or plane wave source (1 OX, 2 OY, 3 OZ)");
  args.AddOption(&scale, "-scale", "--source-scale", "Scaling factor for the source");
  args.AddOption(&Mxx, "-mxx", "--moment-tensor-xx", "xx-component of a moment tensor source");
  args.AddOption(&Mxy, "-mxy", "--moment-tensor-xy", "xy-component of a moment tensor source");
  args.AddOption(&Mxz, "-mxz", "--moment-tensor-xz", "xz-component of a moment tensor source");
  args.AddOption(&Myy, "-myy", "--moment-tensor-yy", "yy-component of a moment tensor source");
  args.AddOption(&Myz, "-myz", "--moment-tensor-yz", "yz-component of a moment tensor source");
  args.AddOption(&Mzz, "-mzz", "--moment-tensor-zz", "zz-component of a moment tensor source");
  args.AddOption(&type, "-type", "--source-type", "Type of the source (pointforce, momenttensor)");
  args.AddOption(&spatial_function, "-spatial", "--source-spatial", "Spatial function of the source (delta, gauss)");
  args.AddOption(&gauss_support, "-gs", "--gauss-support", "Gauss support for 'gauss' spatial function of the source");
  args.AddOption(&plane_wave, "-planewave", "--plane-wave", "-noplanewave", "--no-plane-wave", "Plane wave as a source");
}

void SourceParameters::check_parameters() const
{
  MFEM_VERIFY(frequency > 0, "Frequency (" + d2s(frequency) + ") must be >0");
  MFEM_VERIFY(direction == 1 || direction == 2 || direction == 3, "Unsupported "
              "direction of the source: " + d2s(direction));
  MFEM_VERIFY(!strcmp(type, "pointforce") || !strcmp(type, "momenttensor"),
              "Unknown source type: " + string(type));
  MFEM_VERIFY(!strcmp(spatial_function, "delta") ||
              !strcmp(spatial_function, "gauss"), "Unknown spatial function of "
              "the source: " + string(spatial_function));
  if (!strcmp(spatial_function, "gauss"))
    MFEM_VERIFY(gauss_support > 0, "Gauss support (" + d2s(gauss_support) +
                ") must be >0");
}



//------------------------------------------------------------------------------
//
// Media properties parameters
//
//------------------------------------------------------------------------------
MediaPropertiesParameters::MediaPropertiesParameters()
  : rho(2500)
  , vp(3500)
  , vs(2000)
  , rhofile(DEFAULT_FILE_NAME)
  , vpfile(DEFAULT_FILE_NAME)
  , vsfile(DEFAULT_FILE_NAME)
  , rho_array(nullptr)
  , vp_array(nullptr)
  , vs_array(nullptr)
  , min_rho(DBL_MAX), max_rho(DBL_MIN)
  , min_vp (DBL_MAX), max_vp (DBL_MIN)
  , min_vs (DBL_MAX), max_vs (DBL_MIN)
{ }

MediaPropertiesParameters::~MediaPropertiesParameters()
{
  delete[] rho_array;
  delete[] vp_array;
  delete[] vs_array;
}

void MediaPropertiesParameters::AddOptions(OptionsParser& args)
{
  args.AddOption(&rho, "-rho", "--rho", "Density of homogeneous model, kg/m^3");
  args.AddOption(&vp, "-vp", "--vp", "P-wave velocity of homogeneous model, m/s");
  args.AddOption(&vs, "-vs", "--vs", "S-wave velocity of homogeneous model, m/s");
  args.AddOption(&rhofile, "-rhofile", "--rhofile", "Density file, in kg/m^3");
  args.AddOption(&vpfile, "-vpfile", "--vpfile", "P-wave velocity file, in m/s");
  args.AddOption(&vsfile, "-vsfile", "--vsfile", "S-wave velocity file, in m/s");
}

void MediaPropertiesParameters::check_parameters() const
{
  // no checks here
}

void MediaPropertiesParameters::init(int n_elements)
{
  rho_array = new double[n_elements];
  vp_array = new double[n_elements];
  vs_array = new double[n_elements];

  if (!strcmp(rhofile, DEFAULT_FILE_NAME))
  {
    for (int i = 0; i < n_elements; ++i) rho_array[i] = rho;
    min_rho = max_rho = rho;
  }
  else
  {
    read_binary(rhofile, n_elements, rho_array);
    get_minmax(rho_array, n_elements, min_rho, max_rho);
  }

  if (!strcmp(vpfile, DEFAULT_FILE_NAME))
  {
    for (int i = 0; i < n_elements; ++i) vp_array[i] = vp;
    min_vp = max_vp = vp;
  }
  else
  {
    read_binary(vpfile, n_elements, vp_array);
    get_minmax(vp_array, n_elements, min_vp, max_vp);
  }

  if (!strcmp(vsfile, DEFAULT_FILE_NAME))
  {
    for (int i = 0; i < n_elements; ++i) vs_array[i] = vs;
    min_vs = max_vs = vs;
  }
  else
  {
    read_binary(vsfile, n_elements, vs_array);
    get_minmax(vs_array, n_elements, min_vs, max_vs);
  }
}



//------------------------------------------------------------------------------
//
// Boundary conditions parameters
//
//------------------------------------------------------------------------------
BoundaryConditionsParameters::BoundaryConditionsParameters()
  : left  ("abs")
  , right ("abs")
  , bottom("abs")
  , top   ("abs")
  , front ("abs")
  , back  ("abs")
  , damp_layer(100.0)
  , damp_power(3.0)
{ }

void BoundaryConditionsParameters::AddOptions(OptionsParser& args)
{
  // Left, right, front and back surfaces are usually absorbing, so we
  // don't need to set up program options for them, but this can be changed if
  // desired.
//  args.AddOption(&left, "-left", "--left-surface", "Left surface: abs or free");
//  args.AddOption(&right, "-right", "--right-surface", "Right surface: abs or free");
//  args.AddOption(&front, "-front", "--front-surface", "Front surface: abs or free");
//  args.AddOption(&back, "-back", "--back-surface", "Back surface: abs or free");

  args.AddOption(&bottom, "-bottom", "--bottom-surface", "Bottom surface: abs or free");
  args.AddOption(&top, "-top", "--top-surface", "Top surface: abs or free");
  args.AddOption(&damp_layer, "-dlayer", "--damp-layer", "Thickness of damping layer, m");
  args.AddOption(&damp_power, "-dpower", "--damp-power", "Power in damping coefficient functions");
}

void BoundaryConditionsParameters::check_parameters() const
{
  MFEM_VERIFY(!strcmp(left, "abs") || !strcmp(left, "free"), "Unknown boundary "
              "condition on the left surface: " + string(left));
  MFEM_VERIFY(!strcmp(right, "abs") || !strcmp(right, "free"), "Unknown boundary "
              "condition on the right surface: " + string(right));
  MFEM_VERIFY(!strcmp(bottom, "abs") || !strcmp(bottom, "free"), "Unknown boundary "
              "condition on the bottom surface: " + string(bottom));
  MFEM_VERIFY(!strcmp(top, "abs") || !strcmp(top, "free"), "Unknown boundary "
              "condition on the top surface: " + string(top));
  MFEM_VERIFY(!strcmp(front, "abs") || !strcmp(front, "free"), "Unknown boundary "
              "condition on the front surface: " + string(front));
  MFEM_VERIFY(!strcmp(back, "abs") || !strcmp(back, "free"), "Unknown boundary "
              "condition on the back surface: " + string(back));
  if (!strcmp(left, "abs") || !strcmp(right, "abs") || !strcmp(bottom, "abs") ||
      !strcmp(top, "abs") || !strcmp(front, "abs") || !strcmp(back, "abs"))
    MFEM_VERIFY(damp_layer > 0, "Damping layer (" + d2s(damp_layer) +
                ") must be >0");
}



//------------------------------------------------------------------------------
//
// Method parameters
//
//------------------------------------------------------------------------------
MethodParameters::MethodParameters()
  : order(1)
  , name("sem")
  , dg_sigma(-1.) // SIPDG
  , dg_kappa(1.)
  , gms_Nx(1), gms_Ny(1), gms_Nz(1)
  , gms_nb(1), gms_ni(1)
{ }

void MethodParameters::AddOptions(OptionsParser& args)
{
  args.AddOption(&order, "-o", "--order", "Finite element order (polynomial degree)");
  args.AddOption(&name, "-method", "--method", "Finite elements (fem), spectral elements (sem), discontinuous Galerkin (dg)");
  args.AddOption(&dg_sigma, "-dg-sigma", "--dg-sigma", "Sigma in the DG method");
  args.AddOption(&dg_kappa, "-dg-kappa", "--dg-kappa", "Kappa in the DG method");
  args.AddOption(&gms_Nx, "-gms-Nx", "--gms-Nx", "Number of coarse cells in x-direction");
  args.AddOption(&gms_Ny, "-gms-Ny", "--gms-Ny", "Number of coarse cells in y-direction");
  args.AddOption(&gms_Nz, "-gms-Nz", "--gms-Nz", "Number of coarse cells in z-direction");
  args.AddOption(&gms_nb, "-gms-nb", "--gms-nb", "Number of boundary basis functions");
  args.AddOption(&gms_ni, "-gms-ni", "--gms-ni", "Number of interior basis functions");
}

void MethodParameters::check_parameters() const
{
  MFEM_VERIFY(order >= 0, "Order is negative");
  MFEM_VERIFY(!strcmp(name, "FEM") || !strcmp(name, "fem") ||
              !strcmp(name, "SEM") || !strcmp(name, "sem") ||
              !strcmp(name, "DG")  || !strcmp(name, "dg")  ||
              !strcmp(name, "GMsFEM") || !strcmp(name, "gmsfem"),
              "Unknown method: " + string(name));
}



//------------------------------------------------------------------------------
//
// All parameters of the problem to be solved
//
//------------------------------------------------------------------------------
Parameters::Parameters()
  : dimension(2)
  , grid()
  , source()
  , media()
  , bc()
  , method()
  , mesh(nullptr)
  , T(1.0)
  , dt(1e-3)
  , step_snap(1000)
  , step_seis(1)
  , receivers_file(DEFAULT_FILE_NAME)
  , output_dir("output")
  , extra_string("")
{ }

Parameters::~Parameters()
{
  for (size_t i = 0; i < sets_of_receivers.size(); ++i)
    delete sets_of_receivers[i];

  delete mesh;
}

void Parameters::init(int argc, char **argv)
{
  OptionsParser args(argc, argv);

  args.AddOption(&dimension, "-d", "--dim", "Dimension of wave simulation (2 or 3)");

  grid.AddOptions(args);
  source.AddOptions(args);
  media.AddOptions(args);
  bc.AddOptions(args);
  method.AddOptions(args);

  args.AddOption(&T, "-T", "--time-end", "Simulation time, s");
  args.AddOption(&dt, "-dt", "--time-step", "Time step, s");
  args.AddOption(&step_snap, "-step-snap", "--step-snapshot", "Time step for outputting snapshots");
  args.AddOption(&step_seis, "-step-seis", "--step-seismogram", "Time step for outputting seismograms");
  args.AddOption(&receivers_file, "-rec-file", "--receivers-file", "File with information about receivers");
  args.AddOption(&output_dir, "-outdir", "--output-dir", "Directory to save results of computations");
  args.AddOption(&extra_string, "-extra", "--extra", "Extra string for naming output files");

  args.Parse();
  if (!args.Good())
  {
    args.PrintUsage(cout);
    throw 1;
  }
  args.PrintOptions(cout);

  check_parameters();


  cout << "Mesh initialization..." << endl;
  const int generate_edges = 1;
  if (strcmp(grid.meshfile, DEFAULT_FILE_NAME))
  {
    cout << "  Reading mesh from " << grid.meshfile << endl;
    ifstream in(grid.meshfile);
    MFEM_VERIFY(in, "File can't be opened");
    const int refine = 0;
    mesh = new Mesh(in, generate_edges, refine, grid.fix_orientation);
    double xmin = DBL_MAX, xmax = DBL_MIN;
    double ymin = DBL_MAX, ymax = DBL_MIN;
    double zmin = DBL_MAX, zmax = DBL_MIN;
    for (int i = 0; i < mesh->GetNV(); ++i)
    {
      const double* v = mesh->GetVertex(i);
      xmin = std::min(xmin, v[0]);
      xmax = std::max(xmax, v[0]);
      ymin = std::min(ymin, v[1]);
      ymax = std::max(ymax, v[1]);
      zmin = std::min(zmin, v[2]);
      zmax = std::max(zmax, v[2]);
    }
    cout << "min coord: x " << xmin << " y " << ymin << " z " << zmin
         << "\nmax coord: x " << xmax << " y " << ymax << " z " << zmax
         << "\n";
    grid.sx = xmax - xmin;
    grid.sy = ymax - ymin;
    grid.sz = zmax - zmin;
  }
  else
  {
    cout << "  Generating mesh" << endl;
    if (dimension == 2)
    {
      mesh = new Mesh(grid.nx, grid.ny, Element::QUADRILATERAL,
                      generate_edges, grid.sx, grid.sy);
    }
    else
    {
      mesh = new Mesh(grid.nx, grid.ny, grid.nz, Element::HEXAHEDRON,
                      generate_edges, grid.sx, grid.sy, grid.sz);
    }
  }

  MFEM_VERIFY(mesh->Dimension() == dimension, "Unexpected mesh dimension");
  for (int el = 0; el < mesh->GetNE(); ++el)
    mesh->GetElement(el)->SetAttribute(el+1);
  cout << "Mesh initialization is done" << endl;


  media.init(mesh->GetNE());

  const double min_wavelength = min(media.min_vp, media.min_vs) /
                                (2.0*source.frequency);
  cout << "min wavelength = " << min_wavelength << endl;

  if (bc.damp_layer < 2.5*min_wavelength)
    mfem_warning("damping layer for absorbing bc should be about 3*wavelength\n");

  {
    ifstream in(receivers_file);
    MFEM_VERIFY(in, "The file '" + string(receivers_file) + "' can't be opened");
    string line; // we read the file line-by-line
    string type; // type of the set of receivers
    while (getline(in, line))
    {
      // ignore empty lines and lines starting from '#'
      if (line.empty() || line[0] == '#') continue;
      // every meaningfull line should start with the type of the receivers set
      istringstream iss(line);
      iss >> type;
      ReceiversSet *rec_set = nullptr;
      if (type == "Line")
        rec_set = new ReceiversLine(dimension);
      else if (type == "Plane")
        rec_set = new ReceiversPlane(dimension);
      else MFEM_ABORT("Unknown type of receivers set: " + type);

      rec_set->init(in); // read the parameters
      rec_set->distribute_receivers();
      rec_set->find_cells_containing_receivers(*mesh);
      sets_of_receivers.push_back(rec_set); // put this set in the vector
    }
  }

  {
    string cmd = "mkdir -p " + (string)output_dir + " ; ";
    cmd += "mkdir -p " + (string)output_dir + "/" + SNAPSHOTS_DIR + " ; ";
    cmd += "mkdir -p " + (string)output_dir + "/" + SEISMOGRAMS_DIR + " ; ";
    const int res = system(cmd.c_str());
    MFEM_VERIFY(res == 0, "Failed to create a directory " + (string)output_dir);
  }
}

void Parameters::check_parameters() const
{
  MFEM_VERIFY(dimension == 2 || dimension == 3, "Dimension (" + d2s(dimension) +
              ") must be 2 or 3");

  grid.check_parameters();
  source.check_parameters();
  media.check_parameters();
  bc.check_parameters();
  method.check_parameters();

  MFEM_VERIFY(T > 0, "Time (" + d2s(T) + ") must be >0");
  MFEM_VERIFY(dt < T, "dt (" + d2s(dt) + ") must be < T (" + d2s(T) + ")");
  MFEM_VERIFY(step_snap > 0, "step_snap (" + d2s(step_snap) + ") must be >0");
  MFEM_VERIFY(step_seis > 0, "step_seis (" + d2s(step_seis) + ") must be >0");
}

