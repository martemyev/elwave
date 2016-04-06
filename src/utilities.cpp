#include "mfem.hpp"
#include "utilities.hpp"

#include <cmath>
#include <fstream>

using namespace std;
using namespace mfem;



double to_radians(double x)
{
  return x*M_PI/180.0;
}



void read_binary(const char *filename, int n_values, double *values)
{
  ifstream in(filename, ios::binary);
  MFEM_VERIFY(in, "File '" + string(filename) + "' can't be opened");

  in.seekg(0, in.end); // jump to the end of the file
  int length = in.tellg(); // total length of the file in bytes
  int size_value = length / n_values; // size (in bytes) of one value

  MFEM_VERIFY(length % n_values == 0, "The number of bytes in the file '" +
              string(filename) + "' is not divisible by the number of elements "
              + d2s(n_values));

  in.seekg(0, in.beg); // jump to the beginning of the file

  if (size_value == sizeof(double))
  {
    in.read((char*)values, n_values*size_value); // read all at once

    MFEM_VERIFY(n_values == static_cast<int>(in.gcount()), "The number of "
                "successfully read elements is different from the expected one");
  }
  else if (size_value == sizeof(float))
  {
    float val = 0;
    for (int i = 0; i < n_values; ++i)  // read element-by-element
    {
      in.read((char*)&val, size_value); // read a 'float' value
      values[i] = val;                  // convert it to a 'double' value
    }
  }
  else MFEM_VERIFY(0, "Unknown size of an element (" + d2s(size_value) + ") in "
                   "bytes. Expected one is either sizeof(float) = " +
                   d2s(sizeof(float)) + ", or sizeof(double) = " +
                   d2s(sizeof(double)));

  in.close();
}



void write_binary(const char *filename, int n_values, double *values)
{
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    throw std::runtime_error("File '" + std::string(filename) +
                             "' can't be opened");
  }
  float val;
  for (int i = 0; i < n_values; ++i) {
    val = values[i];
    out.write(reinterpret_cast<char*>(&val), sizeof(val));
  }
}



void get_minmax(double *a, int n_elements, double &min_val, double &max_val)
{
  min_val = max_val = a[0];
  for (int i = 1; i < n_elements; ++i)
  {
    min_val = min(min_val, a[i]);
    max_val = max(max_val, a[i]);
  }
}



void write_vts_vector(const std::string& filename, const std::string& solname,
                      double sx, double sy, double sz, int nx, int ny, int nz,
                      const Vector& sol_x, const Vector& sol_y,
                      const Vector& sol_z)
{
  ofstream out(filename.c_str());
  MFEM_VERIFY(out, "File '" + filename + "' can't be opened");

  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"StructuredGrid\" version=\"0.1\">\n";
  out << "  <StructuredGrid WholeExtent=\"1 " << nx+1 << " 1 " << ny+1 << " 1 " << nz+1 << "\">\n";
  out << "    <Piece Extent=\"1 " << nx+1 << " 1 " << ny+1 << " 1 " << nz+1 << "\">\n";
  out << "      <PointData Vectors=\"" << solname << "\" Scalars=\"" << solname << "_scalar_mag\">\n";
  out << "        <DataArray type=\"Float64\" Name=\"" << solname << "\" format=\"ascii\" NumberOfComponents=\"3\">\n";
  for (int iz = 0; iz < nz+1; ++iz)
    for (int iy = 0; iy < ny+1; ++iy)
      for (int ix = 0; ix < nx+1; ++ix)
      {
        const int glob_vert_index = iz*(nx+1)*(ny+1) + iy*(nx+1) + ix;
        out << sol_x(glob_vert_index) << " "
            << sol_y(glob_vert_index) << " "
            << sol_z(glob_vert_index) << " ";
      }
  out << "\n";
  out << "        </DataArray>\n";
  out << "        <DataArray type=\"Float64\" Name=\"" << solname << "_scalar_mag\" format=\"ascii\" NumberOfComponents=\"1\">\n";
  for (int iz = 0; iz < nz+1; ++iz)
    for (int iy = 0; iy < ny+1; ++iy)
      for (int ix = 0; ix < nx+1; ++ix)
      {
        const int glob_vert_index = iz*(nx+1)*(ny+1) + iy*(nx+1) + ix;
        double mag = pow(sol_x(glob_vert_index), 2) +
                     pow(sol_y(glob_vert_index), 2) +
                     pow(sol_z(glob_vert_index), 2);
        out << sqrt(mag) << " ";
      }
  out << "\n";
  out << "        </DataArray>\n";
  out << "      </PointData>\n";
  out << "      <Points>\n";
  out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";

  const double hx = sx / nx;
  const double hy = sy / ny;
  const double hz = sz / nz;

  for (int iz = 0; iz < nz+1; ++iz)
  {
    const double z = (iz == nz ? sz : iz*hz);
    for (int iy = 0; iy < ny+1; ++iy)
    {
      const double y = (iy == ny ? sy : iy*hy);
      for (int ix = 0; ix < nx+1; ++ix)
      {
        const double x = (ix == nx ? sx : ix*hx);
        out << x << " " << y << " " << z << " ";
      }
    }
  }

  out << "\n";
  out << "        </DataArray>\n";
  out << "      </Points>\n";
  out << "    </Piece>\n";
  out << "  </StructuredGrid>\n";
  out << "</VTKFile>\n";

  out.close();
}



void write_vts_scalar(const std::string& filename, const std::string& solname,
                      double sx, double sy, double sz, int nx, int ny, int nz,
                      const Vector& sol)
{
  ofstream out(filename.c_str());
  MFEM_VERIFY(out, "File '" + filename + "' can't be opened");

  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"StructuredGrid\" version=\"0.1\">\n";
  out << "  <StructuredGrid WholeExtent=\"1 " << nx+1 << " 1 " << ny+1 << " 1 " << nz+1 << "\">\n";
  out << "    <Piece Extent=\"1 " << nx+1 << " 1 " << ny+1 << " 1 " << nz+1 << "\">\n";
  out << "      <PointData Scalars=\"" << solname << "\">\n";
  out << "        <DataArray type=\"Float64\" Name=\"" << solname << "\" format=\"ascii\" NumberOfComponents=\"1\">\n";

  for (int iz = 0; iz < nz+1; ++iz)
    for (int iy = 0; iy < ny+1; ++iy)
      for (int ix = 0; ix < nx+1; ++ix)
      {
        const int glob_vert_index = iz*(nx+1)*(ny+1) + iy*(nx+1) + ix;
        out << sol(glob_vert_index) << " ";
      }

  out << "\n";
  out << "        </DataArray>\n";
  out << "      </PointData>\n";
  out << "      <Points>\n";
  out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";

  const double hx = sx / nx;
  const double hy = sy / ny;
  const double hz = sz / nz;

  for (int iz = 0; iz < nz+1; ++iz)
  {
    const double z = (iz == nz ? sz : iz*hz);
    for (int iy = 0; iy < ny+1; ++iy)
    {
      const double y = (iy == ny ? sy : iy*hy);
      for (int ix = 0; ix < nx+1; ++ix)
      {
        const double x = (ix == nx ? sx : ix*hx);
        out << x << " " << y << " " << z << " ";
      }
    }
  }

  out << "\n";
  out << "        </DataArray>\n";
  out << "      </Points>\n";
  out << "    </Piece>\n";
  out << "  </StructuredGrid>\n";
  out << "</VTKFile>\n";

  out.close();
}



void get_limits(const Mesh &mesh, const Element &element,
                std::vector<double> &limits)
{
  const int dim = mesh.Dimension();

  Array<int> vert_indices;
  element.GetVertices(vert_indices);

  const double *vert0 = mesh.GetVertex(vert_indices[0]); // min coords
  double x0 = vert0[0];
  double y0 = vert0[1];
  double z0 = (dim == 3 ? vert0[2] : 0.);
  double x1 = x0, y1 = y0, z1 = z0;
  for (int i = 1; i < vert_indices.Size(); ++i)
  {
    const double *vert = mesh.GetVertex(vert_indices[i]);
    x0 = std::min(x0, vert[0]);
    x1 = std::max(x1, vert[0]);
    y0 = std::min(y0, vert[1]);
    y1 = std::max(y1, vert[1]);
    if (dim == 3)
    {
      z0 = std::min(z0, vert[2]);
      z1 = std::max(z1, vert[2]);
    }
  }

  limits[0] = x0;
  limits[1] = y0;
  limits[2] = z0;
  limits[3] = x1;
  limits[4] = y1;
  limits[5] = z1;
}



int find_element(const Mesh &mesh, const Vertex &point, bool throw_exception)
{
  const int dim = mesh.Dimension();
  MFEM_VERIFY(dim == 2 || dim == 3, "Wrong dimension");

  const double px = point(0); // coordinates of the point of interest
  const double py = point(1);
  const double pz = (dim == 3 ? point(2) : 0.);

  const double tol = FIND_CELL_TOLERANCE;

  for (int el = 0; el < mesh.GetNE(); ++el)
  {
    const Element *element = mesh.GetElement(el);
    if (dim == 2)
    {
      MFEM_VERIFY(dynamic_cast<const Quadrilateral*>(element), "The mesh "
                  "element has to be a quadrilateral");
    }
    else // 3D
    {
      MFEM_VERIFY(dynamic_cast<const Hexahedron*>(element), "The mesh "
                  "element has to be a hexahedron");
    }

    std::vector<double> limits(6);
    get_limits(mesh, *element, limits);

    const double x0 = limits[0];
    const double y0 = limits[1];
    const double z0 = limits[2];
    const double x1 = limits[3];
    const double y1 = limits[4];
    const double z1 = limits[5];

    if (dim == 2)
    {
      if (px > x0 - tol && px < x1 + tol &&
          py > y0 - tol && py < y1 + tol)
        return el;
    }
    else // 3D
    {
      if (px > x0 - tol && px < x1 + tol &&
          py > y0 - tol && py < y1 + tol &&
          pz > z0 - tol && pz < z1 + tol)
        return el;
    }
  }

  if (throw_exception)
  {
    if (dim == 2)
    {
      MFEM_ABORT("The given point [" + d2s(px) + "," + d2s(py) +
                 "] doesn't belong to the mesh");
    }
    else
    {
      MFEM_ABORT("The given point [" + d2s(px) + "," + d2s(py) + "," + d2s(pz) +
                 "] doesn't belong to the mesh");
    }
  }

  return -1; // to show that the point in not here
}



//------------------------------------------------------------------------------
//
// Check endianness
//
//------------------------------------------------------------------------------
bool is_big_endian()
{
  union
  {
    int i;
    char c[sizeof(int)];
  } x;
  x.i = 1;
  return x.c[0] == 1;
}

//------------------------------------------------------------------------------
//
// Get the endianness of the machine
//
//------------------------------------------------------------------------------
std::string endianness()
{
  return (is_big_endian() ? "BigEndian" : "LittleEndian");
}



//------------------------------------------------------------------------------
//
// Name of a file without a path
//
//------------------------------------------------------------------------------
std::string file_name(const std::string &path)
{
  if (path == "") return path;

  size_t pos = 0;
#if defined(__linux__) || defined(__APPLE__)
  pos = path.find_last_of('/');
#elif defined(_WIN32)
  pos = path.find_last_of('\\');
#endif

  if (pos == std::string::npos)
    return path; // there is no '/' in the path, so this is the filename

  return path.substr(pos + 1);
}

//------------------------------------------------------------------------------
//
// Path of a given file
//
//------------------------------------------------------------------------------
std::string file_path(const std::string &path)
{
  if (path == "") return path;

  size_t pos = 0;
#if defined(__linux__) || defined(__APPLE__)
  pos = path.find_last_of('/');
#elif defined(_WIN32)
  pos = path.find_last_of('\\');
#endif

  if (pos == std::string::npos)
    return ""; // there is no '/' in the path, the path is "" then

  return path.substr(0, pos + 1);
}

//------------------------------------------------------------------------------
//
// Stem of a given file (no path, no extension)
//
//------------------------------------------------------------------------------
std::string file_stem(const std::string &path)
{
  if (path == "") return path;

  // get a file name from the path
  const std::string fname = file_name(path);

  // extract a stem and return it
  size_t pos = fname.find_last_of('.');
  if (pos == std::string::npos)
    return fname; // there is no '.', so this is the stem

  return fname.substr(0, pos);
}

//------------------------------------------------------------------------------
//
// Extension of a given file
//
//------------------------------------------------------------------------------
std::string file_extension(const std::string &path)
{
  if (path == "") return path;

  // extract a file name from the path
  const std::string fname = file_name(path);

  size_t pos = fname.find_last_of('.');
  if (pos == std::string::npos)
    return ""; // there is no '.', so there is no extension

  // extract an extension and return it
  return fname.substr(pos);
}

//------------------------------------------------------------------------------
//
// Check if the given file exists
//
//------------------------------------------------------------------------------
bool file_exists(const std::string &path)
{
  if (path == "") return false; // no file - no existence

  // This is not the fastest method, but it should work on all operating
  // systems. Some people also not that this method check 'availibility' of the
  // file, not its 'existance'. But that's what we actually need. If a file
  // exists, but it's not available (even for reading), we believe, that the
  // file doesn't exist.
  bool exists = false;
  std::ifstream in(path.c_str());
  if (in.good())
    exists = true; // file exists and is in a good state
  in.close();

  return exists;
}

