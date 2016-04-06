#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include "config.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace mfem
{
  class Mesh;
  class Vector;
}

/**
 * Convert the data of any type which has oveloaded operator '<<' to string
 * @param data - the data
 * @param scientific - use scientific (exponential) format or not
 * @param precision - if scientific format is used, we can change the precision
 * @param noperiod - if true and the floating point number is being converted,
 * the period separating the number is removed
 * @param width - the width of a resulting string (filled with 0 if the number
 * is shorter)
 * @return data in string format
 */
template <typename T>
inline std::string d2s(T data,
                       bool scientific = false,
                       int precision = 0,
                       bool noperiod = false,
                       int width = 0)
{
  const char filler = '0';
  std::ostringstream o;
  if (scientific)    o.setf(std::ios::scientific);
  if (precision > 0) o.precision(precision);
  if (width > 0)     o << std::setfill(filler) << std::setw(width);

  if (!(o << data))
    throw std::runtime_error("Bad conversion of data to string!");

  // eliminate a period in case of floating-point numbers in non-scientific
  // format
  if (!scientific && noperiod)
  {
    std::string res = o.str(); // resulting string
    std::string::size_type pos = res.find('.');
    if (pos != std::string::npos)
      res.erase(pos, 1);
    if (width > 0 && static_cast<int>(res.size()) < width)
      res.insert(0, width-res.size(), filler);
    return res;
  }

  return o.str();
}

/**
 * Convert an angle in degrees to radians.
 */
double to_radians(double x);

/**
 * Read a binary file
 */
void read_binary(const char *filename, int n_values, double *values);

/**
 * Write a binary file
 */
void write_binary(const char *filename, int n_values, double *values);

/**
 * Find min and max values of the given array (vector) a
 */
void get_minmax(double *a, int n_elements, double &min_val, double &max_val);

/**
 * Write a snapshot of a vector wavefield in a VTS format
 * @param filename - output file name
 * @param solname - name of the wavefield
 * @param sx - size of domain in x-direction
 * @param sy - size of domain in y-direction
 * @param sz - size of domain in z-direction
 * @param nx - number of cells in x-direction
 * @param ny - number of cells in y-direction
 * @param nz - number of cells in z-direction
 * @param sol_x - x-component of the vector wavefield
 * @param sol_y - y-component of the vector wavefield
 */
void write_vts_vector(const std::string& filename, const std::string& solname,
                      double sx, double sy, double sz, int nx, int ny, int nz,
                      const mfem::Vector& sol_x, const mfem::Vector& sol_y,
                      const mfem::Vector& sol_z);

/**
 * Write scalar values in a VTS format
 * @param filename - output file name
 * @param solname - name of the wavefield
 * @param sx - size of domain in x-direction
 * @param sy - size of domain in y-direction
 * @param sz - size of domain in z-direction
 * @param nx - number of cells in x-direction
 * @param ny - number of cells in y-direction
 * @param nz - number of cells in z-direction
 * @param sol - scalar values
 */
void write_vts_scalar(const std::string& filename, const std::string& solname,
                      double sx, double sy, double sz, int nx, int ny, int nz,
                      const mfem::Vector& sol);

void get_limits(const mfem::Mesh &mesh, const mfem::Element &element,
                std::vector<double> &limits);

int find_element(const mfem::Mesh &mesh, const mfem::Vertex &point,
                 bool throw_exception);

std::string endianness();

std::string file_name(const std::string &path);

std::string file_path(const std::string &path);

std::string file_stem(const std::string &path);

std::string file_extension(const std::string &path);

bool file_exists(const std::string &path);

#endif // UTILITIES_HPP
