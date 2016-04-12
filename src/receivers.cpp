#include "mfem.hpp"
#include "receivers.hpp"

using namespace mfem;

//==============================================================================
//
// ReceiversSet
//
//==============================================================================
ReceiversSet::ReceiversSet(int d)
  : _variable(""),
    _n_receivers(0),
    _receivers(),
    _cells_containing_receivers(),
    _dimension(d)
{
  MFEM_VERIFY(_dimension == 2 || _dimension == 3, "Incorrect dimension");
}

void ReceiversSet::
find_cells_containing_receivers(const Mesh &mesh)
{
  MFEM_VERIFY(!_receivers.empty(), "The receivers haven't been distributed yet");

  _cells_containing_receivers.clear();
  _cells_containing_receivers.resize(_n_receivers);

  // we throw an exception if we don't find a cell containing a receiver
  const bool throw_exception = true;

  for (int p = 0; p < _n_receivers; ++p)
  {
    _cells_containing_receivers[p] = find_element(mesh, _receivers[p],
                                                  throw_exception);
#if defined(SHOW_CELLS_CONTAINING_RECEIVERS)
    std::cout << p << " ";
    for (int i = 0; i < mesh.Dimension(); ++i)
      std::cout << _receivers[p](i) << " ";
    std::cout << _cells_containing_receivers[p] << "\n";
#endif // SHOW_CELLS_CONTAINING_RECEIVERS
  }
}




//==============================================================================
//
// ReceiversLine
//
//==============================================================================
ReceiversLine::ReceiversLine(int d)
  : ReceiversSet(d),
    _start(),
    _end()
{ }

void ReceiversLine::init(std::ifstream &in)
{
  MFEM_VERIFY(in.is_open(), "The stream for reading receivers is not open");

  std::string tmp;

  in >> _variable;
  in >> _n_receivers; getline(in, tmp);

  if (_dimension == 2)
  {
    in >> _start(0) >> _start(1); getline(in, tmp);
    in >> _end(0)   >> _end(1);   getline(in, tmp);
  }
  else // 3D
  {
    in >> _start(0) >> _start(1) >> _start(2); getline(in, tmp);
    in >> _end(0)   >> _end(1)   >> _end(2);   getline(in, tmp);
  }

  MFEM_VERIFY(_n_receivers > 0, "The number of receivers (" + d2s(_n_receivers)+
              ") must be >0");
}

void ReceiversLine::distribute_receivers()
{
  const double x0 = _start(0);
  const double y0 = _start(1);
  const double z0 = (_dimension == 3 ? _start(2) : 0.);
  const double x1 = _end(0);
  const double y1 = _end(1);
  const double z1 = (_dimension == 3 ? _end(2) : 0.);

  _receivers.resize(_n_receivers);

  const double dx = (x1 - x0) / (_n_receivers-1);
  const double dy = (y1 - y0) / (_n_receivers-1);
  const double dz = (z1 - z0) / (_n_receivers-1);

  for (int i = 0; i < _n_receivers; ++i)
  {
    const double x = (i == _n_receivers-1 ? x1 : x0 + i*dx);
    const double y = (i == _n_receivers-1 ? y1 : y0 + i*dy);
    const double z = (i == _n_receivers-1 ? z1 : z0 + i*dz);
    if (_dimension == 2)
      _receivers[i] = Vertex(x, y);
    else // 3D
      _receivers[i] = Vertex(x, y, z);
  }
}

std::string ReceiversLine::description() const
{
  if (_dimension == 2)
    return "_rec_line_x" + d2s(_start(0)) + "_" + d2s(_end(0)) +
           "_y" + d2s(_start(1)) + "_" + d2s(_end(1));
  else
    return "_rec_line_x" + d2s(_start(0)) + "_" + d2s(_end(0)) +
           "_y" + d2s(_start(1)) + "_" + d2s(_end(1)) +
           "_z" + d2s(_start(2)) + "_" + d2s(_end(2));
}




//==============================================================================
//
// ReceiversPlane
//
//==============================================================================
ReceiversPlane::ReceiversPlane(int d)
  : ReceiversSet(d), _n_points_1(0), _n_points_2(0), _plane("unknown")
{
  MFEM_VERIFY(_dimension == 3, "Plane receivers doesn't make sense in 2D");
}

void ReceiversPlane::init(std::ifstream &in)
{
  MFEM_VERIFY(in.is_open(), "The stream for reading receivers is not open");

  std::string tmp;

  in >> _variable; getline(in, tmp);
  in >> _n_points_1 >> _n_points_2; getline(in, tmp);
  for (int v = 0; v < N_VERTICES; ++v) {
    in >> _vertices[v](0) >> _vertices[v](1) >> _vertices[v](2);
    getline(in, tmp);
  }

  _n_receivers = _n_points_1*_n_points_2;
  MFEM_VERIFY(_n_receivers > 0, "The number of receivers (" + d2s(_n_receivers)+
              ") must be >0");
}

void ReceiversPlane::distribute_receivers()
{
   _receivers.resize(_n_receivers);

  double x0 = _vertices[0](0);
  double y0 = _vertices[0](1);
  double z0 = _vertices[0](2);
  double x1 = _vertices[0](0);
  double y1 = _vertices[0](1);
  double z1 = _vertices[0](2);
  for (int v = 1; v < N_VERTICES; ++v)
  {
    x0 = std::min(x0, _vertices[v](0));
    y0 = std::min(y0, _vertices[v](1));
    z0 = std::min(z0, _vertices[v](2));
    x1 = std::max(x1, _vertices[v](0));
    y1 = std::max(y1, _vertices[v](1));
    z1 = std::max(z1, _vertices[v](2));
  }

  if (fabs(x1 - x0) < FLOAT_NUMBERS_EQUALITY_TOLERANCE)
    distribute_receivers_YZ_plane(x0, y0, y1, z0, z1);
  else if (fabs(y1 - y0) < FLOAT_NUMBERS_EQUALITY_TOLERANCE)
    distribute_receivers_XZ_plane(x0, x1, y0, z0, z1);
  else if (fabs(z1 - z0) < FLOAT_NUMBERS_EQUALITY_TOLERANCE)
    distribute_receivers_XY_plane(x0, x1, y0, y1, z0);
  else MFEM_ABORT("Receivers plane is not parallel to XY, XZ or YZ");
}

void ReceiversPlane::distribute_receivers_YZ_plane(double x,
                                                   double y0, double y1,
                                                   double z0, double z1)
{
  _plane = "YZ";

  const double dy = (y1 - y0) / (_n_points_1-1);
  const double dz = (z1 - z0) / (_n_points_2-1);

  int p = 0;
  for (int i = 0; i < _n_points_2; ++i)
  {
    double z = (i == _n_points_2-1 ? z1 : z0 + i*dz);
    for (int j = 0; j < _n_points_1; ++j)
    {
      double y = (j == _n_points_1-1 ? y1 : y0 + j*dy);
      _receivers[p++] = Vertex(x, y, z);
    }
  }
}

void ReceiversPlane::distribute_receivers_XZ_plane(double x0, double x1,
                                                   double y,
                                                   double z0, double z1)
{
  _plane = "XZ";

  const double dx = (x1 - x0) / (_n_points_1-1);
  const double dz = (z1 - z0) / (_n_points_2-1);

  int p = 0;
  for (int i = 0; i < _n_points_2; ++i)
  {
    double z = (i == _n_points_2-1 ? z1 : z0 + i*dz);
    for (int j = 0; j < _n_points_1; ++j)
    {
      double x = (j == _n_points_1-1 ? x1 : x0 + j*dx);
      _receivers[p++] = Vertex(x, y, z);
    }
  }
}

void ReceiversPlane::distribute_receivers_XY_plane(double x0, double x1,
                                                   double y0, double y1,
                                                   double z)
{
  _plane = "XY";

  const double dx = (x1 - x0) / (_n_points_1-1);
  const double dy = (y1 - y0) / (_n_points_2-1);

  int p = 0;
  for (int i = 0; i < _n_points_2; ++i)
  {
    double y = (i == _n_points_2-1 ? y1 : y0 + i*dy);
    for (int j = 0; j < _n_points_1; ++j)
    {
      double x = (j == _n_points_1-1 ? x1 : x0 + j*dx);
      _receivers[p++] = Vertex(x, y, z);
    }
  }
}

std::string ReceiversPlane::description() const
{
  const double x = _vertices[0](0);
  const double y = _vertices[0](1);
  const double z = _vertices[0](2);
  std::string coord = (_plane == "YZ" ? "_x" + d2s(x) :
                                        (_plane == "XZ" ? "_y" + d2s(y) :
                                                          "_z" + d2s(z)));
  return "_rec_plane" + _plane + coord;
}

