#include "source.hpp"
#include "utilities.hpp"
#include "parameters.hpp"

using namespace std;
using namespace mfem;

double RickerWavelet(const SourceParameters& source, double t)
{
  const double f = source.frequency;
  const double a  = M_PI*f*(t-1.0/f);
  return source.scale*(1.-2.*a*a)*exp(-a*a);
}

double GaussFirstDerivative(const SourceParameters& source, double t)
{
  const double f = source.frequency;
  const double a = M_PI*f*(t-1.0/f);
  return source.scale*(t-1.0/f)*exp(-a*a);
}

void PointForce(const SourceParameters& source, const Vector& location,
                const Vector& x, Vector& f)
{
  if (!strcmp(source.spatial_function, "delta"))
    DeltaPointForce(source, location, x, f);
  else if (!strcmp(source.spatial_function, "gauss"))
    GaussPointForce(source, location, x, f);
  else
    MFEM_ABORT("Unknown spatial function: " + string(source.spatial_function));
}

void MomentTensor(const SourceParameters& source, const Vector& location,
                  const Vector& x, Vector& f)
{
  if (!strcmp(source.spatial_function, "delta"))
    DivDeltaMomentTensor(source, location, x, f);
  else if (!strcmp(source.spatial_function, "gauss"))
    DivGaussMomentTensor(source, location, x, f);
  else
    MFEM_ABORT("Unknown spatial function: " + string(source.spatial_function));
}

void DeltaPointForce(const SourceParameters& source,
                     const Vector& location, const Vector& x, Vector& f)
{
  const double tol = FLOAT_NUMBERS_EQUALITY_TOLERANCE;
  const double loc[] = { location(0), location(1), location(2) };
  double value = 0.0;
  if (x.DistanceTo(loc) < tol)
    value = 1.0;

  f = 0.0;
  f(source.direction-1) = value;
}

void GaussPointForce(const SourceParameters& source,
                     const Vector& location, const Vector& x, Vector& f)
{
  const double xdiff  = x(0) - location(0);
  const double ydiff  = x(1) - location(1);
  const double zdiff  = x(2) - location(2);
  const double xdiff2 = xdiff*xdiff;
  const double ydiff2 = ydiff*ydiff;
  const double zdiff2 = zdiff*zdiff;
  const double h2 = source.gauss_support * source.gauss_support;
  const double G = exp(-(xdiff2 + ydiff2 + zdiff2) / h2);
  f = 0.0;
  f(source.direction-1) = G;
}

void DivDeltaMomentTensor(const SourceParameters&, const Vector&, const Vector&,
                          Vector&)
{
  MFEM_ABORT("NOT implemented");
}

void DivGaussMomentTensor(const SourceParameters& source,
                          const Vector& location, const Vector& x, Vector& f)
{
  const double xdiff  = x(0) - location(0);
  const double ydiff  = x(1) - location(1);
  const double zdiff  = x(2) - location(2);
  const double xdiff2 = xdiff*xdiff;
  const double ydiff2 = ydiff*ydiff;
  const double zdiff2 = zdiff*zdiff;
  const double h2 = source.gauss_support * source.gauss_support;
  const double exp_val = exp(-(xdiff2 + ydiff2 + zdiff2) / h2);
  const double Gx = -2.*xdiff/h2 * exp_val;
  const double Gy = -2.*ydiff/h2 * exp_val;
  const double Gz = -2.*zdiff/h2 * exp_val;

  f(0) = source.Mxx*Gx + source.Mxy*Gy + source.Mxz*Gz;
  f(1) = source.Mxy*Gx + source.Myy*Gy + source.Myz*Gz;
  f(2) = source.Mxz*Gx + source.Myz*Gy + source.Mzz*Gz;
}



//------------------------------------------------------------------------------
//
// A source represented by a vector point force.
//
//------------------------------------------------------------------------------
VectorPointForce::VectorPointForce(int dim, const Parameters& p)
  : VectorCoefficient(dim)
  , param(p)
{
  location.SetSize(vdim);
  for (int i = 0; i < vdim; ++i)
    location(i) = param.source.location(i);
}

void VectorPointForce::Eval(Vector &V, ElementTransformation &T,
                            const IntegrationPoint &ip)
{
  Vector transip;
  T.Transform(ip, transip);
  V.SetSize(vdim);
  PointForce(param.source, location, transip, V);
}



//------------------------------------------------------------------------------
//
// A source represented by a divergence of the moment tensor density.
//
//------------------------------------------------------------------------------
MomentTensorSource::MomentTensorSource(int dim, const Parameters& p)
  : VectorCoefficient(dim)
  , param(p)
{
  location.SetSize(vdim);
  for (int i = 0; i < vdim; ++i)
    location(i) = param.source.location(i);
}

void MomentTensorSource::Eval(Vector &V, ElementTransformation &T,
                              const IntegrationPoint &ip)
{
  Vector transip;
  T.Transform(ip, transip);
  V.SetSize(vdim);
  MomentTensor(param.source, location, transip, V);
}



//------------------------------------------------------------------------------
//
// A set of source distributed along a plane.
//
//------------------------------------------------------------------------------
PlaneWaveSource::PlaneWaveSource(int dim, const Parameters& p)
  : VectorCoefficient(dim)
  , param(p)
{ }

void PlaneWaveSource::Eval(Vector &V, ElementTransformation &T,
                           const IntegrationPoint &ip)
{
  Vector transip;
  T.Transform(ip, transip);
  V.SetSize(vdim);

//  const double px = transip(0); // coordinates of the point
  const double py = transip(1);
//  const double pz = transip(2);

//  const double X0 = 0.0;
//  const double Z0 = 0.0;
//  const double X1 = param.grid.sx;
//  const double Z1 = param.grid.sz;
//  const double layer = param.bc.damp_layer;
  const double tol = FLOAT_NUMBERS_EQUALITY_TOLERANCE;

  // if the point 'transip' is on the plane of the plane wave, and it's not in
  // the absorbing layer, we have a source located at the exact same point
  if (fabs(py - param.source.location(1)) < tol) // &&
      //px-layer+tol > X0 && px+layer-tol < X1  &&
      //pz-layer+tol > Z0 && pz+layer-tol < Z1)
  {
    if (!strcmp(param.source.type, "pointforce"))
      PointForce(param.source, transip, transip, V);
    else if (!strcmp(param.source.type, "momenttensor"))
      MomentTensor(param.source, transip, transip, V);
    else MFEM_ABORT("Unknown source type: " + string(param.source.type));
  }
  else
    V = 0.0;
}
