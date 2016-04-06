#ifndef SOURCE_HPP
#define SOURCE_HPP

#include "config.hpp"
#include "mfem.hpp"

class Parameters;
class SourceParameters;



double RickerWavelet(const SourceParameters& source, double t);
double GaussFirstDerivative(const SourceParameters& source, double t);

void PointForce(const SourceParameters& source, const mfem::Vector& location,
                const mfem::Vector& x, mfem::Vector& f);
void MomentTensor(const SourceParameters& source, const mfem::Vector& location,
                  const mfem::Vector& x, mfem::Vector& f);

void DeltaPointForce(const SourceParameters& source,
                     const mfem::Vector& location, const mfem::Vector& x,
                     mfem::Vector& f);
void GaussPointForce(const SourceParameters& source,
                     const mfem::Vector& location, const mfem::Vector& x,
                     mfem::Vector& f);
void DivDeltaMomentTensor(const SourceParameters&, const mfem::Vector&,
                          const mfem::Vector&, mfem::Vector&);
void DivGaussMomentTensor(const SourceParameters& source,
                          const mfem::Vector& location, const mfem::Vector& x,
                          mfem::Vector& f);



/**
 * Implementation of a vector point force type of source.
 */
class VectorPointForce: public mfem::VectorCoefficient
{
public:
  VectorPointForce(int dim, const Parameters& p);
  ~VectorPointForce() { }

  void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip);

private:
  const Parameters& param;
  mfem::Vector location;
};



/**
 * Implementation of a moment tensor type of source.
 */
class MomentTensorSource: public mfem::VectorCoefficient
{
public:
  MomentTensorSource(int dim, const Parameters& p);
  ~MomentTensorSource() { }

  void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip);

private:
  const Parameters& param;
  mfem::Vector location;
};



/**
 * Implementation of a plane wave type of source.
 */
class PlaneWaveSource: public mfem::VectorCoefficient
{
public:
  PlaneWaveSource(int dim, const Parameters& p);
  ~PlaneWaveSource() { }

  void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip);

private:
  const Parameters& param;
};

#endif // SOURCE_HPP
