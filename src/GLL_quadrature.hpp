#ifndef GLL_QUADRATURE_HPP
#define GLL_QUADRATURE_HPP

#include "config.hpp"
#include "mfem.hpp"

/**
 * Initialization of a Gauss-Lobatto-Legendre quadrature rule on a reference
 * segment [0, 1].
 * @param p - order
 * @param segment_GLL - integration rule to be set up
 */
void create_segment_GLL_rule(int p, mfem::IntegrationRule& segment_GLL);

/**
 * Computation of Gauss-Lobatto-Legendre quadrature points at a reference
 * segment [-1, 1] and corresponding weights. Adopted from a Matlab code written
 * by Greg von Winckel.
 * @param p - order
 * @param x - GLL points
 * @param w - GLL weights
 */
void segment_GLL_quadrature(int p, mfem::Vector& x, mfem::Vector& w,
                            double tol = 2e-16, int maxiter = 10000);

/**
 * ||a-b||_{L^Infinity}
 */
double LInfDiff(const mfem::Vector& a, const mfem::Vector& b);

#endif // GLL_QUADRATURE_HPP
