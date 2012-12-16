#ifndef SRC_RAND_BOUND_GENERATOR_H
#define SRC_RAND_BOUND_GENERATOR_H
#include <math.h>
#include <cstdlib>
#include <assert.h>

namespace blr
{
namespace math
{

/// <summary> Partition consecutive intervals of size bound mapped to the
///   numbers [0, bound - 1].
/// </summary>
inline
int RandBound(const int bound)
{
  assert(bound <= RAND_MAX);
  int factor = ((RAND_MAX - bound) / bound) + 1;
  int limit = factor * bound;
  int r;
  do
  {
    r = rand();
  } while (r >= limit);
  return r / factor;
}

/// <summary> Partition consecutive intervals of size bound mapped to the
///   numbers [0, bound - 1].
/// </summary>
struct RandBoundGenerator
{
  RandBoundGenerator(const int bound_);
  int operator()() const;
  int bound;
  int factor;
  int limit;
};

template <typename NumericType>
inline
NumericType RandUniform()
{
  static RandBoundGenerator s_rand(RAND_MAX);
  return static_cast<NumericType>(s_rand()) /
         static_cast<NumericType>(RAND_MAX - 1);
}

// **ATTENTION*******************ATTENTION*************************
// This code has a numerical BUG. It can return NaN and INF!
// **ATTENTION*******************ATTENTION*************************
///// <summary> Generate values from a normal distribution using ratio of uniforms. </summary>
///// <remarks>
/////   <para> Uses two uniform random variables to generate values from a normal
/////     distribution. Code taken from:
/////       William H. Press, Saul A. Teukolsky, William T. Vetterling, and Brian
/////       P. Flannery. 2007. Numerical Recipes 3rd Edition: The Art of
/////       Scientific Computing (3 ed.). Cambridge University Press, New York,
/////       NY, USA.
/////   </para>
///// </remarks>
//inline
//double RatioOfUniforms(const double mu, const double sig)
//{
//  // Uses a squeeze on the cartesion plot of standard distribution region
//  // to reject efficiently (u,v) not in the allowed region. Since (u,v) is
//  // selected uniformly, the coordinates allowed model the normal distribution
//  // with the desired mean and variance.
//  double u, v, x, y, q;
//  do
//  {
//    u = RandUniform<double>();
//    {
//      v = 1.7156 * (RandUniform<double>() - 0.5);
//    }
//    x = u - 0.449871;
//    y = fabs(v) + 0.386596;
//    q = (x * x) + (y * ((0.19600 * y) - (0.25472 * x)));
//  } while ((q > 0.27597) &&
//           ((q > 0.27846) || ((v * v) > (-4.0 * log(u) * (u * u)))));
//  return mu + (sig * (v / u));
//}
//
//struct RatioUniformGenerator
//{
//  RatioUniformGenerator(double mu_, double sig_);
//
//  double operator()() const;
//
//  double mu;
//  double sig;
//};

////////////////////////////////////////////////////////////////////////////////
// Inline definitions.
////////////////////////////////////////////////////////////////////////////////
inline
RandBoundGenerator::RandBoundGenerator(const int bound_)
: bound(bound_),
  factor(((RAND_MAX - bound) / bound) + 1),
  limit(factor * bound)
{
  assert(bound <= RAND_MAX);
}

inline
int RandBoundGenerator::operator()() const
{
  int r;
  do
  {
    r = rand();
  } while (r >= limit);
  return r / factor;
}

//inline
//RatioUniformGenerator::RatioUniformGenerator(double mu_, double sig_)
//: mu(mu_), sig(sig_)
//{}

//inline
//double RatioUniformGenerator::operator()() const
//{
//  return RatioOfUniforms(mu, sig);
//}

} // end ns math
using namespace math;
} // end ns blr

#endif //SRC_RAND_BOUND_GENERATOR_H
