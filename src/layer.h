/* Copyright (C) 2012 Brandon L. Reiss
   brandon@brandonreiss.com

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/
#ifndef SRC_LAYER_H
#define SRC_LAYER_H
#include "type_utils.h"
#include "static_assert.h"

#include "opencv/cv.h"
#include <sstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <assert.h>

// br - Enable to perform lots of numerical checks during computation.
#define DETECT_NUMERICAL_ERRORS_ENABED
#if defined(DETECT_NUMERICAL_ERRORS_ENABED)
#define DETECT_NUMERICAL_ERRORS(cvMat)                                                            \
  {                                                                                               \
    /* Compute the sum and norm squared. */                                                       \
    const double __detect_numerical_errors_norm_sq = (cvMat).dot((cvMat));                        \
    const double __detect_numerical_errors_sum = cv::sum((cvMat)).val[0];                         \
    const bool __detect_numerical_errors_nans =                                                   \
      (my_isnan(__detect_numerical_errors_norm_sq) || my_isnan(__detect_numerical_errors_sum));   \
    const bool __detect_numerical_errors_inf =                                                    \
      (my_isinf(__detect_numerical_errors_norm_sq) || my_isinf(__detect_numerical_errors_sum));   \
    const bool __detect_numerical_errors_valid =                                                  \
      (!__detect_numerical_errors_nans && !__detect_numerical_errors_inf) &&                      \
      (-1e20 < __detect_numerical_errors_sum && __detect_numerical_errors_sum < 1e20);            \
    if (!__detect_numerical_errors_valid)                                                         \
    {                                                                                             \
      std::stringstream ssMsg;                                                                    \
      ssMsg << __FILE__ << "(" << __LINE__ << ") : "                                              \
                << "Detected numerical issue "                                                    \
                   "||" << #cvMat << "||^2 = " << __detect_numerical_errors_norm_sq << ", "       \
                   "SUM(" << #cvMat << ") = " << __detect_numerical_errors_sum << std::endl;      \
      std::cout << ssMsg.str(); std::cout.flush();                                                \
    }                                                                                             \
  }
#else
#define DETECT_NUMERICAL_ERRORS(...)
#endif

namespace blr
{
namespace nn
{

// Numeric test function from http://www.devx.com/tips/Tip/42853.
inline
int my_isnan(double x)
{
  return x != x;
}
inline
int my_isinf(double x)
{
  if ((x == x) && ((x - x) != 0.0))
  {
    return (x < 0.0 ? -1 : 1);
  }
  else
  {
    return 0;
  }
}

template <typename LayerType_>
class StandardLayer
{
public:
  typedef LayerType_ LayerType;

  StandardLayer();
};

template <int NumInputs_, typename NumericType_ = double>
class Passthrough
  : public StandardLayer<Passthrough<NumInputs_, NumericType_> >
{
public:
  // API definitions.
  typedef NumericType_ NumericType;
  enum { NumInputs = NumInputs_, };
  enum { NumOutputs = NumInputs_, };
  enum { NumParameters = 0, };

  static void Forward(const cv::Mat& X, const cv::Mat& W, cv::Mat* Y);

  static void Backward(const cv::Mat& X, const cv::Mat& W, const cv::Mat& Y,
                       const cv::Mat& dLdY, cv::Mat* dLdW, cv::Mat* dLdX);
  static double ComputeTruncateL2Factor(const cv::Mat& W, NumericType maxNormSq);
};

template <int NumInputs_, int NumOutputs_, typename NumericType_ = double>
class Linear
  : public StandardLayer<Linear<NumInputs_, NumOutputs_, NumericType_> >
{
public:
  // API definitions.
  typedef NumericType_ NumericType;
  enum { NumInputs = NumInputs_, };
  enum { NumOutputs = NumOutputs_, };
  enum { NumParameters = (NumInputs * NumOutputs) + NumOutputs, };

  // Non-API definitions.
  enum { ParamsLinearMat = NumInputs * NumOutputs };

  static void Forward(const cv::Mat& X, const cv::Mat& W, cv::Mat* Y);

  static void Backward(const cv::Mat& X, const cv::Mat& W, const cv::Mat& Y,
                       const cv::Mat& dLdY, cv::Mat* dLdW, cv::Mat* dLdX);
  static double ComputeTruncateL2Factor(const cv::Mat& W, NumericType maxNormSq);
};

template <int NumInputs_, typename NumericType_ = double>
class Tanh
  : public StandardLayer<Tanh<NumInputs_, NumericType_> >
{
public:
  // API definitions.
  typedef NumericType_ NumericType;
  enum { NumInputs = NumInputs_, };
  enum { NumOutputs = NumInputs_, };
  enum { NumParameters = 0, };

  static void Forward(const cv::Mat& X, const cv::Mat& W, cv::Mat* Y);

  static void Backward(const cv::Mat& X, const cv::Mat& W, const cv::Mat& Y,
                       const cv::Mat& dLdY, cv::Mat* dLdW, cv::Mat* dLdX);
  static double ComputeTruncateL2Factor(const cv::Mat& W, NumericType maxNormSq);
};

template <int NumClasses_, typename NumericType_ = double>
class SoftMax
  : public StandardLayer<SoftMax<NumClasses_, NumericType_> >
{
public:
  // API definitions.
  typedef NumericType_ NumericType;
  enum { NumInputs = NumClasses_, };
  enum { NumOutputs = NumClasses_, };
  enum { NumParameters = 0, };

  static void Forward(const cv::Mat& X, const cv::Mat& W, cv::Mat* Y);

  static void Backward(const cv::Mat& X, const cv::Mat& W, const cv::Mat& Y,
                       const cv::Mat& dLdY, cv::Mat* dLdW, cv::Mat* dLdX);
  static double ComputeTruncateL2Factor(const cv::Mat& W, NumericType maxNormSq);
};

struct NLLCriterion
{
  template <typename NNType>
  static const cv::Mat* SampleLoss(const NNType& nn, const cv::Mat& xi, const cv::Mat& yi,
                                   double* loss, int* error);

  template <typename NNType>
  static void DatasetLoss(const NNType& nn, const cv::Mat& X, const cv::Mat& Y,
                          double* loss, int* errors);

  template <typename NNType>
  static const cv::Mat* SampleGradient(NNType* nn, const cv::Mat& xi, const cv::Mat& yi,
                                       cv::Mat* dLdY, double* loss, int* error);
};

////////////////////////////////////////////////////////////////////////////////
// Inline definitions.
template <typename LayerType_>
StandardLayer<LayerType_>::StandardLayer()
{
  STATIC_ASSERT(LayerType::NumInputs > 0, "NumInputs > 0");
  STATIC_ASSERT(LayerType::NumOutputs > 0, "NumOutputs > 0");
  STATIC_ASSERT(LayerType::NumParameters >= 0, "NumParameters >= 0");
}

template <int NumInputs_, typename NumericType_>
inline
void Passthrough<NumInputs_, NumericType_>
::Forward(const cv::Mat& X, const cv::Mat& /*W*/, cv::Mat* Y)
{
  assert(X.rows == Y->rows && X.cols == Y->cols && X.type() == Y->type());
  DETECT_NUMERICAL_ERRORS(X);
  X.copyTo(*Y);
  DETECT_NUMERICAL_ERRORS(*Y);
}

template <int NumInputs_, typename NumericType_>
inline
void Passthrough<NumInputs_, NumericType_>
::Backward(const cv::Mat& /*X*/, const cv::Mat& /*W*/, const cv::Mat& /*Y*/,
           const cv::Mat& dLdY, cv::Mat* /*dLdW*/, cv::Mat* dLdX)
{
  assert(dLdY.rows == dLdX->rows && dLdY.cols == dLdX->cols && dLdY.type() == dLdX->type());
  DETECT_NUMERICAL_ERRORS(dLdY);
  dLdY.copyTo(*dLdX);
  DETECT_NUMERICAL_ERRORS(*dLdX);
}

template <int NumInputs_, typename NumericType_>
inline
double Passthrough<NumInputs_, NumericType_>
::ComputeTruncateL2Factor(const cv::Mat& /*W*/, NumericType /*maxNormSq*/)
{
  return 1;
}

template <int NumInputs_, int NumOutputs_, typename NumericType_>
inline
void Linear<NumInputs_, NumOutputs_, NumericType_>
::Forward(const cv::Mat& X, const cv::Mat& W, cv::Mat* Y)
{
  // Compute linear, Y = M X + B.
  const cv::Mat& M = W.rowRange(0, ParamsLinearMat).reshape(1, NumOutputs);
  const cv::Mat& B = W.rowRange(ParamsLinearMat, NumParameters);
  assert(X.rows == M.cols && Y->rows == M.rows && Y->rows == B.rows &&
         X.type() == W.type() && W.type() == Y->type());
  DETECT_NUMERICAL_ERRORS(X);
  DETECT_NUMERICAL_ERRORS(W);
  DETECT_NUMERICAL_ERRORS(M);
  DETECT_NUMERICAL_ERRORS(B);
  //*Y = M * X + B;
  cv::gemm(M, X, 1.0, B, 1.0, *Y);
  DETECT_NUMERICAL_ERRORS(*Y);
}

template <int NumInputs_, int NumOutputs_, typename NumericType_>
inline
void Linear<NumInputs_, NumOutputs_, NumericType_>
::Backward(const cv::Mat& X, const cv::Mat& W, const cv::Mat& /*Y*/,
              const cv::Mat& dLdY, cv::Mat* dLdW, cv::Mat* dLdX)
{
  const cv::Mat& M = W.rowRange(0, ParamsLinearMat).reshape(1, NumOutputs);
  cv::Mat dLdM = dLdW->rowRange(0, ParamsLinearMat).reshape(1, NumOutputs);
  cv::Mat dLdB = dLdW->rowRange(ParamsLinearMat, NumParameters);
  assert(X.rows == M.cols && dLdY.rows == M.rows && dLdY.rows == dLdB.rows &&
         dLdX->rows == X.rows && dLdW->rows == W.rows &&
         X.type() == W.type() && W.type() == dLdY.type() &&
         dLdY.type() == dLdW->type() && dLdW->type() == dLdX->type());
  DETECT_NUMERICAL_ERRORS(X);
  DETECT_NUMERICAL_ERRORS(W);
  DETECT_NUMERICAL_ERRORS(M);
  DETECT_NUMERICAL_ERRORS(dLdY);

  // dLdX = M^T dLdY
  cv::gemm(M, dLdY, 1.0, cv::Mat(), 0.0, *dLdX, CV_GEMM_A_T);
  // dLdM = dLdY X^T
  cv::gemm(dLdY, X, 1.0, cv::Mat(), 0.0, dLdM, CV_GEMM_B_T);
  // dLdB = dLdY
  dLdY.copyTo(dLdB);

  DETECT_NUMERICAL_ERRORS(dLdM);
  DETECT_NUMERICAL_ERRORS(dLdB);
  DETECT_NUMERICAL_ERRORS(*dLdW);
  DETECT_NUMERICAL_ERRORS(*dLdX);
}

template <int NumInputs_, int NumOutputs_, typename NumericType_>
double Linear<NumInputs_, NumOutputs_, NumericType_>
::ComputeTruncateL2Factor(const cv::Mat& W, NumericType maxNormSq)
{
  cv::Mat M = W.rowRange(0, ParamsLinearMat).reshape(1, NumOutputs);
  cv::Mat B = W.rowRange(ParamsLinearMat, NumParameters);
  // Check each row's norm (and his offset!).
  const double scaleNumerator = std::sqrt(maxNormSq);
  for (int i = 0; i < M.rows; ++i)
  {
    cv::Mat r = M.row(i);
    cv::Mat b = B.row(i);
    const double rowNormSq = r.dot(r) + b.dot(b);
    if (rowNormSq > maxNormSq)
    {
      const double hiddenUnitRenormScale = scaleNumerator / std::sqrt(rowNormSq);
      r *= hiddenUnitRenormScale;
      b *= hiddenUnitRenormScale;
      std::cout << "Scaling hidden unit " << i << " by " << hiddenUnitRenormScale << "\n";
      assert(((maxNormSq + 1e-1) - (r.dot(r) + b.dot(b))) > 0);
    }
    DETECT_NUMERICAL_ERRORS(r);
    DETECT_NUMERICAL_ERRORS(b);
  }
  return 1;
}

template <int NumInputs_, typename NumericType_>
void Tanh<NumInputs_, NumericType_>
::Forward(const cv::Mat& X, const cv::Mat& /*W*/, cv::Mat* Y)
{
  assert(X.rows == Y->rows && X.type() == Y->type());
  DETECT_NUMERICAL_ERRORS(X);
  // Compute Tanh, Y = tanh(X)
  // Perform nonlinear operation (ideally, vectorized).
  cv::MatConstIterator_<NumericType> x = X.begin<NumericType>();
  cv::MatIterator_<NumericType> y = Y->begin<NumericType>();
  const cv::MatConstIterator_<NumericType> yEnd = Y->end<NumericType>();
  for(; y != yEnd; ++x, ++y)
  {
    *y = std::tanh(*x);
  }
  DETECT_NUMERICAL_ERRORS(*Y);
}

template <int NumInputs_, typename NumericType_>
void Tanh<NumInputs_, NumericType_>
::Backward(const cv::Mat& X, const cv::Mat& /*W*/, const cv::Mat& /*Y*/,
           const cv::Mat& dLdY, cv::Mat* /*dLdW*/, cv::Mat* dLdX)
{
  assert(X.rows == dLdY.rows && X.rows == dLdX->rows &&
         X.type() == dLdY.type() && dLdY.type() == dLdX->type());
  DETECT_NUMERICAL_ERRORS(X);
  DETECT_NUMERICAL_ERRORS(dLdY);
  // Compute dLdx = sech^2(x).dLdY (ideally vectorized).
  cv::MatConstIterator_<NumericType> x = X.begin<NumericType>();
  cv::MatConstIterator_<NumericType> dy = dLdY.begin<NumericType>();
  cv::MatIterator_<NumericType> dx = dLdX->begin<NumericType>();
  const cv::MatConstIterator_<NumericType> dxEnd = dLdX->end<NumericType>();
  for(; dx != dxEnd; ++x, ++dy, ++dx)
  {
    //(*dx) = static_cast<NumericType>(1.0 / std::max<double>(std::cosh(*x), 1e-100));
    (*dx) = static_cast<NumericType>(1.0 / std::cosh(*x));
    (*dx) *= (*dx) * (*dy);
  }
  DETECT_NUMERICAL_ERRORS(*dLdX);
}

template <int NumInputs_, typename NumericType_>
inline
double Tanh<NumInputs_, NumericType_>
::ComputeTruncateL2Factor(const cv::Mat& /*W*/, NumericType /*maxNormSq*/)
{
  return 1;
}

namespace detail
{
template <typename NumericType>
void MaxElement(const cv::Mat& m, NumericType* maxVal, int* maxIdx)
{
  // Compute index of output category.
  *maxIdx = 0;
  cv::MatConstIterator_<NumericType> y = m.begin<NumericType>();
  const cv::MatConstIterator_<NumericType> yEnd = m.end<NumericType>();
  *maxVal = *y;
  int i = 1;
  for (++y; y != yEnd; ++y, ++i)
  {
    if (*y > *maxVal)
    {
      *maxVal = *y;
      *maxIdx = i;
    }
  }
}
}

template <int NumClasses_, typename NumericType_>
inline
void SoftMax<NumClasses_, NumericType_>
::Forward(const cv::Mat& X, const cv::Mat& /*W*/, cv::Mat* Y)
{
  using detail::MaxElement;

  assert(X.rows == Y->rows && X.type() == Y->type());
  DETECT_NUMERICAL_ERRORS(X);
  // Compute softmax as in Y = exp(X) / \sum{exp(X)}
  int maxIdx;
  NumericType maxVal;
  MaxElement(X, &maxVal, &maxIdx);
  *Y = X - maxVal;
  cv::exp(*Y, *Y);
  // Perform Gibbs normalization.
  const double preNormSum = cv::sum(*Y).val[0];
  const bool isNan = my_isnan(preNormSum) != 0;
  assert(!isNan); (void)isNan;
  const bool isInf = my_isinf(preNormSum) != 0;
  assert(!isInf); (void)isInf;
//  if (!isInf)
  {
    //const NumericType normFactor = static_cast<NumericType>(1.0 / std::max(preNormSum, 1e-100));
    const NumericType normFactor = static_cast<NumericType>(1.0 / preNormSum);
    (*Y) *= normFactor;
  }
//  else
//  {
//    std::stringstream ssMsg;
//    ssMsg << "Whoa, bro, preNormSum = " << preNormSum << "\n";//", preNormSq = " << preNormSq << "\n";
//    std::cout << ssMsg.str(); std::cout.flush();
//    // Make all infinities 1, all non-infinities 0.
//    cv::MatIterator_<NumericType> y = Y->begin<NumericType>();
//    const cv::MatConstIterator_<NumericType> yEnd = Y->end<NumericType>();
//    int infCount = 0;
//    for(; y != yEnd; ++y)
//    {
//      if (*y > std::numeric_limits<NumericType>::max())
//      {
//        ++infCount;
//        *y = static_cast<NumericType>(1);
//      }
//      else
//      {
//        *y = 0;
//      }
//    }
//    if (infCount > 0)
//    {
//      *Y *= static_cast<NumericType>(1.0 / infCount);
//    }
//    else
//    {
//      *Y = static_cast<NumericType>(1.0 / Y->rows);
//    }
//  }
  DETECT_NUMERICAL_ERRORS(*Y);
}

template <int NumClasses_, typename NumericType_>
void SoftMax<NumClasses_, NumericType_>
::Backward(const cv::Mat& /*X*/, const cv::Mat& /*W*/, const cv::Mat& Y,
           const cv::Mat& dLdY, cv::Mat* /*dLdW*/, cv::Mat* dLdX)
{
  assert(Y.rows == dLdY.rows && Y.rows == dLdX->rows &&
         Y.type() == dLdY.type() && dLdY.type() == dLdX->type());
  DETECT_NUMERICAL_ERRORS(Y);
  DETECT_NUMERICAL_ERRORS(dLdY);
  // dLdX = Y (e - Y) dLdY
  cv::multiply(dLdY, Y, *dLdX);
  cv::scaleAdd(Y, -Y.dot(dLdY), *dLdX, *dLdX);
  DETECT_NUMERICAL_ERRORS(*dLdX);
}

template <int NumClasses_, typename NumericType_>
inline
double SoftMax<NumClasses_, NumericType_>
::ComputeTruncateL2Factor(const cv::Mat& /*W*/, NumericType /*maxNormSq*/)
{
  return 1;
}

template <typename NNType>
const cv::Mat* NLLCriterion::
SampleLoss(const NNType& nn, const cv::Mat& xi, const cv::Mat& yi, double* loss, int* error)
{
  using detail::MaxElement;
  typedef typename NNType::NumericType NumericType;
  DETECT_NUMERICAL_ERRORS(xi);
  const cv::Mat* yOut = nn.Forward(xi);
  const double minProb = 1e-16;
  (*yOut) += cv::Scalar::all(minProb);
  (*yOut) *= 1.0 / (1.0 + (yOut->rows * minProb));
  DETECT_NUMERICAL_ERRORS(*yOut);
  // Find max class label.
  int label;
  NumericType maxP;
  MaxElement(*yOut, &maxP, &label);
  const int trueLabel = yi.at<unsigned char>(0, 0);
  *error = (trueLabel != label);
  const NumericType pClass = yOut->at<NumericType>(trueLabel, 0);
  assert(pClass > 0);
  *loss = -std::log(pClass);
  return yOut;
}

template <typename NNType>
void NLLCriterion::
DatasetLoss(const NNType& nn, const cv::Mat& X, const cv::Mat& Y, double* loss, int* errors)
{
  typedef typename NNType::NumericType NumericType;
  *errors = 0;
  *loss = 0;
  double sampleLoss;
  int sampleError;
  for (int i = 0; i < X.rows; ++i)
  {
    const cv::Mat xi = X.row(i).t();
    const cv::Mat yi = Y.row(i);
    SampleLoss(nn, xi, yi, &sampleLoss, &sampleError);
    *loss += sampleLoss;
    *errors += sampleError;
  }
}

template <typename NNType>
const cv::Mat* NLLCriterion::
SampleGradient(NNType* nn, const cv::Mat& xi, const cv::Mat& yi,
               cv::Mat* dLdY, double* loss, int* error)
{
  typedef typename NNType::NumericType NumericType;
  DETECT_NUMERICAL_ERRORS(xi);
  // Forward pass.
  const cv::Mat* yOut = SampleLoss(*nn, xi, yi, loss, error);
  DETECT_NUMERICAL_ERRORS(*yOut);
  // Compute loss gradient to get this party started.
  const int trueLabel = yi.at<unsigned char>(0, 0);
  const double minProb = 1e-20;
  const NumericType pClass =
    static_cast<NumericType>((minProb + yOut->at<NumericType>(trueLabel, 0)) / (1.0 + minProb));
  assert(pClass > 0);
  const NumericType nllGrad = static_cast<NumericType>(-1.0 / pClass);
  *dLdY = cv::Scalar::all(0);
  dLdY->at<NumericType>(trueLabel, 0) = nllGrad;
  DETECT_NUMERICAL_ERRORS(*dLdY);
  // Backward pass.
  return nn->Backward(*dLdY);
}

} // end ns nn
using namespace nn;
} // end ns blr

#endif //SRC_LAYER_H
