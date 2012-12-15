#ifndef SRC_LAYER_H
#define SRC_LAYER_H
#include "type_utils.h"
#include "static_assert.h"
#include "opencv/cv.h"
#include "opencv/cv.hpp"
#include "opencv/cxmat.hpp"
#include <iostream>
#include <limits>
#include <cmath>
#include <assert.h>

namespace blr
{
namespace nn
{

std::ostream& operator<<(std::ostream& stream, const cv::Size& size)
{
  return stream << "(" << size.height << "," << size.width << ")";
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
  static void TruncateL2(NumericType maxNormSq, cv::Mat* W);
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
  static void TruncateL2(NumericType maxNormSq, cv::Mat* W);
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
  static void TruncateL2(NumericType maxNormSq, cv::Mat* W);
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
  static void TruncateL2(NumericType maxNormSq, cv::Mat* W);
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
  X.copyTo(*Y);
}

template <int NumInputs_, typename NumericType_>
inline
void Passthrough<NumInputs_, NumericType_>
::Backward(const cv::Mat& /*X*/, const cv::Mat& /*W*/, const cv::Mat& /*Y*/,
           const cv::Mat& dLdY, cv::Mat* /*dLdW*/, cv::Mat* dLdX)
{
  assert(dLdY.rows == dLdX->rows && dLdY.cols == dLdX->cols && dLdY.type() == dLdX->type());
  dLdY.copyTo(*dLdX);
}

template <int NumInputs_, typename NumericType_>
inline
void Passthrough<NumInputs_, NumericType_>
::TruncateL2(NumericType /*maxNormSq*/, cv::Mat* /*W*/)
{}

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
  //*Y = M * X + B;
  cv::gemm(M, X, 1.0, B, 1.0, *Y);
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
  // dLdX = M^T dLdY
  cv::gemm(M, dLdY, 1.0, cv::Mat(), 0.0, *dLdX, CV_GEMM_A_T);
  // dLdM = dLdY X^T
  cv::gemm(dLdY, X, 1.0, cv::Mat(), 0.0, dLdM, CV_GEMM_B_T);
  // dLdB = dLdY
  dLdY.copyTo(dLdB);
}

template <int NumInputs_, int NumOutputs_, typename NumericType_>
inline
void Linear<NumInputs_, NumOutputs_, NumericType_>
::TruncateL2(NumericType maxNormSq, cv::Mat* W)
{
  cv::Mat M = W->rowRange(0, ParamsLinearMat).reshape(1, NumOutputs);
  cv::Mat B = W->rowRange(ParamsLinearMat, NumParameters);
  // Check each row's norm.
  const double scaleNumerator = std::sqrt(maxNormSq);
  for (int i = 0; i < M.rows; ++i)
  {
    cv::Mat r = M.row(i);
    const double normSq = r.dot(r);
    if (normSq > maxNormSq)
    {
      const NumericType scaleFactor = static_cast<NumericType>(scaleNumerator / std::sqrt(normSq));
      r *= scaleFactor;
      std::cout << "normSq = " << normSq << ", "
                   "updated normSq = " << r.dot(r) << ", "
                   "l = " << maxNormSq << std::endl;
    }
    assert(((maxNormSq + 1e-6) - r.dot(r)) > 0);
  }
  const double normBSq = B.dot(B);
  if (normBSq > maxNormSq)
  {
    const NumericType scaleFactor = static_cast<NumericType>(scaleNumerator / std::sqrt(normBSq));
    B *= scaleFactor;
    std::cout << "normBSq = " << normBSq << ", "
                 "updated normSq = " << B.dot(B) << ", "
                 "l = " << maxNormSq << std::endl;
  }
  assert(((maxNormSq + 1e-6) - B.dot(B)) > 0);
}

template <int NumInputs_, typename NumericType_>
void Tanh<NumInputs_, NumericType_>
::Forward(const cv::Mat& X, const cv::Mat& /*W*/, cv::Mat* Y)
{
  assert(X.rows == Y->rows && X.type() == Y->type());
  // Compute Tanh, Y = tanh(X)
  // Perform nonlinear operation (ideally, vectorized).
  cv::MatConstIterator_<NumericType> x = X.begin<NumericType>();
  cv::MatIterator_<NumericType> y = Y->begin<NumericType>();
  const cv::MatConstIterator_<NumericType> yEnd = Y->end<NumericType>();
  for(; y != yEnd; ++x, ++y)
  {
    *y = std::tanh(*x);
  }
}

template <int NumInputs_, typename NumericType_>
void Tanh<NumInputs_, NumericType_>
::Backward(const cv::Mat& X, const cv::Mat& /*W*/, const cv::Mat& /*Y*/,
              const cv::Mat& dLdY, cv::Mat* /*dLdW*/, cv::Mat* dLdX)
{
  assert(X.rows == dLdY.rows && X.rows == dLdX->rows &&
         X.type() == dLdY.type() && dLdY.type() == dLdX->type());
  // Compute dLdx = sech^2(x).dLdY (ideally vectorized).
  cv::MatConstIterator_<NumericType> x = X.begin<NumericType>();
  cv::MatConstIterator_<NumericType> dy = dLdY.begin<NumericType>();
  cv::MatIterator_<NumericType> dx = dLdX->begin<NumericType>();
  const cv::MatConstIterator_<NumericType> dxEnd = dLdX->end<NumericType>();
  for(; dx != dxEnd; ++x, ++dy, ++dx)
  {
    *dx = static_cast<NumericType>(1.0 / std::max<double>(std::cosh(*x), 1e-16));
    *dx *= *dx * *dy;
  }
}

template <int NumInputs_, typename NumericType_>
inline
void Tanh<NumInputs_, NumericType_>
::TruncateL2(NumericType /*maxNormSq*/, cv::Mat* /*W*/)
{}

template <int NumClasses_, typename NumericType_>
inline
void SoftMax<NumClasses_, NumericType_>
::Forward(const cv::Mat& X, const cv::Mat& /*W*/, cv::Mat* Y)
{
  assert(X.rows == Y->rows && X.type() == Y->type());
  // Compute softmax as in Y = exp(-X) / \sum{exp(-X)}
  X.copyTo(*Y);
  *Y *= -1;
  cv::exp(*Y, *Y);
  // Perform Gibbs normalization. Sometimes this goes badly with infinities.
  const double preNormSum = cv::sum(*Y).val[0];
  if (preNormSum <= std::numeric_limits<double>::max())
  {
    const NumericType normFactor = static_cast<NumericType>(1.0 / std::max(preNormSum, 1e-16));
    *Y *= normFactor;
  }
  else
  {
    // Make all infinities 1, all non-inities 0.
    cv::MatIterator_<NumericType> y = Y->begin<NumericType>();
    const cv::MatConstIterator_<NumericType> yEnd = Y->end<NumericType>();
    int infCount = 0;
    for(; y != yEnd; ++y)
    {
      if (*y > std::numeric_limits<NumericType>::max())
      {
        ++infCount;
        *y = static_cast<NumericType>(1);
      }
      else
      {
        *y = 0;
      }
    }
    if (infCount > 0)
    {
      *Y *= static_cast<NumericType>(1.0 / infCount);
    }
    else
    {
      *Y = static_cast<NumericType>(1.0 / Y->rows);
    }
  }
//  const double dbgNorm = cv::sum(*Y).val[0];
//  if (!(std::abs(1.0 - dbgNorm) < 1.0e-6))
//  {
//    bool why = true;
//  }
}

template <int NumClasses_, typename NumericType_>
void SoftMax<NumClasses_, NumericType_>
::Backward(const cv::Mat& /*X*/, const cv::Mat& /*W*/, const cv::Mat& Y,
           const cv::Mat& dLdY, cv::Mat* /*dLdW*/, cv::Mat* dLdX)
{
  assert(Y.rows == dLdY.rows && Y.rows == dLdX->rows &&
         Y.type() == dLdY.type() && dLdY.type() == dLdX->type());
  // Compute index of output category.
  int classIdx = 0; 
  {
    cv::MatConstIterator_<NumericType> y = Y.begin<NumericType>();
    const cv::MatConstIterator_<NumericType> yEnd = Y.end<NumericType>();
    NumericType maxVal = *y;
    int i = 1;
    for (++y; y != yEnd; ++y, ++i)
    {
      if (*y > maxVal)
      {
        maxVal = *y;
        classIdx = i;
      }
    }
  }
  cv::multiply(dLdY, Y, *dLdX, -1.0);
  cv::scaleAdd(Y, Y.dot(dLdY), *dLdX, *dLdX);
}

template <int NumClasses_, typename NumericType_>
inline
void SoftMax<NumClasses_, NumericType_>
::TruncateL2(NumericType /*maxNormSq*/, cv::Mat* /*W*/)
{}

template <typename NNType>
const cv::Mat* NLLCriterion::
SampleLoss(const NNType& nn, const cv::Mat& xi, const cv::Mat& yi, double* loss, int* error)
{
  typedef typename NNType::NumericType NumericType;
  const cv::Mat* yOut = nn.Forward(xi);
  // Find max class label.
  int label = 0;
  double maxP = yOut->at<NumericType>(0, 0);
  const int yOutSize = yOut->rows;
  for (int i = 1; i < yOutSize; ++i) {
    const double p = yOut->at<NumericType>(i, 0);
    if (p > maxP)
    {
      maxP = p;
      label = i;
    }
  }
  const int trueLabel = yi.at<unsigned char>(0, 0);
  *error = (trueLabel != label);
  *loss = -std::log(yOut->at<NumericType>(trueLabel, 0));
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
  // Forward pass.
  const cv::Mat* yOut = SampleLoss(*nn, xi, yi, loss, error);
  // Compute loss gradient to get this party started.
  const int trueLabel = yi.at<unsigned char>(0, 0);
  const NumericType nllGrad =
    static_cast<NumericType>(-1.0 / std::max<double>(yOut->at<NumericType>(trueLabel, 0), 1e-16));
  *dLdY *= 0;
  dLdY->at<NumericType>(trueLabel, 0) = nllGrad;
  // Backward pass.
  return nn->Backward(*dLdY);
}

} // end ns nn
using namespace nn;
} // end ns blr

#endif //SRC_LAYER_H
