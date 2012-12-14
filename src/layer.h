#ifndef SRC_LAYER_H
#define SRC_LAYER_H
#include "static_assert.h"
#include "opencv/cv.h"
#include "opencv/cv.hpp"
#include "opencv/cxmat.hpp"
#include <iostream>
#include <limits>
#include <cmath>

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

template <typename LayerType_>
class DropoutLayer
{
public:
  typedef LayerType_ LayerType;

  DropoutLayer();

  bool DropoutEnabled() const;
  void EnableDropout();
  void DisableDropout();

private:
  bool dropoutEnabled;
  double dropoutProbability;
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

template <typename LayerType_>
DropoutLayer<LayerType_>::DropoutLayer()
: dropoutEnabled(false),
  dropoutProbability(DropoutProbability / 100.0)
{
  STATIC_ASSERT((LayerType::DropoutProbability <= 100) && (LayerType::DropoutProbability >= 0),
                "0 <= DropoutProbability <= 100");
}

template <typename LayerType_>
inline
bool DropoutLayer<LayerType_>::DropoutEnabled() const
{
  return dropoutEnabled;
}

template <typename LayerType_>
inline
void DropoutLayer<LayerType_>::EnableDropout()
{
  dropoutEnabled = true;
}

template <typename LayerType_>
inline
void DropoutLayer<LayerType_>::DisableDropout()
{
  dropoutEnabled = false;
}

template <int NumInputs_, typename NumericType_>
inline
void Passthrough<NumInputs_, NumericType_>
::Forward(const cv::Mat& X, const cv::Mat& /*W*/, cv::Mat* Y)
{
  X.copyTo(*Y);
}

template <int NumInputs_, typename NumericType_>
inline
void Passthrough<NumInputs_, NumericType_>
::Backward(const cv::Mat& /*X*/, const cv::Mat& /*W*/, const cv::Mat& /*Y*/,
           const cv::Mat& dLdY, cv::Mat* /*dLdW*/, cv::Mat* dLdX)
{
  dLdY.copyTo(*dLdX);
}

template <int NumInputs_, int NumOutputs_, typename NumericType_>
inline
void Linear<NumInputs_, NumOutputs_, NumericType_>
::Forward(const cv::Mat& X, const cv::Mat& W, cv::Mat* Y)
{
  // Compute linear, Y = M X + B.
  const cv::Mat& M =
    W(cv::Range(0, ParamsLinearMat), cv::Range::all()).reshape(1, NumOutputs);
  const cv::Mat& B =
    W(cv::Range(ParamsLinearMat, NumParameters), cv::Range::all());
  //*Y = M * X + B;
  cv::gemm(M, X, 1.0, B, 1.0, *Y);
}

template <int NumInputs_, int NumOutputs_, typename NumericType_>
inline
void Linear<NumInputs_, NumOutputs_, NumericType_>
::Backward(const cv::Mat& X, const cv::Mat& W, const cv::Mat& /*Y*/,
              const cv::Mat& dLdY, cv::Mat* dLdW, cv::Mat* dLdX)
{
  const cv::Mat& M =
    W(cv::Range(0, ParamsLinearMat), cv::Range::all()).reshape(1, NumOutputs);
  cv::Mat dLdM =
    (*dLdW)(cv::Range(0, ParamsLinearMat), cv::Range::all()).reshape(1, NumOutputs);
  cv::Mat dLdB =
    (*dLdW)(cv::Range(ParamsLinearMat, NumParameters), cv::Range::all());
  // dLdX = M^T dLdY
  cv::gemm(M, dLdY, 1.0, cv::Mat(), 0.0, *dLdX, CV_GEMM_A_T);
  // dLdM = dLdY X^T
  cv::gemm(dLdY, X, 1.0, cv::Mat(), 0.0, dLdM, CV_GEMM_B_T);
  // dLdB = dLdY
  dLdY.copyTo(dLdB);
}

template <int NumInputs_, typename NumericType_>
void Tanh<NumInputs_, NumericType_>
::Forward(const cv::Mat& X, const cv::Mat& /*W*/, cv::Mat* Y)
{
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
  // Compute dLdx = sech^2(x).dLdY (ideally vectorized).
  cv::MatConstIterator_<NumericType> x = X.begin<NumericType>();
  cv::MatConstIterator_<NumericType> dy = dLdY.begin<NumericType>();
  cv::MatIterator_<NumericType> dx = dLdX->begin<NumericType>();
  const cv::MatConstIterator_<NumericType> dxEnd = dLdX->end<NumericType>();
  for(; dx != dxEnd; ++x, ++dy, ++dx)
  {
    *dx = 1.0 / std::cosh(*x);
    *dx *= *dx * *dy;
  }
}

template <int NumClasses_, typename NumericType_>
inline
void SoftMax<NumClasses_, NumericType_>
::Forward(const cv::Mat& X, const cv::Mat& /*W*/, cv::Mat* Y)
{
  // Compute softmax as in Y = exp(-X) / \sum{exp(-X)}
  *Y = -X;
  cv::exp(*Y, *Y);
  // Perform Gibbs normalization.
  const NumericType normFactor =
    static_cast<NumericType>(std::max(1.0 / cv::sum(*Y).val[0], 1e-10));;
  *Y *= normFactor;
}

template <int NumClasses_, typename NumericType_>
void SoftMax<NumClasses_, NumericType_>
::Backward(const cv::Mat& /*X*/, const cv::Mat& /*W*/, const cv::Mat& Y,
           const cv::Mat& dLdY, cv::Mat* /*dLdW*/, cv::Mat* dLdX)
{
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

} // end ns nn
using namespace nn;
} // end ns blr

#endif //SRC_LAYER_H
