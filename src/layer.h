#ifndef SRC_LAYER_H
#define SRC_LAYER_H
#include "static_assert.h"
#include "opencv/cv.h"
#include "opencv/cv.hpp"
#include "opencv/cxmat.hpp"
#include <cmath>
#include <iostream>

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

  void Forward(const cv::Mat& X, const cv::Mat& /*W*/, cv::Mat* Y) const
  {
    *Y = X;
  }

  void Backward(const cv::Mat& dLdY, const cv::Mat& /*X*/, const cv::Mat& /*W*/,
                cv::Mat* /*dLdW*/, cv::Mat* dLdX)
  {
    *dLdX = dLdY;
  }
};

/**
 * <summary>Hidden layer with N inputs and M outputs.</summary>
 */
template <int NumInputs_, int NumOutputs_, typename NumericType_ = double>
class HiddenLinearTanh
  : public StandardLayer<HiddenLinearTanh<NumInputs_, NumOutputs_, NumericType_> >
{
public:
  // API definitions.
  typedef NumericType_ NumericType;
  enum { NumInputs = NumInputs_, };
  enum { NumOutputs = NumOutputs_, };
  enum { NumParameters = (NumInputs * NumOutputs) + NumOutputs, };

  // Non-API definitions.
  enum { ParamsLinearMat = NumInputs * NumOutputs };

  /**
   * <summary>Forward propagation F(X, W) = Y.</summary>
   */
  void Forward(const cv::Mat& X, const cv::Mat& W, cv::Mat* Y) const
  {
    // Compute linear -> tanh as in
    //  Y = tanh(Mx + b).
    const cv::Mat& M =
      W(cv::Range(0, ParamsLinearMat), cv::Range::all()).reshape(1, NumOutputs);
    const cv::Mat& b =
      W(cv::Range(ParamsLinearMat, NumParameters), cv::Range::all());

    // Perform linear operation.
    *Y = M * X + b;
    // Perform nonlinear operation (ideally, vectorized).
    const cv::MatConstIterator_<NumericType> yEnd = Y->end<NumericType>();
    for(cv::MatIterator_<NumericType> y = Y->begin<NumericType>(); y != yEnd; ++y)
    {
      *y = std::tanh(*y);
    }
  }

  void Backward(const cv::Mat& dLdY, const cv::Mat& X, const cv::Mat& W,
                cv::Mat* dLdW, cv::Mat* dLdX)
  {
    const cv::Mat& M =
      W(cv::Range(0, ParamsLinearMat), cv::Range::all()).reshape(1, NumOutputs);
    const cv::Mat& b =
      W(cv::Range(ParamsLinearMat, NumParameters), cv::Range::all());
    cv::Mat dLdM =
      (*dLdW)(cv::Range(0, ParamsLinearMat), cv::Range::all()).reshape(1, NumOutputs);
    cv::Mat dLdB =
      (*dLdW)(cv::Range(ParamsLinearMat, NumParameters), cv::Range::all());
    // Compute Wx + b.
    dLdB = M * X + b;
    // Compute dLdB = sech^2(Mx + b).dLdY
    const cv::MatConstIterator_<NumericType> dLdBEnd = dLdB.end<NumericType>();
    for(cv::MatIterator_<NumericType> v = dLdB.begin<NumericType>(); v != dLdBEnd; ++v)
    {
      *v = 1.0 / std::cosh(*v);
      *v *= *v;
    }
    dLdB.mul(dLdY);
    // Compute dLdX = M^T sech^2(Mx + b).dLdY
    *dLdX = M.t() * dLdB;
    // Compute dLdM = sech^2(Mx + b).dLdY X^T
    dLdM  = dLdB * X.t();
  }
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

} // end ns nn
using namespace nn;
} // end ns blr

#endif //SRC_LAYER_H
