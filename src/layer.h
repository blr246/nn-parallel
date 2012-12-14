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

  static void Forward(const cv::Mat& X, const cv::Mat& W, cv::Mat* Y);

  static void Backward(const cv::Mat& X, const cv::Mat& W, const cv::Mat& Y,
                       const cv::Mat& dLdY, cv::Mat* dLdW, cv::Mat* dLdX);
};

template <int NumInputs_, int NumClasses_, typename NumericType_ = double>
class SoftMax
  : public StandardLayer<SoftMax<NumInputs_, NumClasses_, NumericType_> >
{
public:
  // API definitions.
  typedef NumericType_ NumericType;
  enum { NumInputs = NumInputs_, };
  enum { NumOutputs = NumClasses_, };
  enum { NumParameters = (NumOutputs - 1) * (NumInputs + 1), };

  // Non-API definitions.
  enum { ParamsLinearMat = (NumOutputs * NumInputs) - NumInputs };

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
void HiddenLinearTanh<NumInputs_, NumOutputs_, NumericType_>
::Forward(const cv::Mat& X, const cv::Mat& W, cv::Mat* Y)
{
  // Compute linear -> tanh as in
  //  Y = tanh(Mx + B).
  const cv::Mat& M =
    W(cv::Range(0, ParamsLinearMat), cv::Range::all()).reshape(1, NumOutputs);
  const cv::Mat& B =
    W(cv::Range(ParamsLinearMat, NumParameters), cv::Range::all());

  // Perform linear operation Y = Mx + B.
  //*Y = M * X + B;
  cv::gemm(M, X, 1.0, B, 1.0, *Y);
  // Perform nonlinear operation (ideally, vectorized).
  const cv::MatConstIterator_<NumericType> yEnd = Y->end<NumericType>();
  for(cv::MatIterator_<NumericType> y = Y->begin<NumericType>(); y != yEnd; ++y)
  {
    *y = std::tanh(*y);
  }
}

template <int NumInputs_, int NumOutputs_, typename NumericType_>
void HiddenLinearTanh<NumInputs_, NumOutputs_, NumericType_>
::Backward(const cv::Mat& X, const cv::Mat& W, const cv::Mat& /*Y*/,
              const cv::Mat& dLdY, cv::Mat* dLdW, cv::Mat* dLdX)
{
  const cv::Mat& M =
    W(cv::Range(0, ParamsLinearMat), cv::Range::all()).reshape(1, NumOutputs);
  const cv::Mat& B =
    W(cv::Range(ParamsLinearMat, NumParameters), cv::Range::all());
  cv::Mat dLdM =
    (*dLdW)(cv::Range(0, ParamsLinearMat), cv::Range::all()).reshape(1, NumOutputs);
  cv::Mat dLdB =
    (*dLdW)(cv::Range(ParamsLinearMat, NumParameters), cv::Range::all());
  // Compute Mx + B.
  //*Y = M * X + B;
  cv::gemm(M, X, 1.0, B, 1.0, dLdB);
  // Compute dLdB = sech^2(Mx + B).dLdY (ideally vectorized).
  const cv::MatConstIterator_<NumericType> dLdBEnd = dLdB.end<NumericType>();
  for(cv::MatIterator_<NumericType> v = dLdB.begin<NumericType>(); v != dLdBEnd; ++v)
  {
    *v = 1.0 / std::cosh(*v);
    *v *= *v;
  }
  dLdB.mul(dLdY);
  // Compute dLdX = M^T sech^2(Mx + B).dLdY = M^T dLdB
  //*dLdX = M.t() * dLdB;
  cv::gemm(M, dLdB, 1.0, cv::Mat(), 0.0, *dLdX, CV_GEMM_A_T);
  // Compute dLdM = sech^2(Mx + B).dLdY X^T = dLdB x^T
  //dLdM = dLdB * X.t();
  cv::gemm(dLdB, X, 1.0, cv::Mat(), 0.0, dLdM, CV_GEMM_B_T);
}

template <int NumInputs_, int NumClasses_, typename NumericType_>
void SoftMax<NumInputs_, NumClasses_, NumericType_>
::Forward(const cv::Mat& X, const cv::Mat& W, cv::Mat* Y)
{
  // Compute linear -> exp as in
  //  Y = exp(Mx + B).
  const cv::Mat& M =
    W(cv::Range(0, ParamsLinearMat), cv::Range::all()).reshape(1, NumOutputs - 1);
  const cv::Mat& B =
    W(cv::Range(ParamsLinearMat, NumParameters), cv::Range::all());
  cv::Mat Yk_1 = Y->rowRange(0, NumOutputs - 1);
  // Perform linear operation. Last set of weights is implictly 0.
  //Yk_1 = M * X + B;
  cv::gemm(M, X, 1.0, B, 1.0, Yk_1);
  //Y->rowRange(0, NumOutputs - 1) = M * X + B;
  Y->at<NumericType>(NumOutputs - 1, 0) = 1;
  // Perform nonlinear operation (ideally, vectorized).
  cv::exp(*Y, *Y);
  // Perform Gibbs normalization.
  const NumericType normFactor =
    static_cast<NumericType>(std::max(1.0 / cv::sum(*Y).val[0], 1e-10));;
  *Y *= normFactor;
}

template <int NumInputs_, int NumClasses_, typename NumericType_>
void SoftMax<NumInputs_, NumClasses_, NumericType_>
::Backward(const cv::Mat& X, const cv::Mat& W, const cv::Mat& Y,
           const cv::Mat& /*dLdY*/, cv::Mat* dLdW, cv::Mat* dLdX)
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
  const cv::Mat& M =
    W(cv::Range(0, ParamsLinearMat), cv::Range::all()).reshape(1, NumOutputs - 1);
  cv::Mat dLdM =
    (*dLdW)(cv::Range(0, ParamsLinearMat), cv::Range::all()).reshape(1, NumOutputs - 1);
  cv::Mat dLdB =
    (*dLdW)(cv::Range(ParamsLinearMat, NumParameters), cv::Range::all());
  //const NumericType sumDLdY = static_cast<NumericType>(cv::sum(dLdY).val[0]);
  const cv::Mat Yk_1 = Y.rowRange(0, NumOutputs - 1);
  // Complete gradients when output is not the last class label. 
  if (classIdx < (NumOutputs - 1))
  {
    // dL/dX = M^T Y - w_{y_i}^T
    //*dLdX = M.t() * Yk_1 - M.row(classIdx).t();
    cv::gemm(M, Yk_1, 1.0, M.row(classIdx), -1.0, *dLdX, CV_GEMM_A_T | CV_GEMM_C_T);
    // dL/dB = F(X, W) - e_{y_i}
    Yk_1.copyTo(dLdB);
    dLdB.row(classIdx) -= 1;
    // dL/dW = Y X^T - e_{y_i} X^T
    //dLdM = Yk_1 * X.t();
    cv::gemm(Yk_1, X, 1.0, cv::Mat(), 0.0, dLdM, CV_GEMM_B_T);
    dLdM.row(classIdx) -= X.t();
  }
  else
  {
    // dL/dX = M^T Y - w_{y_i}^T
    //*dLdX = M.t() * Y.rowRange(0, NumOutputs - 1);
    cv::gemm(M, Y, 1.0, cv::Mat(), 0.0, *dLdX, CV_GEMM_A_T);
    // dL/dB = F(X, W) - e_{y_i}
    Yk_1.copyTo(dLdB);
    // dL/dW = Y X^T - e_{y_i} X^T
    //dLdM = Yk_1 * X.t();
    cv::gemm(Yk_1, X, 1.0, cv::Mat(), 0.0, dLdM, CV_GEMM_B_T);
  }
}

} // end ns nn
using namespace nn;
} // end ns blr

#endif //SRC_LAYER_H
