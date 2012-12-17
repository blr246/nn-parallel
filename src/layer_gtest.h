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
#ifndef SRC_LAYER_GTEST
#define SRC_LAYER_GTEST
#include "layer.h"
#include "type_utils.h"
#include "rand_bound.h"

#include "opencv/cv.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <limits>

namespace _dropout_nn_parallel_gtest_
{
using namespace blr;

template <typename T>
void MatExpectNear(const cv::Mat& a, const cv::Mat& b, const T eps)
{
  int maxDifferenceCoords[2] = {0, 0};
  T maxDifference = std::numeric_limits<T>::min();
  for (int r = 0; r < a.rows; ++r)
  {
    for (int c = 0; c < a.cols; ++c)
    {
      const T& aij = a.at<T>(r, c);
      const T& bij = b.at<T>(r, c);
      const T diff = std::abs(aij - bij);
      if (diff > maxDifference)
      {
        maxDifference = diff;
        maxDifferenceCoords[0] = r;
        maxDifferenceCoords[1] = c;
      }
    }
  }
  EXPECT_LT(maxDifference, eps) << "Max difference violated by element ("
                                << maxDifferenceCoords[0] << ","
                                << maxDifferenceCoords[1] << ")";
}

template <typename T>
void MatExpectNotNear(const cv::Mat& a, const cv::Mat& b, const T eps)
{
  for (int r = 0; r < a.rows; ++r)
  {
    for (int c = 0; c < a.cols; ++c)
    {
      const T& aij = a.at<T>(r, c);
      const T& bij = b.at<T>(r, c);
      EXPECT_GT(std::abs(aij - bij), eps);
    }
  }
}

template <typename T>
inline void MatRandUniform(cv::Mat* m)
{
  std::generate(m->begin<T>(), m->end<T>(), &RandUniform<T>);
}

namespace detail
{
template <int N, typename LayerType>
struct RandomizeMatHelper {
  static void Init(cv::Mat* m)
  {
    MatRandUniform<typename LayerType::NumericType>(m);
  }
};
template <typename LayerType>
struct RandomizeMatHelper<0, LayerType> { static void Init(cv::Mat* /*m*/) { } };
}

template <typename LayerType_>
struct ForwardBackwardTest
{
public:
  typedef LayerType_ LayerType;
  typedef typename LayerType::NumericType NumericType;
  typedef typename detail::RandomizeMatHelper<LayerType::NumParameters, LayerType> InitWHelper;

  static void Run()
  {
    cv::Mat W(LayerType::NumParameters, 1, CV_64F);
    InitWHelper::Init(&W);
    cv::Mat X(LayerType::NumInputs, 1, CV_64F);
    MatRandUniform<NumericType>(&X);
    X *= static_cast<NumericType>(10);
    cv::Mat Y = cv::Mat(LayerType::NumOutputs, 1, CV_64F,
                        cv::Scalar(std::numeric_limits<NumericType>::max()));
    {
      SCOPED_TRACE("Forward");
      LayerType hl;
      const cv::Mat Y0 = Y.clone();
      hl.Forward(X, W, &Y);
      MatExpectNotNear<NumericType>(Y0, Y, 1.0e-6);
    }
    {
      SCOPED_TRACE("Backward");
      LayerType hl;
      cv::Mat dLdY(LayerType::NumOutputs, 1, CV_64F);
      MatRandUniform<NumericType>(&dLdY);
      cv::Mat dLdW = cv::Mat(LayerType::NumParameters, 1, CV_64F,
                             cv::Scalar(std::numeric_limits<NumericType>::max()));
      const cv::Mat dLdW0 = dLdW.clone();
      cv::Mat dLdX = cv::Mat(LayerType::NumInputs, 1, CV_64F,
                             cv::Scalar(std::numeric_limits<NumericType>::max()));
      const cv::Mat dLdX0 = dLdX.clone();
      hl.Backward(X, W, Y, dLdY, &dLdW, &dLdX);
      MatExpectNotNear<NumericType>(dLdW0, dLdW, 1.0e-6);
      MatExpectNotNear<NumericType>(dLdX0, dLdX, 1.0e-6);
    }
  }
};

template <typename LayerType_, int TolExp = -6, int XScale = 10>
struct LayerForwardBackwardFiniteDifferenceTester
{
  typedef LayerType_ LayerType;
  typedef typename LayerType::NumericType NumericType;
  enum { CvType = NumericTypeToCvType<NumericType>::CvType, };
  typedef typename detail::RandomizeMatHelper<LayerType::NumParameters, LayerType> InitWHelper;

  // Test dLdX.
  static void TestDLdXForward(const cv::Mat& X, const cv::Mat& W, cv::Mat* ddX)
  {
    const NumericType eps = static_cast<NumericType>(1e-6);
    // Perturb input by epsilon.
    cv::Mat outA(LayerType::NumOutputs, 1, CvType);
    cv::Mat outB(LayerType::NumOutputs, 1, CvType);
    // Compute discrete difference formula for derivative.
    for (int i = 0; i < LayerType::NumInputs; ++i)
    {
      cv::Mat dX = X.clone();
      dX.at<NumericType>(i, 0) -= eps;
      LayerType::Forward(dX, W, &outA);
      dX.at<NumericType>(i, 0) += 2 * eps;
      LayerType::Forward(dX, W, &outB);
      outB -= outA;
      outB /= 2 * eps;
      ddX->row(i) = outB.t();
    }
  }

  static void TestDLdXBackward(const cv::Mat& X, const cv::Mat& W, const cv::Mat& Y, cv::Mat* ddX)
  {
    cv::Mat dLdW = cv::Mat(W.size(), W.type());
    cv::Mat dLdX = cv::Mat(X.size(), X.type());
    cv::Mat dLdY = cv::Mat::zeros(LayerType::NumOutputs, 1, CvType);
    // Complte dLdX directly per-output channel.
    for (int i = 0; i < LayerType::NumOutputs; ++i)
    {
      dLdY.at<double>(i, 0) = 1;
      LayerType::Backward(X, W, Y, dLdY, &dLdW, &dLdX);
      dLdY.at<double>(i, 0) = 0;
      ddX->row(i) = dLdX.t();
    }
  }

  static void TestDLdWForward(const cv::Mat& X, const cv::Mat& W, cv::Mat* ddW)
  {
    const NumericType eps = static_cast<NumericType>(1e-6);
    // Perturb input by epsilon.
    cv::Mat dW = W.clone();
    cv::Mat outA(LayerType::NumOutputs, 1, CvType);
    cv::Mat outB(LayerType::NumOutputs, 1, CvType);
    // Compute discrete difference formula for derivative.
    for (int i = 0; i < LayerType::NumParameters; ++i)
    {
      dW.at<NumericType>(i, 0) -= eps;
      LayerType::Forward(X, dW, &outA);
      dW.at<NumericType>(i, 0) += 2 * eps;
      LayerType::Forward(X, dW, &outB);
      dW.at<NumericType>(i, 0) -= eps;
      outB -= outA;
      outB /= 2 * eps;
      ddW->row(i) = outB.t();
    }
  }

  static void TestDLdWBackward(const cv::Mat& X, const cv::Mat& W, const cv::Mat& Y, cv::Mat* ddX)
  {
    cv::Mat dLdW = cv::Mat(W.size(), W.type());
    cv::Mat dLdX = cv::Mat(X.size(), X.type());
    cv::Mat dLdY = cv::Mat::zeros(LayerType::NumOutputs, 1, CvType);
    // Complte dLdW directly per-output channel.
    for (int i = 0; i < LayerType::NumOutputs; ++i)
    {
      dLdY.at<double>(i, 0) = 1;
      LayerType::Backward(X, W, Y, dLdY, &dLdW, &dLdX);
      dLdY.at<double>(i, 0) = 0;
      ddX->row(i) = dLdW.t();
    }
  }

  static void Run()
  {
    // Get a random input.
    cv::Mat X(LayerType::NumInputs, 1, CvType);
    MatRandUniform<NumericType>(&X);
    X *= static_cast<NumericType>(XScale);
    X -= cv::Mat::ones(X.size(), X.type());
    cv::Mat W(LayerType::NumParameters, 1, CvType);
    InitWHelper::Init(&W);
    cv::Mat Y(LayerType::NumOutputs, 1, CvType);
    LayerType::Forward(X, W, &Y);
    const NumericType eps = static_cast<NumericType>(std::pow(10.0, TolExp));

    {
      SCOPED_TRACE("X");
      // Get a matrix to store gradients.
      cv::Mat ddXForward(LayerType::NumInputs, LayerType::NumOutputs, CvType);
      MatRandUniform<NumericType>(&ddXForward);
      cv::Mat ddXBackward(LayerType::NumOutputs, LayerType::NumInputs, CvType);
      MatRandUniform<NumericType>(&ddXBackward);
      // Compute forward/backward difference.
      TestDLdXForward(X, W, &ddXForward);
      TestDLdXBackward(X, W, Y, &ddXBackward);
      // Look for absolute differences.
      MatExpectNear<NumericType>(ddXForward, ddXBackward.t(), eps);
    }
    if (W.rows > 0)
    {
      SCOPED_TRACE("W");
      // Get a matrix to store gradients.
      cv::Mat ddWForward(LayerType::NumParameters, LayerType::NumOutputs, CvType);
      MatRandUniform<NumericType>(&ddWForward);
      cv::Mat ddWBackward(LayerType::NumOutputs, LayerType::NumParameters, CvType);
      MatRandUniform<NumericType>(&ddWBackward);
      // Compute forward/backward difference.
      TestDLdWForward(X, W, &ddWForward);
      TestDLdWBackward(X, W, Y, &ddWBackward);
      MatExpectNear<NumericType>(ddWForward, ddWBackward.t(), eps);
    }
  }
};

TEST(Passthrough, All)
{
#if defined(NDEBUG)
  typedef Passthrough<100, double> LayerType;
#else
  typedef Passthrough<10, double> LayerType;
#endif
  {
    SCOPED_TRACE("Simple Forward-Backward");
    ForwardBackwardTest<LayerType>::Run();
  }
  {
    SCOPED_TRACE("Gradients");
    LayerForwardBackwardFiniteDifferenceTester<LayerType>::Run();
  }
}


TEST(Linear, All)
{
#if defined(NDEBUG)
  typedef Linear<100, 200, double> LayerType;
#else
  typedef Linear<10, 50, double> LayerType;
#endif
  {
    SCOPED_TRACE("Simple Forward-Backward");
    ForwardBackwardTest<LayerType>::Run();
  }
  {
    SCOPED_TRACE("Gradients");
    LayerForwardBackwardFiniteDifferenceTester<LayerType>::Run();
  }
}

TEST(Tanh, All)
{
#if defined(NDEBUG)
  typedef Tanh<100, double> LayerType;
#else
  typedef Tanh<10, double> LayerType;
#endif
  {
    SCOPED_TRACE("Simple Forward-Backward");
    ForwardBackwardTest<LayerType>::Run();
  }
  {
    SCOPED_TRACE("Gradients");
    LayerForwardBackwardFiniteDifferenceTester<LayerType>::Run();
  }
}

template <int NumInputs_, int NumOutputs_, typename NumericType_>
struct HiddenLinearTanh
  : public StandardLayer<HiddenLinearTanh<NumInputs_, NumOutputs_, NumericType_> >
{
  // API definitions.
  typedef NumericType_ NumericType;
  enum { NumInputs = NumInputs_, };
  enum { NumOutputs = NumOutputs_, };
  typedef Linear<NumInputs, NumOutputs, NumericType> MyLinear;
  typedef Tanh<NumOutputs, NumericType> MyTanh;
  enum { NumParameters = MyLinear::NumParameters + MyTanh::NumParameters, };

  static void Forward(const cv::Mat& X, const cv::Mat& W, cv::Mat* Y);

  static void Backward(const cv::Mat& X, const cv::Mat& W, const cv::Mat& Y,
                       const cv::Mat& dLdY, cv::Mat* dLdW, cv::Mat* dLdX);
};

template <int NumInputs_, int NumOutputs_, typename NumericType_>
inline
void HiddenLinearTanh<NumInputs_, NumOutputs_, NumericType_>
::Forward(const cv::Mat& X, const cv::Mat& W, cv::Mat* Y)
{
  cv::Mat Y0(Y->size(), Y->type());
  MyLinear::Forward(X, W, &Y0);
  MyTanh::Forward(Y0, cv::Mat(), Y);
}

template <int NumInputs_, int NumOutputs_, typename NumericType_>
inline
void HiddenLinearTanh<NumInputs_, NumOutputs_, NumericType_>
::Backward(const cv::Mat& X, const cv::Mat& W, const cv::Mat& Y,
           const cv::Mat& dLdY, cv::Mat* dLdW, cv::Mat* dLdX)
{
  cv::Mat XTanh(Y.size(), Y.type());
  MyLinear::Forward(X, W, &XTanh);
  cv::Mat dLdX0(dLdY.size(), dLdY.type());
  MyTanh::Backward(XTanh, cv::Mat(), Y, dLdY, NULL, &dLdX0);
  MyLinear::Backward(X, W, XTanh, dLdX0, dLdW, dLdX);
}

TEST(HiddenLinearTanh, All)
{
#if defined(NDEBUG)
  typedef HiddenLinearTanh<100, 200, double> LayerType;
#else
  typedef HiddenLinearTanh<10, 50, double> LayerType;
#endif
  {
    SCOPED_TRACE("Simple Forward-Backward");
    ForwardBackwardTest<LayerType>::Run();
  }
  {
    SCOPED_TRACE("Gradients");
    LayerForwardBackwardFiniteDifferenceTester<LayerType>::Run();
  }
}

TEST(Softmax, All)
{
  typedef SoftMax<4, double> LayerType;
  {
    SCOPED_TRACE("Simple Forward-Backward");
    ForwardBackwardTest<LayerType>::Run();
  }
  {
    SCOPED_TRACE("Gradients");
    LayerForwardBackwardFiniteDifferenceTester<LayerType>::Run();
  }
}

TEST(Softmax, IsNormalized)
{
  typedef SoftMax<10, double> LayerType;
  typedef LayerType::NumericType NumericType;
  cv::Mat X(LayerType::NumInputs, 1, CV_64F);
  for (int testIdx = 0; testIdx < 10000; ++testIdx)
  {
    MatRandUniform<NumericType>(&X);
    X *= static_cast<NumericType>(RandBound(10));
    cv::Mat Y = cv::Mat(LayerType::NumOutputs, 1, CV_64F,
                        cv::Scalar(std::numeric_limits<NumericType>::max()));
    LayerType hl;
    const cv::Mat Y0 = Y.clone();
    hl.Forward(X, cv::Mat(), &Y);
    const NumericType sum = static_cast<NumericType>(cv::sum(Y).val[0]);
    EXPECT_NEAR(sum, 1, 1e-6);
  }
}

}

#endif //SRC_LAYER_GTEST
