#ifndef SRC_DROPOUT_NN_PARALLEL_GTEST
#define SRC_DROPOUT_NN_PARALLEL_GTEST
#include "neural_network.h"
#include "layer.h"
#include "rand_bound.h"

#include "opencv/cxoperations.hpp"
#include "gtest/gtest.h"
#include <algorithm>
#include <limits>

template <typename T>
struct MatTypeWrapper
{
public:
  explicit MatTypeWrapper(const cv::Mat& m_) : m(&m_) {}
  const cv::Mat* m;
};

template <typename T>
std::ostream& operator<<(std::ostream& stream, const MatTypeWrapper<T>& matTyped)
{
  const cv::Mat& mat = *matTyped.m;
  // Loop over elements dims-wise.
  const cv::MatConstIterator_<T> matEnd = mat.end<T>();
  cv::MatConstIterator_<T> v = mat.begin<T>();
  for (int r = 0; r < mat.rows; ++r)
  {
    stream << "[ ";
    for (int c = 0; c < mat.cols; ++c, ++v)
    {
      stream << *v << " ";
    }
    stream << "]\n";
  }
  return stream.flush();
}


namespace _dropout_nn_parallel_gtest_
{
using namespace blr;

template <typename T>
void MatAssertNear(const cv::Mat& a, const cv::Mat& b, const T eps)
{
  for (int r = 0; r < a.rows; ++r)
  {
    for (int c = 0; c < a.cols; ++c)
    {
      const T& aij = a.at<T>(r, c);
      const T& bij = b.at<T>(r, c);
      ASSERT_NEAR(aij, bij, eps);
    }
  }
}

template <typename T>
void MatAssertNotNear(const cv::Mat& a, const cv::Mat& b, const T eps)
{
  for (int r = 0; r < a.rows; ++r)
  {
    for (int c = 0; c < a.cols; ++c)
    {
      const T& aij = a.at<T>(r, c);
      const T& bij = b.at<T>(r, c);
      ASSERT_GT(std::abs(aij - bij), eps);
    }
  }
}

template <typename T>
inline void MatRandUniform(cv::Mat* m)
{
  std::generate(m->begin<T>(), m->end<T>(), &RandUniform<T>);
}

template <typename LayerType_>
struct TestUtils
{
public:
  typedef LayerType_ LayerType;
  typedef typename LayerType::NumericType NumericType;

  template <int N>
  struct RandomizeMatHelper { static void Init(cv::Mat* m) { MatRandUniform<NumericType>(m); } };
  template <>
  struct RandomizeMatHelper<0> { static void Init(cv::Mat* /*m*/) { } };
};

template <typename LayerType_>
struct ForwardBackwardTest
{
public:
  typedef LayerType_ LayerType;
  typedef typename LayerType::NumericType NumericType;
  typedef typename TestUtils<LayerType>::RandomizeMatHelper<LayerType::NumParameters> InitWHelper;

  static void Run()
  {
    {
      SCOPED_TRACE("Forward");
      LayerType hl;
      cv::Mat W(LayerType::NumParameters, 1, CV_64F);
      InitWHelper::Init(&W);
      cv::Mat X(LayerType::NumInputs, 1, CV_64F);
      MatRandUniform<NumericType>(&X);
      X *= static_cast<NumericType>(10);
      cv::Mat Y = cv::Mat(LayerType::NumOutputs, 1, CV_64F,
                          cv::Scalar(std::numeric_limits<NumericType>::max()));
      const cv::Mat Y0 = Y.clone();
      hl.Forward(X, W, &Y);
      MatAssertNotNear<NumericType>(Y0, Y, 1.0e-6);
    }
    {
      SCOPED_TRACE("Backward");
      LayerType hl;
      cv::Mat dLdY(LayerType::NumOutputs, 1, CV_64F);
      MatRandUniform<NumericType>(&dLdY);
      cv::Mat X(LayerType::NumInputs, 1, CV_64F);
      MatRandUniform<NumericType>(&X);
      cv::Mat W(LayerType::NumParameters, 1, CV_64F);
      InitWHelper::Init(&W);
      cv::Mat dLdW = cv::Mat(LayerType::NumParameters, 1, CV_64F,
                             cv::Scalar(std::numeric_limits<NumericType>::max()));
      const cv::Mat dLdW0 = dLdW.clone();
      cv::Mat dLdX = cv::Mat(LayerType::NumInputs, 1, CV_64F,
                             cv::Scalar(std::numeric_limits<NumericType>::max()));
      const cv::Mat dLdX0 = dLdX.clone();
      hl.Backward(dLdY, X, W, &dLdW, &dLdX);
      MatAssertNotNear<NumericType>(dLdW0, dLdW, 1.0e-6);
      MatAssertNotNear<NumericType>(dLdX0, dLdX, 1.0e-6);
    }
  }
};

template <typename LayerType_>
struct LayerForwardBackwardFiniteDifferenceTester
{
  template <typename T> struct NumericTypeToCvType;
  template <> struct NumericTypeToCvType<double> { enum { CvType = CV_64F, }; };
  template <> struct NumericTypeToCvType<float> { enum { CvType = CV_32F, }; };

  typedef LayerType_ LayerType;
  typedef typename LayerType::NumericType NumericType;
  enum { CvType = NumericTypeToCvType<NumericType>::CvType, };

  // Test dLdX.
  static void TestDLdXForward(const cv::Mat& X, const cv::Mat& W,
                              LayerType* layer, cv::Mat* ddX)
  {
    const NumericType eps = static_cast<NumericType>(1e-6);
    // Perturb input by epsilon.
    cv::Mat dX = X.clone();
    cv::Mat outA(LayerType::NumOutputs, 1, CvType);
    cv::Mat outB(LayerType::NumOutputs, 1, CvType);
    // Compute discrete difference formula for derivative.
    for (int i = 0; i < LayerType::NumInputs; ++i)
    {
      dX.at<NumericType>(i, 0) -= eps;
      layer->Forward(dX, W, &outA);
      dX.at<NumericType>(i, 0) += 2 * eps;
      layer->Forward(dX, W, &outB);
      dX.at<NumericType>(i, 0) -= eps;
      outB -= outA;
      outB /= 2 * eps;
      ddX->row(i) = outB.t();
    }
  }

  static void TestDLdXBackward(const cv::Mat& X, const cv::Mat& W,
                               LayerType* layer, cv::Mat* ddX)
  {
    cv::Mat dLdW = cv::Mat(W.size(), W.type());
    cv::Mat dLdX = cv::Mat(X.size(), X.type());
    cv::Mat dLdY = cv::Mat::zeros(LayerType::NumOutputs, 1, CvType);
    // Complte dLdX directly per-output channel.
    for (int i = 0; i < LayerType::NumOutputs; ++i)
    {
      dLdY.at<double>(i, 0) = 1;
      layer->Backward(dLdY, X, W, &dLdW, &dLdX);
      dLdY.at<double>(i, 0) = 0;
      ddX->row(i) = dLdX.t();
    }
  }

  static void TestDLdWForward(const cv::Mat& X, const cv::Mat& W,
                              LayerType* layer, cv::Mat* ddW)
  {
    const NumericType eps = static_cast<NumericType>(1e-6);
    // Perturb input by epsilon.
    cv::Mat dW = W.clone();
    cv::Mat outA(LayerType::NumParameters, 1, CvType);
    cv::Mat outB(LayerType::NumParameters, 1, CvType);
    // Compute discrete difference formula for derivative.
    for (int i = 0; i < LayerType::NumParameters; ++i)
    {
      dW.at<NumericType>(i, 0) -= eps;
      layer->Forward(X, dW, &outA);
      dW.at<NumericType>(i, 0) += 2 * eps;
      layer->Forward(X, dW, &outB);
      dW.at<NumericType>(i, 0) -= eps;
      outB -= outA;
      outB /= 2 * eps;
      ddW->row(i) = outB.t();
    }
  }

  static void TestDLdWBackward(const cv::Mat& X, const cv::Mat& W,
                               LayerType* layer, cv::Mat* ddX)
  {
    cv::Mat dLdW = cv::Mat(W.size(), W.type());
    cv::Mat dLdX = cv::Mat(X.size(), X.type());
    cv::Mat dLdY = cv::Mat::zeros(LayerType::NumOutputs, 1, CvType);
    // Complte dLdW directly per-output channel.
    for (int i = 0; i < LayerType::NumOutputs; ++i)
    {
      dLdY.at<double>(i, 0) = 1;
      layer->Backward(dLdY, X, W, &dLdW, &dLdX);
      dLdY.at<double>(i, 0) = 0;
      ddX->row(i) = dLdW.t();
    }
  }

  static void Run()
  {
    LayerType layer;

    // Get a random input.
    cv::Mat X(LayerType::NumInputs, 1, CvType);
    MatRandUniform<NumericType>(&X);
    X *= static_cast<NumericType>(10);
    cv::Mat W(LayerType::NumParameters, 1, CvType);
    MatRandUniform<NumericType>(&W);

    {
      SCOPED_TRACE("X");
      // Get a matrix to store gradients.
      cv::Mat ddXForward(LayerType::NumInputs, LayerType::NumOutputs, CvType);
      MatRandUniform<NumericType>(&ddXForward);
      cv::Mat ddXBackward(LayerType::NumOutputs, LayerType::NumInputs, CvType);
      MatRandUniform<NumericType>(&ddXBackward);
      // Compute forward/backward difference.
      TestDLdXForward(X, W, &layer, &ddXForward);
      TestDLdXBackward(X, W, &layer, &ddXBackward);
      // Look for absolute differences.
      const NumericType eps = static_cast<NumericType>(1e-6);
      MatAssertNear<NumericType>(ddXForward, ddXBackward.t(), eps);
    }
    {
      SCOPED_TRACE("W");
      // Get a matrix to store gradients.
      cv::Mat ddWForward(LayerType::NumParameters, LayerType::NumOutputs, CvType);
      MatRandUniform<NumericType>(&ddWForward);
      cv::Mat ddWBackward(LayerType::NumOutputs, LayerType::NumParameters, CvType);
      MatRandUniform<NumericType>(&ddWBackward);
      // Compute forward/backward difference.
      TestDLdWForward(X, W, &layer, &ddWForward);
      TestDLdWBackward(X, W, &layer, &ddWBackward);
      // Look for absolute differences.
      const NumericType eps = static_cast<NumericType>(1e-6);
      MatAssertNear<NumericType>(ddWForward, ddWBackward.t(), eps);
    }
  }
};

TEST(HiddenLinearTanh, ForwardBackward)
{
  typedef HiddenLinearTanh<100, 200, double> LayerType;
  ForwardBackwardTest<LayerType>::Run();
}

TEST(HiddenLinearTanh, LayerGradients)
{
#if defined(NDEBUG)
  typedef HiddenLinearTanh<100, 200, double> LayerType;
#else
  typedef HiddenLinearTanh<10, 50, double> LayerType;
#endif
  LayerForwardBackwardFiniteDifferenceTester<LayerType>::Run();
}

TEST(Passthrough, ForwardBackward)
{
  typedef Passthrough<100, double> LayerType;
  ForwardBackwardTest<LayerType>::Run();
}

}

#endif //SRC_DROPOUT_NN_PARALLEL_GTEST
