#include "layer.h"
#include "neural_network.h"
#include "opencv/cxoperations.hpp"
#include "gtest/gtest.h"

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

TEST(nn, Forward)
{
  typedef HiddenLinearTanh<5, 10, 0> HiddenLayer;
  HiddenLayer hl;
  cv::Mat W(HiddenLayer::ParamsTotal, 1, CV_64F);
  cv::Mat X(HiddenLayer::NumInputs, 1, CV_64F);
  cv::Mat Y(HiddenLayer::NumOutputs, 1, CV_64F);
  hl.Forward(X, W, &Y);
  //std::cout << "X=" << MatTypeWrapper<double>(X) << std::endl;
  //std::cout << "W=" << MatTypeWrapper<double>(W) << std::endl;
  //std::cout << "Y=" << MatTypeWrapper<double>(Y) << std::endl;
}

TEST(nn, Backward)
{
  typedef HiddenLinearTanh<5, 10, 0> HiddenLayer;
  HiddenLayer hl;
  cv::Mat W(HiddenLayer::ParamsTotal, 1, CV_64F);
  cv::Mat dLdW(HiddenLayer::ParamsTotal, 1, CV_64F);
  cv::Mat X(HiddenLayer::NumInputs, 1, CV_64F);
  cv::Mat dLdX(HiddenLayer::NumInputs, 1, CV_64F);
  cv::Mat dLdY(HiddenLayer::NumOutputs, 1, CV_64F);
  hl.Backward(dLdY, X, W, &dLdW, &dLdX);
}

}
