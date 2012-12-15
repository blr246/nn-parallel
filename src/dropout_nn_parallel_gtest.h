#include "layer.h"
#include "neural_network.h"
#include "type_utils.h"

#include "opencv/cxoperations.hpp"
#include "gtest/gtest.h"

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
