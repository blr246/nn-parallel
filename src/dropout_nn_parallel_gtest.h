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
