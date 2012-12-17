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
#ifndef SRC_IDX_CV_GTEST_H
#define SRC_IDX_CV_GTEST_H
#include "idx_cv.h"
#include "gtest/gtest.h"

namespace _src_idx_cv_gtest_h_
{
using namespace blr;

TEST(idx_cv, LoadFile)
{
  cv::Mat X, Y;
  ASSERT_TRUE(IdxToCvMat<float>("data/t10k-images.idx3-ubyte", &X));
  ASSERT_TRUE(IdxToCvMat<unsigned char>("data/t10k-labels.idx1-ubyte", &Y));
}

TEST(idx_cv, ZeroMeanUnitVar)
{
  typedef float NumericType;
  cv::Mat X, Y;
  ASSERT_TRUE(IdxToCvMat<NumericType>("data/t10k-images.idx3-ubyte", &X));
  cv::Mat mu, stddev;
  ZeroMeanUnitVar<NumericType>(&X, &mu, &stddev);
  // Expect zero mean, unit variance.
  for (int i = 0; i < X.cols; ++i)
  {
    cv::Scalar cMu, cStddev;
    cv::meanStdDev(X.col(i), cMu, cStddev);
    EXPECT_NEAR(0.0, cMu.val[0], 1e-6);
    if (std::abs(cStddev.val[0]) > 1e-6)
    {
      EXPECT_NEAR(1.0, cStddev.val[0], 1e-6);
    }
  }
  ASSERT_TRUE(IdxToCvMat<NumericType>("data/t10k-images.idx3-ubyte", &X));
  ApplyZeroMeanUnitVarTform<NumericType>(mu, stddev, &X);
  // Expect zero mean, unit variance.
  for (int i = 0; i < X.cols; ++i)
  {
    cv::Scalar cMu, cStddev;
    cv::meanStdDev(X.col(i), cMu, cStddev);
    EXPECT_NEAR(0.0, cMu.val[0], 1e-6);
    if (std::abs(cStddev.val[0]) > 1e-6)
    {
      EXPECT_NEAR(1.0, cStddev.val[0], 1e-6);
    }
  }
}

}

#endif //SRC_IDX_CV_GTEST_H
