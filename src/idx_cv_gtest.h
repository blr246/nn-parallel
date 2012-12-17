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
