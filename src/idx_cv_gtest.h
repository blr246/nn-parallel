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
  ASSERT_TRUE(IdxToCvMat("data/t10k-images.idx3-ubyte", &X));
  ASSERT_TRUE(IdxToCvMat("data/t10k-labels.idx1-ubyte", &Y));
}

}

#endif //SRC_IDX_CV_GTEST_H
