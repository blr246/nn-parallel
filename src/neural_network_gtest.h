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
#ifndef SRC_NEURAL_NETWORK_GTEST
#define SRC_NEURAL_NETWORK_GTEST
#include "neural_network.h"
#include "idx_cv.h"
#include "type_utils.h"
#include "rand_bound.h"

#include "opencv/cv.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <limits>
#include <deque>

namespace _dropout_nn_parallel_gtest_
{
using namespace blr;

template <int CvTypePoints, int CvTypeLabels>
void LoadDataset(cv::Mat* points, cv::Mat* labels)
{
  ASSERT_TRUE(IdxToCvMat("data/t10k-images.idx3-ubyte", CvTypePoints, -1, points));
  ASSERT_TRUE(IdxToCvMat("data/t10k-labels.idx1-ubyte", CvTypeLabels, -1, labels));
  ASSERT_TRUE(points->rows == labels->rows);
}

TEST(DualLayerNNSoftmax, CopyConstructor)
{
  typedef double NumericType;
  enum { CvType = NumericTypeToCvType<NumericType>::CvType, };
  enum { TrialCount = 3, };
  enum { HiddenUnits = 800, };
  typedef DualLayerNNSoftmax<784, 10, HiddenUnits, 20, 50, NumericType> NNType;
  for (int testIdx = 0; testIdx < TrialCount; ++testIdx)
  {
    NNType nn0;
    NNType nn1 = nn0;
    // Better be the same, but different.
    {
      CvMatPtr WPtr0 = nn0.GetWPtr();
      CvMatPtr WPtr1 = nn1.GetWPtr();
      const cv::Mat diff = (*WPtr0) - (*WPtr1);
      const double diffSum = cv::sum(diff).val[0];
      EXPECT_EQ(0, diffSum);
    }
    cv::Mat points;
    cv::Mat labels;
    LoadDataset<CvType, CV_8U>(&points, &labels);
    const int datasetSize = points.rows;
    const int randomIdx = RandBound(datasetSize);
    const cv::Mat randomXi = points.row(randomIdx).t();
    const cv::Mat randomYi = labels.row(randomIdx);
    double lossInitial0, lossInitial1;
    int errorInitial0, errorInitial1;
    // Same outputs.
    {
      NLLCriterion::SampleLoss(nn0, randomXi, randomYi, &lossInitial0, &errorInitial0);
      NLLCriterion::SampleLoss(nn1, randomXi, randomYi, &lossInitial1, &errorInitial1);
      EXPECT_EQ(lossInitial0, lossInitial1);
      EXPECT_EQ(errorInitial0, errorInitial1);
    }
    // Perturb network 1.
    (*nn1.GetWPtr()) += cv::Scalar::all(1.3003);
    // Different outputs.
    {
      double loss0, loss1;
      int error0, error1;
      NLLCriterion::SampleLoss(nn0, randomXi, randomYi, &loss0, &error0);
      NLLCriterion::SampleLoss(nn1, randomXi, randomYi, &loss1, &error1);
      EXPECT_NE(loss0, loss1);
    }
    // Perturb newtork 2;
    NNType nn2 = nn0;
    nn2.RefreshDropoutMask();
    // Different outputs.
    {
      ScopedDropoutEnabler<NNType> de0(&nn0);
      ScopedDropoutEnabler<NNType> de1(&nn1);
      double loss0, loss2;
      int error0, error2;
      NLLCriterion::SampleLoss(nn0, randomXi, randomYi, &loss0, &error0);
      NLLCriterion::SampleLoss(nn2, randomXi, randomYi, &loss2, &error2);
      EXPECT_NE(loss0, loss2);
    }
  }
}

TEST(DualLayerNNSoftmax, Dropout)
{
}

TEST(DualLayerNNSoftmax, OverfitSinglePoint)
{
  enum { LastN = 5, };
  enum { FwdBackIters = 50, };

  typedef double NumericType;
  enum { CvType = NumericTypeToCvType<NumericType>::CvType, };
  typedef DualLayerNNSoftmax<784, 10, 800, 20, 50, NumericType> NNType;
  cv::Mat points;
  cv::Mat labels;
  LoadDataset<CvType, CV_8U>(&points, &labels);
  // Perform forward-backward iterations on a single point and track the loss.
  const int datasetSize = points.rows;
  NNType nn;
  std::deque<double> lastNLoss(LastN, std::numeric_limits<NumericType>::infinity());
  const int randomIdx = RandBound(datasetSize);
  const cv::Mat randomXi = points.row(randomIdx).t();
  const cv::Mat randomYi = labels.row(randomIdx);
  const int trueLabel = randomYi.at<unsigned char>(0, 0);
  cv::Mat dLdY = cv::Mat(NNType::NumClasses, 1, CvType);
  cv::Mat dLdYMask = cv::Mat::ones(NNType::NumClasses, 1, CvType);
  dLdYMask.at<NumericType>(trueLabel, 0) = 0;
  for (int i = 0; i < FwdBackIters; ++i)
  {
    double loss;
    int error;
    ScopedDropoutDisabler<NNType> disableDropout(&nn);
    const cv::Mat* dLdW = NLLCriterion::SampleGradient(
      &nn, randomXi, randomYi, &dLdY, &loss, &error);
    // Apply gradient update.
    CvMatPtr WPtr = nn.GetWPtr();
    cv::scaleAdd(*dLdW, -0.001, *WPtr, *WPtr);
    // Check loss is decreasing.
    lastNLoss.pop_front();
    lastNLoss.push_back(loss);
    EXPECT_LT(lastNLoss.back(), lastNLoss.front());
    // Gradient should live only in label's row.
    EXPECT_LT(dLdY.at<NumericType>(trueLabel, 0), 0);
    const double lossGradientCheck = dLdY.dot(dLdYMask);
    EXPECT_EQ(0, lossGradientCheck);
  }
}

}

#endif //SRC_NEURAL_NETWORK_GTEST
