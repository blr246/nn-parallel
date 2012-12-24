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
#ifndef SRC_MINIBATCH_TRAINER_GTEST_H
#define SRC_MINIBATCH_TRAINER_GTEST_H
#include "minibatch_trainer.h"
#include "cvmat_pool.h"
#include "omp_lock.h"

#include "gtest/gtest.h"
#include <vector>

namespace _src_minibatch_trainer_gtest_h_
{
using namespace blr;

class MockWeightUpdateType
{
public:
  MockWeightUpdateType();

  template <typename NNType>
  void SubmitGradient(CvMatPtr update, NNType* nn);

  template <typename NNType>
  void ApplyWTo(NNType* nn) const;

  OmpLock lock;
  int submitCount;

private:
  MockWeightUpdateType(const MockWeightUpdateType&);
  MockWeightUpdateType& operator=(const MockWeightUpdateType&);
};

MockWeightUpdateType::MockWeightUpdateType()
: lock(),
  submitCount(0)
{}

template <typename NNType>
void MockWeightUpdateType::SubmitGradient(CvMatPtr update, NNType* /*nn*/)
{
  OmpLock::ScopedLock lock(&lock);
  ++submitCount;
}

template <typename NNType>
inline
void MockWeightUpdateType::ApplyWTo(NNType* /*nn*/) const
{}

TEST(MinibatchTrainer, CopyCtor)
{
  enum { DataDims = 10, };
  enum { DataClasses = 10, };
  enum { DataPoints = 100, };
  enum { BatchSize = 10, };
  typedef float NumericType;
  enum { CvType = NumericTypeToCvType<NumericType>::CvType, };
  typedef DualLayerNNSoftmax<10, DataClasses, 800, 20, 50, NumericType> NNType;
  typedef WeightUpdatePtrWrapper<MockWeightUpdateType> MockWeightUpdateTypeWrapper;
  typedef MiniBatchTrainer<NNType, MockWeightUpdateTypeWrapper> MiniBatchTrainer;

  Dataset dataTrain;
  dataTrain.first = cv::Mat(DataPoints, DataDims, CvType);
  dataTrain.second = cv::Mat(DataPoints, 1, CV_8U);
  cv::randn(dataTrain.first, 0.0, 1.0);
  cv::randu(dataTrain.second, cv::Scalar::all(0), cv::Scalar::all(DataClasses));
  Dataset dataTest;

  MockWeightUpdateType mockWeightUpdate;
//  mockWeightUpdate.latestW->create(NNType::NumParameters, 1, CvType);
//  (*mockWeightUpdate.latestW) = cv::Scalar::all(1);
  std::vector<MiniBatchTrainer> trainers(10, MiniBatchTrainer(NULL,
    MockWeightUpdateTypeWrapper(&mockWeightUpdate),
    &dataTrain, &dataTest, BatchSize));
  NNType nn;
  trainers[0].SetNN(&nn);
  trainers[0].Run(1);
}

}

#endif //SRC_MINIBATCH_TRAINER_GTEST_H
