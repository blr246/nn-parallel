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
#ifndef SRC_MINIBATCH_TRAINER_H
#define SRC_MINIBATCH_TRAINER_H
#include "neural_network.h"
#include "cvmat_pool.h"
#include "type_utils.h"
#include "rand_bound.h"
#include "log.h"

#include "opencv/cv.h"
#include <algorithm>
#include <functional>
#include <utility>
#include <cstring>
#include <iostream>
#if defined(__APPLE__) || defined(UNIX)
#include <sys/types.h>
#endif

namespace blr
{
namespace nn
{

typedef std::pair<cv::Mat, cv::Mat> Dataset;

/// <summary>Parallelizable neural network trainer.</summary>
template <typename NNType_, typename WeightUpdateType_>
class MiniBatchTrainer
{
public:
  typedef NNType_ NNType;
  typedef typename NNType::NumericType NumericType;
  typedef WeightUpdateType_ WeightUpdateType;
  enum { CvType = NumericTypeToCvType<NumericType>::CvType, };

  MiniBatchTrainer(NNType* nn_,
                   WeightUpdateType weightUpdater_,
                   const Dataset* dataTrain_, const Dataset* dataTest_, int batchSize_);
  MiniBatchTrainer(const MiniBatchTrainer& rhs);

  void RefreshSampleIdx();
  void SetNN(NNType* nn_);

  void Run(int t);

private:
  MiniBatchTrainer& operator=(const MiniBatchTrainer&);

  NNType* nn;
  WeightUpdateType weightUpdater;
  const Dataset* dataTrain;
  const Dataset* dataTest;
  int batchSize;
  int sampleIdx;
  cv::Mat dLdY;
};

struct WeightExponentialDecay
{
  enum { T = 500, };

  WeightExponentialDecay();
  WeightExponentialDecay(
      double eps0_, double exp_, double rho0_, double rho1_, double epsMin_);

  template <typename NNType>
  void Initialize(const NNType* nn);
  const cv::Mat* ComputeDeltaW(int t, const cv::Mat& dLdW);

  double eps0;
  double exp;
  double rho0;
  double rho1;
  double epsMin;
  CvMatPtr deltaWPrev;
};

template <typename T>
class ThreadsafeVector
{
public:
  ThreadsafeVector();
  ThreadsafeVector(size_t size);
  ~ThreadsafeVector();

  void Push(const T& ele);
  void Swap(std::vector<T>* rhs);

private:
  ThreadsafeVector(const ThreadsafeVector&);
  ThreadsafeVector& operator=(const ThreadsafeVector&);

  OmpLock lock;
  std::vector<T> v;
};

struct UpdateDelegator
{
  explicit UpdateDelegator(double maxL2_);
  UpdateDelegator(const WeightExponentialDecay& learningRate_, double maxL2_);

  template <typename NNType>
  void Initialize(NNType* nn, const Dataset* dataTrain_, const Dataset* dataTest_);

  template <typename NNType>
  void SubmitGradient(CvMatPtr update, NNType* nn);

  template <typename NNType>
  void Flush(NNType* nn);
  template <typename NNType>
  void ApplyWTo(NNType* nn) const;

private:
  UpdateDelegator(const UpdateDelegator&);
  UpdateDelegator& operator=(const UpdateDelegator&);

  template <typename NNType>
  CvMatPtr ProcessUpdates(const std::vector<CvMatPtr>& myUpdates, NNType* nn);

  CvMatPtr latestWPtr;
  const Dataset* dataTrain;
  const Dataset* dataTest;
  WeightExponentialDecay learningRate;
  double maxL2;
  int t;

  ThreadsafeVector<CvMatPtr> updates;
  OmpLock busyLock;
  mutable OmpLock latestWLock;
};

/// <summary>Share a copiable WeightUpdatePtr* with reference syntax.</summary>
template<typename WeightUpdateType>
struct WeightUpdatePtrWrapper
{
  WeightUpdatePtrWrapper();
  WeightUpdatePtrWrapper(WeightUpdateType* wut_);

  template <typename NNType>
  void SubmitGradient(CvMatPtr update, NNType* nn);
  template <typename NNType>
  void ApplyWTo(NNType* nn) const;

  WeightUpdateType* wut;
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Inline definitions.
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename NNType_, typename WeightUpdateType_>
MiniBatchTrainer<NNType_, WeightUpdateType_>
::MiniBatchTrainer(NNType* nn_,
                   WeightUpdateType weightUpdater_,
                   const Dataset* dataTrain_, const Dataset* dataTest_, int batchSize_)
: nn(nn_),
  weightUpdater(weightUpdater_),
  dataTrain(dataTrain_),
  dataTest(dataTest_),
  batchSize(batchSize_),
  sampleIdx(RandBound(dataTrain->first.rows)),
  dLdY(NNType::NumClasses, 1, CvType)
{}

template <typename NNType_, typename WeightUpdateType_>
MiniBatchTrainer<NNType_, WeightUpdateType_>
::MiniBatchTrainer(const MiniBatchTrainer& rhs)
: nn(rhs.nn),
  weightUpdater(rhs.weightUpdater),
  dataTrain(rhs.dataTrain),
  dataTest(rhs.dataTest),
  batchSize(rhs.batchSize),
  sampleIdx(rhs.sampleIdx),
  dLdY(NNType::NumClasses, 1, CvType)
{}

template <typename NNType_, typename WeightUpdateType_>
void MiniBatchTrainer<NNType_, WeightUpdateType_>
::RefreshSampleIdx()
{
  sampleIdx = RandBound(dataTrain->first.rows);
}

template <typename NNType_, typename WeightUpdateType_>
void MiniBatchTrainer<NNType_, WeightUpdateType_>
::SetNN(NNType* nn_)
{
  nn = nn_;
}

namespace detail
{
inline
int GetThreadId()
{
#if defined(__APPLE__) || defined(UNIX)
  return static_cast<int>(gettid());
#else
  return static_cast<int>(::GetCurrentThreadId());
#endif
}
}

template <typename NNType_, typename WeightUpdateType_>
void MiniBatchTrainer<NNType_, WeightUpdateType_>
::Run(int t)
{
  using detail::GetThreadId;
  Timer batchTimer;
  ScopedDropoutEnabler<NNType> dropoutEnabled(nn);
  std::stringstream ssMsg;
  const int threadId = GetThreadId();

  double sampleLoss;
  int sampleError;
  const int dataTrainSize = dataTrain->first.rows;
  const NumericType avgScale = static_cast<NumericType>(1.0 / batchSize);
  // Do one batch.
  ssMsg.str("");
  ssMsg << "Thread " << threadId << " starting batch " << t << "\n";
  Log(ssMsg.str(), &std::cout);
  const CvMatPtr WPtr = nn->GetWPtr();
  DETECT_NUMERICAL_ERRORS(*WPtr);
  // Get a gradient to accumulate into.
  CvMatPtr dwAccum = CreateCvMatPtr();
  dwAccum->create(WPtr->size(), CvType);
  *dwAccum = cv::Scalar::all(0);
  for (int batchIdxJ = 0; batchIdxJ < batchSize; ++batchIdxJ, ++sampleIdx)
  {
    // Sync latest
    weightUpdater.ApplyWTo(nn);
    nn->RefreshDropoutMask();
    sampleIdx %= dataTrainSize;
    const cv::Mat xi = dataTrain->first.row(sampleIdx).t();
    const cv::Mat yi = dataTrain->second.row(sampleIdx);
    const cv::Mat* dLdW = NLLCriterion::SampleGradient(nn, xi, yi,
                                                       &dLdY, &sampleLoss, &sampleError);
    DETECT_NUMERICAL_ERRORS(*dLdW);
    cv::scaleAdd(*dLdW, avgScale, *dwAccum, *dwAccum);
    DETECT_NUMERICAL_ERRORS(*dwAccum);
  }
  // Compute and submit this update.
  DETECT_NUMERICAL_ERRORS(*dwAccum);
  weightUpdater.SubmitGradient(dwAccum, nn);

  ssMsg.str("");
  ssMsg << "Thread " << threadId << " completed batch " << t << " after "
        << batchTimer.GetTime() << "S\n";
  Log(ssMsg.str(), &std::cout);
}

template <typename T>
ThreadsafeVector<T>::ThreadsafeVector()
: lock(),
  v()
{}

template <typename T>
ThreadsafeVector<T>::ThreadsafeVector(size_t size)
: lock(),
  v(size)
{}

template <typename T>
ThreadsafeVector<T>::~ThreadsafeVector()
{}

template <typename T>
inline
void ThreadsafeVector<T>::Push(const T& ele)
{
  OmpLock::ScopedLock myLock(&lock);
  v.push_back(ele);
}

template <typename T>
inline
void ThreadsafeVector<T>::Swap(std::vector<T>* rhs)
{
  OmpLock::ScopedLock myLock(&lock);
  v.swap(*rhs);
}

template <typename NNType>
inline
void UpdateDelegator::Initialize(NNType* nn,
                                 const Dataset* dataTrain_, const Dataset* dataTest_)
{
  dataTrain = dataTrain_;
  dataTest = dataTest_;
  nn->GetWPtr()->copyTo(*latestWPtr);
  learningRate.Initialize(nn);
  t = 0;
}

template <typename NNType>
void UpdateDelegator::SubmitGradient(CvMatPtr update, NNType* nn)
{
  // Always update when not busy.
  OmpLock::ScopedLock myBusyLock(&busyLock, true);
  if (myBusyLock.Acquired())
  {
    // Get all pending updates.
    std::vector<CvMatPtr> myUpdates;
    updates.Swap(&myUpdates);
    myUpdates.push_back(update);
    CvMatPtr newWPtr = ProcessUpdates(myUpdates, nn);
    // Lock again for update to W.
    OmpLock::ScopedLock myLockW(&latestWLock);
    latestWPtr = newWPtr;
    // Make busy unlocked first so we have order ABAB.
    myBusyLock.Unlock();
    myLockW.Unlock();
    LogMatrix(*newWPtr, "Set latest W", &std::cout);

    DETECT_NUMERICAL_ERRORS(*newWPtr);

//    {
//      ScopedDropoutDisabler<NNType> disableDropout(nn);
//      double lossTrain;
//      int errorsTrain;
//      ssMsg.str("");
//      ssMsg << std::setfill('.') << std::setw(HexAddrLabelColW)
//            << "Computing train loss nn.W "
//            << std::hex << static_cast<void*>(nn->GetWPtr()->data) << "\n";
//      std::cout << ssMsg.str(); std::cout.flush();
//      NLLCriterion::DatasetLoss(*nn, dataTrain->first, dataTrain->second, &lossTrain, &errorsTrain);
//      ssMsg.str("");
//      ssMsg << std::setfill('.') << std::setw(HexAddrLabelColW)
//            << "Train loss for nn.W " << std::hex << static_cast<void*>(nn->GetWPtr()->data) << "\n"
//               "train loss: " << lossTrain << ", train errors: " << std::dec << errorsTrain << "\n";
//      std::cout << ssMsg.str(); std::cout.flush();
//      double lossTest;
//      int errorsTest;
//      ssMsg.str("");
//      ssMsg << std::setfill('.') << std::setw(HexAddrLabelColW)
//            << "Computing test loss nn.W "
//            << std::hex << static_cast<void*>(nn->GetWPtr()->data) << "\n";
//      std::cout << ssMsg.str(); std::cout.flush();
//      NLLCriterion::DatasetLoss(*nn, dataTest->first, dataTest->second, &lossTest, &errorsTest);
//      ssMsg.str("");
//      ssMsg << std::setfill('.') << std::setw(HexAddrLabelColW)
//            << "Test loss for nn.W " << std::hex << static_cast<void*>(nn->GetWPtr()->data) << "\n"
//               "test loss: " << lossTest << ", test errors: " << std::dec << errorsTest << "\n";
//      ssMsg << "Updated weights for all t < " << tNow << std::endl;
//      std::cout << ssMsg.str(); std::cout.flush();
//    }
  }
  else
  {
    // Push update, grab newest W, load, and go.
    LogMatrix(*update, "Pushing gradient", &std::cout);
    updates.Push(update);
  }
}

template <typename NNType>
void UpdateDelegator::Flush(NNType* nn)
{
  OmpLock::ScopedLock myBusyLock(&busyLock);
  std::vector<CvMatPtr> myUpdates;
  updates.Swap(&myUpdates);
  CvMatPtr newWPtr = ProcessUpdates(myUpdates, nn);
  // Lock again for update to W.
  OmpLock::ScopedLock myLockW(&latestWLock);
  latestWPtr = newWPtr;
  // Make busy unlocked first so we have order ABAB.
  myBusyLock.Unlock();
  myLockW.Unlock();

  DETECT_NUMERICAL_ERRORS(*newWPtr);
}

template <typename NNType>
void UpdateDelegator::ApplyWTo(NNType* nn) const
{
  CvMatPtr newWPtr;
  OmpLock::ScopedLock myLockW(&latestWLock);
  newWPtr = latestWPtr;
  myLockW.Unlock();
  nn->SetWPtr(newWPtr);
  //LogMatrix(*newWPtr, "Applying W", &std::cout);

  DETECT_NUMERICAL_ERRORS(*newWPtr);
}

template <typename NNType>
void WeightExponentialDecay::Initialize(const NNType* /*nn*/)
{
  deltaWPrev->create(
      static_cast<int>(NNType::NumParameters), 1,
      NumericTypeToCvType<typename NNType::NumericType>::CvType);
  (*deltaWPrev) = cv::Scalar::all(0);
}

template <typename NNType>
CvMatPtr UpdateDelegator::ProcessUpdates(const std::vector<CvMatPtr>& myUpdates, NNType* nn)
{
  typedef typename NNType::NumericType NumericType;
  // Process the updates.
  CvMatPtr newWPtr = CreateCvMatPtr();
  latestWPtr->copyTo(*newWPtr);
  LogMatrix(*newWPtr, "Processing new W", &std::cout);
  const size_t numUpdates = myUpdates.size();
  std::stringstream ssMsg;
  for (size_t i = 0; i < numUpdates; ++i, ++t)
  {
    DETECT_NUMERICAL_ERRORS(*myUpdates[i]);
    const cv::Mat* update = myUpdates[i];
    const double gradNormSq = update->dot(*update);
    LogMatrix(*update, "Processing G", &std::cout);
    ssMsg.str("");
    ssMsg << "||g_" << std::dec << t << "||^2 = " << gradNormSq << "\n";
    Log(ssMsg.str(), &std::cout);
    const cv::Mat* deltaW = learningRate.ComputeDeltaW(t, *update);
    DETECT_NUMERICAL_ERRORS(*deltaW);
    const double deltaWNormSq = deltaW->dot(*deltaW);
    LogMatrix(*deltaW, "Applying deltaW", &std::cout);
    ssMsg.str("");
    ssMsg << "||delta_" << std::dec << t << "||^2 = " << deltaWNormSq << "\n";
    Log(ssMsg.str(), &std::cout);
    (*newWPtr) += *deltaW;
    DETECT_NUMERICAL_ERRORS(*newWPtr);
  }
  nn->SetWPtr(newWPtr);
  nn->TruncateL2(static_cast<NumericType>(maxL2));
  DETECT_NUMERICAL_ERRORS(*newWPtr);
  ssMsg.str("");
  ssMsg << "||W_" << std::dec << t << "||^2 = " << newWPtr->dot(*newWPtr) << "\n";
  Log(ssMsg.str(), &std::cout);
  LogMatrix(*newWPtr, "Finished processing new W", &std::cout);
  DETECT_NUMERICAL_ERRORS(*newWPtr);
  return newWPtr;
}

template <typename WeightUpdateType>
WeightUpdatePtrWrapper<WeightUpdateType>::WeightUpdatePtrWrapper()
  : wut(NULL)
{}

template <typename WeightUpdateType>
WeightUpdatePtrWrapper<WeightUpdateType>::WeightUpdatePtrWrapper(WeightUpdateType* wut_)
  : wut(wut_)
{}

template <typename WeightUpdateType>
template <typename NNType>
inline
void WeightUpdatePtrWrapper<WeightUpdateType>
::SubmitGradient(CvMatPtr update, NNType* nn)
{
  wut->SubmitGradient(update, nn);
}

template <typename WeightUpdateType>
template <typename NNType>
inline
void WeightUpdatePtrWrapper<WeightUpdateType>
::ApplyWTo(NNType* nn) const
{
  wut->ApplyWTo(nn);
}

} // end ns nn
using namespace nn;
} // end ns blr

#endif //SRC_MINIBATCH_TRAINER_H
