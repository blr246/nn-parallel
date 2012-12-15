#include "neural_network.h"
#include "idx_cv.h"
#include "omp_lock.h"

#include <algorithm>
#include <functional>
#include <vector>
#include <utility>
#include <iostream>
#include <omp.h>

using namespace blr;

enum { MaxL2 = 15, };

struct WeightExponentialDecay
{
  enum { T = 500, };
  WeightExponentialDecay();

  template <typename NNType>
  void Initialize(const NNType* nn);
  const cv::Mat* ComputeDeltaW(int t, const cv::Mat& dLdW);

  double eps0;
  double exp;
  double rho0;
  double rho1;
  double gradScaleMin;
  CvMatPtr deltaWPrev;
};

WeightExponentialDecay::WeightExponentialDecay()
: eps0(0.01),
  exp(0.972),
  rho0(0.5),
  rho1(0.99),
  gradScaleMin(0.0001),
  deltaWPrev(CreateCvMatPtr())
{}

template <typename NNType>
void WeightExponentialDecay::Initialize(const NNType* /*nn*/)
{
  deltaWPrev->create(
      static_cast<int>(NNType::NumParameters), 1,
      NumericTypeToCvType<typename NNType::NumericType>::CvType);
  (*deltaWPrev) *= 0;
}

const cv::Mat* WeightExponentialDecay::ComputeDeltaW(int t, const cv::Mat& dLdW)
{
  const double rhoScale = t / static_cast<double>(T);
  const double rhoT = (t >= T) ? rho1 : (rhoScale * rho1) + ((1.0 - rhoScale) * rho0);
  const double epsT = eps0 * std::pow(exp, t);
  const double gradScale = -std::max((1 - rhoT) * epsT, gradScaleMin);
  std::cout << "t = " << t << ", rhoScale = " << rhoScale << ", rhoT = " << rhoT << ", "
               "epsT = " << epsT << ", gradScale = " << gradScale << std::endl;
  (*deltaWPrev) *= rhoT;
  cv::scaleAdd(dLdW, gradScale, *deltaWPrev, *deltaWPrev);
  return deltaWPrev;
}

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

struct UpdateDelegator
{
  UpdateDelegator();

  template <typename NNType>
  void Initialize(NNType* nn, const Dataset* dataTest_);

  template <typename NNType>
  void SubmitGradient(int t, const CvMatPtr& update, NNType* nn);

  template <typename NNType>
  void Flush(NNType* nn);
  template <typename NNType>
  void ApplyWTo(NNType* nn);

private:
  UpdateDelegator(const UpdateDelegator&);
  UpdateDelegator& operator=(const UpdateDelegator&);

  template <typename NNType>
  CvMatPtr ProcessUpdates(const std::vector<CvMatPtr>& myUpdates, NNType* nn);

  CvMatPtr latestW;
  const Dataset* dataTest;
  WeightExponentialDecay learningRate;
  int t;

  ThreadsafeVector<CvMatPtr> updates;
  OmpLock busyLock;
  OmpLock latestWLock;
};

UpdateDelegator::UpdateDelegator()
: latestW(CreateCvMatPtr()),
  dataTest(NULL),
  learningRate(),
  t(0),
  updates(),
  busyLock(),
  latestWLock()
{}

template <typename NNType>
inline
void UpdateDelegator::Initialize(NNType* nn, const Dataset* dataTest_)
{
  dataTest = dataTest_;
  cv::Mat* W;
  nn->GetW(&W);
  W->copyTo(*latestW);
  learningRate.Initialize(nn);
  t = 0;
}

template <typename NNType>
void UpdateDelegator::SubmitGradient(int t, const CvMatPtr& update, NNType* nn)
{
  // Always update when not busy.
  OmpLock::ScopedLock myBusyLock(&busyLock, true);
  if (myBusyLock.Acquired())
  {
    // Get all pending updates.
    std::vector<CvMatPtr> myUpdates;
    updates.Swap(&myUpdates);
    myUpdates.push_back(update);
    CvMatPtr newW = ProcessUpdates(myUpdates, nn);
    // Lock again for update to W.
    OmpLock::ScopedLock myLockW(&latestWLock);
    latestW = newW;
    // Make busy unlocked first so we have order ABAB.
    myBusyLock.Unlock();
    myLockW.Unlock();
    // Compute loss on testing data.
    {
      double lossTest;
      int errorsTest;
      ScopedDropoutDisabler<NNType> disableDropout(nn);
      NLLCriterion::DatasetLoss(*nn, dataTest->first, dataTest->second,
                                &lossTest, &errorsTest);
      myLockW.Lock();
      std::cout << "loss: " << lossTest << ", errors: " << errorsTest << std::endl;
      std::cout << "Updated weights up to batch " << t << std::endl;
      myLockW.Unlock();
    }
  }
  else
  {
    // Push update, grab newest W, load, and go.
    updates.Push(update);
  }
}

template <typename NNType>
void UpdateDelegator::Flush(NNType* nn)
{
  OmpLock::ScopedLock myBusyLock(&busyLock);
  std::vector<CvMatPtr> myUpdates;
  updates.Swap(&myUpdates);
  CvMatPtr newW = ProcessUpdates(myUpdates, nn);
  // Lock again for update to W.
  OmpLock::ScopedLock myLockW(&latestWLock);
  latestW = newW;
  // Make busy unlocked first so we have order ABAB.
  myBusyLock.Unlock();
  myLockW.Unlock();
}

template <typename NNType>
void UpdateDelegator::ApplyWTo(NNType* nn)
{
  CvMatPtr newW;
  OmpLock::ScopedLock myLockW(&latestWLock);
  newW = latestW;
  myLockW.Unlock();
  nn->SetW(newW);
}

template <typename NNType>
CvMatPtr UpdateDelegator::ProcessUpdates(const std::vector<CvMatPtr>& myUpdates, NNType* nn)
{
  typedef typename NNType::NumericType NumericType;
  // Process the updates.
  CvMatPtr newW = CreateCvMatPtr();
  latestW->copyTo(*newW);
  const size_t numUpdates = myUpdates.size();
//  const NumericType avgScale = static_cast<NumericType>(1.0 / numUpdates);
  for (size_t i = 0; i < numUpdates; ++i, ++t)
  {
    const double dbgGrad = cv::norm(*myUpdates[i]);
    std::cout << "||g_" << t << "|| = " << dbgGrad << std::endl;
    const cv::Mat* deltaW = learningRate.ComputeDeltaW(t, *myUpdates[i]);
    const double dbgDeltaNorm = cv::norm(*deltaW);
    std::cout << "||delta_" << t << "|| = " << dbgDeltaNorm << std::endl;
    (*newW) += *deltaW;
//    cv::scaleAdd(*myUpdates[i], avgScale, *newW, *newW);
    nn->SetW(newW);
    nn->TruncateL2(static_cast<typename NNType::NumericType>(MaxL2));
  }
  return newW;
}

struct UpdateDelegatorWrapper
{
  UpdateDelegatorWrapper();
  UpdateDelegatorWrapper(UpdateDelegator* ud_);

  template <typename NNType>
  void SubmitGradient(int t, const CvMatPtr& update, NNType* nn);
  template <typename NNType>
  void ApplyWTo(NNType* nn);

  UpdateDelegator* ud;
};

UpdateDelegatorWrapper::UpdateDelegatorWrapper()
  : ud(NULL)
{}

UpdateDelegatorWrapper::UpdateDelegatorWrapper(UpdateDelegator* ud_)
  : ud(ud_)
{}

template <typename NNType>
void UpdateDelegatorWrapper::SubmitGradient(int t, const CvMatPtr& update, NNType* nn)
{
  ud->SubmitGradient(t, update, nn);
}

template <typename NNType>
void UpdateDelegatorWrapper::ApplyWTo(NNType* nn)
{
  ud->ApplyWTo(nn);
}

/// <summary>Parallelizable neural network trainer.</summary>
template <typename NNType_, typename WeightUpdateType_>
struct MiniBatchTrainer
{
  typedef NNType_ NNType;
  typedef typename NNType::NumericType NumericType;
  typedef WeightUpdateType_ WeightUpdateType;
  enum { CvType = NumericTypeToCvType<NumericType>::CvType, };

  MiniBatchTrainer();
  MiniBatchTrainer(NNType* nn_,
                   WeightUpdateType weightUpdater_,
                   const Dataset* dataTrain_, const Dataset* dataTest_,
                   int numBatches_, int batchSize_);

  void Run(int t);

  NNType* nn;
  WeightUpdateType weightUpdater;
  const Dataset* dataTrain;
  const Dataset* dataTest;
  int numBatches;
  int batchSize;
  int sampleIdx;
  cv::Mat dLdY;
};

template <typename NNType_, typename WeightUpdateType_>
MiniBatchTrainer<NNType_, WeightUpdateType_>
::MiniBatchTrainer()
: nn(NULL),
  weightUpdater(),
  dataTrain(NULL),
  dataTest(NULL),
  numBatches(0),
  batchSize(0),
  sampleIdx(0),
  dLdY(NNType::NumClasses, 1, CvType)
{}

template <typename NNType_, typename WeightUpdateType_>
MiniBatchTrainer<NNType_, WeightUpdateType_>
::MiniBatchTrainer(NNType* nn_,
                   WeightUpdateType weightUpdater_,
                   const Dataset* dataTrain_, const Dataset* dataTest_,
                   int numBatches_, int batchSize_)
: nn(nn_),
  weightUpdater(weightUpdater_),
  dataTrain(dataTrain_),
  dataTest(dataTest_),
  numBatches(numBatches_),
  batchSize(batchSize_)
{}

template <typename NNType_, typename WeightUpdateType_>
void MiniBatchTrainer<NNType_, WeightUpdateType_>
::Run(int t)
{
  ScopedDropoutEnabler<NNType> dropoutEnabled(nn);
  // Sync latest
  weightUpdater.ApplyWTo(nn);

  double sampleLoss;
  int sampleError;
  const int dataTrainSize = dataTrain->first.rows;
  const NumericType avgScale = static_cast<NumericType>(1.0 / batchSize);
  // Do one batch.
//  std::cout << "Starting batch " << t << std::endl;
  cv::Mat* W;
  nn->GetW(&W);
  // Get a gradient to accumulate into.
  CvMatPtr dwAccum = CreateCvMatPtr();
  dwAccum->create(W->size(), CvType);
  *dwAccum *= 0;
  for (int batchIdxJ = 0; batchIdxJ < batchSize; ++batchIdxJ, ++sampleIdx)
  {
    // Get a new dropout state.
    nn->RefreshDropoutMask();
    sampleIdx %= dataTrainSize;
    const cv::Mat xi = dataTrain->first.row(sampleIdx).t();
    const cv::Mat yi = dataTrain->second.row(sampleIdx);
    const cv::Mat* dLdW = NLLCriterion::SampleGradient(nn, xi, yi,
                                                       &dLdY, &sampleLoss, &sampleError);
    cv::scaleAdd(*dLdW, avgScale, *dwAccum, *dwAccum);
  }
  // Compute and submit this update.
  weightUpdater.SubmitGradient(t, dwAccum, nn);
}

enum
{
  ERROR_NONE           = 0,
  ERROR_BAD_ARGUMENTS  = 1 << 0,
  ERROR_BAD_TRAIN_DATA = 1 << 1,
  ERROR_BAD_TEST_DATA  = 1 << 2,
  ERROR_BAD_DATA_DIMS  = 1 << 3,
};

struct Args
{
  typedef std::pair<std::string, std::string> PathPair;
  enum {
    Argv_DataTrainPoints,
    Argv_DataTrainLabels,
    Argv_DataTestPoints,
    Argv_DataTestLabels,
    Argv_Count,
  };

  Args() : asList(Argv_Count) {}

  static void Usage(const char* argv0, std::ostream& stream)
  {
    const std::string strArgv0(argv0);
    const size_t lastPathSep = strArgv0.find_last_of("\\/");
    const size_t appNameBegin = (lastPathSep != std::string::npos) ? lastPathSep + 1: 0;
    const std::string strAppName = strArgv0.substr(appNameBegin);
    stream << "Usage: " << strAppName
           << " PATH_TRAIN_DATA PATH_TRAIN_LABELS PATH_TEST_DATA PATH_TEST_LABELS" << std::endl;
  }

  std::vector<std::string> asList;
};

int main(int argc, char** argv)
{
#if defined(NDEBUG)
  enum { MaxRows = 5000, };
#else
  enum { MaxRows = 500, };
#endif

  // Parse args.
  Args args;
  {
    if (argc != (Args::Argv_Count + 1))
    {
      std::cerr << "Error: invalid arguments\n" << std::endl;
      Args::Usage(*argv, std::cerr);
      return ERROR_BAD_ARGUMENTS;
    }
    for (int i = 0; i < Args::Argv_Count; ++i)
    {
      args.asList[i] = argv[i + 1];
    }
  }
  typedef std::pair<std::string, std::string> PathPair;
  const PathPair dataTrainPaths = std::make_pair(args.asList[Args::Argv_DataTrainPoints],
                                                 args.asList[Args::Argv_DataTrainLabels]);
  const PathPair dataTestPaths = std::make_pair(args.asList[Args::Argv_DataTestPoints],
                                                args.asList[Args::Argv_DataTestLabels]);

  typedef float NumericType;
  typedef DualLayerNNSoftmax<784, 10, 800, 20, 50, NumericType> NNType;
  std::cout << "Network architecture [I > H > H > O] is ["
            << NNType::NumInputs << " -> " << NNType::NumHiddenUnits << " -> "
            << NNType::NumHiddenUnits << " -> " << NNType::NumClasses << "]" << std::endl;
  enum { CvType = NumericTypeToCvType<NumericType>::CvType, };
  std::cout << "CvType = " << CvType << std::endl;
  //         Load data.
  std::cout << "Loading data..." << std::endl;
  Dataset dataTrain;
  Dataset dataTest;
  int errorCode = ERROR_NONE;
  if (!IdxToCvMat(dataTrainPaths.first, CvType, MaxRows, &dataTrain.first) ||
      !IdxToCvMat(dataTrainPaths.second, CV_8U, MaxRows, &dataTrain.second) ||
      (dataTrain.first.rows != dataTrain.second.rows))
  {
    std::cerr << "Error loading training data" << std::endl;
    errorCode |= ERROR_BAD_TRAIN_DATA;
  }
  std::cout << "Loaded " << dataTrain.first.rows << " training data points." << std::endl;
  if (!IdxToCvMat(dataTestPaths.first, CvType, MaxRows, &dataTest.first) ||
      !IdxToCvMat(dataTestPaths.second, CV_8U, MaxRows, &dataTest.second) ||
      (dataTest.first.rows != dataTest.second.rows))
  {
    std::cerr << "Error loading testing data" << std::endl;
    errorCode |= ERROR_BAD_TEST_DATA;
  }
  std::cout << "Loaded " << dataTest.first.rows << " testing data points." << std::endl;
  assert(dataTrain.first.type() == dataTest.first.type());
  if (dataTrain.first.cols != dataTest.first.cols)
  {
    std::cerr << "Error: train/test input dims unmatched" << std::endl;
    errorCode |= ERROR_BAD_DATA_DIMS;
  }
  if (errorCode)
  {
    return errorCode;
  }
  assert(NNType::NumInputs == dataTrain.first.cols);
  // Instantiate networks and train.
  const int numProcessors = omp_get_num_procs();
  std::vector<NNType> networks(numProcessors);
  // There is only one update delegator for threadsafe updates.
  UpdateDelegator updateDelegator;
  updateDelegator.Initialize(&networks[0], &dataTest);
  // Setup mini batches.
  enum { NumBatches = 100, };
  enum { NumWarmStartEpochs = 5, };
  enum { BatchSize = 100, };
  typedef MiniBatchTrainer<NNType, UpdateDelegatorWrapper> MiniBatchTrainerType;
  std::vector<MiniBatchTrainerType> miniBatchTrainers(numProcessors);
  for (int i = 0; i < numProcessors; ++i)
  {
    NNType& nn = networks[i];
    MiniBatchTrainerType& trainer = miniBatchTrainers[i];
    trainer.nn = &nn;
    trainer.weightUpdater = UpdateDelegatorWrapper(&updateDelegator);
    trainer.dataTrain = &dataTrain;
    trainer.dataTest = &dataTest;
    trainer.numBatches = NumBatches;
    trainer.batchSize = BatchSize;
    trainer.sampleIdx = RandBound(dataTrain.first.rows);
  }
  // Run a few single-threaded epochs.
  for (int batchIdx = 0;  batchIdx < NumWarmStartEpochs; ++batchIdx)
  {
    MiniBatchTrainerType& trainer = miniBatchTrainers[0];
    std::cout << "Warm start epoch " << (batchIdx + 1) << " of " << NumWarmStartEpochs << std::endl;
    trainer.Run(batchIdx);
  }
  // Parallel pandemonium!
#pragma omp parallel for schedule(dynamic, 1)
  for (int batchIdx = NumWarmStartEpochs; batchIdx < NumBatches; ++batchIdx)
  {
    const int whoAmI = omp_get_thread_num();
//    std::cout << "Processor " << whoAmI << " got batch " << batchIdx << std::endl;
    MiniBatchTrainerType& trainer = miniBatchTrainers[whoAmI];
    trainer.Run(batchIdx);
  }
  // Do final weight update.
  updateDelegator.Flush(&networks[0]);
  // Compute loss on testing data.
  {
    std::cout << "Computing loss after " << NumBatches << " training epochs..." << std::endl;
    double lossTest;
    int errorsTest;
    ScopedDropoutDisabler<NNType> disableDropout(&networks[0]);
    NLLCriterion::DatasetLoss(networks[0], dataTest.first, dataTest.second,
                              &lossTest, &errorsTest);
    std::cout << "loss: " << lossTest << ", errors: " << errorsTest << std::endl;
  }
  return 0;
}
