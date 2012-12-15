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
: eps0(0.02),
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
//  std::cout << "t = " << t << ", rhoScale = " << rhoScale << ", rhoT = " << rhoT << ", "
//               "epsT = " << epsT << ", gradScale = " << gradScale << std::endl;
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
  OmpLock::ScopedLock lock(&lock);
  v.push_back(ele);
}

template <typename T>
inline
void ThreadsafeVector<T>::Swap(std::vector<T>* rhs)
{
  OmpLock::ScopedLock lock(&lock);
  v.swap(*rhs);
}

struct UpdateDelegator
{
  UpdateDelegator();

  template <typename NNType>
  void Initialize(NNType* nn);
  template <typename NNType>
  void SubmitWeightUpdate(const CvMatPtr& update, NNType* nn);

private:
  UpdateDelegator(const UpdateDelegator&);
  UpdateDelegator& operator=(const UpdateDelegator&);

  CvMatPtr latestW;
  ThreadsafeVector<CvMatPtr> updates;
  OmpLock busyLock;
  OmpLock latestWLock;
};

UpdateDelegator::UpdateDelegator()
: latestW(CreateCvMatPtr()),
  updates(),
  busyLock(),
  latestWLock()
{}

template <typename NNType>
inline
void UpdateDelegator::Initialize(NNType* nn)
{
  cv::Mat* W;
  nn->GetW(&W);
  W->copyTo(*latestW);
}

template <typename NNType>
void UpdateDelegator::SubmitWeightUpdate(const CvMatPtr& update, NNType* nn)
{
  // Always update when not busy.
  OmpLock::ScopedLock myBusyLock(busyLock);
  if (myBusyLock.Acquired())
  {
    // Get all pending updates.
    std::vector<CvMatPtr> myUpdates;
    updates.Swap(&myUpdates);
    myUpdates.push_back(update);
    // Process the updates.
    CvMatPtr newW = CreateCvMatPtr();
    latestW->copyTo(*newW);
    const size_t numUpdates = myUpdates.size();
    for (size_t i = 0; i < numUpdates; ++i)
    {
      (*newW) += *myUpdates[i];
    }
    nn->SetW(newW);
    nn->TruncateL2(static_cast<typename NNType::NumericType>(MaxL2));
    // Lock again for update to W.
    OmpLock::ScopedLock myLockW(&latestWLock);
    latestW = newW;
    // Make busy unlocked first so we have order ABAB.
    myBusyLock.Unlock();
  }
  else
  {
    // Push update, grab newest W, load, and go.
    updates.Push(update);
    CvMatPtr newW;
    OmpLock::ScopedLock myLockW(&latestWLock);
    newW = latestW;
    myLockW.Unlock();
    nn->SetW(newW);
  }
}

struct UpdateDelegatorWrapper
{
  UpdateDelegatorWrapper(UpdateDelegator* ud_);
  template <typename NNType>
  void SubmitWeightUpdate(const CvMatPtr& update, NNType* nn);
  UpdateDelegator* ud;
};

UpdateDelegatorWrapper::UpdateDelegatorWrapper(UpdateDelegator* ud_)
  : ud(ud_)
{}

template <typename NNType>
void UpdateDelegatorWrapper::SubmitWeightUpdate(const CvMatPtr& update, NNType* nn)
{
  ud->SubmitWeightUpdate(update, nn);
}

/// <summary>Parallelizable neural network trainer.</summary>
template <typename NNType_, typename WeightUpdateType_, typename LearningRateDecay_>
struct MiniBatchTrainer
{
  typedef NNType_ NNType;
  typedef typename NNType::NumericType NumericType;
  typedef WeightUpdateType_ WeightUpdateType;
  typedef LearningRateDecay_ LearningRateDecay;
  enum { CvType = NumericTypeToCvType<NumericType>::CvType, };

  MiniBatchTrainer(NNType* nn_,
                   WeightUpdateType weightUpdater_,
                   LearningRateDecay learningRateDecay_,
                   Dataset dataTrain_, Dataset dataTest_,
                   int numBatches_, int batchSize_);

  void Run();

  NNType* nn;
  WeightUpdateType weightUpdater;
  LearningRateDecay learningRateDecay;
  Dataset dataTrain;
  Dataset dataTest;
  int numBatches;
  int batchSize;
};

template <typename NNType_, typename WeightUpdateType_, typename LearningRateDecay_>
MiniBatchTrainer<NNType_, WeightUpdateType_, LearningRateDecay_>
::MiniBatchTrainer(NNType* nn_,
                   WeightUpdateType weightUpdater_,
                   LearningRateDecay learningRateDecay_,
                   Dataset dataTrain_, Dataset dataTest_,
                   int numBatches_, int batchSize_)
: nn(nn_),
  weightUpdater(weightUpdater_),
  learningRateDecay(learningRateDecay_),
  dataTrain(dataTrain_),
  dataTest(dataTest_),
  numBatches(numBatches_),
  batchSize(batchSize_)
{}

template <typename NNType_, typename WeightUpdateType_, typename LearningRateDecay_>
void MiniBatchTrainer<NNType_, WeightUpdateType_, LearningRateDecay_>
::Run()
{
  ScopedDropoutEnabler<NNType> dropoutEnabled(nn);
  // Backprop.
  double sampleLoss;
  int sampleError;
  cv::Mat dLdY(NNType::NumClasses, 1, CvType);
  const int dataTrainSize = dataTrain.first.rows;
  int totalSamples = 0;
  int sampleIdx = 0;
  weightUpdater.Initialize(nn);
  const NumericType avgScale = static_cast<NumericType>(1.0 / batchSize);
  for (int batchIdx = 0; batchIdx < numBatches; ++batchIdx)
  {
    std::cout << "Starting batch " << batchIdx << std::endl;
    cv::Mat* W;
    nn->GetW(&W);
    // Get a gradient to accumulate into.
    CvMatPtr dwAccum = CreateCvMatPtr();
    dwAccum->create(W->size(), CvType);
    *dwAccum *= 0;
    for (int batchIdxJ = 0; batchIdxJ < batchSize; ++batchIdxJ, ++sampleIdx, ++totalSamples)
    {
      // Get a new dropout state.
      nn->RefreshDropoutMask();
      sampleIdx %= dataTrainSize;
      const cv::Mat xi = dataTrain.first.row(sampleIdx).t();
      const cv::Mat yi = dataTrain.second.row(sampleIdx);
      const cv::Mat* dLdW = NLLCriterion::SampleGradient(nn, xi, yi,
                                                         &dLdY, &sampleLoss, &sampleError);
//      const double checkUpdate = cv::norm(*dwAccum);
//      if ((checkUpdate > 1e100) || (checkUpdate != checkUpdate) || (checkUpdate < 0))
//      {
//        bool what = true;
//      }
      cv::scaleAdd(*dLdW, avgScale, *dwAccum, *dwAccum);
    }
    // Apply gradient update once.
    const cv::Mat* deltaW = weightUpdater.ComputeDeltaW(batchIdx, *dwAccum);
    (*W) += *deltaW;
    // Truncate value from Hinton et. al. http://arxiv.org/abs/1207.0580.
    nn->TruncateL2(static_cast<NumericType>(MaxL2));
    // Compute loss on testing data.
    {
      double lossTest;
      int errorsTest;
      ScopedDropoutDisabler<NNType> disableDropout(nn);
      NLLCriterion::DatasetLoss(*nn, dataTest.first, dataTest.second,
                                &lossTest, &errorsTest);
      std::cout << "loss: " << lossTest << ", errors: " << errorsTest << std::endl;
      std::cout << "Completed batch " << batchIdx << std::endl;
    }
  }
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
  // Instantiate network and train.
  NNType nn0;
  // Parallel backprop.
  // TODO: Create vector of networks and mini batch trainers holding them.
  // Create parallel loop that runs around trying to perform all epochs.
  UpdateDelegator updateDelegator;
  enum { NumBatches = 100, };
  enum { BatchSize = 100, };
  typedef MiniBatchTrainer<
    NNType, WeightExponentialDecay, UpdateDelegatorWrapper> MiniBatchTrainerType;
  MiniBatchTrainerType miniMatchTrainer(
      &nn0, (WeightExponentialDecay()), (UpdateDelegatorWrapper(&updateDelegator)),
      dataTrain, dataTest, NumBatches, BatchSize);
  miniMatchTrainer.Run();

  //const int numProcessors = omp_get_num_procs();
  std::vector<int> workerDoneIds;
  return 0;
}
