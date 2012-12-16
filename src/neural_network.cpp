#include "neural_network.h"
#include "idx_cv.h"
#include "omp_lock.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include <utility>
#include <iostream>
#include <iomanip>
#include <omp.h>

using namespace blr;

enum { MaxL2 = 15, };
enum { HexAddrLabelColW = 60, };

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
//: eps0(0.001),
: eps0(10.0),
  exp(0.998),
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
  (*deltaWPrev) = cv::Scalar::all(0);
}

const cv::Mat* WeightExponentialDecay::ComputeDeltaW(int t, const cv::Mat& dLdW)
{
//  // Just do a scaled update (exponential decay).
//  const double gradScale = -eps0 * std::pow(exp, t);
//  dLdW.copyTo(*deltaWPrev);
//  (*deltaWPrev) *= gradScale;

  const double rhoScale = t / static_cast<double>(T);
  const double rhoT = (t >= T) ? rho1 : (rhoScale * rho1) + ((1.0 - rhoScale) * rho0);
  const double epsT = eps0 * std::pow(exp, t);
  const double gradScale = -std::max((1 - rhoT) * epsT, gradScaleMin);
  std::stringstream ssMsg;
  ssMsg <<  std::dec
        << "t = " << t << ", rhoScale = " << rhoScale << ", rhoT = " << rhoT << ", "
           "epsT = " << epsT << ", gradScale = " << gradScale << std::endl;
  std::cout << ssMsg.str(); std::cout.flush();
  (*deltaWPrev) *= rhoT;
  cv::scaleAdd(dLdW, gradScale, *deltaWPrev, *deltaWPrev);

  DETECT_NUMERICAL_ERRORS(dLdW);
  DETECT_NUMERICAL_ERRORS(*deltaWPrev);
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
  int t;

  ThreadsafeVector<CvMatPtr> updates;
  OmpLock busyLock;
  mutable OmpLock latestWLock;
};

UpdateDelegator::UpdateDelegator()
: latestWPtr(CreateCvMatPtr()),
  dataTrain(NULL),
  dataTest(NULL),
  learningRate(),
  t(0),
  updates(),
  busyLock(),
  latestWLock()
{}

template <typename NNType>
inline
void UpdateDelegator::Initialize(NNType* nn,
                                 const Dataset* dataTrain_, const Dataset* dataTest_)
{
  dataTrain = dataTest_;
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
    cv::Mat* newW = newWPtr; (void)newW;
    DETECT_NUMERICAL_ERRORS(*newW);
    // Lock again for update to W.
    OmpLock::ScopedLock myLockW(&latestWLock);
    latestWPtr = newWPtr;
    // Make busy unlocked first so we have order ABAB.
    const int tNow = t;
    myBusyLock.Unlock();
    std::stringstream ssMsg;
    ssMsg << std::setfill('.') << std::setw(HexAddrLabelColW)
          << "Set latest W " << std::hex << static_cast<void*>(latestWPtr->data) << "\n";
    std::cout << ssMsg.str(); std::cout.flush();
    const cv::Mat* latestW = latestWPtr; (void)latestW;
    myLockW.Unlock();
    DETECT_NUMERICAL_ERRORS(*latestW);

    // Compute train loss on data.
    {
      double lossTrain;
      int errorsTrain;
      ScopedDropoutDisabler<NNType> disableDropout(nn);
      ssMsg.str("");
      ssMsg << std::setfill('.') << std::setw(HexAddrLabelColW)
            << "Computing train loss nn.W "
            << std::hex << static_cast<void*>(nn->GetWPtr()->data) << "\n";
      std::cout << ssMsg.str(); std::cout.flush();
      NLLCriterion::DatasetLoss(*nn, dataTrain->first, dataTrain->second, &lossTrain, &errorsTrain);
      ssMsg.str("");
      ssMsg << std::setfill('.') << std::setw(HexAddrLabelColW)
            << "Train loss for nn.W " << std::hex << static_cast<void*>(nn->GetWPtr()->data) << "\n"
               "train loss: " << lossTrain << ", train errors: " << std::dec << errorsTrain << "\n";
      std::cout << ssMsg.str(); std::cout.flush();
    }
    // Compute test loss on data.
    {
      double lossTest;
      int errorsTest;
      ScopedDropoutDisabler<NNType> disableDropout(nn);
      ssMsg.str("");
      ssMsg << std::setfill('.') << std::setw(HexAddrLabelColW)
            << "Computing test loss nn.W "
            << std::hex << static_cast<void*>(nn->GetWPtr()->data) << "\n";
      std::cout << ssMsg.str(); std::cout.flush();
      NLLCriterion::DatasetLoss(*nn, dataTest->first, dataTest->second, &lossTest, &errorsTest);
      ssMsg.str("");
      ssMsg << std::setfill('.') << std::setw(HexAddrLabelColW)
            << "Test loss for nn.W " << std::hex << static_cast<void*>(nn->GetWPtr()->data) << "\n"
               "test loss: " << lossTest << ", test errors: " << std::dec << errorsTest << "\n";
      ssMsg << "Updated weights for all t < " << tNow << std::endl;
      std::cout << ssMsg.str(); std::cout.flush();
    }
  }
  else
  {
    // Push update, grab newest W, load, and go.
    std::stringstream ssMsg;
    ssMsg << std::setfill('.') << std::setw(HexAddrLabelColW)
          << "Pushing gradient " << std::hex << static_cast<void*>(update->data) << "\n";
    std::cout << ssMsg.str(); std::cout.flush();
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
  cv::Mat* newW = newWPtr; (void)newW;
  DETECT_NUMERICAL_ERRORS(*newW);
  // Lock again for update to W.
  OmpLock::ScopedLock myLockW(&latestWLock);
  latestWPtr = newWPtr;
  const cv::Mat* latestW = latestWPtr; (void)latestW;
  DETECT_NUMERICAL_ERRORS(*latestW);
  // Make busy unlocked first so we have order ABAB.
  myBusyLock.Unlock();
  myLockW.Unlock();
}

template <typename NNType>
void UpdateDelegator::ApplyWTo(NNType* nn) const
{
  CvMatPtr newWPtr;
  OmpLock::ScopedLock myLockW(&latestWLock);
  newWPtr = latestWPtr;
  myLockW.Unlock();
  nn->SetWPtr(newWPtr);
  const cv::Mat* newW = newWPtr; (void)newW;
  std::stringstream ssMsg;
  ssMsg << std::setfill('.') << std::setw(HexAddrLabelColW)
        << "Applying W " << std::hex << static_cast<void*>(newW->data) << "\n";
  std::cout << ssMsg.str(); std::cout.flush();
  DETECT_NUMERICAL_ERRORS(*newW);
}

template <typename NNType>
CvMatPtr UpdateDelegator::ProcessUpdates(const std::vector<CvMatPtr>& myUpdates, NNType* nn)
{
  typedef typename NNType::NumericType NumericType;
  // Process the updates.
  CvMatPtr newWPtr = CreateCvMatPtr();
  cv::Mat* newW = newWPtr;
  const cv::Mat* latestW = latestWPtr;
  latestW->copyTo(*newW);
  std::stringstream ssMsg;
  ssMsg << std::setfill('.') << std::setw(HexAddrLabelColW)
        << "Processing new W " << std::hex << static_cast<void*>(newW->data) << "\n";
  std::cout << ssMsg.str(); std::cout.flush();
  const size_t numUpdates = myUpdates.size();
//  const NumericType avgScale = static_cast<NumericType>(1.0 / numUpdates);
  for (size_t i = 0; i < numUpdates; ++i, ++t)
  {
    DETECT_NUMERICAL_ERRORS(*myUpdates[i]);
    const cv::Mat* update = myUpdates[i];
    const double dbgGrad = cv::norm(*update);
    ssMsg.str("");
    ssMsg << std::setfill('.') << std::setw(HexAddrLabelColW)
          << "Processing G " << std::hex << static_cast<void*>(update->data) << "\n"
          << "||g_" << std::dec << t << "|| = " << dbgGrad << "\n";
    std::cout << ssMsg.str(); std::cout.flush();
    const cv::Mat* deltaW = learningRate.ComputeDeltaW(t, *update);
    DETECT_NUMERICAL_ERRORS(*deltaW);
    const double dbgDeltaNorm = cv::norm(*deltaW);
    ssMsg.str("");
    ssMsg << "||delta_" << std::dec << t << "|| = " << dbgDeltaNorm << "\n";
    std::cout << ssMsg.str(); std::cout.flush();
    (*newW) += *deltaW;
//    cv::scaleAdd(*update, avgScale, *newW, *newW);
    nn->SetWPtr(newWPtr);
    DETECT_NUMERICAL_ERRORS(*newW);
    nn->TruncateL2(static_cast<typename NNType::NumericType>(MaxL2));
    DETECT_NUMERICAL_ERRORS(*newW);
  }
  ssMsg.str("");
  ssMsg << std::setfill('.') << std::setw(HexAddrLabelColW)
        << "Finished processing new W " << std::hex << static_cast<void*>(newW->data) << "\n"
           "||W_" << std::dec << t << "|| = " << cv::norm(*newW) << "\n";
  std::cout << ssMsg.str(); std::cout.flush();
  DETECT_NUMERICAL_ERRORS(*newW);
  return newWPtr;
}

struct UpdateDelegatorWrapper
{
  UpdateDelegatorWrapper();
  UpdateDelegatorWrapper(UpdateDelegator* ud_);

  template <typename NNType>
  void SubmitGradient(CvMatPtr update, NNType* nn);
  template <typename NNType>
  void ApplyWTo(NNType* nn) const;

  UpdateDelegator* ud;
};

UpdateDelegatorWrapper::UpdateDelegatorWrapper()
  : ud(NULL)
{}

UpdateDelegatorWrapper::UpdateDelegatorWrapper(UpdateDelegator* ud_)
  : ud(ud_)
{}

template <typename NNType>
void UpdateDelegatorWrapper::SubmitGradient(CvMatPtr update, NNType* nn)
{
  ud->SubmitGradient(update, nn);
}

template <typename NNType>
void UpdateDelegatorWrapper::ApplyWTo(NNType* nn) const
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
  batchSize(batchSize_),
  sampleIdx(RandBound(dataTrain->first.rows)),
  dLdY(NNType::NumClasses, 1, CvType)
{}

template <typename NNType_, typename WeightUpdateType_>
void MiniBatchTrainer<NNType_, WeightUpdateType_>
::Run(int /*t*/)
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
  const CvMatPtr WPtr = nn->GetWPtr();
  DETECT_NUMERICAL_ERRORS(*WPtr);
  // Get a gradient to accumulate into.
  CvMatPtr dwAccum = CreateCvMatPtr();
  dwAccum->create(WPtr->size(), CvType);
  *dwAccum = cv::Scalar::all(0);
  // br - What is the best policy for dropout refresh?
  nn->RefreshDropoutMask();
  for (int batchIdxJ = 0; batchIdxJ < batchSize; ++batchIdxJ, ++sampleIdx)
  {
//    nn->RefreshDropoutMask();
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
}

template <typename NNType, typename WeightUpdateType>
void ComputeDatasetLossParallel(const Dataset& dataset,
                                const WeightUpdateType& weightUpdate,
                                std::vector<NNType>* networks,
                                double* loss, int* errors)
{
  const int numNetworks = static_cast<int>(networks->size());
  std::vector<double> lossPerThread(numNetworks, 0);
  std::vector<int> errorsPerThread(numNetworks, 0);
  std::vector<Dataset> datasetPerThread(numNetworks);
  const int datasetSize = dataset.first.rows;
  const int numPointsPerThread = datasetSize / numNetworks;
  for (int i = 0; i < (numNetworks - 1); ++i)
  {
    Dataset& threadDataset = datasetPerThread[i];
    const int rBegin = numPointsPerThread * i;
    threadDataset.first = dataset.first.rowRange(rBegin, rBegin + numPointsPerThread);
    threadDataset.second = dataset.second.rowRange(rBegin, rBegin + numPointsPerThread);
  }
  {
    const int rBegin = numPointsPerThread * (numNetworks - 1);
    datasetPerThread.back().first = dataset.first.rowRange(rBegin, datasetSize);
    datasetPerThread.back().second = dataset.second.rowRange(rBegin, datasetSize);
  }
#pragma omp parallel for num_threads(numNetworks)
  for (int i = 0; i < numNetworks; ++i)
  {
    const Dataset& data = datasetPerThread[i];
    double& loss = lossPerThread[i];
    int& errors = errorsPerThread[i];
    NNType& nn = networks->at(i);
    weightUpdate.ApplyWTo(&nn);
    ScopedDropoutDisabler<NNType> disableDropout(&nn);
    NLLCriterion::DatasetLoss(nn, data.first, data.second, &loss, &errors);
  }
  *loss = std::accumulate(lossPerThread.begin(), lossPerThread.end(), 0.0);
  *errors = std::accumulate(errorsPerThread.begin(), errorsPerThread.end(), 0);
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
#if defined (NDEBUG)
  enum { MaxRows = -1, }; // No limit.
  enum { NumBatches = 100, };
  enum { BatchSize = 100, };
#else
  enum { MaxRows = 500, };
  enum { NumBatches = 100, };
  enum { BatchSize = 100, };
#endif
  enum { NumWarmStartEpochs = 100, };
//  enum { NumWarmStartEpochs = NumBatches, };
  typedef double NumericType;
  enum { CvType = NumericTypeToCvType<NumericType>::CvType, };
  typedef DualLayerNNSoftmax<784, 10, 1200, 20, 50, NumericType> NNType;
//  typedef DualLayerNNSoftmax<784, 10, 800, 0, 0, NumericType> NNType;

  // Parse args.
  typedef std::pair<std::string, std::string> PathPair;
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
  const PathPair dataTrainPaths = std::make_pair(args.asList[Args::Argv_DataTrainPoints],
                                                 args.asList[Args::Argv_DataTrainLabels]);
  const PathPair dataTestPaths = std::make_pair(args.asList[Args::Argv_DataTestPoints],
                                                args.asList[Args::Argv_DataTestLabels]);

  std::cout << "Network architecture [I > H > H > O] is ["
            << NNType::NumInputs << " -> " << NNType::NumHiddenUnits << " -> "
            << NNType::NumHiddenUnits << " -> " << NNType::NumClasses << "]" << std::endl;
  std::cout << "CvType = " << CvType << std::endl;
  // Load data.
  std::cout << "Loading data..." << std::endl;
  int errorCode = ERROR_NONE;
  Dataset dataTrain;
  if (!IdxToCvMat(dataTrainPaths.first, CvType, MaxRows, &dataTrain.first) ||
      !IdxToCvMat(dataTrainPaths.second, CV_8U, MaxRows, &dataTrain.second) ||
      (dataTrain.first.rows != dataTrain.second.rows))
  {
    std::cerr << "Error loading training data" << std::endl;
    errorCode |= ERROR_BAD_TRAIN_DATA;
  }
  std::cout << "Loaded " << dataTrain.first.rows << " training data points "
               "from file " << dataTrainPaths.first << "." << std::endl;
  Dataset dataTest;
  if (!IdxToCvMat(dataTestPaths.first, CvType, MaxRows, &dataTest.first) ||
      !IdxToCvMat(dataTestPaths.second, CV_8U, MaxRows, &dataTest.second) ||
      (dataTest.first.rows != dataTest.second.rows))
  {
    std::cerr << "Error loading testing data" << std::endl;
    errorCode |= ERROR_BAD_TEST_DATA;
  }
  std::cout << "Loaded " << dataTest.first.rows << " testing data points "
               "from file " << dataTestPaths.first << "." << std::endl;
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
//  enum { ForceTestThreads = 32, };
//  const int numProcessors = ForceTestThreads;
//  omp_set_num_threads(ForceTestThreads);
  std::cout << "Creating networks." << std::endl;
  std::vector<NNType> networks(numProcessors);
  std::cout << "Created networks." << std::endl;
  // There is only one update delegator for threadsafe updates.
  UpdateDelegator updateDelegator;
  updateDelegator.Initialize(&networks[0], &dataTrain, &dataTest);
  // Setup mini batches.
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
  {
    double lossTrain;
    int errorsTrain;
    ComputeDatasetLossParallel(dataTrain, updateDelegator, &networks, &lossTrain, &errorsTrain);
    std::cout << "TRAIN loss initial:\n"
                 "loss: " << lossTrain << ", errors: " << std::dec << errorsTrain << std::endl;
    double lossTest;
    int errorsTest;
    ComputeDatasetLossParallel(dataTest, updateDelegator, &networks, &lossTest, &errorsTest);
    std::cout << "TEST loss initial:\n"
                 "loss: " << lossTest << ", errors: " << std::dec << errorsTest << std::endl;
  }
  // Run a few single-threaded epochs.
  for (int batchIdx = 0;  batchIdx < NumWarmStartEpochs; ++batchIdx)
  {
    MiniBatchTrainerType& trainer = miniBatchTrainers[0];
    std::cout << "Warm start epoch " << (batchIdx + 1) << " of " << NumWarmStartEpochs << std::endl;
    trainer.Run(batchIdx);
  }
  {
    double lossTrain;
    int errorsTrain;
    ComputeDatasetLossParallel(dataTrain, updateDelegator, &networks, &lossTrain, &errorsTrain);
    std::cout << "TRAIN loss:\n"
                 "loss: " << lossTrain << ", errors: " << std::dec << errorsTrain << std::endl;
    double lossTest;
    int errorsTest;
    ComputeDatasetLossParallel(dataTest, updateDelegator, &networks, &lossTest, &errorsTest);
    std::cout << "TEST loss:\n"
                 "loss: " << lossTest << ", errors: " << std::dec << errorsTest << std::endl;
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
  std::cout << "Computing loss after " << NumBatches << " training epochs..." << std::endl;
  {
    double lossTrain;
    int errorsTrain;
    ComputeDatasetLossParallel(dataTrain, updateDelegator, &networks, &lossTrain, &errorsTrain);
    std::cout << "TRAIN loss:\n"
                 "loss: " << lossTrain << ", errors: " << std::dec << errorsTrain << std::endl;
    double lossTest;
    int errorsTest;
    ComputeDatasetLossParallel(dataTest, updateDelegator, &networks, &lossTest, &errorsTest);
    std::cout << "TEST loss:\n"
                 "loss: " << lossTest << ", errors: " << std::dec << errorsTest << std::endl;
  }
  return 0;
}
