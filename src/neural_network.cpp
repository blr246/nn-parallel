#include "neural_network.h"
#include "idx_cv.h"

#include <algorithm>
#include <functional>
#include <vector>
#include <utility>
#include <iostream>
#include <omp.h>

using namespace blr;

struct WeightExponentialDecay
{
  WeightExponentialDecay();

  template <typename NNType>
  void Initialize();

  const cv::Mat* ComputeDeltaW(int t, const cv::Mat& dLdW);

  enum { T = 500, };

  double eps0;
  double exp;
  double rho0;
  double rho1;
  CvMatPtr deltaWPrev;
};

WeightExponentialDecay::WeightExponentialDecay()
: eps0(10.0),
  exp(0.998),
  rho0(0.5),
  rho1(0.99),
  deltaWPrev(CreateCvMatPtr())
{}

template <typename NNType>
void WeightExponentialDecay::Initialize()
{
  deltaWPrev->create(
      NNType::NumParameters, NumericTypeToCvType<typename NNType::NumericType>::CvType);
  (*deltaWPrev) = 0;
}

const cv::Mat* WeightExponentialDecay::ComputeDeltaW(int t, const cv::Mat& dLdW)
{
  const double epsT = eps0 * std::pow(exp, t);
  const double tInv = 1.0 / std::max(t, 1);
  const double rhoT = (t >= T) ? rho1 : (tInv * rho0) + ((1.0 - tInv) * rho1);
  const double gradScale = -(1 - rhoT) * epsT;
  (*deltaWPrev) *= rhoT;
  cv::scaleAdd(dLdW, gradScale, *deltaWPrev, *deltaWPrev);
  return deltaWPrev;
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

  MiniBatchTrainer(NNType* nn,
                   WeightUpdateType weightUpdater_,
                   LearningRateDecay learningRateDecay_,
                   Dataset dataTrain_,
                   int numBatches_, int batchSize_);

  void Run();

  NNType* nn;
  WeightUpdateType weightUpdater;
  LearningRateDecay learningRateDecay;
  Dataset dataTrain;
  int numBatches;
  int batchSize;
};

template <typename NNType_, typename WeightUpdateType_, typename LearningRateDecay_>
MiniBatchTrainer<NNType_, WeightUpdateType_, LearningRateDecay_>
::MiniBatchTrainer(NNType* nn_,
                   WeightUpdateType weightUpdater_, LearningRateDecay learningRateDecay_,
                   Dataset dataTrain_, int numBatches_, int batchSize_)
: nn(nn_),
  weightUpdater(weightUpdater_),
  learningRateDecay(learningRateDecay_),
  dataTrain(dataTrain_),
  numBatches(numBatches_),
  batchSize(batchSize_)
{}

template <typename NNType_, typename WeightUpdateType_, typename LearningRateDecay_>
void MiniBatchTrainer<NNType_, WeightUpdateType_, LearningRateDecay_>
::Run()
{
  const bool wasDropoutEnabled = nn->DropoutEnabled();
  nn->EnableDropout();
  // Backprop.
  double sampleLoss;
  int sampleError;
  cv::Mat dLdY(NNType::NumClasses, 1, CvType);
  enum { DebugPrintEveryNSamples = 500, };
  const int dataTrainSize = dataTrain.first.rows;
  int totalSamples = 0;
  int sampleIdx = 0;
  //weightUpdater.Initialize<NNType>();
  for (int batchIdx = 0; batchIdx < numBatches; ++batchIdx)
  {
    cv::Mat* W;
    nn->GetW(&W);
    // Get a gradient to accumulate into.
    CvMatPtr dwAccum = CreateCvMatPtr();
    dwAccum->create(W->size(), CvType);
    const NumericType avgEtaScale = static_cast<NumericType>(0.001 / batchSize);
    for (int batchIdxJ = 0; batchIdxJ < batchSize; ++batchIdxJ, ++sampleIdx, ++totalSamples)
    {
      sampleIdx %= dataTrainSize;
      const cv::Mat xi = dataTrain.first.row(sampleIdx).t();
      const cv::Mat yi = dataTrain.second.row(sampleIdx);
      const cv::Mat* dLdW = NLLCriterion::SampleGradient(nn, xi, yi,
                                                         &dLdY, &sampleLoss, &sampleError);
      cv::scaleAdd(*dLdW, -avgEtaScale, *dwAccum, *dwAccum);
      if (0 == (totalSamples % DebugPrintEveryNSamples))
      {
        std::cout << "Processed sample " << totalSamples << " of " << dataTrainSize << std::endl;
      }
    }
    // Apply gradient update once.
    const cv::Mat* deltaW = weightUpdater.ComputeDeltaW(batchIdx + 1, *deltaW);
    (*W) += *deltaW;
    // Get a new dropout state.
    nn->RefreshDropoutMask();
    // Truncate value from Hinton et. al. http://arxiv.org/abs/1207.0580.
    nn->TruncateL2(static_cast<NumericType>(15));
  }
  // Return dropout state.
  if (!wasDropoutEnabled)
  {
    nn->DisableDropout();
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
  // Initial loss and errors.
  typedef std::pair<double, int> LossErrorPair;
  std::vector<LossErrorPair> lossErrors;
  LossErrorPair lossErrorPair;
  std::cout << "Computing initial loss.." << std::endl;
  NLLCriterion::DatasetLoss(nn0, dataTrain.first, dataTrain.second,
                            &lossErrorPair.first, &lossErrorPair.second);
  lossErrors.push_back(lossErrorPair);
  // Backprop.
  enum { NumBatches = 100, };
  enum { BatchSize = 100, };
  typedef MiniBatchTrainer<NNType, WeightExponentialDecay, int> MiniBatchTrainerType;
  MiniBatchTrainerType miniMatchTrainer(&nn0, WeightExponentialDecay(),
                                        0, dataTrain, NumBatches, BatchSize);
  miniMatchTrainer.Run();
  // Compute loss on training data.
  NLLCriterion::DatasetLoss(nn0, dataTrain.first, dataTrain.second,
                            &lossErrorPair.first, &lossErrorPair.second);
  lossErrors.push_back(lossErrorPair);
  // Printout all lossies.
  std::cout << "ROUND : (TRAINING LOSS, TRAINING ERRORS)\n";
  const int lossErrorsSize = lossErrors.size();
  for (int i = 0; i < lossErrorsSize; ++i)
  {
    const LossErrorPair& lePair = lossErrors[i];
    std::cout << i << " : (" << lePair.first << ", " << lePair.second << ")\n";
  }
  std::cout.flush();
  return 0;
}
