#include "minibatch_trainer.h"
#include "neural_network.h"
#include "idx_cv.h"
#include "omp_lock.h"
#include "timer.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include <utility>
#include <iostream>
#include <iomanip>
#include <omp.h>

using namespace blr;

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
//  enum { NumWarmStartEpochs = 5, };
  enum { NumWarmStartEpochs = NumBatches, };
//  typedef double NumericType;
  typedef float NumericType;
  enum { CvType = NumericTypeToCvType<NumericType>::CvType, };
  typedef DualLayerNNSoftmax<784, 10, 800, 20, 50, NumericType> NNType;
//  typedef DualLayerNNSoftmax<784, 10, 800, 0, 0, NumericType> NNType;
  std::stringstream ssMsg;

  // Parse args.
  typedef std::pair<std::string, std::string> PathPair;
  Args args;
  {
    if (argc != (Args::Argv_Count + 1))
    {
      std::cerr << "Error: invalid arguments\n";
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

  ssMsg.str("");
  ssMsg << "Network architecture [I > H > H > O] is ["
         << NNType::NumInputs << " -> " << NNType::NumHiddenUnits << " -> "
         << NNType::NumHiddenUnits << " -> " << NNType::NumClasses << "]\n";
  Log(ssMsg.str(), &std::cout);
  ssMsg.str("");
  ssMsg << "CvType = " << CvType << "\n";
  Log(ssMsg.str(), &std::cout);
  // Load data.
  ssMsg.str("");
  ssMsg << "Loading data...\n";
  Log(ssMsg.str(), &std::cout);
  int errorCode = ERROR_NONE;
  Dataset dataTrain;
  if (!IdxToCvMat(dataTrainPaths.first, CvType, MaxRows, &dataTrain.first) ||
      !IdxToCvMat(dataTrainPaths.second, CV_8U, MaxRows, &dataTrain.second) ||
      (dataTrain.first.rows != dataTrain.second.rows))
  {
    std::cerr << "Error loading training data\n";
    errorCode |= ERROR_BAD_TRAIN_DATA;
  }
  ssMsg.str("");
  ssMsg << "Loaded " << dataTrain.first.rows << " training data points "
           "from file " << dataTrainPaths.first << ".\n";
  Log(ssMsg.str(), &std::cout);
  Dataset dataTest;
  if (!IdxToCvMat(dataTestPaths.first, CvType, MaxRows, &dataTest.first) ||
      !IdxToCvMat(dataTestPaths.second, CV_8U, MaxRows, &dataTest.second) ||
      (dataTest.first.rows != dataTest.second.rows))
  {
    std::cerr << "Error loading testing data\n";
    errorCode |= ERROR_BAD_TEST_DATA;
  }
  ssMsg.str("");
  ssMsg << "Loaded " << dataTest.first.rows << " testing data points "
           "from file " << dataTestPaths.first << ".\n";
  Log(ssMsg.str(), &std::cout);
  assert(dataTrain.first.type() == dataTest.first.type());
  if (dataTrain.first.cols != dataTest.first.cols)
  {
    std::cerr << "Error: train/test input dims unmatched\n";
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
  ssMsg.str("");
  ssMsg << "Creating networks.\n";
  Log(ssMsg.str(), &std::cout);
  ssMsg.str("");
  std::vector<NNType> networks(numProcessors);
  ssMsg << "Created networks.\n";
  Log(ssMsg.str(), &std::cout);
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
    ssMsg.str("");
    ssMsg << "TRAIN loss initial:\n"
             "loss: " << lossTrain << ", errors: " << std::dec << errorsTrain << "\n";
    Log(ssMsg.str(), &std::cout);
    double lossTest;
    int errorsTest;
    ComputeDatasetLossParallel(dataTest, updateDelegator, &networks, &lossTest, &errorsTest);
    ssMsg.str("");
    ssMsg << "TEST loss initial:\n"
             "loss: " << lossTest << ", errors: " << std::dec << errorsTest << "\n";
    Log(ssMsg.str(), &std::cout);
  }
  // Run a few single-threaded epochs.
  for (int batchIdx = 0;  batchIdx < NumWarmStartEpochs; ++batchIdx)
  {
    MiniBatchTrainerType& trainer = miniBatchTrainers[0];
    ssMsg.str("");
    ssMsg << "Warm start epoch " << (batchIdx + 1) << " of " << NumWarmStartEpochs << "\n";
    Log(ssMsg.str(), &std::cout);
    trainer.Run(batchIdx);
    {
      double lossTrain;
      int errorsTrain;
      ComputeDatasetLossParallel(dataTrain, updateDelegator, &networks, &lossTrain, &errorsTrain);
      ssMsg.str("");
      ssMsg << "TRAIN loss:\n"
               "loss: " << lossTrain << ", errors: " << std::dec << errorsTrain << "\n";
      Log(ssMsg.str(), &std::cout);
      double lossTest;
      int errorsTest;
      ComputeDatasetLossParallel(dataTest, updateDelegator, &networks, &lossTest, &errorsTest);
      ssMsg.str("");
      ssMsg << "TEST loss:\n"
               "loss: " << lossTest << ", errors: " << std::dec << errorsTest << "\n";
      Log(ssMsg.str(), &std::cout);
    }
  }
//  {
//    double lossTrain;
//    int errorsTrain;
//    ComputeDatasetLossParallel(dataTrain, updateDelegator, &networks, &lossTrain, &errorsTrain);
//    ssMsg.str("");
//    ssMsg << "TRAIN loss:\n"
//             "loss: " << lossTrain << ", errors: " << std::dec << errorsTrain << "\n";
//    Log(ssMsg.str(), &std::cout);
//    double lossTest;
//    int errorsTest;
//    ComputeDatasetLossParallel(dataTest, updateDelegator, &networks, &lossTest, &errorsTest);
//    ssMsg.str("");
//    ssMsg << "TEST loss:\n"
//             "loss: " << lossTest << ", errors: " << std::dec << errorsTest << "\n";
//    Log(ssMsg.str(), &std::cout);
//  }
  // Parallel pandemonium!
#pragma omp parallel for schedule(dynamic, 1)
  for (int batchIdx = NumWarmStartEpochs; batchIdx < NumBatches; ++batchIdx)
  {
    const int whoAmI = omp_get_thread_num();
    MiniBatchTrainerType& trainer = miniBatchTrainers[whoAmI];
    trainer.Run(batchIdx);
  }
  // Do final weight update.
  updateDelegator.Flush(&networks[0]);
  ssMsg.str("");
  ssMsg << "Computing loss after " << NumBatches << " training epochs...\n";
  Log(ssMsg.str(), &std::cout);
  {
    double lossTrain;
    int errorsTrain;
    ComputeDatasetLossParallel(dataTrain, updateDelegator, &networks, &lossTrain, &errorsTrain);
    ssMsg.str("");
    ssMsg << "TRAIN loss:\n"
             "loss: " << lossTrain << ", errors: " << std::dec << errorsTrain << "\n";
    Log(ssMsg.str(), &std::cout);
    double lossTest;
    int errorsTest;
    ComputeDatasetLossParallel(dataTest, updateDelegator, &networks, &lossTest, &errorsTest);
    ssMsg.str("");
    ssMsg << "TEST loss:\n"
             "loss: " << lossTest << ", errors: " << std::dec << errorsTest << "\n";
    Log(ssMsg.str(), &std::cout);
  }
  return 0;
}
