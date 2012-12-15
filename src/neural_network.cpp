#include "layer.h"
#include "idx_cv.h"
#include "rand_bound.h"
#include "cv.h"
#include <algorithm>
#include <functional>
#include <vector>
#include <utility>
#include <iostream>

using namespace blr;

template <typename LayerType>
int PartitionOutputs(cv::Mat* Y, cv::Mat* part, int rStart);

template <typename LayerType>
int PartitionParameters(cv::Mat* Y, cv::Mat* part, int rStart);

template <typename LayerType>
int PartitionParameters(cv::Mat* Y, cv::Mat* part, int rStart);

template <typename LayerType>
inline
int PartitionOutputs(cv::Mat* Y, cv::Mat* part, int rStart)
{
  int rEnd = rStart + LayerType::NumOutputs;
  *part = Y->rowRange(rStart, rEnd);
  return rEnd;
}

template <typename LayerType>
inline
int PartitionParameters(cv::Mat* Y, cv::Mat* part, int rStart)
{
  int rEnd = rStart + LayerType::NumParameters;
  *part = Y->rowRange(rStart, rEnd);
  return rEnd;
}

template <typename LayerType>
inline
int PartitionInputs(cv::Mat* Y, cv::Mat* part, int rStart)
{
  int rEnd = rStart + LayerType::NumInputs;
  *part = Y->rowRange(rStart, rEnd);
  return rEnd;
}

struct NLLCriterion
{
  template <typename NNType>
  static const cv::Mat* SampleLoss(const NNType& nn, const cv::Mat& xi, const cv::Mat& yi,
                                   double* loss, int* error);

  template <typename NNType>
  static void DatasetLoss(const NNType& nn, const cv::Mat& X, const cv::Mat& Y,
                          double* loss, int* errors);

  template <typename NNType>
  static const cv::Mat* SampleGradient(NNType* nn, const cv::Mat& xi, const cv::Mat& yi,
                                       cv::Mat* dLdY, double* loss, int* error);
};

template <typename NNType>
const cv::Mat* NLLCriterion::
SampleLoss(const NNType& nn, const cv::Mat& xi, const cv::Mat& yi, double* loss, int* error)
{
  typedef typename NNType::NumericType NumericType;
  const cv::Mat* yOut = nn.Forward(xi);
  // Find max class label.
  int label = 0;
  double maxP = yOut->at<NumericType>(0, 0);
  const int yOutSize = yOut->rows;
  for (int i = 1; i < yOutSize; ++i) {
    const double p = yOut->at<NumericType>(i, 0);
    if (p > maxP)
    {
      maxP = p;
      label = i;
    }
  }
  const int trueLabel = yi.at<unsigned char>(0, 0);
  *error = (trueLabel != label);
  *loss = -std::log(yOut->at<NumericType>(trueLabel, 0));
  return yOut;
}

template <typename NNType>
void NLLCriterion::
DatasetLoss(const NNType& nn, const cv::Mat& X, const cv::Mat& Y, double* loss, int* errors)
{
  typedef typename NNType::NumericType NumericType;
  *errors = 0;
  *loss = 0;
  double sampleLoss;
  int sampleError;
  for (int i = 0; i < X.rows; ++i)
  {
    const cv::Mat xi = X.row(i).t();
    const cv::Mat yi = Y.row(i);
    SampleLoss(nn, xi, yi, &sampleLoss, &sampleError);
    *loss += sampleLoss;
    *errors += sampleError;
  }
}

template <typename NNType>
const cv::Mat* NLLCriterion::
SampleGradient(NNType* nn, const cv::Mat& xi, const cv::Mat& yi,
               cv::Mat* dLdY, double* loss, int* error)
{
  typedef typename NNType::NumericType NumericType;
  // Forward pass.
  const cv::Mat* yOut = SampleLoss(*nn, xi, yi, loss, error);
  // Compute loss gradient to get this party started.
  const int trueLabel = yi.at<unsigned char>(0, 0);
  const NumericType nllGrad = -static_cast<NumericType>(1.0 / yOut->at<NumericType>(trueLabel, 0));
  *dLdY = cv::Scalar(0);
  dLdY->at<NumericType>(trueLabel, 0) = nllGrad;
  // Backward pass.
  return nn->Backward(*dLdY);
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_, typename NumericType_ = double>
struct DualLayerNNSoftmax
{
  typedef NumericType_ NumericType;
  enum { NumInputs = NumInputs_, };
  enum { NumHiddenUnits = NumHiddenUnits_, };
  enum { NumClasses = NumClasses_, };
  enum { CvType = NumericTypeToCvType<NumericType>::CvType, };

  DualLayerNNSoftmax();

  //  v INPUT (~20% dropout, Hinton et. al.)
  typedef Passthrough<NumInputs, NumericType> Layer0;
  // vv Linear + Tahn (~50% dropout, Honton et. al.)
  typedef Linear<NumInputs, NumHiddenUnits, NumericType> Layer1a;
  typedef Tanh<NumHiddenUnits, NumericType> Layer1b;
  // vv Linear + Tahn (~50% dropout, Honton et. al.)
  typedef Linear<NumHiddenUnits, NumHiddenUnits, NumericType> Layer2a;
  typedef Tanh<NumHiddenUnits, NumericType> Layer2b;
  // vv Softmax
  typedef Linear<NumHiddenUnits, NumClasses, NumericType> Layer3a;
  typedef SoftMax<NumClasses, NumericType> Layer3b;
  //  v OUTPUT

  enum { NumSublayers = 7, };

  enum { NumParameters = Layer0::NumParameters +
                         Layer1a::NumParameters + Layer1b::NumParameters +
                         Layer2a::NumParameters + Layer2b::NumParameters +
                         Layer3a::NumParameters + Layer3b::NumParameters, };
  enum { NumInternalOutputs = Layer0::NumOutputs +
                              Layer1a::NumOutputs + Layer1b::NumOutputs +
                              Layer2a::NumOutputs + Layer2b::NumOutputs +
                              Layer3a::NumOutputs + Layer3b::NumOutputs, };
  enum { NumInternalInputs = Layer0::NumInputs +
                             Layer1a::NumInputs + Layer1b::NumInputs +
                             Layer2a::NumInputs + Layer2b::NumInputs +
                             Layer3a::NumInputs + Layer3b::NumInputs, };

  const cv::Mat* Forward(const cv::Mat& X) const;
  const cv::Mat* Backward(const cv::Mat dLdY);
  void Reset();

  // Flattened outputs for the entire network.
  cv::Mat Y;
  // Flattened parameter matrix for the entire network.
  cv::Mat W;
  // Flattened parameter gradient matrix for the entire network.
  cv::Mat dLdW;
  // Flattened gradient matrix for the entire network.
  cv::Mat dLdX;

  mutable cv::Mat yPartitions[NumSublayers];
  cv::Mat wPartitions[NumSublayers];
  cv::Mat dwPartitions[NumSublayers];
  cv::Mat dxPartitions[NumSublayers];
};

struct ParameterPartitionerIterator
{
  ParameterPartitionerIterator(cv::Mat* W_, cv::Mat* Y_, cv::Mat* dLdW_, cv::Mat* dLdX_)
    : W(W_),
      Y(Y_),
      dLdW(dLdW_),
      dLdX(dLdX_),
      wIdx(0),
      yIdx(0),
      dwIdx(0),
      dxIdx(0)
  {}

  template <typename LayerType>
  void Next(cv::Mat* wPart, cv::Mat* yPart, cv::Mat* dLdWPart, cv::Mat* dLdXPart)
  {
    {
      int wEnd = wIdx + LayerType::NumParameters;
      *wPart = W->rowRange(wIdx, wEnd);
      wIdx = wEnd;
    }
    {
      int yEnd = yIdx + LayerType::NumOutputs;
      *yPart = Y->rowRange(yIdx, yEnd);
      yIdx = yEnd;
    }
    {
      int dwEnd = dwIdx + LayerType::NumParameters;
      *dLdWPart = dLdW->rowRange(dwIdx, dwEnd);
      dwIdx = dwEnd;
    }
    {
      int dxEnd = dxIdx + LayerType::NumInputs;
      *dLdXPart = dLdX->rowRange(dxIdx, dxEnd);
      dxIdx = dxEnd;
    }
  }

  cv::Mat* W;
  cv::Mat* Y;
  cv::Mat* dLdW;
  cv::Mat* dLdX;
  int wIdx;
  int yIdx;
  int dwIdx;
  int dxIdx;
};

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_, typename NumericType_>
DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_, NumericType_>
::DualLayerNNSoftmax()
: Y(NumInternalOutputs, 1, CvType),
  W(NumParameters, 1, CvType),
  dLdW(NumParameters, 1, CvType),
  dLdX(NumInternalInputs, 1, CvType),
  yPartitions(),
  wPartitions(),
  dwPartitions(),
  dxPartitions()
{
  ParameterPartitionerIterator partitionIter(&W, &Y, &dLdW, &dLdX);
  partitionIter.Next<Layer0> (wPartitions + 0, yPartitions + 0, dwPartitions + 0, dxPartitions + 0);
  partitionIter.Next<Layer1a>(wPartitions + 1, yPartitions + 1, dwPartitions + 1, dxPartitions + 1);
  partitionIter.Next<Layer1b>(wPartitions + 2, yPartitions + 2, dwPartitions + 2, dxPartitions + 2);
  partitionIter.Next<Layer2a>(wPartitions + 3, yPartitions + 3, dwPartitions + 3, dxPartitions + 3);
  partitionIter.Next<Layer2b>(wPartitions + 4, yPartitions + 4, dwPartitions + 4, dxPartitions + 4);
  partitionIter.Next<Layer3a>(wPartitions + 5, yPartitions + 5, dwPartitions + 5, dxPartitions + 5);
  partitionIter.Next<Layer3b>(wPartitions + 6, yPartitions + 6, dwPartitions + 6, dxPartitions + 6);
  assert(Y.rows == partitionIter.yIdx);
  assert(W.rows == partitionIter.wIdx);
  assert(dLdW.rows == partitionIter.dwIdx);
  assert(dLdX.rows == partitionIter.dxIdx);
  Reset();
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_, typename NumericType_>
inline
const cv::Mat* DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_, NumericType_>
::Forward(const cv::Mat& X) const
{
  int layerIdx = 0;
  Layer0::Forward(X, wPartitions[layerIdx], &yPartitions[layerIdx]);
  cv::Mat* yPrev = yPartitions; ++layerIdx;
  Layer1a::Forward(*yPrev, wPartitions[layerIdx], &yPartitions[layerIdx]);
  ++yPrev; ++layerIdx;
  Layer1b::Forward(*yPrev, wPartitions[layerIdx], &yPartitions[layerIdx]);
  ++yPrev; ++layerIdx;
  Layer2a::Forward(*yPrev, wPartitions[layerIdx], &yPartitions[layerIdx]);
  ++yPrev; ++layerIdx;
  Layer2b::Forward(*yPrev, wPartitions[layerIdx], &yPartitions[layerIdx]);
  ++yPrev; ++layerIdx;
  Layer3a::Forward(*yPrev, wPartitions[layerIdx], &yPartitions[layerIdx]);
  ++yPrev; ++layerIdx;
  Layer3b::Forward(*yPrev, wPartitions[layerIdx], &yPartitions[layerIdx]);
  ++yPrev; ++layerIdx;
  return yPrev;
}

struct BackpropIterator
{
  BackpropIterator(const cv::Mat* X_, const cv::Mat* W_, const cv::Mat* Y_, const cv::Mat* dLdY_,
                   cv::Mat* dLdW_, cv::Mat* dLdX_)
    : X(X_),
      W(W_),
      Y(Y_),
      dLdY(dLdY_),
      dLdW(dLdW_),
      dLdX(dLdX_)
  {}

  void Next()
  {
    --X;
    --W;
    --Y;
    --dLdY;
    --dLdW;
    --dLdX;
  }

  const cv::Mat* X;
  const cv::Mat* W;
  const cv::Mat* Y;
  const cv::Mat* dLdY;
  cv::Mat* dLdW;
  cv::Mat* dLdX;
};

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_, typename NumericType_>
inline
const cv::Mat* DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_, NumericType_>
::Backward(const cv::Mat dLdY)
{
  // Base case.
  Layer3b::Backward(yPartitions[NumSublayers - 2],
                    wPartitions[NumSublayers - 1],
                    yPartitions[NumSublayers - 1], dLdY,
                    &dwPartitions[NumSublayers - 1], &dxPartitions[NumSublayers - 1]);
  // Initialize iteration.
  BackpropIterator bpIter( yPartitions + NumSublayers - 3,  wPartitions + NumSublayers - 2,
                           yPartitions + NumSublayers - 2, dxPartitions + NumSublayers - 1,
                          dwPartitions + NumSublayers - 2, dxPartitions + NumSublayers - 2);
  Layer3a::Backward(*bpIter.X, *bpIter.W, *bpIter.Y, *bpIter.dLdY, bpIter.dLdW, bpIter.dLdX);
  bpIter.Next();
  Layer2b::Backward(*bpIter.X, *bpIter.W, *bpIter.Y, *bpIter.dLdY, bpIter.dLdW, bpIter.dLdX);
  bpIter.Next();
  Layer2a::Backward(*bpIter.X, *bpIter.W, *bpIter.Y, *bpIter.dLdY, bpIter.dLdW, bpIter.dLdX);
  bpIter.Next();
  Layer1b::Backward(*bpIter.X, *bpIter.W, *bpIter.Y, *bpIter.dLdY, bpIter.dLdW, bpIter.dLdX);
  bpIter.Next();
  Layer1a::Backward(*bpIter.X, *bpIter.W, *bpIter.Y, *bpIter.dLdY, bpIter.dLdW, bpIter.dLdX);
  // Do not backpropagate input.
  return &dLdW;
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_, typename NumericType_>
inline
void DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_, NumericType_>
::Reset()
{
  // As per Hinton, et. al. http://arxiv.org/abs/1207.0580:
  //   w ~ N(0, 0.01)
  std::generate(W.begin<NumericType>(), W.end<NumericType>(), RatioUniformGenerator(0, 0.01));
  
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
  enum { MaxRows = 4000, };
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
  typedef DualLayerNNSoftmax<784, 10, 800, NumericType> NNType;
  std::cout << "Network architecture [I > H > H > O] is ["
            << NNType::NumInputs << " -> " << NNType::NumHiddenUnits << " -> "
            << NNType::NumHiddenUnits << " -> " << NNType::NumClasses << "]" << std::endl;
  enum { CvType = NumericTypeToCvType<NumericType>::CvType, };
  std::cout << "CvType = " << CvType << std::endl;
  //         Load data.
  std::cout << "Loading data..." << std::endl;
  typedef std::pair<cv::Mat, cv::Mat> Dataset;
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
  assert(dataTrain.first.type() == dataTest.first.type());
  if (dataTrain.first.cols != dataTest.first.cols)
  {
    std::cerr << "Error: train/test input dims unmatched" << std::endl;
    errorCode |= ERROR_BAD_DATA_DIMS;
  }
  std::cout << "Loaded " << dataTest.first.rows << " testing data." << std::endl;
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
  double sampleLoss;
  int sampleError;
  cv::Mat dLdY(NNType::NumClasses, 1, CvType);
  const NumericType eta = static_cast<NumericType>(0.001);
  enum { DebugPrintEveryNSamples = 500, };
  const int dataTestSize = dataTest.first.rows;
  for (int i = 0; i < dataTestSize; ++i)
  {
    const cv::Mat xi = dataTest.first.row(i).t();
    const cv::Mat yi = dataTest.second.row(i);
    const cv::Mat* dLdW = NLLCriterion::SampleGradient(&nn0, xi, yi,
                                                       &dLdY, &sampleLoss, &sampleError);
    //nn0.W += -eta * *dLdW;
    cv::scaleAdd(*dLdW, -eta, nn0.W, nn0.W);
    if (0 == (i % DebugPrintEveryNSamples))
    {
      std::cout << "Processed sample " << i << " of " << dataTestSize << std::endl;
    }
  }
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
