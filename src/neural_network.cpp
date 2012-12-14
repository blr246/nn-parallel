#include "layer.h"
#include "idx_cv.h"
#include "cv.h"

using namespace blr;

template <typename T> struct NumericTypeToCvType;
template <> struct NumericTypeToCvType<double> { enum { CvType = CV_64F, }; };
template <> struct NumericTypeToCvType<float> { enum { CvType = CV_32F, }; };

template <typename LayerType>
int PartitionOutputs(cv::Mat* Y, cv::Mat* yPart, int rStart);

template <typename LayerType>
int PartitionParameters(cv::Mat* Y, cv::Mat* yPart, int rStart);

template <typename LayerType>
int PartitionParameters(cv::Mat* Y, cv::Mat* yPart, int rStart);

template <typename NNType>
double NegativeLogLikelihoodLoss(const NNType& nn, const cv::Mat& X, const cv::Mat& Y);

template <typename LayerType>
inline
int PartitionOutputs(cv::Mat* Y, cv::Mat* yPart, int rStart)
{
  int rEnd = rStart + LayerType::NumOutputs;
  *yPart = Y->rowRange(rStart, rEnd);
  return rEnd;
}

template <typename LayerType>
inline
int PartitionParameters(cv::Mat* Y, cv::Mat* yPart, int rStart)
{
  int rEnd = rStart + LayerType::NumParameters;
  *yPart = Y->rowRange(rStart, rEnd);
  return rEnd;
}

template <typename LayerType>
inline
int PartitionInputs(cv::Mat* Y, cv::Mat* yPart, int rStart)
{
  int rEnd = rStart + LayerType::NumInputs;
  *yPart = Y->rowRange(rStart, rEnd);
  return rEnd;
}

template <typename NNType>
double NegativeLogLikelihoodLoss(const NNType& nn, const cv::Mat& X, const cv::Mat& Y)
{
  for (int i = 0; i < X.rows; ++i)
  {
    const cv::Mat xi = X[i];
    const cv::Mat yi = Y[i];
  }
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

  enum { TotalSublayers = 7, };

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

  void Forward(const cv::Mat& X, cv::Mat* Y) const;
  void Backward();

  // Flattened outputs for the entire network.
  cv::Mat Y;
  // Flattened parameter matrix for the entire network.
  cv::Mat W;
  // Flattened parameter gradient matrix for the entire network.
  cv::Mat dLdW;
  // Flattened gradient matrix for the entire network.
  cv::Mat dLdX;

  mutable cv::Mat yPartitions[TotalSublayers];
  cv::Mat wPartitions[TotalSublayers];
  cv::Mat dwPartitions[TotalSublayers];
  cv::Mat dxPartitions[TotalSublayers];
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
  // Partition Y.
  {
    int rStart = 0;
    rStart = PartitionOutputs<Layer0> (&Y, yPartitions + 0, rStart);
    rStart = PartitionOutputs<Layer1a>(&Y, yPartitions + 1, rStart);
    rStart = PartitionOutputs<Layer1b>(&Y, yPartitions + 2, rStart);
    rStart = PartitionOutputs<Layer2a>(&Y, yPartitions + 3, rStart);
    rStart = PartitionOutputs<Layer2b>(&Y, yPartitions + 4, rStart);
    rStart = PartitionOutputs<Layer3a>(&Y, yPartitions + 5, rStart);
    rStart = PartitionOutputs<Layer3b>(&Y, yPartitions + 6, rStart);
    assert(Y.rows == rStart);
  }
  // Partition W.
  {
    int rStart = 0;
    rStart = PartitionParameters<Layer0> (&W, wPartitions + 0, rStart);
    rStart = PartitionParameters<Layer1a>(&W, wPartitions + 1, rStart);
    rStart = PartitionParameters<Layer1b>(&W, wPartitions + 2, rStart);
    rStart = PartitionParameters<Layer2a>(&W, wPartitions + 3, rStart);
    rStart = PartitionParameters<Layer2b>(&W, wPartitions + 4, rStart);
    rStart = PartitionParameters<Layer3a>(&W, wPartitions + 5, rStart);
    rStart = PartitionParameters<Layer3b>(&W, wPartitions + 6, rStart);
    assert(W.rows == rStart);
  }
  // Partition dLdW.
  {
    int rStart = 0;
    rStart = PartitionParameters<Layer0> (&dLdW, dwPartitions + 0, rStart);
    rStart = PartitionParameters<Layer1a>(&dLdW, dwPartitions + 1, rStart);
    rStart = PartitionParameters<Layer1b>(&dLdW, dwPartitions + 2, rStart);
    rStart = PartitionParameters<Layer2a>(&dLdW, dwPartitions + 3, rStart);
    rStart = PartitionParameters<Layer2b>(&dLdW, dwPartitions + 4, rStart);
    rStart = PartitionParameters<Layer3a>(&dLdW, dwPartitions + 5, rStart);
    rStart = PartitionParameters<Layer3b>(&dLdW, dwPartitions + 6, rStart);
    assert(W.rows == rStart);
  }
  // Partition dLdX.
  {
    int rStart = 0;
    rStart = PartitionInputs<Layer0> (&dLdX, dxPartitions + 0, rStart);
    rStart = PartitionInputs<Layer1a>(&dLdX, dxPartitions + 1, rStart);
    rStart = PartitionInputs<Layer1b>(&dLdX, dxPartitions + 2, rStart);
    rStart = PartitionInputs<Layer2a>(&dLdX, dxPartitions + 3, rStart);
    rStart = PartitionInputs<Layer2b>(&dLdX, dxPartitions + 4, rStart);
    rStart = PartitionInputs<Layer3a>(&dLdX, dxPartitions + 5, rStart);
    rStart = PartitionInputs<Layer3b>(&dLdX, dxPartitions + 6, rStart);
    assert(dLdX.rows == rStart);
  }
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_, typename NumericType_>
inline
void DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_, NumericType_>
::Forward(const cv::Mat& X, cv::Mat* Y) const
{
  int layerIdx = 0;
  Layer0::Forward(X,  wPartitons[layerIdx], yPartitons[layerIdx]);
  ++layerIdx;
  Layer1a::Forward(X, wPartitons[layerIdx], yPartitons[layerIdx]);
  ++layerIdx;
  Layer1b::Forward(X, wPartitons[layerIdx], yPartitons[layerIdx]);
  ++layerIdx;
  Layer2a::Forward(X, wPartitons[layerIdx], yPartitons[layerIdx]);
  ++layerIdx;
  Layer2b::Forward(X, wPartitons[layerIdx], yPartitons[layerIdx]);
  ++layerIdx;
  Layer3a::Forward(X, wPartitons[layerIdx], yPartitons[layerIdx]);
  ++layerIdx;
  Layer3b::Forward(X, wPartitons[layerIdx], yPartitons[layerIdx]);
  if (Y) { *Y = yPartitons[layerIdx]; }
}

enum
{
  ERROR_BAD_TRAIN_DATA   = 1,
  ERROR_BAD_TRAIN_LABELS = 1 << 1,
};

int main(int /*argc*/, char** /*argv*/)
{
  typedef DualLayerNNSoftmax<784, 10, 200, double> NNType;
  // Load data.
  cv::Mat X, Y;
  int errorCode = 0;
  if (!IdxToCvMat("data/train-images.idx3-ubyte", &X))
  {
    errorCode |= ERROR_BAD_TRAIN_DATA;
  }
  if (!IdxToCvMat("data/train-labels.idx1-ubyte", &Y))
  {
    errorCode |= ERROR_BAD_TRAIN_LABELS;
  }
  if (errorCode)
  {
    return errorCode;
  }
  assert(NNType::NumInputs == X.cols);
  // Instantiate network and train.
  NNType nn0;
  return 0;
}
