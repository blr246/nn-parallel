#ifndef SRC_NEURAL_NEWTORK_H
#include "layer.h"
#include "rand_bound.h"
#include "cvmat_pool.h"

#include "opencv/cv.h"
#include "opencv/cv.hpp"
#include <algorithm>
#include <functional>
#include <utility>

namespace blr
{
namespace nn
{

typedef std::pair<cv::Mat, cv::Mat> Dataset;

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_, typename NumericType_ = double>
class DualLayerNNSoftmax
{
public:
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

  void SetW(CvMatPtr W);
  void GetW(cv::Mat** W);

private:
  void UpdatePartitions();

  // Flattened outputs for the entire network.
  cv::Mat Y;
  // Flattened parameter matrix for the entire network.
  CvMatPtr WPtr;
  // Flattened parameter gradient matrix for the entire network.
  cv::Mat dLdW;
  // Flattened gradient matrix for the entire network.
  cv::Mat dLdX;

  mutable cv::Mat yPartitions[NumSublayers];
  cv::Mat wPartitions[NumSublayers];
  cv::Mat dwPartitions[NumSublayers];
  cv::Mat dxPartitions[NumSublayers];
};

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_, typename NumericType_>
DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_, NumericType_>
::DualLayerNNSoftmax()
: Y(NumInternalOutputs, 1, CvType),
  WPtr(),
  dLdW(NumParameters, 1, CvType),
  dLdX(NumInternalInputs, 1, CvType),
  yPartitions(),
  wPartitions(),
  dwPartitions(),
  dxPartitions()
{
  SetW(CreateCvMatPtr());
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
  std::generate(WPtr->begin<NumericType>(), WPtr->end<NumericType>(), RatioUniformGenerator(0, 0.01));
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_, typename NumericType_>
inline
void DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_, NumericType_>
::SetW(CvMatPtr W)
{
  WPtr = W;
  assert(NULL != WPtr);
  WPtr->create(NumParameters, 1, CvType);
  UpdatePartitions();
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_, typename NumericType_>
inline
void DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_, NumericType_>
::GetW(cv::Mat** W)
{
  *W = WPtr;
}

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
inline
void DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_, NumericType_>
::UpdatePartitions()
{
  ParameterPartitionerIterator partitionIter(WPtr, &Y, &dLdW, &dLdX);
  partitionIter.Next<Layer0> (wPartitions + 0, yPartitions + 0, dwPartitions + 0, dxPartitions + 0);
  partitionIter.Next<Layer1a>(wPartitions + 1, yPartitions + 1, dwPartitions + 1, dxPartitions + 1);
  partitionIter.Next<Layer1b>(wPartitions + 2, yPartitions + 2, dwPartitions + 2, dxPartitions + 2);
  partitionIter.Next<Layer2a>(wPartitions + 3, yPartitions + 3, dwPartitions + 3, dxPartitions + 3);
  partitionIter.Next<Layer2b>(wPartitions + 4, yPartitions + 4, dwPartitions + 4, dxPartitions + 4);
  partitionIter.Next<Layer3a>(wPartitions + 5, yPartitions + 5, dwPartitions + 5, dxPartitions + 5);
  partitionIter.Next<Layer3b>(wPartitions + 6, yPartitions + 6, dwPartitions + 6, dxPartitions + 6);
  assert(Y.rows == partitionIter.yIdx);
  assert(WPtr->rows == partitionIter.wIdx);
  assert(dLdW.rows == partitionIter.dwIdx);
  assert(dLdX.rows == partitionIter.dxIdx);
}

} // end ns nn
using namespace nn;
} // end ns blr

#endif //SRC_NEURAL_NEWTORK_H
