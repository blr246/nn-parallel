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

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_ = 0, int DropoutProbabilityHidden_ = 0,
          typename NumericType_ = double>
class DualLayerNNSoftmax
{
public:
  typedef NumericType_ NumericType;
  enum { NumInputs = NumInputs_, };
  enum { NumHiddenUnits = NumHiddenUnits_, };
  enum { NumClasses = NumClasses_, };
  enum { CvType = NumericTypeToCvType<NumericType>::CvType, };
  enum { DropoutProbabilityInput = DropoutProbabilityInput_, };
  enum { DropoutProbabilityHidden = DropoutProbabilityHidden_, };

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
  enum { NumDropoutLayers = 3, };
  enum { NumDropoutParameters = Layer0::NumOutputs +
                                Layer1b::NumOutputs +
                                Layer2b::NumOutputs, };

  const cv::Mat* Forward(const cv::Mat& X) const;
  const cv::Mat* Backward(const cv::Mat dLdY);
  void Reset();

  void EnableDropout();
  void DisableDropout();
  bool DropoutEnabled();
  void RefreshDropoutMask();

  void TruncateL2(const NumericType maxNorm);

  void SetW(const CvMatPtr& W);
  void GetW(cv::Mat** W);

private:
  void UpdatePartitions();

  /// <summary>Mask to perform dropout when enabled.</summary>
  cv::Mat dropoutMask;
  /// </summary>Flattened outputs for the entire network.</summary>
  cv::Mat Y;
  /// <summary>Flattened parameter matrix for the entire network.</summary>
  CvMatPtr WPtr;
  /// <summary>Flattened parameter gradient matrix for the entire network.</summary>
  cv::Mat dLdW;
  /// <summary>Flattened gradient matrix for the entire network.</summary>
  cv::Mat dLdX;

  mutable cv::Mat yPartitions[NumSublayers];
  cv::Mat wPartitions[NumSublayers];
  cv::Mat dwPartitions[NumSublayers];
  cv::Mat dxPartitions[NumSublayers];
  cv::Mat dropoutPartitions[NumDropoutLayers];

  bool dropoutEnabled;
};

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                   DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
::DualLayerNNSoftmax()
: dropoutMask(NumDropoutParameters, 1, CvType),
  Y(NumInternalOutputs, 1, CvType),
  WPtr(),
  dLdW(NumParameters, 1, CvType),
  dLdX(NumInternalInputs, 1, CvType),
  yPartitions(),
  wPartitions(),
  dwPartitions(),
  dxPartitions(),
  dropoutEnabled(false)
{
  SetW(CreateCvMatPtr());
  Reset();
  RefreshDropoutMask();
  dropoutPartitions[0] = dropoutMask.rowRange(0, Layer0::NumOutputs);
  dropoutPartitions[1] = dropoutMask.rowRange(0, Layer1b::NumOutputs);
  dropoutPartitions[2] = dropoutMask.rowRange(0, Layer2b::NumOutputs);
}

namespace detail
{
template <typename NumericType, int P>
struct ApplyDropout
{
  static void Apply(bool enabled, const cv::Mat& dropout, cv::Mat* m)
  {
    if (enabled)
    {
      (*m).mul(dropout);
    }
    else
    {
      const NumericType dropoutScale = static_cast<NumericType>(100.0 / P);
      (*m) *= dropoutScale;
    }
  }
};
template <typename NumericType>
struct ApplyDropout<NumericType, 0>
{
  static void Apply(bool enabled, const cv::Mat& dropout, cv::Mat* m) {}
};
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
const cv::Mat* DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                                  DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
::Forward(const cv::Mat& X) const
{
  using detail::ApplyDropout;

  int layerIdx = 0;
  Layer0::Forward(X, wPartitions[layerIdx], &yPartitions[layerIdx]);
  ApplyDropout<NumericType, DropoutProbabilityInput>::Apply(
      dropoutEnabled, dropoutPartitions[0], yPartitions + layerIdx);
  cv::Mat* yPrev = yPartitions; ++layerIdx;
  Layer1a::Forward(*yPrev, wPartitions[layerIdx], &yPartitions[layerIdx]);
  ++yPrev; ++layerIdx;
  Layer1b::Forward(*yPrev, wPartitions[layerIdx], &yPartitions[layerIdx]);
  ApplyDropout<NumericType, DropoutProbabilityHidden>::Apply(
      dropoutEnabled, dropoutPartitions[1], yPartitions + layerIdx);
  ++yPrev; ++layerIdx;
  Layer2a::Forward(*yPrev, wPartitions[layerIdx], &yPartitions[layerIdx]);
  ++yPrev; ++layerIdx;
  Layer2b::Forward(*yPrev, wPartitions[layerIdx], &yPartitions[layerIdx]);
  ApplyDropout<NumericType, DropoutProbabilityHidden>::Apply(
      dropoutEnabled, dropoutPartitions[2], yPartitions + layerIdx);
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

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
const cv::Mat* DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                                  DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
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

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
inline
void DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                        DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
::EnableDropout()
{
  dropoutEnabled = true;
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
inline
void DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                        DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
::DisableDropout()
{
  dropoutEnabled = false;
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
inline
bool DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                        DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
::DropoutEnabled()
{
  return dropoutEnabled;
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
inline
void DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                        DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
::RefreshDropoutMask()
{
  // Perform Bernoulli on each output with given probability.
  for (int i = 0; i < NumDropoutLayers; ++i)
  {
    const int dropoutProbability = (i < 1) ?
                                   static_cast<int>(DropoutProbabilityHidden) :
                                   static_cast<int>(DropoutProbabilityInput);
    cv::Mat& dropout = dropoutPartitions[i];
    const size_t numEles = dropout.rows;
    for (size_t j = 0; j < numEles; ++j)
    {
      dropout.at<NumericType>(j, 0) =
        static_cast<NumericType>((RandBound(100) < dropoutProbability) ? 1 : 0);
    }
  }
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
inline
void DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                        DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
::TruncateL2(const NumericType maxNorm)
{
  Layer0::TruncateL2(maxNorm, WPtr);
  Layer1a::TruncateL2(maxNorm, WPtr);
  Layer1b::TruncateL2(maxNorm, WPtr);
  Layer2a::TruncateL2(maxNorm, WPtr);
  Layer2b::TruncateL2(maxNorm, WPtr);
  Layer3a::TruncateL2(maxNorm, WPtr);
  Layer3b::TruncateL2(maxNorm, WPtr);
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
inline
void DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                        DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
::Reset()
{
  // As per Hinton, et. al. http://arxiv.org/abs/1207.0580:
  //   w ~ N(0, 0.01)
  std::generate(WPtr->begin<NumericType>(), WPtr->end<NumericType>(), RatioUniformGenerator(0, 0.01));
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
inline
void DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                        DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
::SetW(const CvMatPtr& W)
{
  WPtr = W;
  assert(NULL != WPtr);
  WPtr->create(NumParameters, 1, CvType);
  UpdatePartitions();
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
inline
void DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                        DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
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

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
inline
void DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                        DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
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

template <typename NNType>
struct ScopedDropoutEnabler
{
  ScopedDropoutEnabler(NNType* nn_)
    : nn(nn_),
      wasEnabled(nn->DropoutEnabled())
  {
    nn->EnableDropout();
  }
  ~ScopedDropoutEnabler()
  {
    if (!wasEnabled)
    {
      nn->DisableDropout();
    }
  }
  NNType* nn;
  bool wasEnabled;
};

template <typename NNType>
struct ScopedDropoutDisabler
{
  ScopedDropoutDisabler(NNType* nn_)
    : nn(nn_),
      wasEnabled(nn->DropoutEnabled())
  {
    nn->DisableDropout();
  }
  ~ScopedDropoutDisabler()
  {
    if (wasEnabled)
    {
      nn->EnableDropout();
    }
  }
  NNType* nn;
  bool wasEnabled;
};

} // end ns nn
using namespace nn;
} // end ns blr

#endif //SRC_NEURAL_NEWTORK_H
