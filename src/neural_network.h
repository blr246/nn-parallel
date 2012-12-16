#ifndef SRC_NEURAL_NEWTORK_H
#define SRC_NEURAL_NEWTORK_H
#include "layer.h"
#include "log.h"
#include "rand_bound.h"
#include "cvmat_pool.h"
#include "type_utils.h"

#include "opencv/cv.h"
#include <algorithm>
#include <functional>
#include <utility>
#include <cstring>

namespace blr
{
namespace nn
{

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
  DualLayerNNSoftmax(const DualLayerNNSoftmax& rhs);

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

  enum {
    SL_Layer0, 
    SL_Layer1a, 
    SL_Layer1b, 
    SL_Layer2a, 
    SL_Layer2b, 
    SL_Layer3a, 
    SL_Layer3b, 
    NumSublayers,
  };

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
                                Layer1a::NumOutputs +
                                Layer2a::NumOutputs, };

  const cv::Mat* Forward(const cv::Mat& X) const;
  const cv::Mat* Backward(const cv::Mat dLdY);
  void Reset();

  void EnableDropout();
  void DisableDropout();
  bool DropoutEnabled() const;
  void RefreshDropoutMask();

  void TruncateL2(const NumericType maxNorm);

  void SetWPtr(CvMatPtr newWPtr);
  CvMatPtr GetWPtr() const;

private:
  DualLayerNNSoftmax& operator=(const DualLayerNNSoftmax&);

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

#if !defined(NDEBUG)
  enum { CalledForward, CalledBackward, };
  mutable int lastCallState;

  enum { NumPersistentStorage = 4, };
  unsigned char* dataPtrs[NumPersistentStorage];
  unsigned char* dataPartitonPtrs[NumPersistentStorage * NumSublayers];

  void AssertDataPointersValid() const;
  void CollectDataPointers(unsigned char* main[], unsigned char* partitions[]) const;
#endif

  mutable cv::Mat yPartitions[NumSublayers];
  cv::Mat wPartitions[NumSublayers];
  cv::Mat dwPartitions[NumSublayers];
  cv::Mat dxPartitions[NumSublayers];
  cv::Mat dropoutPartitions[NumDropoutLayers];

  bool dropoutEnabled;
};

template <typename NNType>
struct ScopedDropoutEnabler
{
  ScopedDropoutEnabler(NNType* nn_);
  ~ScopedDropoutEnabler();
  NNType* nn;
  bool wasEnabled;
};

template <typename NNType>
struct ScopedDropoutDisabler
{
  ScopedDropoutDisabler(NNType* nn_);
  ~ScopedDropoutDisabler();
  NNType* nn;
  bool wasEnabled;
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Inline definitions.
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                   DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
::DualLayerNNSoftmax()
: dropoutMask(NumDropoutParameters, 1, CvType),
  Y(NumInternalOutputs, 1, CvType),
  WPtr(CreateCvMatPtr()),
  dLdW(NumParameters, 1, CvType),
  dLdX(NumInternalInputs, 1, CvType),
#if !defined(NDEBUG)
  lastCallState(CalledBackward),
  dataPtrs(),
  dataPartitonPtrs(),
#endif
  yPartitions(),
  wPartitions(),
  dwPartitions(),
  dxPartitions(),
  dropoutEnabled(false)
{
  WPtr->create(NumParameters, 1, CvType);
  Reset();
  UpdatePartitions();
#if !defined(NDEBUG)
  CollectDataPointers(dataPtrs, dataPartitonPtrs);
#endif
  SetWPtr(WPtr);
  RefreshDropoutMask();
  DETECT_NUMERICAL_ERRORS(*WPtr);
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                   DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
::DualLayerNNSoftmax(const DualLayerNNSoftmax& rhs)
: dropoutMask(),
  Y(),
  WPtr(CreateCvMatPtr()),
  dLdW(),
  dLdX(),
#if !defined(NDEBUG)
  lastCallState(rhs.lastCallState),
  dataPtrs(),
  dataPartitonPtrs(),
#endif
  yPartitions(),
  wPartitions(),
  dwPartitions(),
  dxPartitions(),
  dropoutEnabled(rhs.dropoutEnabled)
{
  // Copy over.
  rhs.dropoutMask.copyTo(dropoutMask);
  rhs.Y.copyTo(Y);
  rhs.WPtr->copyTo(*WPtr);
  rhs.dLdW.copyTo(dLdW);
  rhs.dLdX.copyTo(dLdX);
  UpdatePartitions();
#if !defined(NDEBUG)
  CollectDataPointers(dataPtrs, dataPartitonPtrs);
#endif
  SetWPtr(WPtr);
  DETECT_NUMERICAL_ERRORS(*WPtr);
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
      const NumericType dropoutScale = static_cast<NumericType>(1.0 - (P / 100.0));
      (*m) *= dropoutScale;
    }
  }
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
  ApplyDropout<NumericType, DropoutProbabilityHidden>::Apply(
      dropoutEnabled, dropoutPartitions[1], yPartitions + layerIdx);
  ++yPrev; ++layerIdx;
  Layer1b::Forward(*yPrev, wPartitions[layerIdx], &yPartitions[layerIdx]);
  ApplyDropout<NumericType, DropoutProbabilityHidden>::Apply(
      dropoutEnabled, dropoutPartitions[1], yPartitions + layerIdx);
  ++yPrev; ++layerIdx;
  Layer2a::Forward(*yPrev, wPartitions[layerIdx], &yPartitions[layerIdx]);
  ApplyDropout<NumericType, DropoutProbabilityHidden>::Apply(
      dropoutEnabled, dropoutPartitions[2], yPartitions + layerIdx);
  ++yPrev; ++layerIdx;
  Layer2b::Forward(*yPrev, wPartitions[layerIdx], &yPartitions[layerIdx]);
  ApplyDropout<NumericType, DropoutProbabilityHidden>::Apply(
      dropoutEnabled, dropoutPartitions[2], yPartitions + layerIdx);
  ++yPrev; ++layerIdx;
  Layer3a::Forward(*yPrev, wPartitions[layerIdx], &yPartitions[layerIdx]);
  ++yPrev; ++layerIdx;
  Layer3b::Forward(*yPrev, wPartitions[layerIdx], &yPartitions[layerIdx]);
  ++yPrev; ++layerIdx;

#if !defined(NDEBUG)
  AssertDataPointersValid();
  lastCallState = CalledForward;
#endif

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

#if !defined(NDEBUG)
  assert(CalledForward == lastCallState);
  lastCallState = CalledBackward;
  AssertDataPointersValid();
#endif

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
::DropoutEnabled() const
{
  return dropoutEnabled;
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
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
#if !defined(NDEBUG)
  AssertDataPointersValid();
#endif
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
inline
void DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                        DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
::TruncateL2(const NumericType maxNorm)
{
  int layerIdx = 0;
  double normFactor = 1.0;
  normFactor = std::min(Layer0::ComputeTruncateL2Factor(wPartitions[layerIdx], maxNorm), normFactor);
  ++layerIdx;
  normFactor = std::min(Layer1a::ComputeTruncateL2Factor(wPartitions[layerIdx], maxNorm), normFactor);
  ++layerIdx;                                                                          
  normFactor = std::min(Layer1b::ComputeTruncateL2Factor(wPartitions[layerIdx], maxNorm), normFactor);
  ++layerIdx;                                                                          
  normFactor = std::min(Layer2a::ComputeTruncateL2Factor(wPartitions[layerIdx], maxNorm), normFactor);
  ++layerIdx;                                                                          
  normFactor = std::min(Layer2b::ComputeTruncateL2Factor(wPartitions[layerIdx], maxNorm), normFactor);
  ++layerIdx;                                                                          
  normFactor = std::min(Layer3a::ComputeTruncateL2Factor(wPartitions[layerIdx], maxNorm), normFactor);
  ++layerIdx;                                                                          
  normFactor = std::min(Layer3b::ComputeTruncateL2Factor(wPartitions[layerIdx], maxNorm), normFactor);
  ++layerIdx;
  assert(normFactor <= 1);
  std::stringstream ssMsg;
  ssMsg << "Scaling W by " << normFactor << "\n";
  Log(ssMsg.str(), &std::cout);
  if (normFactor < 1)
  {
    (*WPtr) *= normFactor;
  }

  DETECT_NUMERICAL_ERRORS(*WPtr);
#if !defined(NDEBUG)
  AssertDataPointersValid();
#endif
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
  cv::randn(*WPtr, cv::Scalar::all(0), cv::Scalar::all(0.01));
  std::cout << "*****************RESET************" << std::endl;
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
inline
void DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                        DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
::SetWPtr(CvMatPtr newWPtr)
{
  assert(NULL != WPtr && NumParameters == WPtr->rows);
  WPtr = newWPtr;
  UpdatePartitions();
  const cv::Mat* W = WPtr; (void)W;
  DETECT_NUMERICAL_ERRORS(*W);
#if !defined(NDEBUG)
  AssertDataPointersValid();
#endif
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
inline
CvMatPtr DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                        DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
::GetWPtr() const
{
  return WPtr;
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
  int i = 0;
  partitionIter.Next<Layer0> (wPartitions + i, yPartitions + i, dwPartitions + i, dxPartitions + i);
  ++i;
  partitionIter.Next<Layer1a>(wPartitions + i, yPartitions + i, dwPartitions + i, dxPartitions + i);
  ++i;
  partitionIter.Next<Layer1b>(wPartitions + i, yPartitions + i, dwPartitions + i, dxPartitions + i);
  ++i;
  partitionIter.Next<Layer2a>(wPartitions + i, yPartitions + i, dwPartitions + i, dxPartitions + i);
  ++i;
  partitionIter.Next<Layer2b>(wPartitions + i, yPartitions + i, dwPartitions + i, dxPartitions + i);
  ++i;
  partitionIter.Next<Layer3a>(wPartitions + i, yPartitions + i, dwPartitions + i, dxPartitions + i);
  ++i;
  partitionIter.Next<Layer3b>(wPartitions + i, yPartitions + i, dwPartitions + i, dxPartitions + i);
  ++i;

  dropoutPartitions[0] = dropoutMask.rowRange(0, Layer0::NumOutputs);
  dropoutPartitions[1] = dropoutMask.rowRange(0, Layer1a::NumOutputs);
  dropoutPartitions[2] = dropoutMask.rowRange(0, Layer2a::NumOutputs);

  assert(Y.rows == partitionIter.yIdx);
  assert(WPtr->rows == partitionIter.wIdx);
  assert(dLdW.rows == partitionIter.dwIdx);
  assert(dLdX.rows == partitionIter.dxIdx);
}

#if !defined(NDEBUG)
template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
inline
void DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                        DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
::AssertDataPointersValid() const
{
  unsigned char* nowDataPtrs[NumPersistentStorage];
  unsigned char* nowDataPartitonPtrs[NumPersistentStorage * NumSublayers];
  CollectDataPointers(nowDataPtrs, nowDataPartitonPtrs);
  assert(0 == std::memcmp(nowDataPtrs, dataPtrs, sizeof(nowDataPtrs)) &&
         0 == std::memcmp(nowDataPartitonPtrs, dataPartitonPtrs, sizeof(nowDataPartitonPtrs)));
}

template <int NumInputs_, int NumClasses_, int NumHiddenUnits_,
          int DropoutProbabilityInput_, int DropoutProbabilityHidden_, typename NumericType_>
inline
void DualLayerNNSoftmax<NumInputs_, NumClasses_, NumHiddenUnits_,
                        DropoutProbabilityInput_, DropoutProbabilityHidden_, NumericType_>
::CollectDataPointers(unsigned char* main[], unsigned char* partitions[]) const
{
  main[0] = dropoutMask.data;
  main[1] = Y.data;
  main[2] = dLdW.data;
  main[3] = dLdX.data;
  for (int i = 0; i < NumDropoutLayers; ++i)
  {
    partitions[0*NumSublayers + i] = dropoutPartitions[i].data;
  }
  for (int i = NumDropoutLayers; i < NumSublayers; ++i)
  {
    partitions[0*NumSublayers + i] = 0;
  }
  for (int i = 0; i < NumSublayers; ++i)
  {
    partitions[1*NumSublayers + i] = yPartitions[i].data;
    partitions[2*NumSublayers + i] = dwPartitions[i].data;
    partitions[3*NumSublayers + i] = dxPartitions[i].data;
  }
}
#endif

template <typename NNType>
ScopedDropoutEnabler<NNType>::ScopedDropoutEnabler(NNType* nn_)
: nn(nn_),
  wasEnabled(nn->DropoutEnabled())
{
  nn->EnableDropout();
}

template <typename NNType>
ScopedDropoutEnabler<NNType>::~ScopedDropoutEnabler()
{
  if (!wasEnabled)
  {
    nn->DisableDropout();
  }
}

template <typename NNType>
ScopedDropoutDisabler<NNType>::ScopedDropoutDisabler(NNType* nn_)
: nn(nn_),
  wasEnabled(nn->DropoutEnabled())
{
  nn->DisableDropout();
}

template <typename NNType>
ScopedDropoutDisabler<NNType>::~ScopedDropoutDisabler()
{
  if (wasEnabled)
  {
    nn->EnableDropout();
  }
}

} // end ns nn
using namespace nn;
} // end ns blr

#endif //SRC_NEURAL_NEWTORK_H
