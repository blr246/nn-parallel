/* Copyright (C) 2012 Brandon L. Reiss
   brandon@brandonreiss.com

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/
#ifndef SRC_IDX_CV_H
#define SRC_IDX_CV_H
#include "type_utils.h"
#include "opencv/cv.h"
#include <string>

namespace blr
{
namespace nn
{

/// <summary>Read matrix of points from an idx file.</summary>
template <typename NumericType>
bool IdxToCvMat(const std::string& path, cv::Mat* mat);

/// <summary>Read matrix of points from an idx file.</summary>
template <typename NumericType>
bool IdxToCvMat(const std::string& path, int maxRows, cv::Mat* mat);

/// <summary>Read matrix of points from an idx file.</summary>
bool IdxToCvMat(const std::string& path, int cvType, int maxRows, cv::Mat* mat);

/// <summary>Convert to zero-mean and unit variance.</summary>
void ZeroMeanUnitVar(cv::Mat* X, cv::Mat* mu, cv::Mat* stddev);

/// <summary>Convert to zero-mean and unit variance.</summary>
void ApplyZeroMeanUnitVarTform(const cv::Mat& mu, const cv::Mat& stddev, cv::Mat* X);

////////////////////////////////////////////////////////////////////////////////
// Inline definitions.
////////////////////////////////////////////////////////////////////////////////
template <typename NumericType>
inline
bool IdxToCvMat(const std::string& path, int maxRows, cv::Mat* mat)
{
  const int cvType = NumericTypeToCvType<NumericType>::CvType;
  return IdxToCvMat(path, cvType, maxRows, mat);
}

template <typename NumericType>
inline
bool IdxToCvMat(const std::string& path, cv::Mat* mat)
{
  const int cvType = NumericTypeToCvType<NumericType>::CvType;
  return IdxToCvMat(path, cvType, -1, mat);
}

namespace detail
{
template <typename NumericType>
void ZeroMeanUnitVarInternal(cv::Mat* X, cv::Mat* mu, cv::Mat* stddev, bool applyOnly)
{
  // Compute sample mean per channel (column).
  for (int i = 0; i < X->cols; ++i)
  {
    cv::Mat Xcol = X->col(i);
    if (!applyOnly)
    {
      cv::Scalar cMu = cv::mean(Xcol);
      mu->row(i) = cMu.val[0];
    }
    const double cMu = mu->at<NumericType>(i, 0);
    Xcol -= cMu;
  }
  // Compute stddev per channel (column).
  for (int i = 0; i < X->cols; ++i)
  {
    cv::Mat Xcol = X->col(i);
    if (!applyOnly)
    {
      cv::Scalar cMu, cStddev;
      cv::meanStdDev(Xcol, cMu, cStddev);
      stddev->row(i) = cStddev.val[0];
    }
    const double cStddev = stddev->at<NumericType>(i, 0);
    if (cStddev > 1e-6)
    {
      Xcol /= cStddev;
    }
  }
}
}

template <typename NumericType>
inline
void ZeroMeanUnitVar(cv::Mat* X, cv::Mat* mu, cv::Mat* stddev)
{
  using detail::ZeroMeanUnitVarInternal;
  mu->create(X->cols, 1, X->type());
  stddev->create(X->cols, 1, X->type());
  ZeroMeanUnitVarInternal<NumericType>(X, mu, stddev, false);
}

template <typename NumericType>
inline
void ApplyZeroMeanUnitVarTform(const cv::Mat& mu, const cv::Mat& stddev, cv::Mat* X)
{
  using detail::ZeroMeanUnitVarInternal;
  ZeroMeanUnitVarInternal<NumericType>(
      X, const_cast<cv::Mat*>(&mu), const_cast<cv::Mat*>(&stddev), true);
}

} // end ns nn
using namespace nn;
} // end ns blr

#endif //SRC_IDX_CV_H
