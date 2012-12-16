#ifndef SRC_IDX_CV_H
#define SRC_IDX_CV_H
#include "type_utils.h"
#include "opencv/cv.h"
#include <string>

namespace blr
{
namespace nn
{

// <summary> Read matrix of points from an idx file.</summary>
template <typename NumericType>
bool IdxToCvMat(const std::string& path, cv::Mat* mat);

// <summary> Read matrix of points from an idx file.</summary>
template <typename NumericType>
bool IdxToCvMat(const std::string& path, int maxRows, cv::Mat* mat);

// <summary> Read matrix of points from an idx file.</summary>
bool IdxToCvMat(const std::string& path, int cvType, int maxRows, cv::Mat* mat);

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

} // end ns nn
using namespace nn;
} // end ns blr

#endif //SRC_IDX_CV_H
