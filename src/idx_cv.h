#ifndef SRC_IDX_CV_H
#define SRC_IDX_CV_H
#include "opencv/cv.h"
#include "opencv/cv.hpp"
#include "opencv/cxmat.hpp"
#include <string>

namespace blr
{
namespace nn
{

template <typename T> struct NumericTypeToCvType;
template <> struct NumericTypeToCvType<double> { enum { CvType = CV_64F, }; };
template <> struct NumericTypeToCvType<float> { enum { CvType = CV_32F, }; };
template <> struct NumericTypeToCvType<char> { enum { CvType = CV_8S, }; };
template <> struct NumericTypeToCvType<unsigned char> { enum { CvType = CV_8U, }; };
template <> struct NumericTypeToCvType<short> { enum { CvType = CV_16S, }; };
template <> struct NumericTypeToCvType<unsigned short> { enum { CvType = CV_16U, }; };

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
