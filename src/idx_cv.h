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

// <summary> Read matrix of points from an idx file.</summary>
bool IdxToCvMat(const std::string& path, cv::Mat* mat);

} // end ns nn
using namespace nn;
} // end ns blr

#endif //SRC_IDX_CV_H
