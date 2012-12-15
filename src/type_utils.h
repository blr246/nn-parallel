#ifndef SRC_TYPE_UTILS_H
#define SRC_TYPE_UTILS_H
#include "opencv/cxtypes.h"

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

} // end ns nn
using namespace nn;
} // end ns blr

#endif //SRC_TYPE_UTILS_H
