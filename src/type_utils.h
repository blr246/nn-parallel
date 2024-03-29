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
#ifndef SRC_TYPE_UTILS_H
#define SRC_TYPE_UTILS_H
#include "opencv/cv.h"
#include <ostream>
#include <iomanip>

template <typename T>
struct MatTypeWrapper
{
public:
  explicit MatTypeWrapper(const cv::Mat& m_) : m(&m_) {}
  const cv::Mat* m;
};

template <typename T>
std::ostream& operator<<(std::ostream& stream, const MatTypeWrapper<T>& matTyped)
{
  const cv::Mat& mat = *matTyped.m;
  // Loop over elements dims-wise.
  const cv::MatConstIterator_<T> matEnd = mat.end<T>();
  cv::MatConstIterator_<T> v = mat.begin<T>();
  for (int r = 0; r < mat.rows; ++r)
  {
    stream << "[ ";
    for (int c = 0; c < mat.cols; ++c, ++v)
    {
      stream << std::scientific << std::setprecision(4) << *v << " ";
    }
    stream << "]\n";
  }
  return stream.flush();
}

inline
std::ostream& operator<<(std::ostream& stream, const cv::Size& size)
{
  return stream << "(" << size.height << "," << size.width << ")";
}

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
