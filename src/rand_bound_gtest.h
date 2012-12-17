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
#ifndef SRC_RAND_BOUND_GTEST_H
#define SRC_RAND_BOUND_GTEST_H
#include "rand_bound.h"
#include "gtest/gtest.h"

namespace _src_rand_bound_gtest_h_
{
using namespace blr;

//TEST(RandBound, RatioOfUniforms)
//{
//  // reissb -- 20111017 -- This is a visual GTest. Inspect the output
//  //   histogram to verify that it has the proper normal distribution shape.
//  enum { Mean = 20, };
//  enum { Var = 5, };
//  enum { HistVar = Var * 3, };
//  std::vector<int> hist(2 * (HistVar) + 1, 0);
//  for (int i = 0; i < 300; ++i)
//  {
//    const double val = RatioOfUniforms(20, 4);
//    const int bin = static_cast<int>((val - Mean)) + HistVar;
//    const int binClamp = std::min(std::max(0, bin), static_cast<int>(hist.size()) - 1);
//    ++hist[binClamp];
//  }
//  for (int i = Mean - HistVar, binIdx = 0; i <= Mean + HistVar; ++i, ++binIdx)
//  {
//    std::cout << std::setw(2) << std::setfill(' ') << std::right << i << ":"
//              << std::setw(hist[binIdx]) << std::setfill('x') << " " << std::endl;
//  }
//}

}

#endif //SRC_RAND_BOUND_GTEST_H
