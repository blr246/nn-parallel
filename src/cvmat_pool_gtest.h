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
#ifndef SRC_CVMAT_POOL_GTEST
#define SRC_CVMAT_POOL_GTEST
#include "cvmat_pool.h"
#include "cv.h"

#include "gtest/gtest.h"
#include <assert.h>
#include <algorithm>

namespace _cvmat_pool_gtest_
{
using namespace blr;

TEST(CvMatPool, ConcurrentAccess)
{
  enum { ForceTestThreads = 16, };
  const cv::Size testSize(1024, 1024);
  const int numEles = testSize.height * testSize.width;
  typedef std::vector<const void*> PointerList;
  std::vector<PointerList> pointersSeen(ForceTestThreads);
#pragma omp parallel for schedule(dynamic, 1) num_threads(ForceTestThreads)
  for (int i = 0; i < 1000; ++i)
  {
    const int whoAmiI = omp_get_thread_num();
    CvMatPtr mat = CreateCvMatPtr();
    mat->create(testSize, CV_32S);
    pointersSeen[whoAmiI].push_back(mat->data);
    *mat = i;
    const double sum = cv::sum(*mat).val[0];
    EXPECT_EQ(sum, i * numEles);
  }
  // Make sure that we pooled something.
  int totalPointersSeen = 0;
  int totalUniquePointersSeen = 0;
  for (int i = 0; i < ForceTestThreads; ++i)
  {
    PointerList& pointers = pointersSeen[i];
    const size_t pointersSize = pointers.size();
    totalPointersSeen += pointersSize;
    const PointerList::iterator uniqueEnd = std::unique(pointers.begin(), pointers.end());
    const size_t numUnique = std::distance(pointers.begin(), uniqueEnd);
    totalUniquePointersSeen += numUnique;
  }
  EXPECT_LT(totalUniquePointersSeen, totalPointersSeen);
}

}

#endif //SRC_CVMAT_POOL_GTEST
