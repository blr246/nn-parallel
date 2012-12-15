#ifndef SRC_CVMAT_POOL_GTEST
#define SRC_CVMAT_POOL_GTEST
#include "cvmat_pool.h"
#include "cxtypes.h"

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
