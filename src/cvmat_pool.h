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
#ifndef SRC_CVMAT_POOL_H
#define SRC_CVMAT_POOL_H
#include "omp_lock.h"

#include "cv.h"
#include <vector>
#include <omp.h>

namespace blr
{
namespace nn
{

typedef cv::Ptr<cv::Mat> CvMatPtr;

class CvMatPool
{
public:
  class ScopedLock
  {
  public:
    explicit ScopedLock(CvMatPool* pool_);
    ~ScopedLock();

    std::vector<cv::Mat*>* GetItems() const;
    void Release() const;

  private:
    ScopedLock(const ScopedLock&);
    ScopedLock& operator=(const ScopedLock&);

    mutable CvMatPool* pool;
    OmpLock::ScopedLock lock;
  };

  CvMatPool();
  ~CvMatPool();

private:
  CvMatPool(const CvMatPool&);
  CvMatPool& operator=(const CvMatPool&);

  OmpLock lock;
  std::vector<cv::Mat*> myLittleMats;
};

namespace detail
{
struct CvMatPoolInitailizer
{
  CvMatPoolInitailizer();
};
extern CvMatPoolInitailizer g_cvMatPoolInitializer;
}

////////////////////////////////////////////////////////////////////////////////
// Inline definitions.
////////////////////////////////////////////////////////////////////////////////
inline
CvMatPool::ScopedLock::ScopedLock(CvMatPool* pool_)
: pool(pool_),
  lock(&pool->lock)
{}

inline
CvMatPool::ScopedLock::~ScopedLock()
{
  if (pool)
  {
    Release();
  }
}

inline
std::vector<cv::Mat*>* CvMatPool::ScopedLock::GetItems() const
{
  if (pool)
  {
    return &pool->myLittleMats;
  }
  else
  {
    return NULL;
  }
}

inline
void CvMatPool::ScopedLock::Release() const
{
  assert(lock.Acquired());
  lock.Unlock();
  pool = NULL;
}

inline
CvMatPool* GetCvMatPool()
{
  static CvMatPool g_cvMatPool;
  return &g_cvMatPool;
}

inline
CvMatPtr CreateCvMatPtr()
{
  CvMatPool::ScopedLock lock(GetCvMatPool());
  std::vector<cv::Mat*>* items = lock.GetItems();
  if (!items->empty())
  {
    CvMatPtr mat = items->back();
    items->pop_back();
    return mat;
  }
  else
  {
    lock.Release();
    return CvMatPtr(new cv::Mat());
  }
}

inline
void DestroyCvMat(cv::Mat* obj)
{
  CvMatPool::ScopedLock lock(GetCvMatPool());
  std::vector<cv::Mat*>* items = lock.GetItems();
  items->push_back(obj);
}

} // end ns nn
using namespace nn;
} // end ns blr

namespace cv
{
template<>
inline
void blr::CvMatPtr::delete_obj()
{
  blr::DestroyCvMat(obj);
}
}

#endif //SRC_CVMAT_POOL_H
