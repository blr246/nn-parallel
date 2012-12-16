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
    ScopedLock(CvMatPool* pool_);
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

CvMatPool
::ScopedLock::ScopedLock(CvMatPool* pool_)
: pool(pool_),
  lock(&pool->lock)
{}

CvMatPool
::ScopedLock::~ScopedLock()
{
  if (pool)
  {
    Release();
  }
}

std::vector<cv::Mat*>* CvMatPool
::ScopedLock::GetItems() const
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

void CvMatPool
::ScopedLock::Release() const
{
  assert(lock.Acquired());
  lock.Unlock();
  pool = NULL;
}

CvMatPool::CvMatPool()
: lock(),
  myLittleMats()
{}

CvMatPool::~CvMatPool()
{
  // These better be released.
  const size_t numMats = myLittleMats.size();
  for (size_t i = 0; i < numMats; ++i)
  {
    delete myLittleMats[i];
  }
}

CvMatPool* GetCvMatPool()
{
  static CvMatPool g_cvMatPool;
  return &g_cvMatPool;
}

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

void DestroyCvMat(cv::Mat* obj)
{
  CvMatPool::ScopedLock lock(GetCvMatPool());
  std::vector<cv::Mat*>* items = lock.GetItems();
  items->push_back(obj);
}

namespace detail
{
struct CvMatPoolInitailizer
{
  CvMatPoolInitailizer()
  {
    // Get things cooking.
    CreateCvMatPtr();
  }
};
CvMatPoolInitailizer g_cvMatPoolInitializer;
extern CvMatPoolInitailizer g_cvMatPoolInitializer;
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
