#ifndef SRC_CVMAT_POOL_H
#define SRC_CVMAT_POOL_H
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
  class Lock
  {
  public:
    Lock(CvMatPool* pool_)
      : pool(pool_)
    {
      omp_set_lock(&pool->lock);
    }
    ~Lock()
    {
      Release();
    }
    std::vector<CvMatPtr>* GetItems() const
    {
      return &pool->myLittleMats;
    }
    void Release()
    {
      if (pool)
      {
        omp_unset_lock(&pool->lock);
      }
      pool = NULL;
    }
  private:
    CvMatPool* pool;
  };

  CvMatPool()
    : lock(),
      myLittleMats()
  {
    omp_init_lock(&lock);
  }

  ~CvMatPool()
  {
    omp_destroy_lock(&lock);
  }

private:
  omp_lock_t lock;
  std::vector<CvMatPtr> myLittleMats;
};

CvMatPool* GetCvMatPool()
{
  static CvMatPool g_cvMatPool;
  return &g_cvMatPool;
}

CvMatPtr CreateCvMatPtr()
{
  CvMatPool::Lock lock(GetCvMatPool());
  std::vector<CvMatPtr>* items = lock.GetItems();
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
  CvMatPool::Lock lock(GetCvMatPool());
  std::vector<CvMatPtr>* items = lock.GetItems();
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
