#ifndef SRC_OMP_LOCK_H
#define SRC_OMP_LOCK_H
#include <omp.h>
#include <assert.h>

namespace blr
{
namespace sys
{

class OmpLock
{
public:
  class ScopedLock
  {
  public:
    explicit ScopedLock(OmpLock* lock_, bool tryAcquire = false);
    ~ScopedLock();

    void Unlock() const;
    void Lock() const;
    bool Acquired() const;

  private:
    ScopedLock(const ScopedLock&);
    ScopedLock& operator=(const ScopedLock&);

    mutable bool acquired;
    OmpLock* lock;
  };

  OmpLock();
  ~OmpLock();

private:
  OmpLock(const OmpLock& lock);
  OmpLock& operator=(const OmpLock& lock);

  omp_lock_t lock;
};

////////////////////////////////////////////////////////////////////////////////
// Inline definitions.
////////////////////////////////////////////////////////////////////////////////
inline
OmpLock::OmpLock()
{
  omp_init_lock(&lock);
}

inline
OmpLock::~OmpLock()
{
  omp_destroy_lock(&lock);
}

inline
OmpLock::ScopedLock::ScopedLock(OmpLock* lock_, bool tryAcquire)
: acquired(true),
  lock(lock_)
{
  if (tryAcquire)
  {
    acquired = omp_test_lock(&lock->lock) ? true : false;
  }
  else
  {
    omp_set_lock(&lock->lock);
  }
}

inline
OmpLock::ScopedLock::~ScopedLock()
{
  if (acquired)
  {
    omp_unset_lock(&lock->lock);
  }
}

inline
void OmpLock::ScopedLock::Unlock() const
{
  assert(acquired);
  omp_unset_lock(&lock->lock);
  acquired = false;
}

inline
void OmpLock::ScopedLock::Lock() const
{
  assert(!acquired);
  omp_set_lock(&lock->lock);
  acquired = true;
}

inline
bool OmpLock::ScopedLock::Acquired() const
{
  return acquired;
}

} // end ns sys
using namespace sys;
} // end ns blr

#endif //SRC_OMP_LOCK_H
