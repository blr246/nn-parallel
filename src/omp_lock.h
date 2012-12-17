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
