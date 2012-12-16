#include "cvmat_pool.h"

namespace blr
{
namespace nn
{

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

namespace detail
{
CvMatPoolInitailizer::CvMatPoolInitailizer()
{
  // Get things cooking.
  CreateCvMatPtr();
}
CvMatPoolInitailizer g_cvMatPoolInitializer;
}

} // end ns nn
using namespace nn;
} // end ns blr
