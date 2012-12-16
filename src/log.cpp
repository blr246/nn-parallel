#include "log.h"

namespace blr
{
namespace nn
{

const Timer* LogTimer()
{
  static Timer s_timer;
  return &s_timer;
}

}
}
