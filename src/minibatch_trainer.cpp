#include "minibatch_trainer.h"
#include "neural_network.h"
#include "idx_cv.h"
#include "omp_lock.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include <utility>
#include <iostream>
#include <iomanip>
#include <omp.h>

namespace blr
{
namespace nn
{

WeightExponentialDecay::WeightExponentialDecay()
: eps0(1.0),
  exp(0.975),
  rho0(0.5),
  rho1(0.99),
  gradScaleMin(0.001),
  deltaWPrev(CreateCvMatPtr())
{}

const cv::Mat* WeightExponentialDecay::ComputeDeltaW(int t, const cv::Mat& dLdW)
{
//  // Just do a scaled update (exponential decay).
//  const double gradScale = -eps0 * std::pow(exp, t);
//  dLdW.copyTo(*deltaWPrev);
//  (*deltaWPrev) *= gradScale;

  const double rhoScale = t / static_cast<double>(T);
  const double rhoT = (t >= T) ? rho1 : (rhoScale * rho1) + ((1.0 - rhoScale) * rho0);
  const double epsT = eps0 * std::pow(exp, t);
  const double gradScale = -std::max((1 - rhoT) * epsT, gradScaleMin);
  std::stringstream ssMsg;
  ssMsg <<  std::dec
        << "t = " << t << ", rhoScale = " << rhoScale << ", rhoT = " << rhoT << ", "
           "epsT = " << epsT << ", gradScale = " << gradScale << std::endl;
  std::cout << ssMsg.str(); std::cout.flush();
  (*deltaWPrev) *= rhoT;
  cv::scaleAdd(dLdW, gradScale, *deltaWPrev, *deltaWPrev);

  DETECT_NUMERICAL_ERRORS(dLdW);
  DETECT_NUMERICAL_ERRORS(*deltaWPrev);
  return deltaWPrev;
}

UpdateDelegator::UpdateDelegator()
: latestWPtr(CreateCvMatPtr()),
  dataTrain(NULL),
  dataTest(NULL),
  learningRate(),
  t(0),
  updates(),
  busyLock(),
  latestWLock()
{}

UpdateDelegatorWrapper::UpdateDelegatorWrapper()
  : ud(NULL)
{}

UpdateDelegatorWrapper::UpdateDelegatorWrapper(UpdateDelegator* ud_)
  : ud(ud_)
{}

} // end ns nn
using namespace nn;
} // end ns blr
