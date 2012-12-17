#include "minibatch_trainer.h"
#include "neural_network.h"
#include "idx_cv.h"
#include "omp_lock.h"
#include "log.h"

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
: eps0(0.05),
  exp(0.998),
  rho0(0.5),
  rho1(0.99),
  epsMin(0.001),
  deltaWPrev(CreateCvMatPtr())
{}

WeightExponentialDecay::WeightExponentialDecay(
    double eps0_, double exp_, double rho0_, double rho1_, double epsMin_)
: eps0(eps0_),
  exp(exp_),
  rho0(rho0_),
  rho1(rho1_),
  epsMin(epsMin_),
  deltaWPrev(CreateCvMatPtr())
{}

const cv::Mat* WeightExponentialDecay::ComputeDeltaW(int t, const cv::Mat& dLdW)
{
//  // Just do a scaled update (exponential decay).
//  const double gradScale = -eps0 * std::pow(exp, t);
//  dLdW.copyTo(*deltaWPrev);
//  (*deltaWPrev) *= gradScale;

  std::stringstream ssMsg;

  const double rhoScale = t / static_cast<double>(T);
  const double rhoT = (t >= T) ? rho1 : (rhoScale * rho1) + ((1.0 - rhoScale) * rho0);
  const double epsT = std::max(eps0 * std::pow(exp, t), epsMin);
  const double gradScale = -(1 - rhoT) * epsT;
  ssMsg.str("");;
  ssMsg <<  std::dec
        << "t = " << t << ", rhoScale = " << rhoScale << ", rhoT = " << rhoT << ", "
           "epsT = " << epsT << ", gradScale = " << gradScale << std::endl;
  Log(ssMsg.str(), &std::cout);
  (*deltaWPrev) *= rhoT;
  cv::scaleAdd(dLdW, gradScale, *deltaWPrev, *deltaWPrev);

  DETECT_NUMERICAL_ERRORS(dLdW);
  DETECT_NUMERICAL_ERRORS(*deltaWPrev);
  return deltaWPrev;
}

UpdateDelegator::UpdateDelegator(double maxL2_)
: latestWPtr(CreateCvMatPtr()),
  dataTrain(NULL),
  dataTest(NULL),
  learningRate(),
  maxL2(maxL2_),
  t(0),
  updates(),
  busyLock(),
  latestWLock()
{}

UpdateDelegator::UpdateDelegator(const WeightExponentialDecay& learningRate_, double maxL2_)
: latestWPtr(CreateCvMatPtr()),
  dataTrain(NULL),
  dataTest(NULL),
  learningRate(learningRate_),
  maxL2(maxL2_),
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
