//#include "cvmat_pool_gtest.h"
//#include "idx_cv_gtest.h"
//#include "layer_gtest.h"
#include "neural_network_gtest.h"
//#include "rand_bound_gtest.h"

#include "gtest/gtest.h"
#include <iostream>
#ifdef WIN32
#include <time.h>
#elif __APPLE__
#include <time.h>
#else
#include <sys/time.h>
#endif

int main(int argc, char** argv)
{
  const unsigned int randSeed = static_cast<unsigned int>(time(NULL));
  std::cout << "Random seed: " << randSeed << "." << std::endl;
  srand(randSeed);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
