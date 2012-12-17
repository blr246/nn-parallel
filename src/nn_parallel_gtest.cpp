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
#include "cvmat_pool_gtest.h"
#include "idx_cv_gtest.h"
#include "layer_gtest.h"
#include "neural_network_gtest.h"
#include "rand_bound_gtest.h"

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
