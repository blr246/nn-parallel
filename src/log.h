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
#ifndef SRC_LOG_H
#define SRC_LOG_H
#include "timer.h"

#include "opencv/cv.h"
#include <sstream>
#include <iostream>
#include <iomanip>

namespace blr
{
namespace nn
{

enum { HexAddrLabelColW = 60, };

const Timer* LogTimer();

inline
void LogMatrix(const cv::Mat& m, const char* msg, std::ostream* stream)
{
  const double tNow = LogTimer()->GetTime();
  std::stringstream ssMsg;
  ssMsg << "[" << std::fixed << std::setprecision(6) << tNow << "] | "
        << std::setfill('.') << std::setw(HexAddrLabelColW)
        << msg << " " << std::hex << static_cast<void*>(m.data) << "\n";
  (*stream) << ssMsg.str();
  std::cout.flush();
}

inline
void LogMatrix(const cv::Mat& m, const std::string& msg, std::ostream* stream)
{
  LogMatrix(m, msg.c_str(), stream);
}

inline
void Log(const char* msg, std::ostream* stream)
{
  const double tNow = LogTimer()->GetTime();
  std::stringstream ssMsg;
  ssMsg << "[" << std::fixed << std::setprecision(6) << tNow << "] | " << msg;
  (*stream) << ssMsg.str();
  std::cout.flush();
}

inline
void Log(const std::string& msg, std::ostream* stream)
{
  Log(msg.c_str(), stream);
}

} // end ns nn
using namespace nn;
} // end ns blr

#endif //SRC_LOG_H
