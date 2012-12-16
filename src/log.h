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
