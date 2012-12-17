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
#ifndef SRC_TIMER_H_
#define SRC_TIMER_H_

#ifdef WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <time.h>
#include <windows.h>
#else
#include <sys/time.h>
#include <cstddef>
#endif

namespace blr
{
namespace util
{

/// <summary> A timer class giving double seconds. </summary>
class Timer
{
#ifdef WIN32
public:
  Timer()
    : m_timeStart(0ULL),
      m_freq(0ULL),
      m_freqFrac(0.0)
  {
    // Capture start time.
    QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&m_freq));
    m_freqFrac = static_cast<double>(m_freq);
    QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&m_timeStart));
  }

  inline double GetTime() const
  {
    unsigned long long timeNow;
    QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&timeNow));
    const unsigned long long totalTime = timeNow - m_timeStart;
    const unsigned long long timeWholeS = totalTime / m_freq;
    const double timeFracS = static_cast<double>(totalTime % m_freq);
    return static_cast<double>(timeWholeS) + (timeFracS / m_freqFrac);
  }

  inline void Reset()
  {
    QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&m_timeStart));
  }

private:
  unsigned long long m_timeStart;
  unsigned long long m_freq;
  double m_freqFrac;
#else
public:
  Timer()
    : m_timeStart()
  {
    gettimeofday(&m_timeStart, NULL);
  }

  inline double GetTime() const
  {
    timeval timeEnd;
    gettimeofday(&timeEnd, NULL);
    time_t timeWholeS = timeEnd.tv_sec - m_timeStart.tv_sec;
    double timeMicroS = static_cast<double>(timeEnd.tv_usec -
                                            m_timeStart.tv_usec);
    return static_cast<double>(timeWholeS) + (timeMicroS * 1.0e-6);
  }

  inline void Reset()
  {
    gettimeofday(&m_timeStart, NULL);
  }

private:
  timeval m_timeStart;
#endif
};

}
using namespace util;
}

#endif //SRC_TIMER_H_
