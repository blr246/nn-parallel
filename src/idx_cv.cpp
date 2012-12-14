#include "idx_cv.h"
#include <fstream>

namespace blr
{
namespace nn
{

bool IdxToCvMat(const std::string& path, cv::Mat* mat)
{
  std::ifstream data(path.c_str());
  union
  {
    int dataInt;
    char dataChar[sizeof(int)];
  };
  // First 4 bytes are the magic number.
  if (!data.good()) { return false; }
  data.read(dataChar, sizeof(dataChar));
  // Check 0x00XX is magic number.
  const bool magicValid = (dataChar[0] + dataChar[1]) == 0;
  if (!magicValid) { return false; }
  // Read type code and dims. This is where the fun begins.
  const unsigned char type = dataChar[2];
  int cvType = 0;
  int bytesPerDatum = 0;
  switch(type)
  {
  case 0x08: bytesPerDatum = 1; cvType = CV_8U; break;
  case 0x09: bytesPerDatum = 1; cvType = CV_8S; break;
  case 0x0B: bytesPerDatum = 2; cvType = CV_16S; break;
  case 0x0C: bytesPerDatum = 4; cvType = CV_32S; break; 
  case 0x0D: bytesPerDatum = 4; cvType = CV_32F; break; 
  case 0x0E: bytesPerDatum = 8; cvType = CV_64F; break;
  default: return false;
  }
  // Read dims sizes.
  const unsigned char dims = dataChar[3];
  assert(0 < dims && dims < 4);
  int sizes[3] = {1, 1, 1};
  for (int d = 0; d < dims; ++d)
  {
    if (!data.good()) { return false; }
    data.read(dataChar, sizeof(dataChar));
    // Endian swap big -> little.
    std::swap(dataChar[0], dataChar[3]);
    std::swap(dataChar[1], dataChar[2]);
    sizes[d] = dataInt;
  }
  // Allocate the matrix.
  mat->create(sizes[0], sizes[1] * sizes[2], cvType);
  // Read the data.
  const int totalBytes = bytesPerDatum * sizes[0] * sizes[1] * sizes[2];
  assert(std::distance(mat->datastart, mat->dataend) == totalBytes);
  data.read(reinterpret_cast<char*>(mat->data), totalBytes);
  return !data.bad();
}

} // end ns nn
} // end ns blr
