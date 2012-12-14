#include "idx_cv.h"
#include <fstream>

namespace blr
{
namespace nn
{

bool IdxToCvMat(const std::string& path, int cvType, int maxRows, cv::Mat* mat)
{
  std::ifstream data(path.c_str(), std::ios::in | std::ios::binary);
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
  int cvTypeFile = 0;
  int bytesPerDatum = 0;
  switch(type)
  {
  case 0x08: bytesPerDatum = 1; cvTypeFile = CV_8U; break;
  case 0x09: bytesPerDatum = 1; cvTypeFile = CV_8S; break;
  case 0x0B: bytesPerDatum = 2; cvTypeFile = CV_16S; break;
  case 0x0C: bytesPerDatum = 4; cvTypeFile = CV_32S; break; 
  case 0x0D: bytesPerDatum = 4; cvTypeFile = CV_32F; break; 
  case 0x0E: bytesPerDatum = 8; cvTypeFile = CV_64F; break;
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
  // Truncate rows?
  if (maxRows > 0)
  {
    sizes[0] = std::min(sizes[0], maxRows);
  }
  // Allocate the matrix.
  mat->create(sizes[0], sizes[1] * sizes[2], cvType);
  const int totalElements = sizes[0] * sizes[1] * sizes[2];
  if (cvTypeFile == cvType)
  {
    // Read the data directly (when types match).
    const int totalBytes = bytesPerDatum * totalElements;
    assert(std::distance(mat->datastart, mat->dataend) == totalBytes);
    if (!data.good()) { return false; }
    data.read(reinterpret_cast<char*>(mat->data), totalBytes);
  }
  else
  {
    // Create a temp buffer for conversion.
    enum { TempBufferElements = 2048, };
    cv::Mat bufferMat(TempBufferElements, 1, cvTypeFile);
    // View the target as a contiguous array.
    cv::Mat matFlat = mat->reshape(1, totalElements);
    // Read all elements.
    int elementsRead = 0;
    while ((elementsRead < totalElements) && data.good())
    {
      const int elementsToRead = totalElements - elementsRead;
      const int tempReadSize = std::min(elementsToRead, static_cast<int>(TempBufferElements));
      const int tempReadBytes = bytesPerDatum * tempReadSize;
      bufferMat.create(tempReadSize, 1, cvTypeFile);
      data.read(reinterpret_cast<char*>(bufferMat.data), tempReadBytes);
      const int elementsReadEnd = elementsRead + tempReadSize;
      cv::Mat readDst = matFlat.rowRange(elementsRead, elementsReadEnd);
      bufferMat.convertTo(readDst, cvType);
      elementsRead = elementsReadEnd;
    }
  }
  return !data.bad();
}

} // end ns nn
} // end ns blr
