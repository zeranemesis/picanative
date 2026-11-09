#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

// Reads a .pntrace (docs/TRACE.md), record after record.
namespace picanative::trace {

enum Kind : uint32_t {
  Memory = 1,
  Register = 2,
  Framebuffer = 3,
  CommandList = 4,
  MemoryFill = 5,
  DisplayTransfer = 6,
  TextureCopy = 7,
  Frame = 8,
  Screen = 9,
};

struct Record {
  uint32_t kind = 0;
  std::vector<uint8_t> payload;

  uint32_t word(size_t index) const {
    uint32_t value = 0;
    for (int b = 0; b < 4; ++b) value |= uint32_t{payload[index * 4 + b]} << (8 * b);
    return value;
  }
};

class Reader {
public:
  // false, with error() set, when the file is not a version 1 trace
  bool open(const std::string& path);
  // false at the end of the trace
  bool next(Record& record);
  const std::string& error() const { return error_; }

private:
  std::ifstream in_;
  std::string error_;
};

} // namespace picanative::trace
