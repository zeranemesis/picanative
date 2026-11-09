#include "picanative/trace.hpp"

#include <cstring>

namespace picanative::trace {

bool Reader::open(const std::string& path) {
  in_.open(path, std::ios::binary);
  char header[16];
  if (!in_.read(header, sizeof header)) {
    error_ = "cannot read " + path;
    return false;
  }
  uint32_t version;
  std::memcpy(&version, header + 8, 4);
  if (std::memcmp(header, "PNTRACE\0", 8) != 0 || version != 1) {
    error_ = path + " is not a version 1 .pntrace";
    return false;
  }
  return true;
}

bool Reader::next(Record& record) {
  uint32_t head[2];
  if (!in_.read(reinterpret_cast<char*>(head), sizeof head)) return false;
  record.kind = head[0];
  record.payload.resize(head[1]);
  return static_cast<bool>(in_.read(reinterpret_cast<char*>(record.payload.data()), head[1]));
}

} // namespace picanative::trace
