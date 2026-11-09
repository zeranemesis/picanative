#pragma once

#include <cstdint>

namespace picanative {

// Physical addresses of the memory the PICA200 sees.
inline constexpr uint32_t kVramBase = 0x18000000;
inline constexpr uint32_t kVramSize = 0x00600000;
inline constexpr uint32_t kFcramBase = 0x20000000;

// The game's memory, seen by physical address. The port maps it its own way
// (the recompiled scaffold, then the native game), the engine only asks for
// contiguous runs of bytes.
class Memory {
public:
  virtual ~Memory() = default;
  // Host bytes for [address, address + size), or nullptr when the range is
  // not mapped in one piece.
  virtual uint8_t* map(uint32_t address, uint32_t size) = 0;
};

} // namespace picanative
