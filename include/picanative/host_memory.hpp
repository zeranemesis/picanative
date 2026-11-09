#pragma once

#include "picanative/memory.hpp"

#include <vector>

namespace picanative {

// VRAM and FCRAM as plain host arrays, for the tools that replay traces and
// test cases without a game.
class HostMemory final : public Memory {
public:
  HostMemory() : vram_(kVramSize), fcram_(0x10000000) {}

  uint8_t* map(uint32_t address, uint32_t size) override {
    if (address >= kVramBase && uint64_t{address} + size <= uint64_t{kVramBase} + vram_.size())
      return &vram_[address - kVramBase];
    if (address >= kFcramBase && uint64_t{address} + size <= uint64_t{kFcramBase} + fcram_.size())
      return &fcram_[address - kFcramBase];
    return nullptr;
  }

private:
  std::vector<uint8_t> vram_, fcram_;
};

} // namespace picanative
