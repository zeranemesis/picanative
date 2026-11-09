#pragma once

#include "picanative/memory.hpp"

#include <cstdint>

// The GPU's memory operations, done on the CPU, byte for byte as the console
// does them. They are the exact reference: the GPU versions of the surface
// cache are checked against them, and they serve whatever the cache does not
// hold. All addresses are physical.
namespace picanative {

// MemoryFill: repeats the first `width` bytes (2, 3 or 4) of `value` over
// [start, end).
void memory_fill(Memory& memory, uint32_t start, uint32_t end, uint32_t value, uint32_t width);

// DisplayTransfer: copies an image, converting its format and its layout
// (tiled or linear), flipped and downscaled when the flags ask.
// Dimensions are width | height << 16.
void display_transfer(Memory& memory, uint32_t input, uint32_t output, uint32_t input_dimensions,
                      uint32_t output_dimensions, uint32_t flags);

// TextureCopy: copies `size` bytes as lines of a given width, skipping a
// gap after each line on either side. Gaps are width | skip << 16, in units
// of 16 bytes.
void texture_copy(Memory& memory, uint32_t input, uint32_t output, uint32_t size, uint32_t input_gap,
                  uint32_t output_gap);

} // namespace picanative
