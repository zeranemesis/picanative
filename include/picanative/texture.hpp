#pragma once

#include <cstdint>
#include <vector>

// PICA200 texture formats, decoded to 8-bit RGBA in rows, top row first as
// the texture's own y counts (the GPU path uploads this).
namespace picanative::texture {

enum class Format : uint32_t {
  RGBA8 = 0, RGB8 = 1, RGB5A1 = 2, RGB565 = 3, RGBA4 = 4, IA8 = 5, HILO8 = 6, I8 = 7, A8 = 8, IA4 = 9,
  I4 = 10, A4 = 11, ETC1 = 12, ETC1A4 = 13,
};

constexpr uint32_t bits_per_pixel(uint32_t format) {
  switch (format) {
  case 0: return 32;
  case 1: return 24;
  case 2: case 3: case 4: case 5: case 6: return 16;
  case 7: case 8: case 9: case 13: return 8;
  case 10: case 11: case 12: return 4;
  default: return 32;
  }
}

// bytes of a width x height image of the format
constexpr uint32_t size(uint32_t format, uint32_t width, uint32_t height) {
  return width * height * bits_per_pixel(format) / 8;
}

// Decodes `width x height` texels (both multiples of 8). Bytes past the end
// of `data` read as zero. `out` gets width * height * 4 bytes, texel (x, y)
// at (y * width + x) * 4.
void decode(uint32_t format, const uint8_t* data, size_t length, uint32_t width, uint32_t height,
            std::vector<uint8_t>& out);

} // namespace picanative::texture
