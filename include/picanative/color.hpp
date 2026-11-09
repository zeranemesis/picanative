#pragma once

#include <array>
#include <cstdint>

namespace picanative {

// Color formats of screens and DisplayTransfer (the numbering of color
// buffers swaps 2 and 3, see color_buffer_format).
enum class ColorFormat : uint8_t { RGBA8 = 0, RGB8 = 1, RGB565 = 2, RGB5A1 = 3, RGBA4 = 4 };

using Rgba = std::array<uint8_t, 4>;

constexpr ColorFormat transfer_format(uint32_t raw) {
  raw &= 7;
  return raw <= 3 ? static_cast<ColorFormat>(raw) : ColorFormat::RGBA4;
}

constexpr ColorFormat color_buffer_format(uint32_t raw) {
  raw &= 7;
  if (raw == 2) return ColorFormat::RGB5A1;
  if (raw == 3) return ColorFormat::RGB565;
  return transfer_format(raw);
}

constexpr uint32_t bytes_per_pixel(ColorFormat format) {
  switch (format) {
  case ColorFormat::RGBA8: return 4;
  case ColorFormat::RGB8: return 3;
  default: return 2;
  }
}

// One pixel to 8-bit RGBA. The GPU stores channels from alpha to red.
Rgba decode(ColorFormat format, const uint8_t* bytes);
// 8-bit RGBA to one pixel, by truncating each channel.
void encode(ColorFormat format, Rgba rgba, uint8_t* bytes);

// Images are stored in 8x8 tiles, row of tiles after row of tiles, the
// pixels of a tile in Morton (Z) order.
constexpr uint32_t morton_in_tile(uint32_t x, uint32_t y) {
  x &= 7;
  y &= 7;
  return (x & 1) | (y & 1) << 1 | (x & 2) << 1 | (y & 2) << 2 | (x & 4) << 2 | (y & 4) << 3;
}

// Byte offset of pixel (x, y) in a tiled image `width` pixels wide.
constexpr uint32_t tiled_offset(uint32_t x, uint32_t y, uint32_t width, uint32_t bpp) {
  return ((y / 8 * (width / 8) + x / 8) * 64 + morton_in_tile(x, y)) * bpp;
}

} // namespace picanative
