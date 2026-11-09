#include "picanative/texture.hpp"

#include "picanative/color.hpp"

#include <algorithm>
#include <cstring>

namespace picanative::texture {
namespace {

constexpr uint8_t e4(uint32_t v) { return static_cast<uint8_t>(v << 4 | v); }
constexpr uint8_t e5(uint32_t v) { return static_cast<uint8_t>(v << 3 | v >> 2); }
constexpr uint8_t e6(uint32_t v) { return static_cast<uint8_t>(v << 2 | v >> 4); }

void texel(uint32_t format, const uint8_t* b, uint8_t* o) {
  const uint32_t v = b[0] | (format >= 2 && format <= 6 ? b[1] << 8 : 0);
  switch (format) {
  case 0: o[0] = b[3], o[1] = b[2], o[2] = b[1], o[3] = b[0]; break;
  case 1: o[0] = b[2], o[1] = b[1], o[2] = b[0], o[3] = 0xFF; break;
  case 2: o[0] = e5(v >> 11 & 0x1F), o[1] = e5(v >> 6 & 0x1F), o[2] = e5(v >> 1 & 0x1F), o[3] = v & 1 ? 0xFF : 0; break;
  case 3: o[0] = e5(v >> 11 & 0x1F), o[1] = e6(v >> 5 & 0x3F), o[2] = e5(v & 0x1F), o[3] = 0xFF; break;
  case 4: o[0] = e4(v >> 12 & 0xF), o[1] = e4(v >> 8 & 0xF), o[2] = e4(v >> 4 & 0xF), o[3] = e4(v & 0xF); break;
  case 5: o[0] = o[1] = o[2] = b[1], o[3] = b[0]; break;
  case 6: o[0] = 0, o[1] = b[1], o[2] = b[0], o[3] = 0xFF; break; // HILO8: (0, g, r, 1)
  case 7: o[0] = o[1] = o[2] = b[0], o[3] = 0xFF; break;
  case 8: o[0] = o[1] = o[2] = 0, o[3] = b[0]; break;
  case 9: o[0] = o[1] = o[2] = e4(b[0] >> 4), o[3] = e4(b[0] & 0xF); break;
  default: o[0] = o[1] = o[2] = o[3] = 0; break;
  }
}

constexpr int kEtcModifiers[8][4] = {{2, 8, -2, -8},     {5, 17, -5, -17},   {9, 29, -9, -29},
                                     {13, 42, -13, -42}, {18, 60, -18, -60}, {24, 80, -24, -80},
                                     {33, 106, -33, -106}, {47, 183, -47, -183}};

int extend3(uint32_t v) { return v & 4 ? static_cast<int>(v) - 8 : static_cast<int>(v); }

// one 4x4 ETC1 block (8 bytes, a little-endian 64-bit word), texel (x, y)
void etc1(const uint8_t* block, uint32_t x, uint32_t y, uint8_t* o) {
  uint32_t low, high;
  std::memcpy(&low, block, 4);
  std::memcpy(&high, block + 4, 4);
  const bool flip = high & 1, diff = high & 2;
  int base[2][3];
  if (diff) {
    const int r = high >> 27 & 0x1F, g = high >> 19 & 0x1F, b = high >> 11 & 0x1F;
    const int c0[3] = {r, g, b};
    const int d[3] = {extend3(high >> 24 & 7), extend3(high >> 16 & 7), extend3(high >> 8 & 7)};
    for (int c = 0; c < 3; ++c) {
      base[0][c] = e5(static_cast<uint32_t>(c0[c]));
      base[1][c] = e5(static_cast<uint32_t>(std::clamp(c0[c] + d[c], 0, 31)));
    }
  } else {
    for (int c = 0; c < 3; ++c) {
      base[0][c] = e4(high >> (28 - c * 8) & 0xF);
      base[1][c] = e4(high >> (24 - c * 8) & 0xF);
    }
  }
  const bool first = flip ? y < 2 : x < 2;
  const int table = static_cast<int>(first ? high >> 5 & 7 : high >> 2 & 7);
  // the index planes are column-major
  const uint32_t index = x * 4 + y;
  const int delta = kEtcModifiers[table][(low >> (index + 16) & 1) << 1 | (low >> index & 1)];
  for (int c = 0; c < 3; ++c) o[c] = static_cast<uint8_t>(std::clamp(base[first ? 0 : 1][c] + delta, 0, 255));
}

} // namespace

void decode(uint32_t format, const uint8_t* data, size_t length, uint32_t width, uint32_t height,
            std::vector<uint8_t>& out) {
  out.assign(size_t{width} * height * 4, 0);
  auto in = [&](size_t offset, size_t bytes) { return offset + bytes <= length; };
  if (format == 12 || format == 13) {
    // 8x8 tiles row by row, four 4x4 blocks per tile in Z order
    const uint32_t block_size = format == 13 ? 16 : 8;
    for (uint32_t y = 0; y < height; ++y)
      for (uint32_t x = 0; x < width; ++x) {
        const uint32_t tile = y / 8 * std::max(width / 8, 1u) + x / 8;
        const uint32_t sub = x % 8 / 4 + 2 * (y % 8 / 4);
        const size_t offset = size_t{tile * 4 + sub} * block_size;
        uint8_t* o = &out[(size_t{y} * width + x) * 4];
        if (!in(offset, block_size)) continue;
        const uint8_t* block = data + offset;
        etc1(block + (format == 13 ? 8 : 0), x % 4, y % 4, o);
        if (format == 13) {
          uint64_t alpha;
          std::memcpy(&alpha, block, 8);
          o[3] = e4(alpha >> ((x % 4 * 4 + y % 4) * 4) & 0xF);
        } else {
          o[3] = 0xFF;
        }
      }
    return;
  }
  const uint32_t bits = bits_per_pixel(format);
  for (uint32_t y = 0; y < height; ++y)
    for (uint32_t x = 0; x < width; ++x) {
      uint8_t* o = &out[(size_t{y} * width + x) * 4];
      if (bits >= 8) {
        const size_t offset = tiled_offset(x, y, width, bits / 8);
        if (in(offset, bits / 8)) texel(format, data + offset, o);
      } else {
        // 4-bit formats: two texels a byte, low nibble first
        const size_t bit = tiled_offset(x, y, width, 4);
        if (!in(bit / 8, 1)) continue;
        const uint8_t byte = data[bit / 8], nibble = bit % 8 ? byte >> 4 : byte & 0xF;
        if (format == 10) o[0] = o[1] = o[2] = e4(nibble), o[3] = 0xFF;
        else if (format == 11) o[3] = e4(nibble); // alpha alone reads black
      }
    }
}

} // namespace picanative::texture
