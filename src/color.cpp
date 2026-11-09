#include "picanative/color.hpp"

namespace picanative {
namespace {

constexpr uint8_t expand4(uint32_t v) { return static_cast<uint8_t>(v << 4 | v); }
constexpr uint8_t expand5(uint32_t v) { return static_cast<uint8_t>(v << 3 | v >> 2); }
constexpr uint8_t expand6(uint32_t v) { return static_cast<uint8_t>(v << 2 | v >> 4); }

uint32_t load16(const uint8_t* bytes) { return bytes[0] | bytes[1] << 8; }

void store16(uint8_t* bytes, uint32_t value) {
  bytes[0] = static_cast<uint8_t>(value);
  bytes[1] = static_cast<uint8_t>(value >> 8);
}

} // namespace

Rgba decode(ColorFormat format, const uint8_t* b) {
  switch (format) {
  case ColorFormat::RGBA8: return {b[3], b[2], b[1], b[0]};
  case ColorFormat::RGB8: return {b[2], b[1], b[0], 0xFF};
  case ColorFormat::RGB565: {
    const uint32_t v = load16(b);
    return {expand5(v >> 11 & 0x1F), expand6(v >> 5 & 0x3F), expand5(v & 0x1F), 0xFF};
  }
  case ColorFormat::RGB5A1: {
    const uint32_t v = load16(b);
    return {expand5(v >> 11 & 0x1F), expand5(v >> 6 & 0x1F), expand5(v >> 1 & 0x1F),
            static_cast<uint8_t>(v & 1 ? 0xFF : 0)};
  }
  case ColorFormat::RGBA4: {
    const uint32_t v = load16(b);
    return {expand4(v >> 12 & 0xF), expand4(v >> 8 & 0xF), expand4(v >> 4 & 0xF), expand4(v & 0xF)};
  }
  }
  return {};
}

void encode(ColorFormat format, Rgba c, uint8_t* b) {
  switch (format) {
  case ColorFormat::RGBA8:
    b[0] = c[3], b[1] = c[2], b[2] = c[1], b[3] = c[0];
    break;
  case ColorFormat::RGB8:
    b[0] = c[2], b[1] = c[1], b[2] = c[0];
    break;
  case ColorFormat::RGB565:
    store16(b, (c[0] >> 3) << 11 | (c[1] >> 2) << 5 | c[2] >> 3);
    break;
  case ColorFormat::RGB5A1:
    store16(b, (c[0] >> 3) << 11 | (c[1] >> 3) << 6 | (c[2] >> 3) << 1 | (c[3] >= 0x80));
    break;
  case ColorFormat::RGBA4:
    store16(b, (c[0] >> 4) << 12 | (c[1] >> 4) << 8 | (c[2] >> 4) << 4 | c[3] >> 4);
    break;
  }
}

} // namespace picanative
