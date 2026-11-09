#include "picanative/transfer.hpp"

#include "picanative/color.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace picanative {
namespace {

// Unmapped memory reads as zero and ignores writes.
void read(Memory& memory, uint32_t address, uint8_t* out, uint32_t size) {
  if (const uint8_t* bytes = memory.map(address, size)) {
    std::memcpy(out, bytes, size);
  } else {
    std::memset(out, 0, size);
  }
}

void write(Memory& memory, uint32_t address, const uint8_t* data, uint32_t size) {
  if (uint8_t* bytes = memory.map(address, size)) {
    std::memcpy(bytes, data, size);
  }
}

} // namespace

void memory_fill(Memory& memory, uint32_t start, uint32_t end, uint32_t value, uint32_t width) {
  if (end <= start) return;
  const uint32_t size = end - start;
  const uint32_t period = width == 2 || width == 3 ? width : 4;
  const uint8_t pattern[4] = {static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8),
                              static_cast<uint8_t>(value >> 16), static_cast<uint8_t>(value >> 24)};
  std::vector<uint8_t> filled;
  uint8_t* bytes = memory.map(start, size);
  if (!bytes) {
    filled.resize(size);
    bytes = filled.data();
  }
  for (uint32_t i = 0; i < size; ++i) bytes[i] = pattern[i % period];
  if (!filled.empty()) write(memory, start, bytes, size);
}

void display_transfer(Memory& memory, uint32_t input_address, uint32_t output_address,
                      uint32_t input_dimensions, uint32_t output_dimensions, uint32_t flags) {
  // flags: bit 0 flips vertically, bit 1 reads a linear image (tiled
  // otherwise), bit 2 crops lines to the output width, bit 5 keeps the
  // layout, bits 8-10 and 12-14 are the formats, bits 24-25 the downscale.
  const bool flip = flags & 1;
  const bool input_linear = flags & 2;
  const bool crop = flags & 4;
  const bool output_tiled = input_linear != ((flags & 0x20) != 0);
  const ColorFormat input_format = transfer_format(flags >> 8);
  const ColorFormat output_format = transfer_format(flags >> 12);
  const uint32_t input_bpp = bytes_per_pixel(input_format);
  const uint32_t output_bpp = bytes_per_pixel(output_format);
  const uint32_t downscale = flags >> 24 & 3;
  const uint32_t scale_x = downscale == 1 || downscale == 2 ? 2 : 1;
  const uint32_t scale_y = downscale == 2 ? 2 : 1;

  const uint32_t input_width = input_dimensions & 0xFFFF, input_height = input_dimensions >> 16;
  // the output dimensions count input pixels: a downscale divides them
  const uint32_t output_width = (output_dimensions & 0xFFFF) / scale_x;
  const uint32_t output_height = (output_dimensions >> 16) / scale_y;
  const uint32_t copy_width = std::min(output_width, input_width / scale_x);
  const uint32_t copy_height = std::min(output_height, input_height / scale_y);
  if (copy_width == 0 || copy_height == 0) return;

  // flipped and cropped, the console starts the output later by the bytes of
  // the lines cropped off, all but one
  if (flip && crop) {
    const uint32_t cropped = input_width > (output_dimensions & 0xFFFF) ? input_width - (output_dimensions & 0xFFFF) : 0;
    output_address += cropped * (output_height ? output_height - 1 : 0) * output_bpp;
  }

  std::vector<uint8_t> input(size_t{input_width} * input_height * input_bpp);
  std::vector<uint8_t> output(size_t{output_width} * output_height * output_bpp);
  read(memory, input_address, input.data(), static_cast<uint32_t>(input.size()));
  // what lies outside the copied rectangle stays as it was
  read(memory, output_address, output.data(), static_cast<uint32_t>(output.size()));

  for (uint32_t y = 0; y < copy_height; ++y) {
    const uint32_t out_y = flip ? output_height - 1 - y : y;
    for (uint32_t x = 0; x < copy_width; ++x) {
      const size_t to = output_tiled ? tiled_offset(x, out_y, output_width, output_bpp)
                                     : (size_t{out_y} * output_width + x) * output_bpp;
      if (to + output_bpp > output.size()) continue;
      // a downscale averages each 2x1 or 2x2 block, rounding to nearest
      uint32_t sum[4] = {};
      uint32_t count = 0;
      for (uint32_t dy = 0; dy < scale_y; ++dy) {
        for (uint32_t dx = 0; dx < scale_x; ++dx) {
          const uint32_t sx = x * scale_x + dx, sy = y * scale_y + dy;
          const size_t from = input_linear ? (size_t{sy} * input_width + sx) * input_bpp
                                           : tiled_offset(sx, sy, input_width, input_bpp);
          if (from + input_bpp > input.size()) continue;
          const Rgba sample = decode(input_format, &input[from]);
          for (int c = 0; c < 4; ++c) sum[c] += sample[c];
          ++count;
        }
      }
      if (count == 0) continue;
      Rgba pixel;
      for (int c = 0; c < 4; ++c) pixel[c] = static_cast<uint8_t>((sum[c] + count / 2) / count);
      encode(output_format, pixel, &output[to]);
    }
  }
  write(memory, output_address, output.data(), static_cast<uint32_t>(output.size()));
}

void texture_copy(Memory& memory, uint32_t input, uint32_t output, uint32_t size, uint32_t input_gap,
                  uint32_t output_gap) {
  const uint32_t input_width = (input_gap & 0xFFFF) * 16, input_skip = (input_gap >> 16) * 16;
  const uint32_t output_width = (output_gap & 0xFFFF) * 16, output_skip = (output_gap >> 16) * 16;
  std::vector<uint8_t> buffer;
  // a zero width on either side: one contiguous run
  if (input_width == 0 || output_width == 0) {
    buffer.resize(size);
    read(memory, input, buffer.data(), size);
    write(memory, output, buffer.data(), size);
    return;
  }
  // a stream of bytes cut into lines on each side, each side skipping its
  // gap at the end of its own lines
  uint32_t in_line = 0, out_line = 0;
  while (size > 0) {
    const uint32_t run = std::min({size, input_width - in_line, output_width - out_line});
    buffer.resize(run);
    read(memory, input, buffer.data(), run);
    write(memory, output, buffer.data(), run);
    input += run, output += run, size -= run, in_line += run, out_line += run;
    if (in_line == input_width) input += input_skip, in_line = 0;
    if (out_line == output_width) output += output_skip, out_line = 0;
  }
}

} // namespace picanative
