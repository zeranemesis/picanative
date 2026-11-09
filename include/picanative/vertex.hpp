#pragma once

#include "picanative/memory.hpp"
#include "picanative/pica.hpp"

#include <array>
#include <cstdint>
#include <vector>

// How a draw's vertices reach the vertex shader: the attribute arrays
// (loaders), their fields, and the input register each attribute lands in.
// The layout is decoded once per draw; the GPU path turns it into vertex
// buffer descriptions, and fetch() reads it on the CPU, the exact reference.
namespace picanative::vertex {

inline constexpr uint32_t kInputRegisters = 16;
using Inputs = std::array<pica::Vec4, kInputRegisters>;

// attribute component types
enum Type : uint32_t { Byte = 0, UnsignedByte = 1, Short = 2, Float = 3 };

struct Field {
  uint32_t offset;   // in the array's vertex, aligned to the component size
  uint32_t type;
  uint32_t count;    // 1 to 4 components; the rest read as (0, 0, 0, 1)
  uint32_t input;    // vertex shader input register
};

struct Loader {
  uint32_t index;    // 0 to 11, whose register holds the array's offset
  uint32_t offset;   // from the attribute base
  uint32_t stride;
  uint32_t size;     // bytes of a vertex its fields span
  std::vector<Field> fields;
};

struct Layout {
  uint32_t base = 0;                 // physical address the arrays count from
  std::vector<Loader> loaders;       // only arrays feeding at least one input
  Inputs defaults{};                 // what each input holds before the arrays
};

Layout decode(const pica::State& state);

// The vertices a draw names, in order: the index buffer's entries, or the
// arrays from the first vertex on.
std::vector<uint32_t> vertices(const pica::State& state, Memory& memory, bool indexed);

// The shader inputs of each vertex, read from memory (unmapped bytes read 0).
void fetch(const Layout& layout, Memory& memory, const std::vector<uint32_t>& vertices, std::vector<Inputs>& out);

// FNV-1a over the vertex count and the inputs, as the reference's
// per-draw fingerprint.
uint64_t fingerprint(uint32_t count, const std::vector<Inputs>& inputs);

} // namespace picanative::vertex
