#include "picanative/vertex.hpp"

#include <bit>
#include <cstring>
#include <optional>

namespace picanative::vertex {
namespace {

constexpr uint32_t kRegAttributeBase = 0x200, kRegAttributeFormatLow = 0x201, kRegAttributeFormatHigh = 0x202,
                   kRegLoaders = 0x203, kLoaderCount = 12, kRegIndexArray = 0x227, kRegVertexCount = 0x228,
                   kRegVertexOffset = 0x22A;
constexpr uint32_t kRegVsInputCount = pica::kRegVsBlock + 0x09, kRegVsInputMapLow = pica::kRegVsBlock + 0x0B,
                   kRegVsInputMapHigh = pica::kRegVsBlock + 0x0C;

constexpr uint32_t component_size(uint32_t type) { return type <= UnsignedByte ? 1 : type == Short ? 2 : 4; }
constexpr uint32_t align(uint32_t value, uint32_t to) { return (value + to - 1) / to * to; }

void read(Memory& memory, uint32_t address, uint8_t* out, uint32_t size) {
  if (const uint8_t* bytes = memory.map(address, size)) {
    std::memcpy(out, bytes, size);
  } else {
    std::memset(out, 0, size);
  }
}

} // namespace

Layout decode(const pica::State& state) {
  const auto& regs = state.regs;
  Layout layout;
  layout.base = (regs[kRegAttributeBase] & 0x0FFFFFFF) << 3;
  const uint64_t formats = regs[kRegAttributeFormatLow] | uint64_t{regs[kRegAttributeFormatHigh]} << 32;

  // every array's fields, and which one gives each attribute (the last)
  struct Source {
    uint32_t loader, offset, type, count;
  };
  std::array<std::optional<Source>, 12> source;
  std::vector<Loader> all;
  for (uint32_t i = 0; i < kLoaderCount; ++i) {
    const uint32_t base = kRegLoaders + i * 3;
    const uint64_t components = regs[base + 1] | uint64_t{regs[base + 2] & 0xFFFF} << 32;
    const uint32_t count = regs[base + 2] >> 28 & 0xF;
    if (count == 0) continue;
    Loader loader{i, regs[base] & 0x0FFFFFFF, regs[base + 2] >> 16 & 0xFF, 0, {}};
    uint32_t offset = 0;
    for (uint32_t slot = 0; slot < std::min(count, 12u); ++slot) {
      const uint32_t id = components >> (slot * 4) & 0xF;
      if (id >= 12) { // padding: to a word, then 4 to 16 bytes
        offset = align(offset, 4) + (id - 11) * 4;
        continue;
      }
      const uint32_t nibble = formats >> (id * 4) & 0xF, type = nibble & 3;
      offset = align(offset, component_size(type));
      source[id] = Source{static_cast<uint32_t>(all.size()), offset, type, (nibble >> 2) + 1};
      offset += component_size(type) * ((nibble >> 2) + 1);
    }
    loader.size = offset;
    all.push_back(std::move(loader));
  }

  // the vertex shader's permutation: which attribute each input takes (a
  // later attribute over an earlier one)
  const uint32_t inputs = std::min((regs[kRegVsInputCount] & 0xF) + 1, 12u);
  const uint64_t map = regs[kRegVsInputMapLow] | uint64_t{regs[kRegVsInputMapHigh]} << 32;
  std::array<std::optional<uint32_t>, kInputRegisters> owner;
  for (uint32_t id = 0; id < inputs; ++id) owner[map >> (id * 4) & 0xF] = id;

  const uint32_t fixed = regs[kRegAttributeFormatHigh] >> 16 & 0xFFF;
  for (uint32_t input = 0; input < kInputRegisters; ++input) {
    if (!owner[input]) continue;
    const uint32_t id = *owner[input];
    if (source[id]) {
      const Source& s = *source[id];
      all[s.loader].fields.push_back({s.offset, s.type, s.count, input});
    } else if (fixed >> id & 1) {
      layout.defaults[input] = state.fixed_attributes[id];
    } else {
      layout.defaults[input] = {0, 0, 0, 1};
    }
  }
  for (auto& loader : all)
    if (!loader.fields.empty()) layout.loaders.push_back(std::move(loader));
  return layout;
}

std::vector<uint32_t> vertices(const pica::State& state, Memory& memory, bool indexed) {
  const auto& regs = state.regs;
  const uint32_t count = regs[kRegVertexCount];
  std::vector<uint32_t> out(count);
  if (!indexed) {
    for (uint32_t i = 0; i < count; ++i) out[i] = regs[kRegVertexOffset] + i;
    return out;
  }
  const uint32_t config = regs[kRegIndexArray];
  const bool wide = config >> 31;
  const uint32_t address = ((regs[kRegAttributeBase] & 0x0FFFFFFF) << 3) + (config & 0x0FFFFFFF);
  std::vector<uint8_t> bytes(size_t{count} * (wide ? 2 : 1));
  read(memory, address, bytes.data(), static_cast<uint32_t>(bytes.size()));
  for (uint32_t i = 0; i < count; ++i) out[i] = wide ? bytes[i * 2] | bytes[i * 2 + 1] << 8 : bytes[i];
  return out;
}

void fetch(const Layout& layout, Memory& memory, const std::vector<uint32_t>& vertices, std::vector<Inputs>& out) {
  out.assign(vertices.size(), layout.defaults);
  uint8_t bytes[256];
  for (const Loader& loader : layout.loaders) {
    for (size_t v = 0; v < vertices.size(); ++v) {
      read(memory, layout.base + loader.offset + vertices[v] * loader.stride, bytes, loader.size);
      for (const Field& field : loader.fields) {
        pica::Vec4 value = {0, 0, 0, 1};
        const uint8_t* at = bytes + field.offset;
        for (uint32_t c = 0; c < field.count; ++c) {
          switch (field.type) {
          case Byte: value[c] = static_cast<float>(static_cast<int8_t>(at[c])); break;
          case UnsignedByte: value[c] = static_cast<float>(at[c]); break;
          case Short: value[c] = static_cast<float>(static_cast<int16_t>(at[c * 2] | at[c * 2 + 1] << 8)); break;
          default: {
            uint32_t word;
            std::memcpy(&word, at + c * 4, 4);
            value[c] = std::bit_cast<float>(word);
          }
          }
        }
        out[v][field.input] = value;
      }
    }
  }
}

uint64_t fingerprint(uint32_t count, const std::vector<Inputs>& inputs) {
  uint64_t hash = 0xCBF29CE484222325ull;
  auto add = [&hash](uint32_t word) {
    for (int b = 0; b < 4; ++b) hash = (hash ^ (word >> (8 * b) & 0xFF)) * 0x100000001B3ull;
  };
  add(count);
  for (const Inputs& vertex : inputs)
    for (const pica::Vec4& vector : vertex)
      for (float component : vector) add(std::bit_cast<uint32_t>(component));
  return hash;
}

} // namespace picanative::vertex
