#include "picanative/pica.hpp"

#include <bit>
#include <cstring>
#include <vector>

namespace picanative::pica {
namespace {

// the bits of a register that the four byte-enable bits of a command write
constexpr uint32_t byte_mask(uint32_t mask) {
  return (mask & 1 ? 0xFFu : 0) | (mask & 2 ? 0xFF00u : 0) | (mask & 4 ? 0xFF0000u : 0) |
         (mask & 8 ? 0xFF000000u : 0);
}

} // namespace

float decode_float24(uint32_t raw) {
  raw &= 0xFFFFFF;
  if (raw == 0) return 0.0f;
  const uint32_t sign = raw >> 23 & 1, exponent = raw >> 16 & 0x7F, mantissa = raw & 0xFFFF;
  if (exponent == 0) return sign ? -0.0f : 0.0f;
  if (exponent == 0x7F) return std::bit_cast<float>(sign << 31 | 0x7F800000u);
  return std::bit_cast<float>(sign << 31 | (exponent + 64) << 23 | mantissa << 7);
}

CommandProcessor::CommandProcessor(Memory& memory) : memory_(memory) {
  for (auto& attribute : state_.fixed_attributes) attribute = {0, 0, 0, 1};
}

void CommandProcessor::run(uint32_t address, uint32_t size) {
  std::vector<uint32_t> words;
  auto load = [&](uint32_t at, uint32_t bytes) {
    words.assign(bytes / 4, 0);
    if (const uint8_t* data = memory_.map(at, bytes / 4 * 4)) std::memcpy(words.data(), data, bytes / 4 * 4);
  };
  load(address, size);
  uint32_t jumps = 0;
  jump_ = false;
  size_t i = 0;
  while (i + 1 < words.size()) {
    // a command: a parameter word, then a header word (register, byte
    // enables, extra word count, consecutive flag), then the extra words
    const uint32_t first = words[i], header = words[i + 1];
    i += 2;
    const uint32_t reg = header & 0xFFFF, mask = header >> 16 & 0xF, extra = header >> 20 & 0xFF;
    const bool consecutive = header >> 31;
    if (reg >= kRegisterCount) break; // not a command: the list is over
    const size_t available = std::min<size_t>(extra, words.size() - i);
    write(reg, first, mask);
    for (size_t k = 0; k < available; ++k) write(consecutive ? reg + 1 + static_cast<uint32_t>(k) : reg, words[i + k], mask);
    i += available;
    if (extra % 2) ++i; // commands are padded to an even number of words
    i = std::min(i, words.size());
    if (jump_) {
      jump_ = false;
      if (++jumps > 0x10000) break;
      load(jump_address_, jump_size_);
      i = 0;
    }
  }
  ++command_lists;
}

void CommandProcessor::write(uint32_t reg, uint32_t value, uint32_t mask) {
  if (reg >= kRegisterCount) return;
  auto& regs = state_.regs;
  const uint32_t bits = byte_mask(mask);
  const uint32_t merged = (regs[reg] & ~bits) | (value & bits);
  regs[reg] = merged;

  if (reg == kRegDrawArrays || reg == kRegDrawElements) {
    ++draws;
    if (on_draw) on_draw(state_, reg == kRegDrawElements);
  } else if (reg == kRegCmdBufJump0 || reg == kRegCmdBufJump1) {
    const uint32_t channel = reg - kRegCmdBufJump0;
    jump_ = true;
    jump_size_ = (regs[kRegCmdBufSize0 + channel] & 0x1FFFFF) * 8;
    jump_address_ = (regs[kRegCmdBufAddr0 + channel] & 0x1FFFFFFF) * 8;
  } else if (reg == kRegFixedAttributeIndex) {
    fixed_words_ = 0;
    immediate_next_ = 0;
  } else if (reg >= kRegFixedAttributeData && reg <= kRegFixedAttributeDataEnd) {
    fixed_staging_[fixed_words_++] = merged;
    if (fixed_words_ == 3) {
      fixed_words_ = 0;
      // four 24-bit floats in three words, w first
      const uint32_t a = fixed_staging_[0], b = fixed_staging_[1], c = fixed_staging_[2];
      const Vec4 attribute = {decode_float24(c & 0xFFFFFF), decode_float24((b & 0xFFFF) << 8 | c >> 24),
                              decode_float24((a & 0xFF) << 16 | b >> 16), decode_float24(a >> 8)};
      const uint32_t index = regs[kRegFixedAttributeIndex] & 0xF;
      if (index < 15) {
        state_.fixed_attributes[index] = attribute;
        regs[kRegFixedAttributeIndex] = (regs[kRegFixedAttributeIndex] & ~0xFu) | (index + 1);
      } else {
        // immediate mode: the last attribute of a vertex completes it
        // shortcut: immediate vertices are only counted for now (SM3DL sends some), drawn in G2
        if (immediate_next_ < (regs[kRegVsAttributeCount] & 0xF)) {
          ++immediate_next_;
        } else {
          immediate_next_ = 0;
          ++immediate_vertices;
        }
      }
    }
  } else if (reg >= kRegLightData && reg <= kRegLightDataEnd) {
    const uint32_t index = regs[kRegLightIndex], entry = index & 0xFF;
    state_.light_tables[index >> 8 & 0x1F][entry] = merged;
    regs[kRegLightIndex] = (index & ~0xFFu) | ((entry + 1) & 0xFF);
  } else if (reg >= kRegProcTexData && reg <= kRegProcTexDataEnd) {
    const uint32_t index = regs[kRegProcTexIndex], entry = index & 0xFF;
    state_.proctex_tables[index >> 8 & 0xF][entry] = merged;
    regs[kRegProcTexIndex] = (index & ~0xFFu) | ((entry + 1) & 0xFF);
  } else if (reg >= kRegFogData && reg <= kRegFogDataEnd) {
    const uint32_t index = regs[kRegFogIndex];
    state_.fog_table[index % state_.fog_table.size()] = merged;
    regs[kRegFogIndex] = index + 1;
  } else if (reg >= kRegGsBlock && reg < kRegGsBlock + kShaderBlockSize) {
    configure(state_.gs, reg - kRegGsBlock, merged);
  } else if (reg >= kRegVsBlock && reg < kRegVsBlock + kShaderBlockSize) {
    const uint32_t offset = reg - kRegVsBlock;
    configure(state_.vs, offset, merged);
    // unless the geometry unit is configured on its own, it takes the vertex
    // shader's program and descriptors too
    const bool program = (offset >= kShaderProgramIndex && offset <= kShaderProgramDataEnd) ||
                         (offset >= kShaderDescriptorIndex && offset <= kShaderDescriptorDataEnd);
    if (program && !(regs[kRegVsComMode] & 1)) configure(state_.gs, offset, merged);
  }
}

void CommandProcessor::configure(ShaderUnit& unit, uint32_t offset, uint32_t value) {
  if (offset == kShaderBoolUniforms) {
    unit.bool_uniforms = static_cast<uint16_t>(value);
  } else if (offset >= kShaderIntUniforms && offset <= kShaderIntUniformsEnd) {
    std::memcpy(unit.int_uniforms[offset - kShaderIntUniforms].data(), &value, 4);
  } else if (offset == kShaderEntryPoint) {
    const uint32_t entry = value & (kProgramSize - 1);
    unit.key_stale |= unit.entry_point != entry;
    unit.entry_point = entry;
  } else if (offset == kShaderUniformIndex) {
    unit.uniform_index = value & 0x7F;
    unit.uniform_component = 0;
    unit.uniform_wide = value >> 31;
  } else if (offset >= kShaderUniformData && offset <= kShaderUniformDataEnd) {
    upload_uniform(unit, value);
  } else if (offset == kShaderProgramIndex) {
    unit.program_cursor = value & (kProgramSize - 1);
  } else if (offset >= kShaderProgramData && offset <= kShaderProgramDataEnd) {
    if (unit.program_cursor < kProgramSize) {
      unit.key_stale |= unit.program[unit.program_cursor] != value;
      unit.program[unit.program_cursor++] = value;
    }
  } else if (offset == kShaderDescriptorIndex) {
    unit.descriptor_cursor = value & (kDescriptorSize - 1);
  } else if (offset >= kShaderDescriptorData && offset <= kShaderDescriptorDataEnd) {
    if (unit.descriptor_cursor < kDescriptorSize) {
      unit.key_stale |= unit.descriptors[unit.descriptor_cursor] != value;
      unit.descriptors[unit.descriptor_cursor++] = value;
    }
  }
}

void CommandProcessor::upload_uniform(ShaderUnit& unit, uint32_t word) {
  // a vector comes as four 32-bit floats, or as four 24-bit floats packed in
  // three words, most significant first
  const uint32_t needed = unit.uniform_wide ? 4 : 3;
  if (unit.uniform_component < 4) unit.uniform_staging[unit.uniform_component] = word;
  if (++unit.uniform_component < needed) return;
  unit.uniform_component = 0;
  const auto& s = unit.uniform_staging;
  Vec4 value;
  if (unit.uniform_wide) {
    value = {std::bit_cast<float>(s[3]), std::bit_cast<float>(s[2]), std::bit_cast<float>(s[1]),
             std::bit_cast<float>(s[0])};
  } else {
    value = {decode_float24(s[2] & 0xFFFFFF), decode_float24((s[1] & 0xFFFF) << 8 | s[2] >> 24),
             decode_float24((s[0] & 0xFF) << 16 | s[1] >> 16), decode_float24(s[0] >> 8)};
  }
  if (unit.uniform_index < kFloatUniforms) unit.float_uniforms[unit.uniform_index] = value;
  ++unit.uniform_index;
}

uint64_t fingerprint(const State& state) {
  uint64_t hash = 0xCBF29CE484222325ull;
  auto add = [&hash](uint32_t word) {
    for (int b = 0; b < 4; ++b) hash = (hash ^ (word >> (8 * b) & 0xFF)) * 0x100000001B3ull;
  };
  for (uint32_t reg : state.regs) add(reg);
  for (const ShaderUnit* unit : {&state.vs, &state.gs}) {
    for (uint32_t word : unit->program) add(word);
    for (uint32_t word : unit->descriptors) add(word);
    for (const Vec4& vector : unit->float_uniforms)
      for (float component : vector) add(std::bit_cast<uint32_t>(component));
    for (const auto& vector : unit->int_uniforms) add(vector[0] | vector[1] << 8 | vector[2] << 16 | vector[3] << 24);
    add(unit->bool_uniforms);
    add(unit->entry_point);
  }
  return hash;
}

} // namespace picanative::pica
