#pragma once

#include "picanative/memory.hpp"

#include <array>
#include <cstdint>
#include <functional>

// The PICA200's command processor: walks command lists, keeps the register
// file and the GPU's internal memories (shader programs and uniforms, the
// lighting, fog and procedural texture tables). No graphics API here: draws
// are handed out with the state they were issued under.
namespace picanative::pica {

inline constexpr uint32_t kRegisterCount = 0x300;

// registers (word index)
inline constexpr uint32_t kRegProcTexIndex = 0x0AF, kRegProcTexData = 0x0B0, kRegProcTexDataEnd = 0x0B7;
inline constexpr uint32_t kRegFogIndex = 0x0E6, kRegFogData = 0x0E8, kRegFogDataEnd = 0x0EF;
inline constexpr uint32_t kRegLightIndex = 0x1C5, kRegLightData = 0x1C8, kRegLightDataEnd = 0x1CF;
inline constexpr uint32_t kRegDrawArrays = 0x22E, kRegDrawElements = 0x22F;
inline constexpr uint32_t kRegFixedAttributeIndex = 0x232, kRegFixedAttributeData = 0x233,
                          kRegFixedAttributeDataEnd = 0x235;
inline constexpr uint32_t kRegCmdBufSize0 = 0x238, kRegCmdBufAddr0 = 0x23A, kRegCmdBufJump0 = 0x23C,
                          kRegCmdBufJump1 = 0x23D;
inline constexpr uint32_t kRegVsAttributeCount = 0x242, kRegVsComMode = 0x244;
inline constexpr uint32_t kRegGsBlock = 0x280, kRegVsBlock = 0x2B0, kShaderBlockSize = 0x2E;

// offsets inside a shader unit's block
inline constexpr uint32_t kShaderBoolUniforms = 0x00, kShaderIntUniforms = 0x01, kShaderIntUniformsEnd = 0x04,
                          kShaderEntryPoint = 0x0A, kShaderUniformIndex = 0x10, kShaderUniformData = 0x11,
                          kShaderUniformDataEnd = 0x18, kShaderProgramIndex = 0x1B, kShaderProgramData = 0x1C,
                          kShaderProgramDataEnd = 0x23, kShaderDescriptorIndex = 0x25,
                          kShaderDescriptorData = 0x26, kShaderDescriptorDataEnd = 0x2D;

inline constexpr uint32_t kProgramSize = 4096, kDescriptorSize = 128, kFloatUniforms = 96;

using Vec4 = std::array<float, 4>;

// 24-bit float (1.7.16, bias 63) to 32-bit; denormals read as zero.
float decode_float24(uint32_t raw);

struct ShaderUnit {
  std::array<uint32_t, kProgramSize> program{};
  std::array<uint32_t, kDescriptorSize> descriptors{};
  std::array<Vec4, kFloatUniforms> float_uniforms{};
  std::array<std::array<uint8_t, 4>, 4> int_uniforms{};
  uint16_t bool_uniforms = 0;
  uint32_t entry_point = 0;
  // upload cursors
  uint32_t program_cursor = 0, descriptor_cursor = 0;
  uint32_t uniform_index = 0, uniform_component = 0;
  bool uniform_wide = false;
  std::array<uint32_t, 4> uniform_staging{};
};

struct State {
  std::array<uint32_t, kRegisterCount> regs{};
  ShaderUnit vs, gs;
  std::array<Vec4, 16> fixed_attributes{};
  // the tables as uploaded, raw words
  std::array<std::array<uint32_t, 256>, 32> light_tables{};
  std::array<uint32_t, 128> fog_table{};
  std::array<std::array<uint32_t, 256>, 16> proctex_tables{};
};

class CommandProcessor {
public:
  explicit CommandProcessor(Memory& memory);

  // Runs the command list at a physical address, following its jumps.
  void run(uint32_t address, uint32_t size);

  State& state() { return state_; }
  const State& state() const { return state_; }

  // Called for each draw (indexed or not) with the state at that moment.
  std::function<void(const State&, bool indexed)> on_draw;
  uint64_t draws = 0, immediate_vertices = 0, command_lists = 0;

private:
  void write(uint32_t reg, uint32_t value, uint32_t mask);
  void configure(ShaderUnit& unit, uint32_t offset, uint32_t value);
  void upload_uniform(ShaderUnit& unit, uint32_t word);

  Memory& memory_;
  State state_;
  uint32_t fixed_staging_[3] = {};
  uint32_t fixed_words_ = 0, immediate_next_ = 0;
  bool jump_ = false;
  uint32_t jump_address_ = 0, jump_size_ = 0;
};

// FNV-1a over the registers and both shader units, to compare a replay with
// the reference image by image.
uint64_t fingerprint(const State& state);

} // namespace picanative::pica
