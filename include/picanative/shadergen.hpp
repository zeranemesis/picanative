#pragma once

#include "picanative/pica.hpp"

#include <string>
#include <unordered_map>
#include <vector>

// PICA200 shader programs to WGSL.
namespace picanative::shadergen {

// What a generated vertex shader expects at module scope, declared by
// whoever assembles the WGSL module: the unit's uniforms.
//   c: float uniforms c0-c95; ints: i0-i3, one byte each (loop count,
//   start, increment); bools.x: b0-b15.
inline constexpr const char* kVertexUniformsWgsl = R"(struct PicaVsUniforms {
  c: array<vec4f, 96>,
  ints: vec4u,
  bools: vec4u,
}
)";

// A WGSL function `fn pica_vs(vin: array<vec4f, 16>) -> array<vec4f, 16>`
// running the unit's program from its entry point, as the hardware does:
// the output registers it leaves. It reads the uniforms through a variable
// named `uniforms_name` of type PicaVsUniforms. Bool and int uniforms stay
// dynamic, so one translation serves every draw of the program.
// whole: the unit as it is rather than reachable(unit), for comparisons.
std::string vertex_function(const pica::ShaderUnit& unit, const std::string& uniforms_name, bool whole = false);

// A key identifying what the translation depends on: program, operand
// descriptors and entry point.
uint64_t program_key(const pica::ShaderUnit& unit);

// The unit as the translation and the key see it: the instructions control
// can reach from the entry point and the descriptors they use, the rest (what
// earlier programs left in the unit) zero. A program read from a shader file
// thus gets the key the game's upload of it gives.
pica::ShaderUnit reachable(const pica::ShaderUnit& unit);

// Translation ahead of time. The vertex programs of a shader binary (DVLB, a
// .shbin): one unit per vertex shader (DVLE), its code and descriptors where
// the game uploads them (from 0); geometry shaders are only counted.
std::vector<pica::ShaderUnit> read_dvlb(const uint8_t* data, size_t size, uint32_t& geometry);

// program_key → vertex_function(unit, "vsu"), kept in a file. Loading fails
// (with the reason) for a missing file or one another build of this
// translator wrote (kVersion).
using Translations = std::unordered_map<uint64_t, std::string>;
bool save_translations(const std::string& path, const Translations& translations);
bool load_translations(const std::string& path, Translations& translations, std::string& error);
extern const char* const kVersion;

} // namespace picanative::shadergen
