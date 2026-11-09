#pragma once

#include "picanative/pica.hpp"

#include <string>

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
std::string vertex_function(const pica::ShaderUnit& unit, const std::string& uniforms_name);

// A key identifying what the translation depends on: program, operand
// descriptors and entry point.
uint64_t program_key(const pica::ShaderUnit& unit);

} // namespace picanative::shadergen
