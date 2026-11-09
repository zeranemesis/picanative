#pragma once

#include "picanative/pica.hpp"
#include "picanative/vertex.hpp"

#include <cstdint>
#include <string>
#include <vector>

// A draw as the GPU pipeline sees it: the part of the PICA state that
// changes the generated shaders or the fixed-function state (the pipeline
// key), and the WGSL of the pipeline. Everything else (uniforms, colors,
// tables, strides, addresses) is per-draw data.
namespace picanative::draw {

struct Field {
  uint32_t offset, type, count, input;
};

struct Config {
  // vertex stage
  uint64_t program = 0; // shadergen::program_key
  uint32_t entry = 0;
  uint32_t output_mask = 0;
  uint32_t output_total = 0;
  uint32_t output_map[7] = {};
  uint32_t loader_count = 0;
  uint32_t loader_fields[12] = {};         // fields per loader
  std::vector<Field> fields;               // all loaders' fields, in order
  uint32_t topology = 0;                   // 0 list, 1 strip, 2 fan, 3 geometry (list)
  bool indexed = false;
  bool upside_down = false;
  // fragment stage
  uint32_t textures = 0;                   // bits 0-2: units on
  bool texture0_projected = false;
  bool texture2_coord1 = false;
  uint32_t tev[6][3] = {};                 // per stage: sources, operands, combiner (+ scale in tev_scale)
  uint32_t tev_scale[6] = {};
  uint32_t buffer_update = 0;              // 0x0E0 bits 8-15
  uint32_t alpha_test = 0;                 // 0: off, else function + 1
  bool fog = false, fog_flip = false;
  bool lighting = false;
  uint32_t light_count = 0, light_slots = 0, light_config0 = 0, light_config1 = 0;
  uint32_t light_flags[8] = {};            // directional, two-sided, geometric 0/1 per light slot
  uint32_t lut_absolute = 0, lut_input = 0, lut_scale = 0;
  uint32_t logic_mode = 0;                 // 0: blend, else 1 + logic op
  // fixed function
  uint32_t blend = 0;                      // 0x101, when blending
  uint32_t color_mask = 0;                 // bits 0-3: r, g, b, a
  uint32_t depth_test = 0;                 // 0: off, else function + 1
  bool depth_write = false;
  bool stencil = false;
  uint32_t stencil_test = 0, stencil_op = 0; // function and masks (reference is per draw), ops
  uint32_t cull = 0;

  std::vector<uint32_t> key() const;
};

// Reads the configuration of a draw from the state. `layout` is the vertex
// layout vertex::decode gave for it.
Config configure(const pica::State& state, const vertex::Layout& layout, bool indexed);

// The WGSL modules of a pipeline, entry points `vs` and `fs`, with the
// bindings documented in draw.cpp. Separate, so that a vertex shader shared by
// many pipelines compiles once.
std::string vertex_wgsl(const Config& config, const pica::ShaderUnit& vertex_unit);
std::string fragment_wgsl(const Config& config);

// Uniform block of a draw, as the WGSL `Draw` struct lays it out: everything
// per draw that is not in the key. Built by the renderer.
struct alignas(16) DrawUniforms {
  uint32_t loaders[12][4];     // x: byte address of vertex 0 in the arena, y: stride, z: size
  uint32_t index[4];           // x: byte address of the indices in the arena, y: 16-bit, z: first vertex, w: count
  float defaults[16][4];       // what each input holds before the arrays
  float clip[4];               // x: x scale, y: x offset, z: y scale, w: y offset (clip-space viewport)
  float depth[4];              // x: scale, y: offset, z: w-buffer (0/1)
  float tev_constant[6][4];
  float buffer_color[4];
  float fog_color[4];
  uint32_t alpha_reference[4];
  uint32_t texture_size[3][4]; // x: width, y: height, z: levels, w: config (filter and wrap bits)
  float texture_border[3][4];
  float texture_lod[3][4];     // x: bias, y: min, z: max, w: mip config (bit 0 linear, bit 1 between)
  uint32_t lut[4];             // x: lighting tables at (in vec2 units) in the LUT buffer, y: fog table
  float global_ambient[4];
  float lights[8][9][4];       // specular0, specular1, diffuse, ambient, position, direction, spot, distance(bias, scale), number
};

} // namespace picanative::draw
