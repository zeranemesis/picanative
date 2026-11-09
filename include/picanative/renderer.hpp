#pragma once

#include "picanative/draw.hpp"
#include "picanative/gpu.hpp"
#include "picanative/memory.hpp"
#include "picanative/pica.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Draws what the command lists describe with WebGPU.
//
// Version 1, memory is the truth: each command list loads the buffers it
// draws into from the game's memory, records its draws, runs them in render
// passes, and writes the buffers back to memory when it ends (or before a
// draw reads one of them as a texture). The memory operations (fills,
// transfers, copies) then work on memory as the CPU versions do. The surface
// cache of the optimisation phase keeps the buffers on the GPU instead.
namespace picanative {

class Renderer {
public:
  Renderer(gpu::Context& gpu, Memory& memory);
  ~Renderer();

  // Runs a command list, its draws included, and writes back what it drew.
  void command_list(uint32_t address, uint32_t size);

  pica::CommandProcessor& processor() { return processor_; }

  struct Stats {
    uint64_t draws = 0, skipped = 0, passes = 0, pipelines = 0, textures_decoded = 0, flushes = 0;
    uint64_t shader_errors = 0;
  };
  const Stats& stats() const { return stats_; }

  // the first WGSL compile errors, for diagnosis
  std::vector<std::string> errors;

private:
  struct Surface;
  struct Texture;
  struct PendingDraw;

  void record(const pica::State& state, bool indexed);
  void flush();
  Surface& surface(uint32_t address, uint32_t width, uint32_t height, bool depth);
  void load(Surface& surface);
  void store(Surface& surface, const std::vector<uint8_t>& data);
  wgpu::TextureView texture(const pica::State& state, int unit, uint32_t uniforms[4], float border[4], float lod[4]);
  wgpu::RenderPipeline pipeline(const draw::Config& config, const pica::ShaderUnit& vs);
  uint32_t arena_copy(uint32_t address, uint32_t size);
  uint32_t lookup_tables(const pica::State& state);

  gpu::Context& gpu_;
  Memory& memory_;
  pica::CommandProcessor processor_;
  Stats stats_;

  wgpu::BindGroupLayout layout_;
  wgpu::PipelineLayout pipeline_layout_;
  wgpu::RenderPipeline depth_upload_;
  wgpu::BindGroupLayout depth_upload_layout_;
  wgpu::TextureView empty_texture_;
  std::unordered_map<std::string, wgpu::RenderPipeline> pipelines_;
  std::map<std::tuple<uint32_t, uint32_t, uint32_t, bool>, std::unique_ptr<Surface>> surfaces_;
  std::unordered_map<uint64_t, std::unique_ptr<Texture>> textures_;

  // what the pending draws need, uploaded when they run
  std::vector<PendingDraw> pending_;
  std::vector<uint8_t> arena_;      // vertex arrays and indices, byte-addressed
  std::vector<float> luts_;         // lookup tables, vec2 entries
  std::vector<uint8_t> uniforms_;   // vertex shader and draw uniform blocks, 256-aligned
  std::vector<uint32_t> last_light_tables_, last_fog_table_;
  uint32_t last_luts_at_ = UINT32_MAX;
};

} // namespace picanative
