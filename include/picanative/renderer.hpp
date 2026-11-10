#pragma once

#include "picanative/draw.hpp"
#include "picanative/gpu.hpp"
#include "picanative/memory.hpp"
#include "picanative/pica.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Draws what the command lists describe with WebGPU.
//
// The buffers a command list draws into are surfaces on the GPU. Two modes:
//
// - safe (the host does not report the CPU's writes): each command list
//   loads its surfaces from memory and writes them back when it ends, and
//   textures are told apart by hashing their memory;
// - cached surfaces (the fills, transfers and copies all go through the
//   renderer, and the host calls sync or before_cpu_write when the CPU
//   touches what the GPU drew): surfaces stay on the GPU from
//   one command list to the next and go back to memory only when something
//   reads them (a transfer, a texture, the CPU through sync); a fill that
//   covers a whole surface clears it on the GPU; decoded textures stay until
//   their memory is written.
//
// The memory operations go through the renderer, which keeps memory and the
// surfaces in step around them.
namespace picanative {

class Renderer {
public:
  Renderer(gpu::Context& gpu, Memory& memory, bool cache_surfaces = false, bool trust_cpu_writes = false);
  ~Renderer();

  // Runs a command list and its draws.
  void command_list(uint32_t address, uint32_t size);

  // The memory operations, with physical addresses.
  void memory_fill(uint32_t start, uint32_t end, uint32_t value, uint32_t width);
  void display_transfer(uint32_t input, uint32_t output, uint32_t input_dimensions, uint32_t output_dimensions,
                        uint32_t flags);
  void texture_copy(uint32_t input, uint32_t output, uint32_t size, uint32_t input_gap, uint32_t output_gap);

  // Memory gets what the GPU drew in a range, for the CPU to read it.
  void sync(uint32_t address, uint32_t size);
  // The CPU is about to write a range (cached mode).
  void before_cpu_write(uint32_t address, uint32_t size);

  pica::CommandProcessor& processor() { return processor_; }

  // The GPU buffer holding what memory holds at [address, address + size),
  // when a display transfer done on the GPU wrote it and nothing has written
  // there since (as far as the renderer is told); null otherwise. A window
  // shows a screen from it without reading anything back.
  wgpu::Buffer picture(uint32_t address, uint32_t size) const;
  gpu::Context& gpu() { return gpu_; }

  struct Stats {
    uint64_t draws = 0, skipped = 0, passes = 0, pipelines = 0, textures_decoded = 0, flushes = 0;
    uint64_t shader_errors = 0, vertex_modules = 0, waiting = 0, gpu_transfers = 0; // waiting: draws skipped while their pipeline compiles
    // where the time goes, in seconds
    double record = 0, textures = 0, pipelines_time = 0, encode = 0, wait = 0, load = 0, store = 0;
  };
  const Stats& stats() const { return stats_; }

  // Pipelines compile on other threads, and draws needing one that is not
  // ready yet are skipped (for a game: no stall; tools comparing pictures
  // keep the default, compiling as it goes).
  void set_async_pipelines(bool on) { async_ = on; }

  // the first WGSL compile errors, for diagnosis
  std::vector<std::string> errors;

private:
  struct Surface;
  struct Texture;
  struct PendingDraw;

  void record(const pica::State& state, bool indexed);
  // runs the pending work; with write_back, every surface drawn goes back to memory
  void flush(bool write_back);
  // runs the pending work, and writes back the surfaces listed
  void submit(const std::vector<Surface*>& back);
  // what the GPU drew over a range goes to memory; with forget, the surfaces
  // over it also stop holding it (memory is about to change there)
  void write_back(uint32_t address, uint32_t size, bool forget);
  void forget_textures(uint32_t address, uint32_t size);
  // a display transfer done on the GPU, when its input is a surface it holds
  bool gpu_transfer(uint32_t input, uint32_t output, uint32_t input_dimensions, uint32_t output_dimensions,
                    uint32_t flags);
  // memory gets the GPU transfers on their way into a range (all of them with
  // the defaults), the oldest first
  void deliver(uint32_t address = 0, uint32_t size = UINT32_MAX);
  Surface& surface(uint32_t address, uint32_t width, uint32_t height, bool depth);
  void load(Surface& surface);
  void store(Surface& surface, const std::vector<uint8_t>& data);
  wgpu::TextureView texture(const pica::State& state, int unit, uint32_t uniforms[4], float border[4], float lod[4]);
  wgpu::Sampler sampler(uint32_t config);
  wgpu::BindGroup texture_group(const wgpu::TextureView views[3], const wgpu::Sampler samplers[3]);
  wgpu::RenderPipeline pipeline(const draw::Config& config, const pica::ShaderUnit& vs);
  uint32_t arena_copy(uint32_t address, uint32_t size);
  uint32_t lookup_tables(const pica::State& state);

  gpu::Context& gpu_;
  Memory& memory_;
  bool cached_, trusted_;
  bool async_ = false;
  std::set<std::string> compiling_;
  std::vector<wgpu::Future> futures_;
  pica::CommandProcessor processor_;
  Stats stats_;

  wgpu::BindGroupLayout layout_, texture_layout_;
  std::map<std::array<void*, 6>, wgpu::BindGroup> texture_groups_;
  wgpu::PipelineLayout pipeline_layout_;
  wgpu::RenderPipeline depth_upload_;
  wgpu::BindGroupLayout depth_upload_layout_;
  wgpu::TextureView empty_texture_;
  wgpu::ComputePipeline transfer_pipeline_;
  // GPU transfers whose bytes are on their way to memory: read back when
  // something reaches their range, so the CPU never waits for them otherwise
  struct Delivery {
    uint32_t address, size;
    wgpu::Buffer buffer;
    wgpu::Future mapped;
  };
  std::vector<Delivery> deliveries_;
  struct Picture {
    uint32_t address, size;
    wgpu::Buffer buffer;
  };
  std::vector<Picture> pictures_;
  std::unordered_map<uint32_t, wgpu::Sampler> samplers_;
  std::unordered_map<std::string, wgpu::RenderPipeline> pipelines_;
  std::unordered_map<std::string, wgpu::ShaderModule> modules_;
  std::map<std::tuple<uint32_t, uint32_t, uint32_t, bool>, std::unique_ptr<Surface>> surfaces_;
  std::unordered_map<uint64_t, std::unique_ptr<Texture>> textures_;
  // cached mode: the texture each (address, format, size, levels) decoded to,
  // while its memory is unchanged
  struct Known {
    uint32_t address, bytes;
    Texture* texture;
  };
  std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, Known> known_;

  // what the pending draws need, uploaded when they run
  std::vector<PendingDraw> pending_;
  std::vector<uint8_t> arena_;      // vertex arrays and indices, byte-addressed
  std::vector<float> luts_;         // lookup tables, vec2 entries
  std::vector<uint8_t> uniforms_;   // vertex shader and draw uniform blocks, 256-aligned
  std::vector<uint32_t> last_light_tables_, last_fog_table_;
  uint32_t last_luts_at_ = UINT32_MAX;
};

} // namespace picanative
