#define PICANATIVE_BUILDING
#include "picanative/api.h"

#include "picanative/gpu.hpp"
#include "picanative/renderer.hpp"
#include "picanative/transfer.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

using namespace picanative;

namespace {

class HostMemory final : public Memory {
public:
  HostMemory(pn_map_fn map, void* user) : map_(map), user_(user) {}
  uint8_t* map(uint32_t address, uint32_t size) override { return map_(user_, address, size); }

private:
  pn_map_fn map_;
  void* user_;
};

} // namespace

struct pn_engine {
  HostMemory memory;
  gpu::Context gpu;
  std::unique_ptr<Renderer> renderer;
  pn_engine(pn_map_fn map, void* user) : memory(map, user) {}
};

extern "C" {

pn_engine* pn_create(pn_map_fn map, void* user, char* error, uint32_t error_size) {
  auto engine = std::make_unique<pn_engine>(map, user);
  std::string message;
  if (!gpu::create(engine->gpu, message)) {
    if (error && error_size) std::snprintf(error, error_size, "%s", message.c_str());
    return nullptr;
  }
  engine->renderer = std::make_unique<Renderer>(engine->gpu, engine->memory);
  return engine.release();
}

void pn_destroy(pn_engine* engine) { delete engine; }

void pn_command_list(pn_engine* e, uint32_t address, uint32_t size) { e->renderer->command_list(address, size); }

// version 1: every command list writes what it drew back to memory, so the
// memory operations work on memory directly
void pn_memory_fill(pn_engine* e, uint32_t start, uint32_t end, uint32_t value, uint32_t width) {
  e->renderer->memory_fill(start, end, value, width);
}

void pn_display_transfer(pn_engine* e, uint32_t input, uint32_t output, uint32_t input_dimensions,
                         uint32_t output_dimensions, uint32_t flags) {
  e->renderer->display_transfer(input, output, input_dimensions, output_dimensions, flags);
}

void pn_texture_copy(pn_engine* e, uint32_t input, uint32_t output, uint32_t size, uint32_t input_gap,
                     uint32_t output_gap) {
  e->renderer->texture_copy(input, output, size, input_gap, output_gap);
}

void pn_sync_for_cpu(pn_engine* e, uint32_t address, uint32_t size) { e->renderer->sync(address, size); }
void pn_cpu_wrote(pn_engine* e, uint32_t address, uint32_t size) { e->renderer->before_cpu_write(address, size); }

void pn_stats(pn_engine* e, uint64_t out[3]) {
  const auto& s = e->renderer->stats();
  out[0] = s.draws;
  out[1] = s.pipelines;
  out[2] = s.shader_errors;
}

} // extern "C"
