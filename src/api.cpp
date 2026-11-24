#define PICANATIVE_BUILDING
#include "picanative/api.h"

#include "engine.hpp"
#include "picanative/transfer.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

using namespace picanative;

extern "C" {

pn_engine* pn_create(pn_map_fn map, void* user, char* error, uint32_t error_size) {
  return pn_create_ex(map, user, 0, error, error_size);
}

pn_engine* pn_create_ex(pn_map_fn map, void* user, uint32_t flags, char* error, uint32_t error_size) {
  auto engine = std::make_unique<pn_engine>(map, user);
  std::string message;
  if (!gpu::create(engine->gpu, message)) {
    if (error && error_size) std::snprintf(error, error_size, "%s", message.c_str());
    return nullptr;
  }
  engine->renderer = std::make_unique<Renderer>(engine->gpu, engine->memory, (flags & PN_CACHE_SURFACES) != 0,
                                                (flags & PN_TRUST_CPU_WRITES) != 0);
  engine->renderer->set_async_pipelines((flags & PN_ASYNC_PIPELINES) != 0);
  return engine.release();
}

void pn_destroy(pn_engine* engine) { delete engine; }

void pn_set_resolution_scale(pn_engine* e, uint32_t scale) { e->renderer->set_scale(scale); }

void pn_command_list(pn_engine* e, uint32_t address, uint32_t size) { e->renderer->command_list(address, size); }

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

void pn_set_pending_hook(pn_engine* e, pn_pending_fn hook, void* user) {
  if (hook) e->renderer->on_pending = [hook, user](uint32_t address, uint32_t size, bool pending) { hook(user, address, size, pending); };
  else e->renderer->on_pending = nullptr;
}

void pn_set_native_operations(pn_engine* e, pn_native_fn run, void* user) {
  if (run) e->renderer->processor().on_native = [run, user](uint32_t operation) { run(user, operation); };
  else e->renderer->processor().on_native = nullptr;
}

void pn_write_register(pn_engine* e, uint32_t reg, uint32_t value, uint32_t mask) {
  e->renderer->processor().write(reg, value, mask);
}
void pn_run_program(pn_engine* e, const uint32_t* words, uint32_t count) {
  e->renderer->processor().run_program(words, count);
}
void pn_print_register_stats(pn_engine* e) {
  e->renderer->processor().print_register_stats();
  const auto& s = e->renderer->stats();
  if (std::getenv("PN_REG_STATS"))
    std::fprintf(stderr, "[picanative] encoding: buffers made %.1f s (%.1f MB: vertices %.1f, tables %.1f), draws "
                 "encoded %.1f s\n", s.submit_buffers, s.submit_bytes / 1e6, s.arena_bytes / 1e6, s.lut_bytes / 1e6,
                 s.submit_draws);
}
void pn_vertex_uniforms(pn_engine* e, uint32_t first, const uint32_t* words, uint32_t count) {
  e->renderer->processor().vertex_uniforms(first, words, count);
}

void pn_geometry_uniforms(pn_engine* e, uint32_t first, const uint32_t* words, uint32_t count) {
  e->renderer->processor().vertex_uniforms(first, words, count, true);
}

void pn_texture_times(pn_engine* e, double out[4], uint64_t* calls) {
  const auto& s = e->renderer->stats();
  out[0] = s.tex_writeback, out[1] = s.tex_known, out[2] = s.tex_hash, out[3] = s.tex_decode;
  calls[0] = s.tex_calls, calls[1] = s.textures_decoded, calls[2] = s.flushes, calls[3] = s.passes;
}

void pn_times(pn_engine* e, double out[7]) {
  const auto& s = e->renderer->stats();
  const double times[7] = {s.record, s.textures, s.pipelines_time, s.encode, s.wait, s.store, s.load};
  for (int i = 0; i < 7; ++i) out[i] = times[i];
}

int pn_load_shaders(pn_engine* e, const char* path, char* error, uint32_t error_size) {
  std::string message;
  if (e->renderer->load_translations(path, message)) return 1;
  if (error && error_size) std::snprintf(error, error_size, "%s", message.c_str());
  return 0;
}

void pn_shader_stats(pn_engine* e, uint64_t out[3]) {
  const auto& s = e->renderer->stats();
  out[0] = s.programs_ahead;
  out[1] = s.programs_translated;
  out[2] = static_cast<uint64_t>(s.translating * 1e6);
}

void pn_stats(pn_engine* e, uint64_t out[3]) {
  const auto& s = e->renderer->stats();
  out[0] = s.draws;
  out[1] = s.pipelines;
  out[2] = s.shader_errors;
}

} // extern "C"
