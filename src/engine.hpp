#pragma once

// What a pn_engine is, shared by the C interface and the window.
#include "picanative/api.h"
#include "picanative/gpu.hpp"
#include "picanative/renderer.hpp"

#include <memory>

namespace picanative {

class HostMemory final : public Memory {
public:
  HostMemory(pn_map_fn map, void* user) : map_(map), user_(user) {}
  uint8_t* map(uint32_t address, uint32_t size) override { return map_(user_, address, size); }

private:
  pn_map_fn map_;
  void* user_;
};

} // namespace picanative

struct pn_engine {
  picanative::HostMemory memory;
  picanative::gpu::Context gpu;
  std::unique_ptr<picanative::Renderer> renderer;
  pn_engine(pn_map_fn map, void* user) : memory(map, user) {}
};
