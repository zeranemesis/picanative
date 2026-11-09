// What a trace's draws use: for each piece of fragment state, the distinct
// values the draws set and how many draws set each. Tells which features
// the engine has to cover for a game.
//
// usage: pn_stats <trace.pntrace>

#include "picanative/host_memory.hpp"
#include "picanative/pica.hpp"
#include "picanative/trace.hpp"
#include "picanative/transfer.hpp"

#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace picanative;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: pn_stats <trace.pntrace>\n");
    return 2;
  }
  trace::Reader reader;
  if (!reader.open(argv[1])) {
    std::fprintf(stderr, "%s\n", reader.error().c_str());
    return 1;
  }
  auto memory = std::make_unique<HostMemory>();
  pica::CommandProcessor gpu(*memory);

  struct Item {
    const char* name;
    std::vector<uint32_t> regs;
    uint32_t mask;
  };
  const std::vector<Item> items = {
      {"culling 0x040", {0x040}, 3},
      {"depth map 0x06D", {0x06D}, 1},
      {"scissor mode 0x065", {0x065}, 3},
      {"texture units 0x080", {0x080}, 0xFFFFFFFF},
      {"texture 0 format 0x08E", {0x08E}, 0xF},
      {"texture 1 format 0x096", {0x096}, 0xF},
      {"texture 2 format 0x09E", {0x09E}, 0xF},
      {"texture 0 params 0x083", {0x083}, 0xFFFFFFFF},
      {"lighting enable 0x08F", {0x08F}, 1},
      {"fog/gas mode 0x0E0", {0x0E0}, 0xFFFFFFFF},
      {"TEV stage combiners", {0x0C0, 0x0C1, 0x0C2, 0x0C3, 0x0C8, 0x0C9, 0x0CA, 0x0CB, 0x0D0, 0x0D1, 0x0D2, 0x0D3,
                               0x0D8, 0x0D9, 0x0DA, 0x0DB, 0x0F0, 0x0F1, 0x0F2, 0x0F3, 0x0F8, 0x0F9, 0x0FA, 0x0FB},
       0xFFFFFFFF},
      {"TEV buffer 0x0E0 bits 8-13", {0x0E0}, 0x3F00},
      {"color operation 0x100", {0x100}, 0xFFFFFFFF},
      {"blend function 0x101", {0x101}, 0xFFFFFFFF},
      {"logic op 0x102", {0x102}, 0xF},
      {"alpha test 0x104", {0x104}, 0xFFFFFFFF},
      {"stencil test 0x105", {0x105}, 0xFFFFFFFF},
      {"stencil op 0x106", {0x106}, 0xFFFFFFFF},
      {"depth/color mask 0x107", {0x107}, 0xFFFFFFFF},
      {"depth format 0x116", {0x116}, 3},
      {"color format 0x117", {0x117}, 0xFFFFFFFF},
      {"framebuffer size 0x11E", {0x11E}, 0xFFFFFFFF},
      {"primitive 0x25E", {0x25E}, 0x300},
      {"vs output count 0x24A", {0x24A}, 0xF},
      {"output map 0x050-0x056", {0x050, 0x051, 0x052, 0x053, 0x054, 0x055, 0x056}, 0xFFFFFFFF},
  };
  std::vector<std::map<std::string, uint64_t>> seen(items.size());
  gpu.on_draw = [&](const pica::State& state, bool) {
    for (size_t i = 0; i < items.size(); ++i) {
      std::string key;
      char word[16];
      for (uint32_t reg : items[i].regs) {
        std::snprintf(word, sizeof word, "%08X ", state.regs[reg] & items[i].mask);
        key += word;
      }
      ++seen[i][key];
    }
  };

  trace::Record r;
  while (reader.next(r)) {
    switch (r.kind) {
    case trace::Memory:
      if (uint8_t* bytes = memory->map(r.word(0), static_cast<uint32_t>(r.payload.size() - 4)))
        std::memcpy(bytes, r.payload.data() + 4, r.payload.size() - 4);
      break;
    case trace::CommandList: gpu.run(r.word(0), r.word(1)); break;
    default: break;
    }
  }
  for (size_t i = 0; i < items.size(); ++i) {
    std::printf("%s: %zu distinct\n", items[i].name, seen[i].size());
    if (seen[i].size() <= 12)
      for (const auto& [key, n] : seen[i]) std::printf("    %s x%llu\n", key.c_str(), static_cast<unsigned long long>(n));
  }
  return 0;
}
