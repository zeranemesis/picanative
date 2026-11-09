// Replays a trace through the command processor and the memory operations,
// and prints a fingerprint of the GPU state at the end of each frame, to
// compare with the reference's (gfxref with GFXREF_STATE=file).
//
// usage: pn_state <trace.pntrace> [output file, stdout otherwise]

#include "picanative/host_memory.hpp"
#include "picanative/pica.hpp"
#include "picanative/trace.hpp"
#include "picanative/transfer.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>

using namespace picanative;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: pn_state <trace.pntrace> [output file]\n");
    return 2;
  }
  trace::Reader reader;
  if (!reader.open(argv[1])) {
    std::fprintf(stderr, "%s\n", reader.error().c_str());
    return 1;
  }
  FILE* out = argc > 2 ? std::fopen(argv[2], "w") : stdout;
  auto memory = std::make_unique<HostMemory>();
  pica::CommandProcessor gpu(*memory);
  const auto started = std::chrono::steady_clock::now();
  trace::Record r;
  uint32_t frames = 0;
  while (reader.next(r)) {
    switch (r.kind) {
    case trace::Memory:
      if (uint8_t* bytes = memory->map(r.word(0), static_cast<uint32_t>(r.payload.size() - 4)))
        std::memcpy(bytes, r.payload.data() + 4, r.payload.size() - 4);
      break;
    case trace::CommandList: gpu.run(r.word(0), r.word(1)); break;
    case trace::MemoryFill: memory_fill(*memory, r.word(0), r.word(1), r.word(2), r.word(3)); break;
    case trace::DisplayTransfer:
      display_transfer(*memory, r.word(0), r.word(1), r.word(2), r.word(3), r.word(4));
      break;
    case trace::TextureCopy: texture_copy(*memory, r.word(0), r.word(1), r.word(2), r.word(3), r.word(4)); break;
    case trace::Frame:
      std::fprintf(out, "%u %016llx\n", r.word(0), static_cast<unsigned long long>(pica::fingerprint(gpu.state())));
      ++frames;
      break;
    default: break;
    }
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  std::fprintf(stderr, "%u frames, %llu command lists, %llu draws, %llu immediate vertices, %.1f s\n", frames,
               static_cast<unsigned long long>(gpu.command_lists), static_cast<unsigned long long>(gpu.draws),
               static_cast<unsigned long long>(gpu.immediate_vertices), seconds);
  if (out != stdout) std::fclose(out);
  return 0;
}
