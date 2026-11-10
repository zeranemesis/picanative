// Replays a trace with the engine: command lists drawn by the renderer, the
// memory operations by the exact CPU versions, and each screen the trace
// recorded compared with what the engine left in memory.
//
// usage: pn_render [--cached] <trace.pntrace> [directory for pictures: ours | recorded | difference]
//   --cached: surfaces stay on the GPU, the trace reporting the CPU's writes

#include "picanative/color.hpp"
#include "picanative/gpu.hpp"
#include "picanative/host_memory.hpp"
#include "picanative/renderer.hpp"
#include "picanative/trace.hpp"
#include "picanative/transfer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace picanative;

namespace {

struct Screen {
  uint32_t left[2] = {}, active = 0, stride = 0, format = 0;
};

// the screen as the LCD shows it, upright RGBA
std::vector<uint8_t> scan(Memory& memory, const Screen& s, uint32_t width, uint32_t height) {
  std::vector<uint8_t> out(size_t{width} * height * 4, 0);
  const uint32_t address = s.left[s.active ? 1 : 0];
  if (address == 0) return out;
  const ColorFormat format = transfer_format(s.format);
  const uint32_t bpp = bytes_per_pixel(format);
  const uint32_t stride = s.stride ? s.stride : height * bpp;
  const uint8_t* bytes = memory.map(address, stride * width);
  if (!bytes) return out;
  // stored column by column, each scanned from the bottom up
  for (uint32_t x = 0; x < width; ++x)
    for (uint32_t y = 0; y < height; ++y) {
      const Rgba c = decode(format, bytes + x * stride + (height - 1 - y) * bpp);
      std::memcpy(&out[(size_t{y} * width + x) * 4], c.data(), 4);
    }
  return out;
}

void write_ppm(const std::filesystem::path& path, uint32_t width, uint32_t height, const std::vector<uint8_t>& rgb) {
  FILE* f = std::fopen(path.string().c_str(), "wb");
  if (!f) return;
  std::fprintf(f, "P6\n%u %u\n255\n", width, height);
  std::fwrite(rgb.data(), 1, rgb.size(), f);
  std::fclose(f);
}

} // namespace

int main(int argc, char** argv) {
  const bool cached = argc > 1 && std::string(argv[1]) == "--cached";
  if (cached) --argc, ++argv;
  if (argc < 2) {
    std::fprintf(stderr, "usage: pn_render [--cached] <trace.pntrace> [directory for pictures]\n");
    return 2;
  }
  const std::filesystem::path pictures = argc > 2 ? argv[2] : "";
  if (!pictures.empty()) std::filesystem::create_directories(pictures);
  trace::Reader reader;
  if (!reader.open(argv[1])) {
    std::fprintf(stderr, "%s\n", reader.error().c_str());
    return 1;
  }
  gpu::Context g;
  std::string message;
  if (!gpu::create(g, message)) {
    std::fprintf(stderr, "GPU: %s\n", message.c_str());
    return 1;
  }
  std::printf("GPU: %s\n", g.adapter_name.c_str());
  auto memory = std::make_unique<HostMemory>();
  Renderer renderer(g, *memory, cached, cached);
  Screen screens[2];
  const auto started = std::chrono::steady_clock::now();
  uint32_t frame = 0, compared = 0;
  double total_psnr = 0;
  trace::Record r;
  while (reader.next(r)) {
    switch (r.kind) {
    case trace::Memory:
      renderer.before_cpu_write(r.word(0), static_cast<uint32_t>(r.payload.size() - 4));
      if (uint8_t* bytes = memory->map(r.word(0), static_cast<uint32_t>(r.payload.size() - 4)))
        std::memcpy(bytes, r.payload.data() + 4, r.payload.size() - 4);
      break;
    case trace::Framebuffer: {
      Screen& s = screens[r.word(0) & 1];
      s.active = r.word(1) & 1;
      s.left[s.active] = r.word(2);
      s.stride = r.word(4);
      s.format = r.word(5);
      break;
    }
    case trace::CommandList: renderer.command_list(r.word(0), r.word(1)); break;
    case trace::MemoryFill: renderer.memory_fill(r.word(0), r.word(1), r.word(2), r.word(3)); break;
    case trace::DisplayTransfer:
      renderer.display_transfer(r.word(0), r.word(1), r.word(2), r.word(3), r.word(4));
      break;
    case trace::TextureCopy: renderer.texture_copy(r.word(0), r.word(1), r.word(2), r.word(3), r.word(4)); break;
    case trace::Frame: frame = r.word(0); break;
    case trace::Screen: {
      const uint32_t index = r.word(0), width = r.word(1), height = r.word(2);
      const Screen& shown = screens[index & 1];
      renderer.sync(shown.left[shown.active ? 1 : 0], (shown.stride ? shown.stride : height * 4) * width);
      const std::vector<uint8_t> ours = scan(*memory, shown, width, height);
      const uint8_t* recorded = r.payload.data() + 12;
      double squared = 0;
      uint32_t off = 0;
      for (size_t p = 0; p < size_t{width} * height; ++p) {
        int worst = 0;
        for (int c = 0; c < 3; ++c) {
          const int e = std::abs(int{ours[p * 4 + c]} - int{recorded[p * 4 + c]});
          squared += double(e) * e;
          worst = std::max(worst, e);
        }
        off += worst > 16;
      }
      const double mse = squared / (double(width) * height * 3);
      const double psnr = mse == 0 ? 99.0 : 10 * std::log10(255.0 * 255.0 / mse);
      total_psnr += psnr;
      ++compared;
      std::printf("frame %5u screen %u: PSNR %5.1f dB, %5.2f%% pixels off by more than 16\n", frame, index, psnr,
                  100.0 * off / (double(width) * height));
      if (!pictures.empty()) {
        std::vector<uint8_t> rgb(size_t{width} * 3 * height * 3);
        for (uint32_t y = 0; y < height; ++y)
          for (uint32_t x = 0; x < width; ++x)
            for (int c = 0; c < 3; ++c) {
              const size_t p = (size_t{y} * width + x) * 4 + c;
              const size_t row = size_t{y} * width * 3 * 3;
              rgb[row + (x)*3 + c] = ours[p];
              rgb[row + (width + x) * 3 + c] = recorded[p];
              rgb[row + (2 * width + x) * 3 + c] = static_cast<uint8_t>(std::min(255, std::abs(int{ours[p]} - int{recorded[p]}) * 4));
            }
        char name[64];
        std::snprintf(name, sizeof name, "f%05u_%u.ppm", frame, index);
        write_ppm(pictures / name, width * 3, height, rgb);
      }
      break;
    }
    default: break;
    }
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  const auto& s = renderer.stats();
  std::printf("%u frames in %.1f s: %llu draws (%llu skipped), %llu pipelines, %llu shader errors, %llu textures, %llu passes\n",
              frame, seconds, (unsigned long long)s.draws, (unsigned long long)s.skipped, (unsigned long long)s.pipelines,
              (unsigned long long)s.shader_errors, (unsigned long long)s.textures_decoded, (unsigned long long)s.passes);
  std::printf("%llu distinct vertex shaders among the pipelines, %llu display transfers on the GPU\n",
              (unsigned long long)s.vertex_modules, (unsigned long long)s.gpu_transfers);
  std::printf("time: record %.1f s (textures %.1f, pipelines %.1f), encode %.1f, GPU wait + store %.1f (store %.1f), load %.1f\n",
              s.record, s.textures, s.pipelines_time, s.encode, s.wait, s.store, s.load);
  if (compared) std::printf("average PSNR over %u screens: %.1f dB\n", compared, total_psnr / compared);
  for (const std::string& e : renderer.errors) std::printf("--- shader error ---\n%.6000s\n", e.c_str());
  return 0;
}
