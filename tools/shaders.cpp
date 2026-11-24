// pn_shaders <translations file> <shader binary or folder of them>...
// Translates the vertex programs of a game's shader binaries (DVLB: .shbin
// files, or the blocks found in its data and code) ahead of time, into a file
// pn_load_shaders reads. Each program is also checked to keep its key when
// the rest of the unit holds garbage, as it does when the game uploads it
// over an earlier one.
#include "picanative/shadergen.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>

using namespace picanative;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: pn_shaders <translations file> <shader binary or folder>...\n");
    return 2;
  }
  const auto start = std::chrono::steady_clock::now();
  shadergen::Translations translations;
  uint32_t binaries = 0, programs = 0, geometry = 0, unstable = 0;
  std::mt19937 random(1);
  std::vector<std::filesystem::path> files;
  for (int i = 2; i < argc; ++i) {
    if (!std::filesystem::is_directory(argv[i])) files.emplace_back(argv[i]);
    else
      for (const auto& entry : std::filesystem::directory_iterator(argv[i])) files.push_back(entry.path());
  }
  std::sort(files.begin(), files.end());
  for (const auto& file : files) {
    std::ifstream in(file, std::ios::binary);
    const std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto units = shadergen::read_dvlb(data.data(), data.size(), geometry);
    if (units.empty()) std::fprintf(stderr, "pn_shaders: %s: no vertex shader\n", file.string().c_str());
    binaries += !units.empty();
    for (const pica::ShaderUnit& unit : units) {
      ++programs;
      const uint64_t key = shadergen::program_key(unit);
      // the same program over garbage: what it does not reach must not count
      const pica::ShaderUnit used = shadergen::reachable(unit);
      pica::ShaderUnit dirty = used;
      for (uint32_t pc = 0; pc < pica::kProgramSize; ++pc)
        if (used.program[pc] == 0 && unit.program[pc] == 0) dirty.program[pc] = random();
      for (uint32_t d = 0; d < pica::kDescriptorSize; ++d)
        if (used.descriptors[d] == 0 && unit.descriptors[d] == 0) dirty.descriptors[d] = random();
      if (shadergen::program_key(dirty) != key) ++unstable;
      if (!translations.count(key)) translations.emplace(key, shadergen::vertex_function(unit, "vsu"));
    }
  }
  if (!shadergen::save_translations(argv[1], translations)) {
    std::fprintf(stderr, "pn_shaders: cannot write %s\n", argv[1]);
    return 1;
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  std::printf("pn_shaders: %u binaries, %u vertex programs (%zu different) translated in %.0f ms, %u geometry "
              "shaders left out, %u keys changed by garbage\n",
              binaries, programs, translations.size(), ms, geometry, unstable);
  return unstable ? 1 : 0;
}
