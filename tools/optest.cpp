// Checks the memory operations against cases extracted from a trace by the
// reference implementation (.pncase files): memory before, the operation,
// memory after as the reference left it.
//
// File: "PNCASE01", kind (5 fill, 6 display transfer, 7 texture copy), word
// count, words, zone count, then per zone: address, size, bytes before, bytes
// after.
//
// usage: pn_optest <directory of .pncase>

#include "picanative/host_memory.hpp"
#include "picanative/transfer.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace picanative;

namespace {


struct Zone {
  uint32_t address, size;
  std::vector<uint8_t> before, after;
};

struct Case {
  uint32_t kind;
  std::vector<uint32_t> words;
  std::vector<Zone> zones;
};

bool load(const std::filesystem::path& path, Case& c) {
  std::ifstream in(path, std::ios::binary);
  char magic[8];
  auto u32 = [&in] {
    uint32_t value = 0;
    in.read(reinterpret_cast<char*>(&value), 4);
    return value;
  };
  if (!in.read(magic, 8) || std::memcmp(magic, "PNCASE01", 8) != 0) return false;
  c.kind = u32();
  c.words.resize(u32());
  for (auto& word : c.words) word = u32();
  c.zones.resize(u32());
  for (auto& zone : c.zones) {
    zone.address = u32();
    zone.size = u32();
    zone.before.resize(zone.size);
    zone.after.resize(zone.size);
    in.read(reinterpret_cast<char*>(zone.before.data()), zone.size);
    in.read(reinterpret_cast<char*>(zone.after.data()), zone.size);
  }
  return static_cast<bool>(in);
}

const char* name(uint32_t kind) {
  return kind == 5 ? "MemoryFill" : kind == 6 ? "DisplayTransfer" : "TextureCopy";
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: pn_optest <directory of .pncase>\n");
    return 2;
  }
  auto owned = std::make_unique<HostMemory>();
  HostMemory& memory = *owned;
  std::map<std::string, std::pair<int, int>> totals; // family -> passed, total
  int failed = 0;
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(argv[1]))
    if (entry.path().extension() == ".pncase") files.push_back(entry.path());
  std::sort(files.begin(), files.end());

  for (const auto& path : files) {
    Case c;
    if (!load(path, c)) {
      std::printf("%s: unreadable\n", path.filename().string().c_str());
      ++failed;
      continue;
    }
    for (const auto& zone : c.zones)
      if (uint8_t* bytes = memory.map(zone.address, zone.size)) std::memcpy(bytes, zone.before.data(), zone.size);
    const auto& w = c.words;
    if (c.kind == 5) memory_fill(memory, w[0], w[1], w[2], w[3]);
    if (c.kind == 6) display_transfer(memory, w[0], w[1], w[2], w[3], w[4]);
    if (c.kind == 7) texture_copy(memory, w[0], w[1], w[2], w[3], w[4]);

    char family[64];
    std::snprintf(family, sizeof family, "%s 0x%08X", name(c.kind), c.kind == 7 ? w[3] ^ w[4] << 1 : w[c.kind == 5 ? 3 : 4]);
    auto& total = totals[family];
    ++total.second;
    size_t differing = 0, first = SIZE_MAX;
    for (const auto& zone : c.zones) {
      const uint8_t* bytes = memory.map(zone.address, zone.size);
      for (uint32_t i = 0; bytes && i < zone.size; ++i)
        if (bytes[i] != zone.after[i]) {
          if (first == SIZE_MAX) first = zone.address + i;
          ++differing;
        }
      // the next case starts from clean memory
      if (uint8_t* clean = memory.map(zone.address, zone.size)) std::memset(clean, 0, zone.size);
    }
    if (differing == 0) {
      ++total.first;
    } else {
      ++failed;
      std::printf("FAIL %s: %zu bytes differ, first at 0x%08zX\n", path.filename().string().c_str(), differing, first);
    }
  }
  for (const auto& [family, total] : totals)
    std::printf("%-28s %d/%d\n", family.c_str(), total.first, total.second);
  std::printf("%zu cases, %d failed\n", files.size(), failed);
  return failed ? 1 : 0;
}
