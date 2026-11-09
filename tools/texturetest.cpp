// Compares the texture decoder with the reference's on .pntx cases (every
// format, several sizes, pseudo-random bytes): "PNTX01\0\0", format, width,
// height, length, bytes, then width x height x 4 bytes of RGBA.
//
// usage: pn_texturetest <directory of .pntx>

#include "picanative/texture.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: pn_texturetest <directory of .pntx>\n");
    return 2;
  }
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(argv[1]))
    if (entry.path().extension() == ".pntx") files.push_back(entry.path());
  std::sort(files.begin(), files.end());
  int failed = 0;
  std::vector<uint8_t> ours;
  for (const auto& path : files) {
    std::ifstream in(path, std::ios::binary);
    char magic[8];
    uint32_t header[4];
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(header), sizeof header);
    const uint32_t format = header[0], width = header[1], height = header[2], length = header[3];
    std::vector<uint8_t> data(length), expected(size_t{width} * height * 4);
    in.read(reinterpret_cast<char*>(data.data()), length);
    in.read(reinterpret_cast<char*>(expected.data()), expected.size());
    if (!in || std::memcmp(magic, "PNTX01", 6) != 0) {
      std::printf("FAIL %s: unreadable\n", path.filename().string().c_str());
      ++failed;
      continue;
    }
    picanative::texture::decode(format, data.data(), data.size(), width, height, ours);
    size_t differing = 0, first = SIZE_MAX;
    for (size_t i = 0; i < ours.size(); ++i)
      if (ours[i] != expected[i]) {
        if (first == SIZE_MAX) first = i;
        ++differing;
      }
    if (differing) {
      ++failed;
      const size_t texel = first / 4;
      std::printf("FAIL %s: %zu bytes differ, first texel (%zu, %zu) channel %zu: %u, reference %u\n",
                  path.filename().string().c_str(), differing, texel % width, texel / width, first % 4, ours[first],
                  expected[first]);
    }
  }
  std::printf("%zu texture cases, %d failed\n", files.size(), failed);
  return failed ? 1 : 0;
}
