#include "picanative/shadergen.hpp"

#include <cstring>
#include <fstream>

// Shader binaries as the SDK builds them (3dbrew, "SHBIN"): a DVLB header
// listing its shaders (DVLE), then the program they share (DVLP).
namespace picanative::shadergen {
namespace {

uint32_t u32(const uint8_t* data, size_t size, size_t at, bool& ok) {
  if (at + 4 > size) {
    ok = false;
    return 0;
  }
  uint32_t v;
  std::memcpy(&v, data + at, 4);
  return v;
}

constexpr char kMagic[8] = {'P', 'N', 'V', 'S', 'W', 'G', 'S', 'L'};

} // namespace

std::vector<pica::ShaderUnit> read_dvlb(const uint8_t* data, size_t size, uint32_t& geometry) {
  std::vector<pica::ShaderUnit> units;
  bool ok = size >= 8 && std::memcmp(data, "DVLB", 4) == 0;
  const uint32_t shaders = u32(data, size, 4, ok);
  if (!ok || shaders > 256) return units;
  const size_t dvlp = 8 + size_t{shaders} * 4;
  if (dvlp + 0x18 > size || std::memcmp(data + dvlp, "DVLP", 4) != 0) return units;
  const size_t code = dvlp + u32(data, size, dvlp + 0x08, ok), words = u32(data, size, dvlp + 0x0C, ok);
  const size_t descriptors = dvlp + u32(data, size, dvlp + 0x10, ok), count = u32(data, size, dvlp + 0x14, ok);
  if (!ok || code + words * 4 > size || descriptors + count * 8 > size) return units;
  pica::ShaderUnit unit;
  for (size_t i = 0; i < words && i < pica::kProgramSize; ++i) unit.program[i] = u32(data, size, code + i * 4, ok);
  for (size_t i = 0; i < count && i < pica::kDescriptorSize; ++i)
    unit.descriptors[i] = u32(data, size, descriptors + i * 8, ok);
  for (uint32_t s = 0; s < shaders; ++s) {
    const size_t dvle = u32(data, size, 8 + s * 4, ok);
    if (!ok || dvle + 0x0C > size || std::memcmp(data + dvle, "DVLE", 4) != 0) continue;
    if (data[dvle + 6] != 0) { // 1: a geometry shader
      ++geometry;
      continue;
    }
    unit.entry_point = u32(data, size, dvle + 0x08, ok) & (pica::kProgramSize - 1);
    units.push_back(unit);
  }
  return units;
}

// the file: the magic, kVersion's length and text, then each translation's
// key, length and text
bool save_translations(const std::string& path, const Translations& translations) {
  std::ofstream out(path, std::ios::binary);
  auto word = [&out](uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
  out.write(kMagic, sizeof kMagic);
  word(static_cast<uint32_t>(std::strlen(kVersion)));
  out << kVersion;
  for (const auto& [key, text] : translations) {
    out.write(reinterpret_cast<const char*>(&key), 8);
    word(static_cast<uint32_t>(text.size()));
    out << text;
  }
  return static_cast<bool>(out);
}

bool load_translations(const std::string& path, Translations& translations, std::string& error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "no such file";
    return false;
  }
  char magic[sizeof kMagic];
  uint32_t length = 0;
  if (!in.read(magic, sizeof magic) || std::memcmp(magic, kMagic, sizeof kMagic) != 0 ||
      !in.read(reinterpret_cast<char*>(&length), 4) || length > 256) {
    error = "not a translation file";
    return false;
  }
  std::string version(length, '\0');
  in.read(version.data(), length);
  if (version != kVersion) {
    error = "made by another build of the translator (" + version + ", this one " + kVersion + ")";
    return false;
  }
  uint64_t key;
  while (in.read(reinterpret_cast<char*>(&key), 8) && in.read(reinterpret_cast<char*>(&length), 4)) {
    std::string text(length, '\0');
    if (!in.read(text.data(), length)) {
      error = "cut short";
      return false;
    }
    translations[key] = std::move(text);
  }
  return true;
}

} // namespace picanative::shadergen
