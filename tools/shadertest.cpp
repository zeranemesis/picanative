// Runs the WGSL translation of each vertex shader on the GPU, as a compute
// shader over the real vertices of a trace, and compares its outputs with
// the reference's (.pnvs cases, extracted by gfxref with GFXREF_SHADERS).
//
// File "PNVS01\0\0": program (4096 words), descriptors (128), entry point,
// bools, ints (4 words), float uniforms (96 x 4 floats), output mask, vertex
// count n, inputs (n x 16 x 4 floats), reference outputs (n x 16 x 4).
//
// usage: pn_shadertest <directory of .pnvs> [--show name]

#include "picanative/gpu.hpp"
#include "picanative/shadergen.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace picanative;

namespace {

struct Case {
  pica::ShaderUnit unit;
  uint32_t ints[4] = {}, bools = 0, mask = 0, count = 0;
  std::vector<float> inputs, outputs;
};

bool load(const std::filesystem::path& path, Case& c) {
  std::ifstream in(path, std::ios::binary);
  char magic[8];
  if (!in.read(magic, 8) || std::memcmp(magic, "PNVS01", 6) != 0) return false;
  auto u32 = [&in] {
    uint32_t value = 0;
    in.read(reinterpret_cast<char*>(&value), 4);
    return value;
  };
  for (auto& word : c.unit.program) word = u32();
  for (auto& word : c.unit.descriptors) word = u32();
  c.unit.entry_point = u32();
  c.bools = u32();
  c.unit.bool_uniforms = static_cast<uint16_t>(c.bools);
  for (auto& i : c.ints) i = u32();
  for (auto& vector : c.unit.float_uniforms)
    for (auto& component : vector) component = std::bit_cast<float>(u32());
  c.mask = u32();
  c.count = u32();
  c.inputs.resize(size_t{c.count} * 64);
  c.outputs.resize(size_t{c.count} * 64);
  in.read(reinterpret_cast<char*>(c.inputs.data()), c.inputs.size() * 4);
  in.read(reinterpret_cast<char*>(c.outputs.data()), c.outputs.size() * 4);
  return static_cast<bool>(in);
}

std::string module_source(const Case& c) {
  std::string wgsl = shadergen::kVertexUniformsWgsl;
  wgsl += "@group(0) @binding(0) var<uniform> vsu: PicaVsUniforms;\n"
          "@group(0) @binding(1) var<storage, read> inputs: array<vec4f>;\n"
          "@group(0) @binding(2) var<storage, read_write> outputs: array<vec4f>;\n";
  wgsl += shadergen::vertex_function(c.unit, "vsu");
  wgsl += "@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {\n"
          "  let n = arrayLength(&inputs) / 16u;\n  if (id.x >= n) { return; }\n"
          "  var vin: array<vec4f, 16>;\n"
          "  for (var k = 0u; k < 16u; k++) { vin[k] = inputs[id.x * 16u + k]; }\n"
          "  let o = pica_vs(vin);\n"
          "  for (var k = 0u; k < 16u; k++) { outputs[id.x * 16u + k] = o[k]; }\n}\n";
  return wgsl;
}

wgpu::Buffer buffer(gpu::Context& g, const void* data, uint64_t size, wgpu::BufferUsage usage) {
  wgpu::BufferDescriptor d{};
  d.size = (size + 15) & ~uint64_t{15};
  d.usage = usage | wgpu::BufferUsage::CopyDst;
  wgpu::Buffer b = g.device.CreateBuffer(&d);
  if (data) g.queue.WriteBuffer(b, 0, data, size);
  return b;
}

// how far apart two floats are, relative to the reference's size
double error(float ours, float reference) {
  if (std::bit_cast<uint32_t>(ours) == std::bit_cast<uint32_t>(reference)) return 0;
  if (std::isnan(ours) && std::isnan(reference)) return 0;
  if (std::isinf(reference) || std::isnan(reference) || std::isnan(ours) || std::isinf(ours)) return INFINITY;
  return std::fabs(double{ours} - reference) / std::max(1.0, std::fabs(double{reference}));
}

// Only multiplications reading what rcp, rsq, ex2 or lg2 gave keep the
// hardware's 0 * inf = 0: rcp r0, v0; mul r1, c0, r0; mul r2, c0, v0; end.
bool check_exact_multiplications() {
  pica::ShaderUnit unit;
  unit.descriptors[0] = 0xF | 0x1Bu << 5 | 0x1Bu << 14 | 0x1Bu << 23; // xyzw everywhere, all written
  unit.program[0] = 0x0Eu << 26 | 0x10u << 21;
  unit.program[1] = 0x08u << 26 | 0x11u << 21 | 0x20u << 12 | 0x10u << 7;
  unit.program[2] = 0x08u << 26 | 0x12u << 21 | 0x20u << 12;
  unit.program[3] = 0x22u << 26;
  const std::string wgsl = shadergen::vertex_function(unit, "u");
  const bool ok = wgsl.find("pmul(cu(0).xyzw, r[0].xyzw)") != std::string::npos &&
                  wgsl.find("fmul(cu(0).xyzw, v[0].xyzw)") != std::string::npos;
  if (!ok) std::printf("FAIL exact multiplications:\n%s\n", wgsl.c_str());
  return ok;
}

} // namespace

int main(int argc, char** argv) {
  if (!check_exact_multiplications()) return 1;
  if (argc < 2) {
    std::fprintf(stderr, "usage: pn_shadertest <directory of .pnvs> [--show name]\n");
    return 2;
  }
  const std::string show = argc > 3 && std::string(argv[2]) == "--show" ? argv[3] : "";
  gpu::Context g;
  std::string message;
  if (!gpu::create(g, message)) {
    std::fprintf(stderr, "GPU: %s\n", message.c_str());
    return 1;
  }
  std::printf("GPU: %s (backend %d)\n", g.adapter_name.c_str(), static_cast<int>(g.backend));

  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(argv[1]))
    if (entry.path().extension() == ".pnvs") files.push_back(entry.path());
  std::sort(files.begin(), files.end());

  constexpr double kTolerance = 1e-4;
  int passed = 0, failed = 0;
  double worst = 0;
  for (const auto& path : files) {
    Case c;
    const std::string name = path.filename().string();
    if (!load(path, c)) {
      std::printf("FAIL %s: unreadable\n", name.c_str());
      ++failed;
      continue;
    }
    const std::string source = module_source(c);
    if (name == show) std::printf("%s\n", source.c_str());

    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = wgpu::StringView(source.data(), source.size());
    wgpu::ShaderModuleDescriptor md{};
    md.nextInChain = &wgsl;
    wgpu::ComputePipeline pipeline;
    const std::string compile_error = gpu::checked(g, [&] {
      const wgpu::ShaderModule module = g.device.CreateShaderModule(&md);
      wgpu::ComputePipelineDescriptor pd{};
      pd.compute.module = module;
      pd.compute.entryPoint = "main";
      pipeline = g.device.CreateComputePipeline(&pd);
    });
    if (!compile_error.empty()) {
      std::printf("FAIL %s: does not compile: %.400s\n", name.c_str(), compile_error.c_str());
      ++failed;
      continue;
    }

    std::vector<uint8_t> uniforms(96 * 16 + 32);
    for (int u = 0; u < 96; ++u) std::memcpy(&uniforms[u * 16], c.unit.float_uniforms[u].data(), 16);
    std::memcpy(&uniforms[96 * 16], c.ints, 16);
    std::memcpy(&uniforms[96 * 16 + 16], &c.bools, 4);
    const uint64_t bytes = c.inputs.size() * 4;
    const wgpu::Buffer ub = buffer(g, uniforms.data(), uniforms.size(), wgpu::BufferUsage::Uniform);
    const wgpu::Buffer ib = buffer(g, c.inputs.data(), bytes, wgpu::BufferUsage::Storage);
    const wgpu::Buffer ob = buffer(g, nullptr, bytes, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    wgpu::BindGroupEntry entries[3] = {};
    entries[0].binding = 0, entries[0].buffer = ub, entries[0].size = uniforms.size();
    entries[1].binding = 1, entries[1].buffer = ib, entries[1].size = bytes;
    entries[2].binding = 2, entries[2].buffer = ob, entries[2].size = bytes;
    wgpu::BindGroupDescriptor bd{};
    bd.layout = pipeline.GetBindGroupLayout(0);
    bd.entryCount = 3;
    bd.entries = entries;
    const wgpu::BindGroup group = g.device.CreateBindGroup(&bd);
    const wgpu::CommandEncoder encoder = g.device.CreateCommandEncoder();
    const wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, group);
    pass.DispatchWorkgroups((c.count + 63) / 64);
    pass.End();
    const wgpu::CommandBuffer commands = encoder.Finish();
    g.queue.Submit(1, &commands);
    const std::vector<uint8_t> raw = gpu::read_back(g, ob, bytes);
    const float* ours = reinterpret_cast<const float*>(raw.data());

    double case_worst = 0;
    size_t worst_at = 0;
    for (size_t i = 0; i < c.outputs.size(); ++i) {
      const double e = error(ours[i], c.outputs[i]);
      if (e > case_worst) case_worst = e, worst_at = i;
    }
    worst = std::max(worst, case_worst == INFINITY ? worst : case_worst);
    if (case_worst <= kTolerance) {
      ++passed;
    } else {
      ++failed;
      const size_t vertex = worst_at / 64, reg = worst_at % 64 / 4, component = worst_at % 4;
      std::printf("FAIL %s: vertex %zu o%zu.%c: %g, reference %g (%u vertices)\n", name.c_str(), vertex, reg,
                  "xyzw"[component], ours[worst_at], c.outputs[worst_at], c.count);
    }
  }
  std::printf("%d/%zu shaders within %g of the reference (largest error %.3g)\n", passed, files.size(), kTolerance,
              worst);
  return failed ? 1 : 0;
}
