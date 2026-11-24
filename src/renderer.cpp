#include "picanative/renderer.hpp"

#include "picanative/color.hpp"
#include "picanative/shadergen.hpp"
#include "picanative/texture.hpp"
#include "picanative/transfer.hpp"
#include "picanative/vertex.hpp"

#include <algorithm>
#include <chrono>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace picanative {
namespace {

constexpr uint32_t kTextureUnits[3] = {0x081, 0x091, 0x099};
constexpr uint32_t kVsUniformsSize = 96 * 16 + 32;
constexpr uint32_t kUniformAlign = 256;

// adds the time from its creation to the end of its scope to a counter
struct Timed {
  double& total;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  ~Timed() { total += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }
};


uint32_t align_up(uint32_t value, uint32_t to) { return (value + to - 1) / to * to; }

float f24(uint32_t raw) { return pica::decode_float24(raw); }

void unpack(uint32_t value, float out[4]) {
  for (int c = 0; c < 4; ++c) out[c] = static_cast<float>(value >> (8 * c) & 0xFF) / 255.0f;
}

// a float of `mantissa` and `exponent` bits with a sign above them
float pica_float(uint32_t bits, uint32_t mantissa, uint32_t exponent) {
  const uint32_t width = mantissa + exponent + 1;
  const uint32_t sign = (bits >> (mantissa + exponent) & 1) << 31;
  if ((bits & ((1u << (width - 1)) - 1)) == 0) return std::bit_cast<float>(sign);
  uint32_t biased = bits >> mantissa & ((1u << exponent) - 1);
  biased = biased == (1u << exponent) - 1 ? 255 : biased + 128 - (1u << (exponent - 1));
  return std::bit_cast<float>(sign | (bits & ((1u << mantissa) - 1)) << (23 - mantissa) | biased << 23);
}

void light_color(uint32_t v, float out[4]) {
  out[0] = static_cast<float>(v >> 20 & 0x3FF) / 255.0f;
  out[1] = static_cast<float>(v >> 10 & 0x3FF) / 255.0f;
  out[2] = static_cast<float>(v & 0x3FF) / 255.0f;
  out[3] = 0;
}

float fixed13(uint32_t v) { return static_cast<float>(static_cast<int32_t>((v & 0x1FFF) << 19) >> 19) / 2047.0f; }

uint64_t fnv(const uint8_t* data, size_t size, uint64_t hash = 0xCBF29CE484222325ull) {
  // whole words where possible, the hash only has to tell contents apart. Four lanes side by side: each
  // multiplication waits only on its own lane's last, so the processor runs four at once (textures are hashed
  // every command list: this was most of the time spent on them)
  size_t i = 0;
  if (size >= 64) {
    uint64_t lane[4] = {hash, hash ^ 0x9E3779B97F4A7C15ull, hash ^ 0xBF58476D1CE4E5B9ull, hash ^ 0x94D049BB133111EBull};
    for (; i + 32 <= size; i += 32) {
      uint64_t w[4];
      std::memcpy(w, data + i, 32);
      for (int k = 0; k < 4; ++k) lane[k] = (lane[k] ^ w[k]) * 0x100000001B3ull;
    }
    hash = lane[0];
    for (int k = 1; k < 4; ++k) hash = (hash ^ lane[k] ^ (lane[k] >> 29)) * 0x100000001B3ull;
  }
  for (; i + 8 <= size; i += 8) {
    uint64_t word;
    std::memcpy(&word, data + i, 8);
    hash = (hash ^ word) * 0x100000001B3ull;
  }
  for (; i < size; ++i) hash = (hash ^ data[i]) * 0x100000001B3ull;
  return hash;
}

wgpu::CompareFunction compare(uint32_t function) {
  static const wgpu::CompareFunction map[8] = {
      wgpu::CompareFunction::Never,   wgpu::CompareFunction::Always,    wgpu::CompareFunction::Equal,
      wgpu::CompareFunction::NotEqual, wgpu::CompareFunction::Less,     wgpu::CompareFunction::LessEqual,
      wgpu::CompareFunction::Greater, wgpu::CompareFunction::GreaterEqual};
  return map[function & 7];
}

wgpu::StencilOperation stencil_op(uint32_t op) {
  static const wgpu::StencilOperation map[8] = {
      wgpu::StencilOperation::Keep,           wgpu::StencilOperation::Zero,
      wgpu::StencilOperation::Replace,        wgpu::StencilOperation::IncrementClamp,
      wgpu::StencilOperation::DecrementClamp, wgpu::StencilOperation::Invert,
      wgpu::StencilOperation::IncrementWrap,  wgpu::StencilOperation::DecrementWrap};
  return map[op & 7];
}

wgpu::BlendFactor blend_factor(uint32_t raw) {
  switch (raw & 0xF) {
  case 0: return wgpu::BlendFactor::Zero;
  case 2: return wgpu::BlendFactor::Src;
  case 3: return wgpu::BlendFactor::OneMinusSrc;
  case 4: return wgpu::BlendFactor::Dst;
  case 5: return wgpu::BlendFactor::OneMinusDst;
  case 6: return wgpu::BlendFactor::SrcAlpha;
  case 7: return wgpu::BlendFactor::OneMinusSrcAlpha;
  case 8: return wgpu::BlendFactor::DstAlpha;
  case 9: return wgpu::BlendFactor::OneMinusDstAlpha;
  // shortcut: WebGPU has one constant: constant alpha reads the constant color, no game seen uses it yet
  case 10: case 12: return wgpu::BlendFactor::Constant;
  case 11: case 13: return wgpu::BlendFactor::OneMinusConstant;
  case 14: return wgpu::BlendFactor::SrcAlphaSaturated;
  default: return wgpu::BlendFactor::One;
  }
}

wgpu::BlendOperation blend_op(uint32_t raw) {
  switch (raw & 7) {
  case 1: return wgpu::BlendOperation::Subtract;
  case 2: return wgpu::BlendOperation::ReverseSubtract;
  case 3: return wgpu::BlendOperation::Min;
  case 4: return wgpu::BlendOperation::Max;
  default: return wgpu::BlendOperation::Add;
  }
}

wgpu::Buffer make_buffer(gpu::Context& g, const void* data, size_t size, wgpu::BufferUsage usage) {
  wgpu::BufferDescriptor d{};
  d.size = align_up(static_cast<uint32_t>(std::max<size_t>(size, 16)), 16);
  d.usage = usage | wgpu::BufferUsage::CopyDst;
  const wgpu::Buffer buffer = g.device.CreateBuffer(&d);
  if (data && size) g.queue.WriteBuffer(buffer, 0, data, align_up(static_cast<uint32_t>(size), 4));
  return buffer;
}

wgpu::ShaderModule module(gpu::Context& g, const std::string& source) {
  wgpu::ShaderSourceWGSL wgsl{};
  wgsl.code = wgpu::StringView(source.data(), source.size());
  wgpu::ShaderModuleDescriptor d{};
  d.nextInChain = &wgsl;
  return g.device.CreateShaderModule(&d);
}

constexpr const char* kDepthUploadWgsl = R"(
@group(0) @binding(0) var<storage, read> depth: array<u32>;
@group(0) @binding(1) var<uniform> size: vec4u;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @builtin(frag_depth) f32 {
  let x = u32(p.x);
  let y = u32(p.y);
  return f32(depth[y * size.x + x] & 0xFFFFFFu) / 16777215.0;
}
)";

// DisplayTransfer on the GPU: each invocation makes a word of the output,
// byte for byte as transfer.cpp encodes it (little endian, linear lines).
// p: output width, height (output pixels), format | flip << 8 | input format << 9 | (scale x - 1) << 12 |
// (scale y - 1) << 13 | first input line << 16 | internal scale << 26, words. A downscale averages each 2x1 or
// 2x2 block of the stored colors, rounding to nearest, as transfer.cpp.
const char* kTransferWgsl = R"(
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> p: vec4u;
// the color as the surface's format in memory holds it (bits dropped, then widened back as the
// transfer engine reads it), so that the output is what transfer.cpp makes from memory
fn stored(c: vec4u, format: u32) -> vec4u {
  switch format {
    case 1u: { return vec4u(c.rgb, 255u); }
    case 2u: {
      let r = c.r >> 3u; let g = c.g >> 2u; let b = c.b >> 3u;
      return vec4u((r << 3u) | (r >> 2u), (g << 2u) | (g >> 4u), (b << 3u) | (b >> 2u), 255u);
    }
    case 3u: {
      let q = c.rgb >> vec3u(3u);
      return vec4u((q << vec3u(3u)) | (q >> vec3u(2u)), select(0u, 255u, c.a >= 128u));
    }
    case 4u: { let q = c >> vec4u(4u); return (q << vec4u(4u)) | q; }
    default: { return c; }
  }
}
fn fetched(x: u32, y: u32) -> vec4u {
  return stored(vec4u(round(textureLoad(source, vec2u(x, y), 0) * 255.0)), (p.z >> 9u) & 7u);
}
fn encoded(c: vec4u, format: u32) -> u32 {
  switch format {
    case 0u: { return c.a | (c.b << 8u) | (c.g << 16u) | (c.r << 24u); }
    case 1u: { return c.b | (c.g << 8u) | (c.r << 16u); }
    case 2u: { return ((c.r >> 3u) << 11u) | ((c.g >> 2u) << 5u) | (c.b >> 3u); }
    case 3u: { return ((c.r >> 3u) << 11u) | ((c.g >> 3u) << 6u) | ((c.b >> 3u) << 1u) | select(0u, 1u, c.a >= 128u); }
    default: { return ((c.r >> 4u) << 12u) | ((c.g >> 4u) << 8u) | ((c.b >> 4u) << 4u) | (c.a >> 4u); }
  }
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.y * 65535u * 64u + id.x;
  if (i >= p.w) { return; }
  let format = p.z & 0xFFu;
  let flip = ((p.z >> 8u) & 1u) != 0u;
  let first = (p.z >> 16u) & 1023u;
  let scale = p.z >> 26u; // one sample per block of the scaled surface
  let sx = ((p.z >> 12u) & 1u) + 1u;
  let sy = ((p.z >> 13u) & 1u) + 1u;
  let bpp = select(2u, select(3u, 4u, format == 0u), format <= 1u);
  var word = 0u;
  for (var b = 0u; b < 4u; b++) {
    let at = i * 4u + b;
    let pixel = at / bpp;
    if (pixel >= p.x * p.y) { break; }
    let line = pixel / p.x;
    let y = select(line, p.y - 1u - line, flip); // the copied line, from the first
    var sum = vec4u(0u);
    for (var dy = 0u; dy < sy; dy++) {
      for (var dx = 0u; dx < sx; dx++) {
        sum += fetched(((pixel % p.x) * sx + dx) * scale, (first + y * sy + dy) * scale);
      }
    }
    let count = sx * sy;
    let c = (sum + vec4u(count / 2u)) / vec4u(count);
    word |= ((encoded(c, format) >> ((at % bpp) * 8u)) & 0xFFu) << (b * 8u);
  }
  output[i] = word;
}
)";

} // namespace

struct Renderer::Surface {
  uint32_t address, width, height;
  bool depth;
  // a color surface's format in memory (the GPU holds it as RGBA8); a depth surface's, register 0x116:
  // 0 D16, 2 D24, 3 D24S8 (the GPU holds all three as Depth32FloatStencil8)
  ColorFormat format = ColorFormat::RGBA8;
  uint32_t depth_format = 3;
  uint32_t pixel_bytes() const { return depth ? (depth_format == 0 ? 2 : depth_format == 2 ? 3 : 4) : bytes_per_pixel(format); }
  uint32_t bytes() const { return width * height * pixel_bytes(); }
  wgpu::Texture texture;
  bool loaded = false; // holds what memory had, in this command list
  bool drawn = false;  // drawn into since, to write back
  // memory as it was when drawing began: the console's GPU writes at once, this one when the range is read; bytes the
  // game changed meanwhile (the buffer freed and given to other data) are its own, the write-back leaves them
  std::vector<uint8_t> before;
  bool pending = false; // told to on_pending
  wgpu::TextureView view; // to sample it as a texture (render to texture), made at the first such use
  uint64_t version = 0;    // drawn into: one more (a depth buffer read as a color texture is made again)
  // a D24S8 depth buffer read as an RGBA8 texture: its words as the console's memory has them, on the GPU
  wgpu::Texture as_color;
  wgpu::TextureView as_color_view;
  uint64_t as_color_version = ~0ull;
};

struct Renderer::Texture {
  wgpu::Texture texture;
  wgpu::TextureView view;
};

struct Renderer::PendingDraw {
  wgpu::RenderPipeline pipeline;
  Surface* color;
  Surface* depth;
  bool depth_upload = false; // not a draw: load the depth surface from the arena
  bool clear = false;        // not a draw: fill color (or depth, stencil_reference) on the GPU
  double clear_color[4] = {};
  uint32_t vs_offset, draw_offset, vertex_count;
  wgpu::TextureView views[3];
  wgpu::Sampler samplers[3];
  uint32_t scissor[4];
  uint32_t stencil_reference;
  float blend_constant[4];
  uint32_t upload_offset = 0; // depth upload: where its D24 words start in the arena
};

Renderer::Renderer(gpu::Context& gpu, Memory& memory, bool cache_surfaces, bool trust_cpu_writes)
    : gpu_(gpu), memory_(memory), cached_(cache_surfaces), trusted_(trust_cpu_writes), processor_(memory) {
  processor_.on_draw = [this](const pica::State& state, bool indexed) { record(state, indexed); };
  processor_.on_immediate = [this](const pica::State& state, const std::vector<pica::CommandProcessor::Vertex>& v) {
    record(state, false, &v);
  };

  wgpu::BindGroupLayoutEntry entries[4] = {};
  for (int i = 0; i < 2; ++i) {
    entries[i].binding = i;
    entries[i].visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    entries[i].buffer.type = wgpu::BufferBindingType::Uniform;
    entries[i].buffer.hasDynamicOffset = true;
  }
  for (int i = 2; i < 4; ++i) {
    entries[i].binding = i;
    entries[i].visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    entries[i].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  }
  wgpu::BindGroupLayoutDescriptor ld{};
  ld.entryCount = 4;
  ld.entries = entries;
  layout_ = gpu_.device.CreateBindGroupLayout(&ld);
  // the textures and their samplers, a group shared by the draws that use the same ones
  wgpu::BindGroupLayoutEntry te[6] = {};
  for (int i = 0; i < 3; ++i) {
    te[i].binding = i;
    te[i].visibility = wgpu::ShaderStage::Fragment;
    te[i].texture.sampleType = wgpu::TextureSampleType::Float;
    te[i].texture.viewDimension = wgpu::TextureViewDimension::e2D;
    te[3 + i].binding = 3 + i;
    te[3 + i].visibility = wgpu::ShaderStage::Fragment;
    te[3 + i].sampler.type = wgpu::SamplerBindingType::Filtering;
  }
  ld.entryCount = 6;
  ld.entries = te;
  texture_layout_ = gpu_.device.CreateBindGroupLayout(&ld);
  const wgpu::BindGroupLayout layouts[2] = {layout_, texture_layout_};
  wgpu::PipelineLayoutDescriptor pld{};
  pld.bindGroupLayoutCount = 2;
  pld.bindGroupLayouts = layouts;
  pipeline_layout_ = gpu_.device.CreatePipelineLayout(&pld);

  // a unit that is on but points at nothing samples transparent black
  wgpu::TextureDescriptor td{};
  td.size = {1, 1, 1};
  td.format = wgpu::TextureFormat::RGBA8Unorm;
  td.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
  const wgpu::Texture empty = gpu_.device.CreateTexture(&td);
  const uint32_t zero = 0;
  wgpu::TexelCopyTextureInfo dst{};
  dst.texture = empty;
  wgpu::TexelCopyBufferLayout layout{};
  layout.bytesPerRow = 4;
  wgpu::Extent3D one{1, 1, 1};
  gpu_.queue.WriteTexture(&dst, &zero, 4, &layout, &one);
  empty_texture_ = empty.CreateView();

  // the pass that loads a depth surface from memory
  wgpu::BindGroupLayoutEntry de[2] = {};
  de[0].binding = 0, de[0].visibility = wgpu::ShaderStage::Fragment;
  de[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  de[1].binding = 1, de[1].visibility = wgpu::ShaderStage::Fragment;
  de[1].buffer.type = wgpu::BufferBindingType::Uniform;
  wgpu::BindGroupLayoutDescriptor dld{};
  dld.entryCount = 2;
  dld.entries = de;
  depth_upload_layout_ = gpu_.device.CreateBindGroupLayout(&dld);
  wgpu::PipelineLayoutDescriptor dpl{};
  dpl.bindGroupLayoutCount = 1;
  dpl.bindGroupLayouts = &depth_upload_layout_;
  const wgpu::ShaderModule dm = module(gpu_, kDepthUploadWgsl);
  wgpu::RenderPipelineDescriptor pd{};
  pd.layout = gpu_.device.CreatePipelineLayout(&dpl);
  pd.vertex.module = dm;
  pd.vertex.entryPoint = "vs";
  wgpu::FragmentState fs{};
  fs.module = dm;
  fs.entryPoint = "fs";
  fs.targetCount = 0;
  pd.fragment = &fs;
  wgpu::DepthStencilState ds{};
  ds.format = wgpu::TextureFormat::Depth32FloatStencil8;
  ds.depthWriteEnabled = wgpu::OptionalBool::True;
  ds.depthCompare = wgpu::CompareFunction::Always;
  pd.depthStencil = &ds;
  depth_upload_ = gpu_.device.CreateRenderPipeline(&pd);

  wgpu::ComputePipelineDescriptor cd{};
  cd.compute.module = module(gpu_, kTransferWgsl);
  cd.compute.entryPoint = "main";
  transfer_pipeline_ = gpu_.device.CreateComputePipeline(&cd);
}

Renderer::~Renderer() { deliver(); }

void Renderer::command_list(uint32_t address, uint32_t size) {
  // pipelines finished compiling since
  if (async_) {
    // pipelines that finished compiling since, without waiting for the others
    for (auto it = futures_.begin(); it != futures_.end();)
      if (gpu_.instance.WaitAny(*it, 0) == wgpu::WaitStatus::Success) it = futures_.erase(it);
      else ++it;
  }
  // the CPU may have changed textures since the last list unless the host
  // reports its writes; within a list it cannot. VRAM is the exception: what
  // writes it (DMA, fills, transfers, copies, the GPU) goes through the engine,
  // which forgets the textures there; so a VRAM texture stays known, unless a
  // surface overlaps it (the GPU may draw into it again in this list).
  // shortcut: the CPU writing VRAM directly goes unseen; no game seen does,
  // keep this clear() for one that does (PN_TRUST_CPU_WRITES aside)
  if (!trusted_) {
    for (auto it = known_.begin(); it != known_.end();) {
      const Known& k = it->second;
      bool keep = k.address >= 0x18000000 && k.address + k.bytes <= 0x18600000;
      for (const auto& [key, s] : surfaces_)
        if (keep && k.address < s->address + s->bytes() && s->address < k.address + k.bytes) keep = false;
      it = keep ? std::next(it) : known_.erase(it);
    }
  }
  processor_.run(address, size);
  if (cached_) {
    flush(false);
    return;
  }
  // safe mode: memory has everything, and is where the next list starts from
  flush(true);
  for (auto& [key, s] : surfaces_) s->loaded = false;
}

Renderer::Surface& Renderer::surface(uint32_t address, uint32_t width, uint32_t height, bool depth, uint32_t format) {
  auto& slot = surfaces_[{address, width, height, depth, format}];
  if (!slot) {
    slot = std::make_unique<Surface>();
    slot->address = address, slot->width = width, slot->height = height, slot->depth = depth;
    if (depth) slot->depth_format = format;
    else slot->format = static_cast<ColorFormat>(format);
    wgpu::TextureDescriptor td{};
    td.size = {width * scale_, height * scale_, 1};
    td.format = depth ? wgpu::TextureFormat::Depth32FloatStencil8 : wgpu::TextureFormat::RGBA8Unorm;
    td.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst;
    if (!depth) td.usage |= wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::StorageBinding; // GPU transfers
    else td.usage |= wgpu::TextureUsage::TextureBinding; // read as a color texture (depth_as_color)
    slot->texture = gpu_.device.CreateTexture(&td);
  }
  return *slot;
}

// what memory holds in a surface, before this command list draws into it
void Renderer::load(Surface& s) {
  if (s.loaded) return;
  deliver(s.address, s.bytes());
  Timed timed{stats_.load};
  s.loaded = true;
  const uint32_t bpp = s.pixel_bytes();
  const uint8_t* bytes = memory_.map(s.address, s.width * s.height * bpp);
  const uint32_t n = scale_, sw = s.width * n, sh = s.height * n;
  // each of the console's pixels, repeated n x n
  auto grow = [&](auto& small) {
    if (n == 1) return small;
    std::remove_reference_t<decltype(small)> big(size_t{sw} * sh * (small.size() / (size_t{s.width} * s.height)));
    const size_t unit = small.size() / (size_t{s.width} * s.height);
    for (uint32_t y = 0; y < sh; ++y)
      for (uint32_t x = 0; x < sw; ++x)
        std::memcpy(&big[(size_t{y} * sw + x) * unit], &small[(size_t{y / n} * s.width + x / n) * unit],
                    unit * sizeof small[0]);
    return big;
  };
  if (!s.depth) {
    std::vector<uint8_t> rgba(size_t{s.width} * s.height * 4, 0);
    if (bytes)
      for (uint32_t y = 0; y < s.height; ++y)
        for (uint32_t x = 0; x < s.width; ++x) {
          const Rgba c = decode(s.format, bytes + tiled_offset(x, y, s.width, bpp));
          std::memcpy(&rgba[(size_t{y} * s.width + x) * 4], c.data(), 4);
        }
    const auto big = grow(rgba);
    wgpu::TexelCopyTextureInfo dst{};
    dst.texture = s.texture;
    wgpu::TexelCopyBufferLayout layout{};
    layout.bytesPerRow = sw * 4;
    layout.rowsPerImage = sh;
    wgpu::Extent3D extent{sw, sh, 1};
    gpu_.queue.WriteTexture(&dst, big.data(), big.size(), &layout, &extent);
    return;
  }
  // the stencil straight into its aspect (0 without one), the depth by a pass, as 24 bits: D16 widened (v * 0xFFFFFF / 0xFFFF)
  std::vector<uint8_t> stencil(size_t{s.width} * s.height, 0);
  std::vector<uint32_t> depth(size_t{s.width} * s.height, 0);
  if (bytes)
    for (uint32_t y = 0; y < s.height; ++y)
      for (uint32_t x = 0; x < s.width; ++x) {
        uint32_t word = 0;
        std::memcpy(&word, bytes + tiled_offset(x, y, s.width, bpp), bpp);
        const size_t at = size_t{y} * s.width + x;
        if (bpp == 2) depth[at] = word << 8 | word >> 8;
        else depth[at] = word & 0xFFFFFF;
        if (bpp == 4) stencil[at] = static_cast<uint8_t>(word >> 24);
      }
  const auto big_stencil = grow(stencil);
  const auto big_depth = grow(depth);
  wgpu::TexelCopyTextureInfo dst{};
  dst.texture = s.texture;
  dst.aspect = wgpu::TextureAspect::StencilOnly;
  wgpu::TexelCopyBufferLayout layout{};
  layout.bytesPerRow = sw;
  layout.rowsPerImage = sh;
  wgpu::Extent3D extent{sw, sh, 1};
  gpu_.queue.WriteTexture(&dst, big_stencil.data(), big_stencil.size(), &layout, &extent);
  PendingDraw upload{};
  upload.depth_upload = true;
  upload.depth = &s;
  // a storage binding starts on 256 bytes
  arena_.resize(align_up(static_cast<uint32_t>(arena_.size()), 256), 0);
  upload.upload_offset = static_cast<uint32_t>(arena_.size());
  arena_.resize(arena_.size() + big_depth.size() * 4);
  std::memcpy(arena_.data() + upload.upload_offset, big_depth.data(), big_depth.size() * 4);
  pending_.push_back(upload);
}

void Renderer::remember(Surface& s) {
  const uint8_t* bytes = memory_.map(s.address, s.bytes());
  if (bytes) s.before.assign(bytes, bytes + s.bytes());
  else s.before.clear();
  if (on_pending && !s.pending) s.pending = true, on_pending(s.address, s.bytes(), true);
}

// back to memory, tiled, as the PICA leaves its buffers; bytes the game wrote since drawing began kept (remember)
void Renderer::store(Surface& s, const std::vector<uint8_t>& data) {
  Timed timed{stats_.store};
  uint8_t* memory = memory_.map(s.address, s.bytes());
  if (!memory) return;
  const size_t size = s.bytes();
  const bool changed = s.before.size() == size && std::memcmp(memory, s.before.data(), size) != 0;
  std::vector<uint8_t> merged;
  if (changed) merged.assign(memory, memory + size), ++stats_.stores_merged;
  uint8_t* bytes = changed ? merged.data() : memory;
  struct Merge { // on leaving: the stored pixels into memory where the game left it alone
    Surface& s; uint8_t* memory; const std::vector<uint8_t>& merged; bool changed;
    ~Merge() {
      if (changed)
        for (size_t i = 0; i < merged.size(); ++i)
          if (memory[i] == s.before[i]) memory[i] = merged[i];
      s.before.clear();
    }
  } merge{s, memory, merged, changed};
  // the readback is at the scale: one sample per block, its top left
  const uint32_t n = scale_, row = align_up(s.width * n * 4, 256);
  if (!s.depth) {
    for (uint32_t y = 0; y < s.height; ++y)
      for (uint32_t x = 0; x < s.width; ++x) {
        const uint8_t* c = &data[size_t{y * n} * row + x * n * 4];
        encode(s.format, {c[0], c[1], c[2], c[3]}, bytes + tiled_offset(x, y, s.width, bytes_per_pixel(s.format)));
      }
    return;
  }
  const uint32_t stencil_row = align_up(s.width * n, 256);
  const size_t stencil_at = align_up(static_cast<uint32_t>(size_t{row} * s.height * n), 256);
  for (uint32_t y = 0; y < s.height; ++y)
    for (uint32_t x = 0; x < s.width; ++x) {
      float d;
      std::memcpy(&d, &data[size_t{y * n} * row + x * n * 4], 4);
      const uint32_t bpp = s.pixel_bytes();
      if (bpp == 2) {
        const uint32_t d16 = static_cast<uint32_t>(std::lround(std::clamp(d, 0.0f, 1.0f) * 65535.0f)); // exact back
        std::memcpy(bytes + tiled_offset(x, y, s.width, 2), &d16, 2);
        continue;
      }
      const uint32_t d24 = static_cast<uint32_t>(std::clamp(d, 0.0f, 1.0f) * 16777215.0f);
      const uint32_t word = d24 | uint32_t{data[stencil_at + size_t{y * n} * stencil_row + x * n]} << 24;
      std::memcpy(bytes + tiled_offset(x, y, s.width, bpp), &word, bpp); // D24: its 3 low bytes
    }
}

uint32_t Renderer::arena_copy(uint32_t address, uint32_t size) {
  // from a word boundary, so the shader reads whole words; the return is
  // where `address` lands in the arena
  const uint32_t aligned = address & ~3u, lead = address - aligned, total = align_up(size + lead, 4);
  // the same range copied before in this submit (one vertex array for many draws): that copy again; memory does
  // not change while the arena fills (a submit's lists run while the CPU waits)
  auto [known, added] = arena_known_.try_emplace(uint64_t{aligned} << 32 | total, 0u);
  if (!added) return known->second + lead;
  const uint32_t at = static_cast<uint32_t>(arena_.size());
  known->second = at;
  arena_.resize(arena_.size() + total, 0);
  if (const uint8_t* bytes = memory_.map(aligned, total)) {
    std::memcpy(arena_.data() + at, bytes, total);
  } else {
    for (uint32_t i = 0; i < total; ++i)
      if (const uint8_t* b = memory_.map(aligned + i, 1)) arena_[at + i] = *b;
  }
  return at + lead;
}

uint32_t Renderer::lookup_tables(const pica::State& state) {
  // the same tables as the last draw's (the usual case): already uploaded
  const uint32_t* lights_now = state.light_tables[0].data(); // 24 tables of 256, one after the other
  if (last_luts_at_ != UINT32_MAX && std::memcmp(last_light_tables_.data(), lights_now, 24 * 256 * 4) == 0 &&
      std::memcmp(last_fog_table_.data(), state.fog_table.data(), 128 * 4) == 0)
    return last_luts_at_;
  last_light_tables_.assign(lights_now, lights_now + 24 * 256);
  last_fog_table_.assign(state.fog_table.begin(), state.fog_table.end());
  // the same tables met before (materials take turns): their place again
  std::vector<uint32_t> content(last_light_tables_);
  content.insert(content.end(), last_fog_table_.begin(), last_fog_table_.end());
  const uint64_t hash = fnv(reinterpret_cast<const uint8_t*>(content.data()), content.size() * 4);
  if (const auto seen = luts_known_.find(hash); seen != luts_known_.end() && seen->second.first == content)
    return last_luts_at_ = seen->second.second;
  const std::vector<uint32_t>& lights = last_light_tables_;
  const std::vector<uint32_t>& fog = last_fog_table_;
  const uint32_t at = static_cast<uint32_t>(luts_.size() / 2);
  for (uint32_t w : lights) {
    const float step = static_cast<float>(w >> 12 & 0x7FF) / 2047.0f;
    luts_.push_back(static_cast<float>(w & 0xFFF) / 4095.0f);
    luts_.push_back(w >> 23 & 1 ? -step : step);
  }
  for (uint32_t w : fog) {
    luts_.push_back(static_cast<float>(w >> 13 & 0x7FF) / 2048.0f);
    luts_.push_back(static_cast<float>(static_cast<int32_t>((w & 0x1FFF) << 19) >> 19) / 2048.0f);
  }
  last_luts_at_ = at;
  luts_known_[hash] = {std::move(content), at};
  return at;
}

// The textures' group of a draw, made once for each set of textures and
// samplers. It holds them, so their handles stay unique while it is cached.
wgpu::BindGroup Renderer::texture_group(const wgpu::TextureView views[3], const wgpu::Sampler samplers[3]) {
  std::array<void*, 6> key;
  for (int u = 0; u < 3; ++u) key[u] = views[u].Get(), key[3 + u] = samplers[u].Get();
  auto& slot = texture_groups_[key];
  if (slot) return slot;
  // shortcut: a game cycling through more sets than this rebuilds them; an LRU if one does
  if (texture_groups_.size() > 8192) {
    texture_groups_.clear();
    return texture_group(views, samplers);
  }
  wgpu::BindGroupEntry e[6] = {};
  for (int u = 0; u < 3; ++u) {
    e[u].binding = u, e[u].textureView = views[u];
    e[3 + u].binding = 3 + u, e[3 + u].sampler = samplers[u];
  }
  wgpu::BindGroupDescriptor bd{};
  bd.layout = texture_layout_;
  bd.entryCount = 6;
  bd.entries = e;
  slot = gpu_.device.CreateBindGroup(&bd);
  return slot;
}

// The sampler a texture unit's configuration register describes: its wraps
// (the border is the shader's), its filters.
wgpu::Sampler Renderer::sampler(uint32_t config) {
  const uint32_t key = config & 0x01007706;
  auto& slot = samplers_[key];
  if (slot) return slot;
  auto wrap = [](uint32_t mode) {
    switch (mode & 7) {
    case 2: case 6: case 7: return wgpu::AddressMode::Repeat;
    case 3: return wgpu::AddressMode::MirrorRepeat;
    default: return wgpu::AddressMode::ClampToEdge; // edge, and border outside the shader's test
    }
  };
  auto filter = [](uint32_t bit) { return bit ? wgpu::FilterMode::Linear : wgpu::FilterMode::Nearest; };
  wgpu::SamplerDescriptor sd{};
  sd.addressModeU = wrap(key >> 12);
  sd.addressModeV = wrap(key >> 8);
  sd.magFilter = filter(key >> 1 & 1);
  sd.minFilter = filter(key >> 2 & 1);
  sd.mipmapFilter = key >> 24 & 1 ? wgpu::MipmapFilterMode::Linear : wgpu::MipmapFilterMode::Nearest;
  // PN_ANISOTROPY=n: textures the game filters linearly read along their slope, n samples at most
  static const int anisotropy = std::getenv("PN_ANISOTROPY") ? std::atoi(std::getenv("PN_ANISOTROPY")) : 0;
  if (anisotropy > 1 && (key >> 2 & 1)) {
    sd.magFilter = sd.minFilter = wgpu::FilterMode::Linear;
    sd.mipmapFilter = wgpu::MipmapFilterMode::Linear;
    sd.maxAnisotropy = static_cast<uint16_t>(std::min(anisotropy, 16));
  }
  slot = gpu_.device.CreateSampler(&sd);
  return slot;
}

wgpu::TextureView Renderer::texture(const pica::State& state, int unit, uint32_t size[4], float border[4],
                                    float lod[4]) {
  Timed timed{stats_.textures};
  const auto& r = state.regs;
  const uint32_t base = kTextureUnits[unit];
  const uint32_t dimensions = r[base + 1], height = dimensions & 0x7FF, width = dimensions >> 16 & 0x7FF;
  const uint32_t config = r[base + 2], lod_reg = r[base + 3];
  const uint32_t address = (r[base + 4] & 0x0FFFFFFF) << 3;
  const uint32_t format = r[unit == 0 ? base + 13 : base + 5] & 0xF;
  unpack(r[base], border);
  if (width == 0 || height == 0 || address == 0 || format > 13) {
    size[0] = size[1] = size[2] = 1, size[3] = 0;
    lod[0] = lod[1] = lod[2] = lod[3] = 0;
    return empty_texture_;
  }
  const uint32_t largest = lod_reg >> 16 & 0xF;
  const uint32_t tz = std::min<uint32_t>(std::countr_zero(width), std::countr_zero(height));
  const uint32_t levels = 1 + std::min(largest, tz >= 3 ? tz - 3 : 0);
  // PN_MIPMAPS=1: a texture with no smaller levels that the game filters linearly gets them, averaged here
  static const bool mipmaps = std::getenv("PN_MIPMAPS") != nullptr;
  const uint32_t made = mipmaps && levels == 1 && (config >> 2 & 1) && std::has_single_bit(width) &&
                                std::has_single_bit(height) && std::min(width, height) >= 16
                            ? 1 + std::countr_zero(std::min(width, height)) - 2 // down to 4 texels
                            : 0;
  size[0] = width, size[1] = height, size[2] = made ? made : levels, size[3] = config;
  const uint32_t max = made ? made - 1 : std::min(largest, levels - 1);
  lod[0] = static_cast<float>(static_cast<int32_t>(lod_reg << 19) >> 19) / 256.0f;
  lod[1] = made ? 0.0f : static_cast<float>(std::min(lod_reg >> 24 & 0xF, max));
  lod[2] = static_cast<float>(max);
  lod[3] = static_cast<float>((config >> 2 & 1) | (config >> 24 & 1) << 1);

  ++stats_.tex_calls;
  // render to texture: a texture that is a color buffer the GPU holds (drawn or loaded), of the same size and
  // format, is read from the GPU as it is, without the trip through memory (finish the work, read back, encode,
  // hash, decode, send again) that cost Smash a quarter of its time. Not the buffer the draw renders into.
  // RGBA8 only: the others would be read at 8 bits where the console has its few (PN_RTT_ALL=1 takes them,
  // to measure); PN_NO_RTT=1 turns it off
  static const bool rtt = std::getenv("PN_NO_RTT") == nullptr, rtt_all = std::getenv("PN_RTT_ALL") != nullptr;
  if (rtt && cached_ && levels == 1 && format <= 4 && address != (r[0x11D] & 0x0FFFFFFF) << 3) {
    static constexpr ColorFormat as_buffer[5] = {ColorFormat::RGBA8, ColorFormat::RGB8, ColorFormat::RGB5A1,
                                                 ColorFormat::RGB565, ColorFormat::RGBA4};
    for (auto& [key, s] : surfaces_) {
      if (s->depth || s->address != address || s->width != width || s->height != height ||
          s->format != as_buffer[format] || !(s->drawn || s->loaded))
        continue;
      ++stats_.rtt_seen[format];
      if (format != 0 && !rtt_all) break;
      ++stats_.textures_from_buffers;
      if (!s->view) s->view = s->texture.CreateView();
      return s->view;
    }
    // a D24S8 depth buffer read as RGBA8 (Pokemon's effects): its words made on the GPU as memory would hold them
    if (format == 0)
      for (auto& [key, s] : surfaces_)
        if (s->depth && s->depth_format == 3 && s->address == address && s->width == width && s->height == height &&
            (s->drawn || s->loaded)) {
          ++stats_.textures_from_depth;
          return depth_as_color(*s);
        }
  }
  uint32_t bytes = 0;
  for (uint32_t l = 0; l < levels; ++l) bytes += texture::size(format, std::max(width >> l, 1u), std::max(height >> l, 1u));
  // PN_RTT_LOG=1: textures over a drawn color buffer the fast way above did not take (the first 40 kinds)
  static const bool rtt_log = std::getenv("PN_RTT_LOG") != nullptr;
  if (rtt_log)
    for (auto& [key, s] : surfaces_)
      if (s->drawn && address < s->address + s->bytes() && s->address < address + bytes) {
        static std::set<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> told;
        const auto kind = std::tuple{width, height, format, levels, s->width, s->height,
                                     static_cast<uint32_t>(s->format) | (s->depth ? 0x100u : 0u) | (address - s->address) << 12};
        if (told.size() < 40 && told.insert(kind).second)
          std::fprintf(stderr, "[picanative] texture %08x %ux%u format %u levels %u over a drawn %s buffer %08x %ux%u format %u\n",
                       address, width, height, format, levels, s->depth ? "depth" : "color", s->address, s->width, s->height,
                       s->depth ? s->depth_format : static_cast<uint32_t>(s->format));
      }
  // drawn into in this command list: memory has to get it first
  {
    Timed t{stats_.tex_writeback};
    write_back(address, bytes, false);
  }
  {
    Timed t{stats_.tex_known};
    const auto found = known_.find({address, format, width | height << 16, levels});
    if (found != known_.end()) return found->second.texture->view;
  }
  Timed hashing{stats_.tex_hash};
  const uint8_t* data = memory_.map(address, bytes);
  std::vector<uint8_t> copy;
  if (!data) {
    copy.assign(bytes, 0);
    for (uint32_t i = 0; i < bytes; ++i)
      if (const uint8_t* b = memory_.map(address + i, 1)) copy[i] = *b;
    data = copy.data();
  }
  uint64_t key = fnv(data, bytes);
  const uint32_t id[4] = {address, format, width | height << 16, levels};
  key = fnv(reinterpret_cast<const uint8_t*>(id), sizeof id, key);
  auto& slot = textures_[key];
  if (slot) known_[{address, format, width | height << 16, levels}] = {address, bytes, slot.get()};
  if (slot) return slot->view;
  Timed decoding{stats_.tex_decode};
  // a large texture the game changes in part (an atlas it streams tiles into): the rows of 8x8 tiles that
  // changed, decoded and sent alone, into the same GPU texture (once the work using its old content is sent)
  const auto place = std::tuple{address, format, width | height << 16, levels};
  constexpr uint32_t kPartial = 0x40000;
  const uint32_t row_bytes = texture::size(format, width, 8), rows = height / 8;
  if (bytes >= kPartial && levels == 1 && rows && row_bytes * rows == bytes) {
    const auto before = previous_.find(place);
    if (before != previous_.end()) {
      const auto old = textures_.find(before->second.key);
      if (old != textures_.end() && old->second && before->second.bytes.size() == bytes) {
        std::vector<uint32_t> changed;
        for (uint32_t r = 0; r < rows; ++r)
          if (std::memcmp(data + size_t{r} * row_bytes, before->second.bytes.data() + size_t{r} * row_bytes, row_bytes))
            changed.push_back(r);
        if (changed.size() * 2 <= rows) {
          if (!pending_.empty()) flush(false); // draws recorded with the old content first
          slot = std::move(old->second);
          textures_.erase(old);
          std::vector<uint8_t> rgba;
          for (size_t i = 0; i < changed.size();) {
            size_t j = i + 1;
            while (j < changed.size() && changed[j] == changed[j - 1] + 1) ++j;
            const uint32_t first = changed[i], count = static_cast<uint32_t>(j - i);
            texture::decode(format, data + size_t{first} * row_bytes, size_t{count} * row_bytes, width, count * 8, rgba);
            wgpu::TexelCopyTextureInfo dst{};
            dst.texture = slot->texture;
            dst.origin = {0, first * 8, 0};
            wgpu::TexelCopyBufferLayout layout{};
            layout.bytesPerRow = width * 4;
            layout.rowsPerImage = count * 8;
            wgpu::Extent3D extent{width, count * 8, 1};
            gpu_.queue.WriteTexture(&dst, rgba.data(), rgba.size(), &layout, &extent);
            i = j;
          }
          ++stats_.textures_updated;
          before->second.key = key;
          before->second.bytes.assign(data, data + bytes);
          known_[place] = {address, bytes, slot.get()};
          return slot->view;
        }
      }
    }
  }
  slot = std::make_unique<Texture>();
  ++stats_.textures_decoded;
  // PN_TEXTURE_LOG: which textures are decoded again and again (address, format, size, drawn there by the GPU?)
  static const bool log = std::getenv("PN_TEXTURE_LOG") != nullptr;
  if (log) {
    static std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint64_t> decoded;
    const uint64_t n = ++decoded[{address, format, width | height << 16}];
    bool surface = false;
    for (const auto& [k, s] : surfaces_)
      if (address < s->address + s->bytes() && s->address < address + bytes) surface = true;
    if (n == 10 || n == 100 || n == 1000 || n == 10000)
      std::fprintf(stderr, "[picanative] texture %08x format %u %ux%u decoded %llu times%s\n", address, format, width,
                   height, static_cast<unsigned long long>(n), surface ? " (a surface the GPU draws)" : "");
  }
  wgpu::TextureDescriptor td{};
  td.size = {width, height, 1};
  td.mipLevelCount = made ? made : levels;
  td.format = wgpu::TextureFormat::RGBA8Unorm;
  td.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
  slot->texture = gpu_.device.CreateTexture(&td);
  std::vector<uint8_t> rgba;
  uint32_t at = 0;
  for (uint32_t l = 0; l < td.mipLevelCount; ++l) {
    const uint32_t w = std::max(width >> l, 1u), h = std::max(height >> l, 1u);
    if (l < levels) {
      const uint32_t level_bytes = texture::size(format, w, h);
      texture::decode(format, data + at, level_bytes, w, h, rgba);
      at += level_bytes;
    } else { // a made level: each texel the average of the four above it
      std::vector<uint8_t> half(size_t{w} * h * 4);
      const uint32_t pw = w * 2;
      for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
          for (uint32_t k = 0; k < 4; ++k) {
            const auto px = [&](uint32_t xx, uint32_t yy) { return uint32_t{rgba[(size_t{yy} * pw + xx) * 4 + k]}; };
            half[(size_t{y} * w + x) * 4 + k] =
                static_cast<uint8_t>((px(2 * x, 2 * y) + px(2 * x + 1, 2 * y) + px(2 * x, 2 * y + 1) + px(2 * x + 1, 2 * y + 1) + 2) / 4);
          }
      rgba.swap(half);
    }
    wgpu::TexelCopyTextureInfo dst{};
    dst.texture = slot->texture;
    dst.mipLevel = l;
    wgpu::TexelCopyBufferLayout layout{};
    layout.bytesPerRow = w * 4;
    layout.rowsPerImage = h;
    wgpu::Extent3D extent{w, h, 1};
    gpu_.queue.WriteTexture(&dst, rgba.data(), rgba.size(), &layout, &extent);
  }
  slot->view = slot->texture.CreateView();
  known_[{address, format, width | height << 16, levels}] = {address, bytes, slot.get()};
  if (bytes >= kPartial && levels == 1) previous_[place] = {key, std::vector<uint8_t>(data, data + bytes)};
  return slot->view;
}

bool Renderer::load_translations(const std::string& path, std::string& error) {
  shadergen::Translations loaded;
  if (!shadergen::load_translations(path, loaded, error)) return false;
  stats_.programs_ahead += loaded.size();
  translations_.merge(loaded);
  return true;
}

struct Renderer::Described {
  wgpu::RenderPipelineDescriptor pd{};
  wgpu::ColorTargetState target{};
  wgpu::BlendState blend{};
  wgpu::FragmentState fs{};
  wgpu::DepthStencilState ds{};
};

wgpu::RenderPipelineDescriptor* Renderer::describe(const Recipe& r, Described& d) {
  auto shared = [&](const std::string& text) {
    auto& slot = modules_[text];
    if (!slot) slot = module(gpu_, text);
    return slot;
  };
  const auto& s = r.state;
  d.pd.layout = pipeline_layout_;
  d.pd.vertex.module = shared(r.vertex);
  d.pd.vertex.entryPoint = "vs";
  d.pd.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
  d.pd.primitive.cullMode = static_cast<wgpu::CullMode>(s[0]);
  d.pd.primitive.frontFace = static_cast<wgpu::FrontFace>(s[1]);
  d.target.format = wgpu::TextureFormat::RGBA8Unorm;
  d.target.writeMask = static_cast<wgpu::ColorWriteMask>(s[2]);
  if (s[3]) {
    d.blend.color = {static_cast<wgpu::BlendOperation>(s[4]), static_cast<wgpu::BlendFactor>(s[5]),
                     static_cast<wgpu::BlendFactor>(s[6])};
    d.blend.alpha = {static_cast<wgpu::BlendOperation>(s[7]), static_cast<wgpu::BlendFactor>(s[8]),
                     static_cast<wgpu::BlendFactor>(s[9])};
    d.target.blend = &d.blend;
  }
  d.fs.module = shared(r.fragment);
  d.fs.entryPoint = "fs";
  d.fs.targetCount = 1;
  d.fs.targets = &d.target;
  d.pd.fragment = &d.fs;
  d.ds.format = wgpu::TextureFormat::Depth32FloatStencil8;
  d.ds.depthWriteEnabled = s[10] ? wgpu::OptionalBool::True : wgpu::OptionalBool::False;
  d.ds.depthCompare = static_cast<wgpu::CompareFunction>(s[11]);
  if (s[12]) {
    wgpu::StencilFaceState face{};
    face.compare = static_cast<wgpu::CompareFunction>(s[13]);
    face.failOp = static_cast<wgpu::StencilOperation>(s[14]);
    face.depthFailOp = static_cast<wgpu::StencilOperation>(s[15]);
    face.passOp = static_cast<wgpu::StencilOperation>(s[16]);
    d.ds.stencilFront = d.ds.stencilBack = face;
  }
  d.ds.stencilReadMask = s[17];
  d.ds.stencilWriteMask = s[18];
  d.pd.depthStencil = &d.ds;
  return &d.pd;
}

// on Dawn's threads; the draws that need it wait for it (pipeline())
void Renderer::start_pipeline(const Recipe& recipe) {
  if (pipelines_.count(recipe.id) || compiling_.count(recipe.id)) return;
  Described d;
  wgpu::RenderPipelineDescriptor* pd = describe(recipe, d);
  compiling_.insert(recipe.id);
  const std::string id = recipe.id, source = recipe.vertex + recipe.fragment;
  wgpu::Future f = gpu_.device.CreateRenderPipelineAsync(
      pd, wgpu::CallbackMode::WaitAnyOnly,
      [this, id, source](wgpu::CreatePipelineAsyncStatus status, wgpu::RenderPipeline ready, wgpu::StringView message) {
        compiling_.erase(id);
        warming_.erase(id);
        if (status != wgpu::CreatePipelineAsyncStatus::Success) {
          ++stats_.shader_errors;
          if (errors.size() < 3) errors.push_back(std::string(message.data, message.length) + "\n--- WGSL ---\n" + source);
        }
        pipelines_[id] = std::move(ready);
      });
  futures_.push_back(f);
  warming_[id] = f;
}

namespace {
void put_string(std::ofstream& out, const std::string& s) {
  const uint32_t n = static_cast<uint32_t>(s.size());
  out.write(reinterpret_cast<const char*>(&n), 4);
  out.write(s.data(), n);
}
bool get_string(std::ifstream& in, std::string& s) {
  uint32_t n = 0;
  if (!in.read(reinterpret_cast<char*>(&n), 4) || n > (64u << 20)) return false;
  s.resize(n);
  return static_cast<bool>(in.read(s.data(), n));
}
} // namespace

size_t Renderer::warm_pipelines(const std::string& path) {
  recipes_path_ = path;
  std::ifstream in(path, std::ios::binary);
  size_t started = 0;
  // records: id, vertex, fragment, state count, state words; a torn last one (a run that crashed) is dropped
  for (Recipe r; in;) {
    uint32_t n = 0;
    if (!get_string(in, r.id) || !get_string(in, r.vertex) || !get_string(in, r.fragment) ||
        !in.read(reinterpret_cast<char*>(&n), 4) || n != 19)
      break;
    r.state.resize(n);
    if (!in.read(reinterpret_cast<char*>(r.state.data()), n * 4)) break;
    if (!recipes_known_.insert(r.id).second) continue;
    start_pipeline(r);
    ++started;
  }
  stats_.pipelines_warmed += started;
  return started;
}

wgpu::RenderPipeline Renderer::pipeline(const draw::Config& c, const pica::ShaderUnit& vs) {
  const std::vector<uint32_t> key = c.key();
  std::string id(reinterpret_cast<const char*>(key.data()), key.size() * 4);
  auto found = pipelines_.find(id);
  if (found != pipelines_.end()) return found->second;
  if (compiling_.count(id)) {
    if (auto p = warming_.find(id); p != warming_.end()) {
      const wgpu::Future f = p->second;
      gpu_.instance.WaitAny(f, async_ ? 20'000'000 : UINT64_MAX);
      if (const auto done = pipelines_.find(id); done != pipelines_.end()) return done->second;
    }
    ++stats_.waiting;
    return nullptr;
  }
  ++stats_.pipelines;
  Timed timed{stats_.pipelines_time};
  auto program = translations_.find(c.program);
  // PN_CHECK_SHADERS=1: each program translated ahead against its translation now, and against the
  // translation of the whole unit (what the engine translated before it kept only what is reachable)
  static const bool check = std::getenv("PN_CHECK_SHADERS") != nullptr;
  if (check && program != translations_.end() && checked_.insert(c.program).second) {
    const std::string now = shadergen::vertex_function(vs, "vsu"), whole = shadergen::vertex_function(vs, "vsu", true);
    std::fprintf(stderr, "[picanative] check %016llx: ahead %s now, %s the whole unit\n",
                 static_cast<unsigned long long>(c.program), program->second == now ? "==" : "!=",
                 program->second == whole ? "==" : "!=");
    if (program->second != whole) // both, in the current folder
      for (const auto& [name, text] : {std::pair{"ahead", program->second}, std::pair{"whole", whole}})
        std::ofstream(std::to_string(c.program) + "-" + name + ".wgsl") << text;
  }
  if (program == translations_.end()) {
    Timed translating{stats_.translating};
    ++stats_.programs_translated;
    if (stats_.programs_ahead)
      std::fprintf(stderr, "[picanative] vertex program %016llx (entry %u) not translated ahead: translating it\n",
                   static_cast<unsigned long long>(c.program), vs.entry_point);
    program = translations_.emplace(c.program, shadergen::vertex_function(vs, "vsu")).first;
  }
  const std::string vertex_source = draw::vertex_wgsl(c, program->second), fragment_source = draw::fragment_wgsl(c);
  const std::string source = vertex_source + fragment_source; // for error reports
  wgpu::RenderPipeline result;
  Recipe recipe{id, vertex_source, fragment_source, std::vector<uint32_t>(19, 0)};
  const std::string error = gpu::checked(gpu_, [&] {
    if (!modules_.count(vertex_source)) ++stats_.vertex_modules;
    auto& s = recipe.state;
    // the PICA's culling, seen with y pointing up: 1 drops counter-clockwise, 2 clockwise
    wgpu::CullMode cull = wgpu::CullMode::None;
    if (c.cull == 1) cull = c.upside_down ? wgpu::CullMode::Back : wgpu::CullMode::Front;
    else if (c.cull == 2) cull = c.upside_down ? wgpu::CullMode::Front : wgpu::CullMode::Back;
    s[0] = static_cast<uint32_t>(cull), s[1] = static_cast<uint32_t>(wgpu::FrontFace::CCW);
    wgpu::ColorWriteMask mask = wgpu::ColorWriteMask::None;
    if (c.color_mask & 1) mask |= wgpu::ColorWriteMask::Red;
    if (c.color_mask & 2) mask |= wgpu::ColorWriteMask::Green;
    if (c.color_mask & 4) mask |= wgpu::ColorWriteMask::Blue;
    if (c.color_mask & 8) mask |= wgpu::ColorWriteMask::Alpha;
    if (c.logic_mode == 7) mask = wgpu::ColorWriteMask::None; // no-op: the buffer keeps its color
    s[2] = static_cast<uint32_t>(mask);
    wgpu::BlendState blend{};
    if (c.logic_mode == 0) {
      blend.color = {blend_op(c.blend), blend_factor(c.blend >> 16), blend_factor(c.blend >> 20)};
      blend.alpha = {blend_op(c.blend >> 8), blend_factor(c.blend >> 24), blend_factor(c.blend >> 28)};
      // min and max take no factors
      if (blend.color.operation == wgpu::BlendOperation::Min || blend.color.operation == wgpu::BlendOperation::Max)
        blend.color.srcFactor = blend.color.dstFactor = wgpu::BlendFactor::One;
      if (blend.alpha.operation == wgpu::BlendOperation::Min || blend.alpha.operation == wgpu::BlendOperation::Max)
        blend.alpha.srcFactor = blend.alpha.dstFactor = wgpu::BlendFactor::One;
      s[3] = 1;
      s[4] = static_cast<uint32_t>(blend.color.operation), s[5] = static_cast<uint32_t>(blend.color.srcFactor);
      s[6] = static_cast<uint32_t>(blend.color.dstFactor), s[7] = static_cast<uint32_t>(blend.alpha.operation);
      s[8] = static_cast<uint32_t>(blend.alpha.srcFactor), s[9] = static_cast<uint32_t>(blend.alpha.dstFactor);
    }
    s[10] = c.depth_write ? 1 : 0;
    s[11] = static_cast<uint32_t>(c.depth_test ? compare(c.depth_test - 1) : wgpu::CompareFunction::Always);
    if (c.stencil) {
      s[12] = 1;
      s[13] = static_cast<uint32_t>(compare(c.stencil_test & 7));
      s[14] = static_cast<uint32_t>(stencil_op(c.stencil_op));
      s[15] = static_cast<uint32_t>(stencil_op(c.stencil_op >> 4));
      s[16] = static_cast<uint32_t>(stencil_op(c.stencil_op >> 8));
      s[17] = c.stencil_test >> 24 & 0xFF;
      s[18] = c.stencil_test >> 8 & 0xFF;
    }
    // kept for the next runs (warm_pipelines), once
    if (!recipes_path_.empty() && recipes_known_.insert(id).second) {
      std::ofstream out(recipes_path_, std::ios::binary | std::ios::app);
      put_string(out, recipe.id), put_string(out, recipe.vertex), put_string(out, recipe.fragment);
      const uint32_t n = static_cast<uint32_t>(s.size());
      out.write(reinterpret_cast<const char*>(&n), 4);
      out.write(reinterpret_cast<const char*>(s.data()), n * 4);
    }
    if (!async_) {
      Described d;
      result = gpu_.device.CreateRenderPipeline(describe(recipe, d));
      return;
    }
    // compiled on Dawn's threads: the draws that need it wait (are skipped)
    // until it is ready, and the game goes on meanwhile
    start_pipeline(recipe);
  });
  if (async_ && error.empty()) {
    // one the disk cache holds is ready in a few milliseconds (about 8 measured): waiting that long spares its draws
    gpu_.instance.WaitAny(futures_.back(), 20'000'000);
    const auto done = pipelines_.find(id);
    if (done != pipelines_.end()) return done->second;
    ++stats_.waiting;
    return nullptr;
  }
  if (!error.empty()) {
    ++stats_.shader_errors;
    if (errors.size() < 3) errors.push_back(error + "\n--- WGSL ---\n" + source);
    result = nullptr;
  }
  pipelines_[id] = result;
  return result;
}


void Renderer::record(const pica::State& state, bool indexed, const std::vector<pica::CommandProcessor::Vertex>* immediate) {
  ++stats_.draws;
  Timed timed{stats_.record};
  const auto& r = state.regs;
  // the color and depth buffers the draw renders into
  const uint32_t dims = r[0x11E], width = dims & 0x7FF, height = (dims >> 12 & 0x3FF) + 1;
  const uint32_t color_address = (r[0x11D] & 0x0FFFFFFF) << 3, depth_address = (r[0x11C] & 0x0FFFFFFF) << 3;
  const uint32_t depth_format = r[0x116] & 3;
  if (width == 0 || color_address == 0 || depth_format == 1) { // 1: no such depth format
    ++stats_.skipped;
    return;
  }
  // immediate vertices: one array of their attributes as floats, each attribute
  // to the input the shader's permutation names, the others zero (as the reference)
  const uint32_t attributes = (r[0x242] & 0xF) + 1;
  vertex::Layout layout;
  if (immediate) {
    vertex::Loader loader{0, 0, attributes * 16, attributes * 16, {}};
    const uint64_t map = r[pica::kRegVsBlock + 0x0B] | uint64_t{r[pica::kRegVsBlock + 0x0C]} << 32;
    for (uint32_t id = 0; id < attributes; ++id)
      loader.fields.push_back({id * 16, vertex::Float, 4, static_cast<uint32_t>(map >> (id * 4) & 0xF)});
    layout.loaders.push_back(std::move(loader));
    for (auto& input : layout.defaults) input = {0, 0, 0, 0};
  } else {
    layout = vertex::decode(state);
  }
  const draw::Config config = draw::configure(state, layout, indexed);
  const uint32_t count = immediate ? static_cast<uint32_t>(immediate->size()) : r[0x228];
  uint32_t corners = 0;
  if (config.topology == 1 || config.topology == 2) corners = count >= 3 ? (count - 2) * 3 : 0;
  else corners = count / 3 * 3;
  // the viewport, applied in clip space, and the scissor it gives
  const float vx = static_cast<float>(static_cast<int32_t>((r[0x068] & 0x3FF) << 22) >> 22);
  const float vy = static_cast<float>(static_cast<int32_t>((r[0x068] >> 16 & 0x3FF) << 22) >> 22);
  const float vw = f24(r[0x041]) * 2.0f, vh = f24(r[0x043]) * 2.0f;
  const int left = std::max(static_cast<int>(std::lround(vx)), 0);
  const int right = std::min(static_cast<int>(std::lround(vx + vw)), static_cast<int>(width));
  const int bottom = std::max(static_cast<int>(std::lround(vy)), 0);
  const int top = std::min(static_cast<int>(std::lround(vy + vh)), static_cast<int>(height));
  if (corners == 0 || !(vw > 0) || !(vh > 0) || left >= right || bottom >= top) {
    ++stats_.skipped;
    return;
  }

  PendingDraw d{};
  d.scissor[0] = left * scale_;
  d.scissor[1] = (config.upside_down ? bottom : height - top) * scale_;
  d.scissor[2] = (right - left) * scale_;
  d.scissor[3] = (top - bottom) * scale_;
  d.vertex_count = corners;
  d.stencil_reference = r[0x105] >> 16 & 0xFF;
  unpack(r[0x103], d.blend_constant);

  draw::DrawUniforms du{};
  // textures first: one drawn into in this command list writes the pending
  // draws back to memory, which starts the arenas over
  for (int unit = 0; unit < 3; ++unit)
    d.views[unit] = config.textures >> unit & 1
                        ? texture(state, unit, du.texture_size[unit], du.texture_border[unit], du.texture_lod[unit])
                        : empty_texture_;
  for (int unit = 0; unit < 3; ++unit) d.samplers[unit] = sampler(du.texture_size[unit][3]);
  if (immediate) {
    arena_.resize((arena_.size() + 3) & ~size_t{3}, 0);
    const uint32_t at = static_cast<uint32_t>(arena_.size());
    for (const auto& v : *immediate)
      for (uint32_t id = 0; id < attributes; ++id) {
        const auto* bytes = reinterpret_cast<const uint8_t*>(v[id].data());
        arena_.insert(arena_.end(), bytes, bytes + 16);
      }
    du.loaders[0][0] = at;
    du.loaders[0][1] = attributes * 16;
    du.loaders[0][2] = attributes * 16;
  }
  // vertices: the arrays from the lowest vertex named to the highest
  const std::vector<uint32_t> names = immediate ? std::vector<uint32_t>{} : vertex::vertices(state, memory_, indexed);
  uint32_t low = UINT32_MAX, high = 0;
  for (uint32_t i = 0; i < std::min<uint32_t>(count, static_cast<uint32_t>(names.size())); ++i)
    low = std::min(low, names[i]), high = std::max(high, names[i]);
  if (low > high) low = high = 0;
  for (size_t l = 0; l < layout.loaders.size() && !immediate; ++l) {
    const auto& loader = layout.loaders[l];
    const uint32_t first = layout.base + loader.offset + low * loader.stride;
    const uint32_t span = (high - low) * loader.stride + loader.size;
    const uint32_t at = arena_copy(first, span);
    du.loaders[l][0] = at - low * loader.stride; // wraps: id * stride brings it back
    du.loaders[l][1] = loader.stride;
    du.loaders[l][2] = loader.size;
  }
  if (indexed) {
    const bool wide = r[0x227] >> 31;
    du.index[0] = arena_copy(layout.base + (r[0x227] & 0x0FFFFFFF), count * (wide ? 2 : 1));
    du.index[1] = wide;
  }
  du.index[2] = immediate ? 0 : r[0x22A];
  du.index[3] = count;
  for (int i = 0; i < 16; ++i) std::memcpy(du.defaults[i], layout.defaults[i].data(), 16);
  du.clip[0] = vw / static_cast<float>(width);
  du.clip[1] = (2.0f * vx + vw) / static_cast<float>(width) - 1.0f;
  du.clip[2] = vh / static_cast<float>(height);
  du.clip[3] = (2.0f * vy + vh) / static_cast<float>(height) - 1.0f;
  du.depth[0] = f24(r[0x04D]);
  du.depth[1] = f24(r[0x04E]);
  du.depth[2] = (r[0x06D] & 1) == 0 ? 1.0f : 0.0f;
  static const uint32_t stages[6] = {0x0C0, 0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8};
  for (int s = 0; s < 6; ++s) unpack(r[stages[s] + 3], du.tev_constant[s]);
  unpack(r[0x0FD], du.buffer_color);
  for (int c = 0; c < 3; ++c) du.fog_color[c] = static_cast<float>(r[0x0E1] >> (8 * c) & 0xFF);
  du.alpha_reference[0] = r[0x104] >> 8 & 0xFF;
  if (config.lighting || config.fog) {
    const uint32_t at = lookup_tables(state);
    du.lut[0] = at;
    du.lut[1] = at + 24 * 256;
  }
  if (config.lighting) {
    light_color(r[0x1C0], du.global_ambient);
    for (uint32_t slot = 0; slot < config.light_count; ++slot) {
      const uint32_t number = config.light_slots >> (slot * 4) & 7;
      const uint32_t* block = &r[0x140 + number * 0x10];
      float(*L)[4] = du.lights[slot];
      for (int k = 0; k < 4; ++k) light_color(block[k], L[k]);
      const float position[3] = {pica_float(block[4] & 0xFFFF, 10, 5), pica_float(block[4] >> 16, 10, 5),
                                 pica_float(block[5] & 0xFFFF, 10, 5)};
      const float length = std::sqrt(position[0] * position[0] + position[1] * position[1] + position[2] * position[2]);
      for (int k = 0; k < 3; ++k) L[4][k] = position[k], L[5][k] = position[k] / length;
      L[6][0] = fixed13(block[6]), L[6][1] = fixed13(block[6] >> 16), L[6][2] = fixed13(block[7]);
      L[7][0] = pica_float(block[10] & 0xFFFFF, 12, 7);
      L[7][1] = pica_float(block[11] & 0xFFFFF, 12, 7);
      L[8][0] = static_cast<float>(number);
    }
  }

  // the uniform blocks
  // the uniform blocks; one the same as the last draw's is that one again (consecutive draws often share
  // all their constants): less to copy and to send
  {
    uint8_t vsu[kVsUniformsSize] = {};
    for (int i = 0; i < 96; ++i) std::memcpy(vsu + i * 16, state.vs.float_uniforms[i].data(), 16);
    for (int i = 0; i < 4; ++i) std::memcpy(vsu + 96 * 16 + i * 4, state.vs.int_uniforms[i].data(), 4);
    const uint32_t bools = state.vs.bool_uniforms;
    std::memcpy(vsu + 96 * 16 + 16, &bools, 4);
    if (last_vs_offset_ != UINT32_MAX && std::memcmp(uniforms_.data() + last_vs_offset_, vsu, sizeof vsu) == 0) {
      d.vs_offset = last_vs_offset_;
    } else {
      d.vs_offset = last_vs_offset_ = static_cast<uint32_t>(uniforms_.size());
      uniforms_.resize(uniforms_.size() + align_up(kVsUniformsSize, kUniformAlign), 0);
      std::memcpy(uniforms_.data() + d.vs_offset, vsu, sizeof vsu);
    }
  }
  if (last_draw_offset_ != UINT32_MAX && std::memcmp(uniforms_.data() + last_draw_offset_, &du, sizeof du) == 0) {
    d.draw_offset = last_draw_offset_;
  } else {
    d.draw_offset = last_draw_offset_ = static_cast<uint32_t>(uniforms_.size());
    uniforms_.resize(uniforms_.size() + align_up(sizeof du, kUniformAlign), 0);
    std::memcpy(uniforms_.data() + d.draw_offset, &du, sizeof du);
  }

  d.pipeline = pipeline(config, state.vs);
  if (!d.pipeline) {
    ++stats_.skipped;
    return;
  }
  // the color buffer's format: blending reads it at 8 bits a channel, as stored
  // only when written back (shortcut: the console blends what the format kept)
  Surface& color = surface(color_address, width, height, false,
                           static_cast<uint32_t>(color_buffer_format(r[0x117] >> 16 & 7)));
  Surface& depth = surface(depth_address, width, height, true, depth_format);
  load(color);
  load(depth);
  if (!color.drawn) forget_textures(color.address, color.bytes()), remember(color);
  if (!depth.drawn) forget_textures(depth.address, depth.bytes()), remember(depth);
  color.drawn = depth.drawn = true;
  ++color.version, ++depth.version;
  d.color = &color;
  d.depth = &depth;
  pending_.push_back(d);
}

void Renderer::flush(bool write_back) {
  std::vector<Surface*> back;
  if (write_back)
    for (auto& [key, s] : surfaces_)
      if (s->drawn) back.push_back(s.get());
  submit(back);
}

void Renderer::submit(const std::vector<Surface*>& back) {
  if (pending_.empty() && back.empty()) return;
  ++stats_.flushes;
  auto encode_start = std::chrono::steady_clock::now();
  // the lookup tables stay on the GPU from one submit to the next: sent again only when some were added
  const size_t added_luts = luts_.size() - std::min(luts_uploaded_, luts_.size());
  stats_.submit_bytes += arena_.size() + added_luts * 4 + uniforms_.size();
  stats_.arena_bytes += arena_.size(), stats_.lut_bytes += added_luts * 4;
  wgpu::Buffer arena, luts, uniforms;
  {
    Timed t{stats_.submit_buffers};
    arena = make_buffer(gpu_, arena_.data(), arena_.size(), wgpu::BufferUsage::Storage);
    if (!luts_buffer_) {
      wgpu::BufferDescriptor bd{};
      bd.size = kLutCapacity * 4;
      bd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
      luts_buffer_ = gpu_.device.CreateBuffer(&bd);
      luts_uploaded_ = 0;
    }
    if (added_luts)
      gpu_.queue.WriteBuffer(luts_buffer_, luts_uploaded_ * 4, luts_.data() + luts_uploaded_, added_luts * 4);
    luts_uploaded_ = luts_.size();
    luts = luts_buffer_;
    uniforms = make_buffer(gpu_, uniforms_.data(), uniforms_.size(), wgpu::BufferUsage::Uniform);
  }
  const auto draws_start = std::chrono::steady_clock::now();
  const wgpu::CommandEncoder encoder = gpu_.device.CreateCommandEncoder();

  wgpu::RenderPassEncoder pass;
  Surface *pass_color = nullptr, *pass_depth = nullptr;
  bool pass_depth_only = false;
  auto end_pass = [&] {
    if (pass) pass.End();
    pass = nullptr;
    pass_color = pass_depth = nullptr;
  };
  auto begin_pass = [&](Surface* color, Surface* depth) {
    end_pass();
    wgpu::RenderPassColorAttachment ca{};
    wgpu::TextureView color_view;
    if (color) {
      color_view = color->texture.CreateView();
      ca.view = color_view;
      ca.loadOp = wgpu::LoadOp::Load;
      ca.storeOp = wgpu::StoreOp::Store;
    }
    const wgpu::TextureView depth_view = depth->texture.CreateView();
    wgpu::RenderPassDepthStencilAttachment da{};
    da.view = depth_view;
    da.depthLoadOp = wgpu::LoadOp::Load;
    da.depthStoreOp = wgpu::StoreOp::Store;
    da.stencilLoadOp = wgpu::LoadOp::Load;
    da.stencilStoreOp = wgpu::StoreOp::Store;
    wgpu::RenderPassDescriptor rp{};
    rp.colorAttachmentCount = color ? 1 : 0;
    rp.colorAttachments = color ? &ca : nullptr;
    rp.depthStencilAttachment = &da;
    pass = encoder.BeginRenderPass(&rp);
    pass_color = color;
    pass_depth = depth;
    pass_depth_only = color == nullptr;
    ++stats_.passes;
  };

  // the buffers' group, the same for every draw of this submit (dynamic
  // offsets), made at the first draw (a submit of clears has no uniforms)
  wgpu::BindGroup buffers;
  auto make_buffers = [&] {
    wgpu::BindGroupEntry e[4] = {};
    e[0].binding = 0, e[0].buffer = uniforms, e[0].size = kVsUniformsSize;
    e[1].binding = 1, e[1].buffer = uniforms, e[1].size = sizeof(draw::DrawUniforms);
    e[2].binding = 2, e[2].buffer = arena, e[2].size = align_up(static_cast<uint32_t>(std::max<size_t>(arena_.size(), 16)), 16);
    e[3].binding = 3, e[3].buffer = luts, e[3].size = align_up(static_cast<uint32_t>(std::max<size_t>(luts_.size() * 4, 16)), 16);
    wgpu::BindGroupDescriptor bd{};
    bd.layout = layout_;
    bd.entryCount = 4;
    bd.entries = e;
    buffers = gpu_.device.CreateBindGroup(&bd);
  };

  const PendingDraw* last = nullptr; // the last draw encoded in the current pass
  for (const PendingDraw& d : pending_) {
    if (d.clear) {
      // a fill over the whole surface, done on the GPU
      stats_.submit_draws += std::chrono::duration<double>(std::chrono::steady_clock::now() - draws_start).count();
  end_pass();
      wgpu::RenderPassColorAttachment ca{};
      wgpu::RenderPassDepthStencilAttachment da{};
      wgpu::RenderPassDescriptor rp{};
      const wgpu::TextureView view = (d.color ? d.color : d.depth)->texture.CreateView();
      if (d.color) {
        ca.view = view;
        ca.loadOp = wgpu::LoadOp::Clear;
        ca.storeOp = wgpu::StoreOp::Store;
        ca.clearValue = {d.clear_color[0], d.clear_color[1], d.clear_color[2], d.clear_color[3]};
        rp.colorAttachmentCount = 1;
        rp.colorAttachments = &ca;
      } else {
        da.view = view;
        da.depthLoadOp = wgpu::LoadOp::Clear;
        da.depthStoreOp = wgpu::StoreOp::Store;
        da.depthClearValue = static_cast<float>(d.clear_color[0]);
        da.stencilLoadOp = wgpu::LoadOp::Clear;
        da.stencilStoreOp = wgpu::StoreOp::Store;
        da.stencilClearValue = d.stencil_reference;
        rp.depthStencilAttachment = &da;
      }
      encoder.BeginRenderPass(&rp).End();
      ++stats_.passes;
      continue;
    }
    if (d.depth_upload) {
      begin_pass(nullptr, d.depth);
      const uint32_t size[4] = {d.depth->width * scale_, d.depth->height * scale_, 0, 0};
      const wgpu::Buffer size_buffer = make_buffer(gpu_, size, sizeof size, wgpu::BufferUsage::Uniform);
      wgpu::BindGroupEntry e[2] = {};
      e[0].binding = 0, e[0].buffer = arena, e[0].offset = d.upload_offset,
      e[0].size = size_t{d.depth->width * scale_} * d.depth->height * scale_ * 4;
      e[1].binding = 1, e[1].buffer = size_buffer, e[1].size = 16;
      wgpu::BindGroupDescriptor bd{};
      bd.layout = depth_upload_layout_;
      bd.entryCount = 2;
      bd.entries = e;
      pass.SetPipeline(depth_upload_);
      pass.SetBindGroup(0, gpu_.device.CreateBindGroup(&bd));
      pass.Draw(3);
      end_pass();
      continue;
    }
    if (!pass || pass_depth_only || pass_color != d.color || pass_depth != d.depth) {
      begin_pass(d.color, d.depth);
      last = nullptr; // a new pass has no state yet
    }
    // what changed since the last draw of this pass, and only that (Dawn checks every call)
    if (!last || last->pipeline.Get() != d.pipeline.Get()) pass.SetPipeline(d.pipeline);
    if (!buffers) make_buffers();
    if (!last || last->vs_offset != d.vs_offset || last->draw_offset != d.draw_offset) {
      const uint32_t offsets[2] = {d.vs_offset, d.draw_offset};
      pass.SetBindGroup(0, buffers, 2, offsets);
    }
    bool same_textures = last != nullptr;
    for (int u = 0; u < 3 && same_textures; ++u)
      same_textures = last->views[u].Get() == d.views[u].Get() && last->samplers[u].Get() == d.samplers[u].Get();
    if (!same_textures) pass.SetBindGroup(1, texture_group(d.views, d.samplers));
    if (!last || std::memcmp(last->scissor, d.scissor, sizeof d.scissor))
      pass.SetScissorRect(d.scissor[0], d.scissor[1], d.scissor[2], d.scissor[3]);
    if (!last || last->stencil_reference != d.stencil_reference) pass.SetStencilReference(d.stencil_reference);
    if (!last || std::memcmp(last->blend_constant, d.blend_constant, sizeof d.blend_constant)) {
      const wgpu::Color constant{d.blend_constant[0], d.blend_constant[1], d.blend_constant[2], d.blend_constant[3]};
      pass.SetBlendConstant(&constant);
    }
    pass.Draw(d.vertex_count);
    last = &d;
  }
  end_pass();

  // what was drawn goes back to memory
  struct Readback {
    Surface* surface;
    wgpu::Buffer buffer;
    uint64_t size;
  };
  std::vector<Readback> readbacks;
  for (Surface* s : back) {
    const uint32_t sw = s->width * scale_, sh = s->height * scale_;
    const uint32_t row = align_up(sw * 4, 256);
    uint64_t size = uint64_t{row} * sh;
    uint64_t stencil_at = 0;
    if (s->depth) {
      stencil_at = align_up(static_cast<uint32_t>(size), 256);
      size = stencil_at + uint64_t{align_up(sw, 256)} * sh;
    }
    wgpu::BufferDescriptor bdesc{};
    bdesc.size = align_up(static_cast<uint32_t>(size), 4);
    bdesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    const wgpu::Buffer buffer = gpu_.device.CreateBuffer(&bdesc);
    wgpu::TexelCopyTextureInfo src{};
    src.texture = s->texture;
    src.aspect = s->depth ? wgpu::TextureAspect::DepthOnly : wgpu::TextureAspect::All;
    wgpu::TexelCopyBufferInfo dst{};
    dst.buffer = buffer;
    dst.layout.bytesPerRow = row;
    dst.layout.rowsPerImage = sh;
    wgpu::Extent3D extent{sw, sh, 1};
    encoder.CopyTextureToBuffer(&src, &dst, &extent);
    if (s->depth) {
      src.aspect = wgpu::TextureAspect::StencilOnly;
      dst.layout.offset = stencil_at;
      dst.layout.bytesPerRow = align_up(sw, 256);
      encoder.CopyTextureToBuffer(&src, &dst, &extent);
    }
    readbacks.push_back({s, buffer, bdesc.size});
  }
  const wgpu::CommandBuffer commands = encoder.Finish();
  gpu_.queue.Submit(1, &commands);
  stats_.encode += std::chrono::duration<double>(std::chrono::steady_clock::now() - encode_start).count();
  Timed waited{stats_.wait};
  for (Readback& rb : readbacks) {
    bool mapped = false;
    gpu::wait(gpu_, rb.buffer.MapAsync(wgpu::MapMode::Read, 0, rb.size, wgpu::CallbackMode::WaitAnyOnly,
                                      [&](wgpu::MapAsyncStatus status, wgpu::StringView) {
                                        mapped = status == wgpu::MapAsyncStatus::Success;
                                      }));
    if (!mapped) continue;
    const auto* bytes = static_cast<const uint8_t*>(rb.buffer.GetConstMappedRange(0, rb.size));
    store(*rb.surface, std::vector<uint8_t>(bytes, bytes + rb.size));
    rb.buffer.Unmap();
    rb.surface->drawn = false;
    if (rb.surface->pending && on_pending) rb.surface->pending = false, on_pending(rb.surface->address, rb.surface->bytes(), false);
  }

  pending_.clear();
  arena_.clear();
  // the tables are kept (and their place, last_luts_at_) unless they grew large
  // full: start over (the tables are added again as draws need them)
  if (luts_.size() + 2 * (24 * 256 + 128) > kLutCapacity)
    luts_.clear(), luts_known_.clear(), luts_uploaded_ = 0, last_luts_at_ = UINT32_MAX;
  uniforms_.clear();
  last_vs_offset_ = last_draw_offset_ = UINT32_MAX;
  arena_known_.clear();
}

void Renderer::write_back(uint32_t address, uint32_t size, bool forget) {
  deliver(address, size);
  std::vector<Surface*> back;
  for (auto& [key, s] : surfaces_) {
    const uint32_t bytes = s->bytes();
    if (address < s->address + bytes && s->address < address + size) {
      if (s->drawn) back.push_back(s.get());
      if (forget) s->loaded = false;
    }
  }
  // nothing drawn there: the pending work can wait
  if (!back.empty()) submit(back);
}

void Renderer::forget_textures(uint32_t address, uint32_t size) {
  std::erase_if(pictures_, [&](const Picture& p) { return address < p.address + p.size && p.address < address + size; });
  for (auto it = known_.begin(); it != known_.end();)
    if (address < it->second.address + it->second.bytes && it->second.address < address + size) it = known_.erase(it);
    else ++it;
}

namespace {

// the ranges a transfer reads and writes, as the transfer engine lays them out
void transfer_ranges(uint32_t input_dimensions, uint32_t output_dimensions, uint32_t flags, uint32_t& input_bytes,
                     uint32_t& output_bytes) {
  const uint32_t in_bpp = bytes_per_pixel(transfer_format(flags >> 8));
  const uint32_t out_bpp = bytes_per_pixel(transfer_format(flags >> 12));
  const uint32_t downscale = flags >> 24 & 3;
  const uint32_t sx = downscale == 1 || downscale == 2 ? 2 : 1, sy = downscale == 2 ? 2 : 1;
  const uint32_t ow = (output_dimensions & 0xFFFF) / sx, oh = (output_dimensions >> 16) / sy;
  const uint32_t iw = input_dimensions & 0xFFFF;
  input_bytes = iw * (input_dimensions >> 16) * in_bpp;
  const uint32_t skew = (flags & 1) && (flags & 4) && iw > (output_dimensions & 0xFFFF)
                            ? (iw - (output_dimensions & 0xFFFF)) * (oh ? oh - 1 : 0) * out_bpp
                            : 0;
  output_bytes = skew + ow * oh * out_bpp;
}

uint32_t copy_span(uint32_t size, uint32_t gap) {
  const uint32_t width = (gap & 0xFFFF) * 16, skip = (gap >> 16) * 16;
  return width == 0 ? size : size + (size + width - 1) / width * skip;
}

} // namespace

void Renderer::memory_fill(uint32_t start, uint32_t end, uint32_t value, uint32_t width) {
  if (end <= start) return;
  deliver(start, end - start);
  if (cached_) {
    // a whole surface filled with pixels of its own size: cleared on the GPU, memory gets it if anything reads it
    for (auto& [key, s] : surfaces_) {
      if (s->address != start || start + s->bytes() != end || s->pixel_bytes() != width) continue;
      PendingDraw clear{};
      clear.clear = true;
      if (s->depth) {
        clear.depth = s.get();
        clear.clear_color[0] = width == 2 ? static_cast<double>(value & 0xFFFF) / 65535.0
                                          : static_cast<double>(value & 0xFFFFFF) / 16777215.0;
        clear.stencil_reference = width == 4 ? value >> 24 : 0;
      } else {
        clear.color = s.get();
        const Rgba c = decode(s->format, reinterpret_cast<const uint8_t*>(&value));
        for (int k = 0; k < 4; ++k) clear.clear_color[k] = c[k] / 255.0;
      }
      pending_.push_back(clear);
      if (!s->drawn) remember(*s);
      s->loaded = s->drawn = true;
      ++s->version;
      forget_textures(start, end - start);
      return;
    }
  }
  if (cached_) write_back(start, end - start, true);
  picanative::memory_fill(memory_, start, end, value, width);
  forget_textures(start, end - start);
}

void Renderer::display_transfer(uint32_t input, uint32_t output, uint32_t input_dimensions,
                                uint32_t output_dimensions, uint32_t flags) {
  uint32_t input_bytes, output_bytes;
  transfer_ranges(input_dimensions, output_dimensions, flags, input_bytes, output_bytes);
  // tiled to tiled (a picture made smaller for an effect, then read as a texture): done on the GPU into a color
  // buffer there, its memory written only if the CPU reads it (PN_NO_TILED_TRANSFER=1: not)
  static const bool tiled_on_gpu = std::getenv("PN_NO_TILED_TRANSFER") == nullptr;
  if (cached_ && tiled_on_gpu && gpu_tiled_transfer(input, output, input_dimensions, output_dimensions, flags)) {
    forget_textures(output, output_bytes);
    return;
  }
  if (cached_) {
    write_back(output, output_bytes, true);
    forget_textures(output, output_bytes);
    if (gpu_transfer(input, output, input_dimensions, output_dimensions, flags)) {
      // PN_CHECK_TRANSFERS: each one also done by transfer.cpp, the bytes compared
      if (std::getenv("PN_CHECK_TRANSFERS")) {
        deliver(output, output_bytes);
        const uint8_t* now = memory_.map(output, output_bytes);
        if (!now) return;
        const std::vector<uint8_t> gpu(now, now + output_bytes);
        write_back(input, input_bytes, false);
        picanative::display_transfer(memory_, input, output, input_dimensions, output_dimensions, flags);
        for (uint32_t k = 0; k < output_bytes; ++k)
          if (gpu[k] != now[k]) {
            std::fprintf(stderr, "transfer %08x -> %08x, flags %08x: byte %u is %02x on the GPU, %02x by transfer.cpp\n",
                         input, output, flags, k, gpu[k], now[k]);
            break;
          }
      }
      return;
    }
    write_back(input, input_bytes, false);
  }
  picanative::display_transfer(memory_, input, output, input_dimensions, output_dimensions, flags);
  forget_textures(output, output_bytes);
}

wgpu::TextureView Renderer::depth_as_color(Surface& s) {
  if (s.as_color && s.as_color_version == s.version) return s.as_color_view;
  if (!depth_color_pipeline_) {
    static const char* source = R"(
@group(0) @binding(0) var depth: texture_depth_2d;
@group(0) @binding(1) var stencil: texture_2d<u32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let p = vec2<u32>(id.xy);
  // as Renderer::store writes it: 24 bits of depth, the stencil above; an RGBA8 texel reads that word's
  // bytes 3, 2, 1, 0 as red, green, blue, alpha
  let d = u32(clamp(textureLoad(depth, p, 0), 0.0, 1.0) * 16777215.0);
  let st = textureLoad(stencil, p, 0).r & 0xFFu;
  textureStore(dst, p, vec4<f32>(f32(st), f32((d >> 16u) & 0xFFu), f32((d >> 8u) & 0xFFu), f32(d & 0xFFu)) / 255.0);
}
)";
    wgpu::ComputePipelineDescriptor cd{};
    cd.compute.module = module(gpu_, source);
    cd.compute.entryPoint = "main";
    depth_color_pipeline_ = gpu_.device.CreateComputePipeline(&cd);
  }
  if (!s.as_color) {
    wgpu::TextureDescriptor td{};
    td.size = {s.width * scale_, s.height * scale_, 1};
    td.format = wgpu::TextureFormat::RGBA8Unorm;
    td.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::StorageBinding;
    s.as_color = gpu_.device.CreateTexture(&td);
    s.as_color_view = s.as_color.CreateView();
  }
  submit({}); // the draws into the depth buffer first
  wgpu::TextureViewDescriptor depth_view{}, stencil_view{};
  depth_view.aspect = wgpu::TextureAspect::DepthOnly;
  stencil_view.aspect = wgpu::TextureAspect::StencilOnly;
  wgpu::BindGroupEntry e[3] = {};
  e[0].binding = 0, e[0].textureView = s.texture.CreateView(&depth_view);
  e[1].binding = 1, e[1].textureView = s.texture.CreateView(&stencil_view);
  e[2].binding = 2, e[2].textureView = s.as_color_view;
  wgpu::BindGroupDescriptor group{};
  group.layout = depth_color_pipeline_.GetBindGroupLayout(0);
  group.entryCount = 3;
  group.entries = e;
  const wgpu::CommandEncoder encoder = gpu_.device.CreateCommandEncoder();
  const wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
  pass.SetPipeline(depth_color_pipeline_);
  pass.SetBindGroup(0, gpu_.device.CreateBindGroup(&group));
  pass.DispatchWorkgroups((s.width * scale_ + 7) / 8, (s.height * scale_ + 7) / 8);
  pass.End();
  const wgpu::CommandBuffer commands = encoder.Finish();
  gpu_.queue.Submit(1, &commands);
  s.as_color_version = s.version;
  return s.as_color_view;
}

bool Renderer::gpu_tiled_transfer(uint32_t input, uint32_t output, uint32_t input_dimensions,
                                  uint32_t output_dimensions, uint32_t flags) {
  const uint32_t width = input_dimensions & 0xFFFF, height = input_dimensions >> 16;
  const uint32_t out_width = output_dimensions & 0xFFFF, out_height = output_dimensions >> 16;
  const bool flip = flags & 1, input_linear = flags & 2;
  const bool output_tiled = input_linear != ((flags & 0x20) != 0);
  const uint32_t downscale = flags >> 24 & 3;
  if (input_linear || !output_tiled || flip || downscale == 3) return false;
  const uint32_t sx = downscale ? 2 : 1, sy = downscale == 2 ? 2 : 1;
  const uint32_t copy_width = out_width / sx, copy_height = out_height / sy;
  if (copy_width == 0 || copy_height == 0 || copy_width * sx > width || copy_height * sy > height) return false;
  if (copy_width % 8 || copy_height % 8) return false; // whole tiles only
  const ColorFormat input_format = transfer_format(flags >> 8), output_format = transfer_format(flags >> 12);
  Surface* in = nullptr;
  for (auto& [key, c] : surfaces_)
    if (!c->depth && c->address == input && c->width == width && c->format == input_format && c->loaded &&
        c->height >= copy_height * sy) {
      in = c.get();
      break;
    }
  if (!in) return false;
  Surface& out = surface(output, copy_width, copy_height, false, static_cast<uint32_t>(output_format));
  if (&out == in) return false;
  if (!tiled_pipeline_) {
    static const char* source = R"(
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
struct P { w: u32, h: u32, sx: u32, sy: u32 };
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= p.w || id.y >= p.h) { return; }
  var sum = vec4<f32>(0.0);
  for (var j = 0u; j < p.sy; j++) {
    for (var i = 0u; i < p.sx; i++) {
      sum += textureLoad(src, vec2<u32>(id.x * p.sx + i, id.y * p.sy + j), 0);
    }
  }
  textureStore(dst, vec2<u32>(id.x, id.y), sum / f32(p.sx * p.sy));
}
)";
    wgpu::ComputePipelineDescriptor cd{};
    cd.compute.module = module(gpu_, source);
    cd.compute.entryPoint = "main";
    tiled_pipeline_ = gpu_.device.CreateComputePipeline(&cd);
  }
  submit({}); // the draws into the input first
  // the output's pixels at the scale: each the average of sx by sy of the input's at the scale
  const uint32_t parameters[4] = {copy_width * scale_, copy_height * scale_, sx, sy};
  const wgpu::Buffer uniform = make_buffer(gpu_, parameters, sizeof parameters, wgpu::BufferUsage::Uniform);
  if (!in->view) in->view = in->texture.CreateView();
  if (!out.view) out.view = out.texture.CreateView();
  wgpu::BindGroupEntry e[3] = {};
  e[0].binding = 0, e[0].textureView = in->view;
  e[1].binding = 1, e[1].textureView = out.view;
  e[2].binding = 2, e[2].buffer = uniform, e[2].size = sizeof parameters;
  wgpu::BindGroupDescriptor group{};
  group.layout = tiled_pipeline_.GetBindGroupLayout(0);
  group.entryCount = 3;
  group.entries = e;
  const wgpu::CommandEncoder encoder = gpu_.device.CreateCommandEncoder();
  const wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
  pass.SetPipeline(tiled_pipeline_);
  pass.SetBindGroup(0, gpu_.device.CreateBindGroup(&group));
  pass.DispatchWorkgroups((copy_width * scale_ + 7) / 8, (copy_height * scale_ + 7) / 8);
  pass.End();
  const wgpu::CommandBuffer commands = encoder.Finish();
  gpu_.queue.Submit(1, &commands);
  ++stats_.gpu_tiled_transfers;
  // the whole output is the GPU's now: memory gets it when read
  if (!out.drawn) remember(out);
  out.loaded = out.drawn = true;
  return true;
}

bool Renderer::gpu_transfer(uint32_t input, uint32_t output, uint32_t input_dimensions,
                            uint32_t output_dimensions, uint32_t flags) {
  // the case games use to show a frame: a color buffer the GPU holds, out to
  // a linear picture of the same size or cropped; the rest is transfer.cpp's
  const uint32_t width = input_dimensions & 0xFFFF, height = input_dimensions >> 16;
  const uint32_t out_width = output_dimensions & 0xFFFF, out_height = output_dimensions >> 16;
  const bool flip = flags & 1, input_linear = flags & 2;
  const bool output_tiled = input_linear != ((flags & 0x20) != 0);
  // any color format: the surface's GPU texture is RGBA8 whatever memory holds, the shader stores it as memory would
  const ColorFormat input_format = transfer_format(flags >> 8);
  const uint32_t downscale = flags >> 24 & 3;
  if (input_linear || output_tiled || downscale == 3) return false;
  // the output dimensions count input pixels: a downscale divides them (transfer.cpp)
  const uint32_t sx = downscale == 1 || downscale == 2 ? 2 : 1, sy = downscale == 2 ? 2 : 1;
  const uint32_t copy_width = out_width / sx, copy_height = out_height / sy;
  // the whole output written: a narrower output keeps the first pixels of each line (a crop), but a flipped crop
  // moves the output (transfer.cpp), and what an output larger than the input leaves alone stays transfer.cpp's
  if (copy_width == 0 || copy_height == 0 || copy_width * sx > width || copy_height * sy > height) return false;
  if (flip && (flags & 4) && width > out_width) return false;
  // a color surface as wide holding the lines read: from its start, or from a
  // line of tiles down (a game drawing both screens into one buffer)
  Surface* found = nullptr;
  uint32_t first = 0;
  for (auto& [key, candidate] : surfaces_) {
    const uint32_t tile_line = width * 8 * bytes_per_pixel(input_format);
    if (candidate->depth || candidate->format != input_format || candidate->width != width || !candidate->loaded ||
        input < candidate->address ||
        (input - candidate->address) % tile_line != 0)
      continue;
    const uint32_t line = (input - candidate->address) / tile_line * 8;
    if (line + copy_height * sy > candidate->height) continue;
    found = candidate.get(), first = line;
    break;
  }
  if (!found) {
    static int said = 0; // DEBUG-TRANSFER
    const char* debug = std::getenv("PN_DEBUG_TRANSFER");
    if (debug && said++ >= std::atoi(debug) && said < std::atoi(debug) + 6) {
      std::fprintf(stderr, "[pn] no surface for transfer %08x %ux%u fmt %d:", input, width, height, int(input_format));
      for (auto& [key, c] : surfaces_)
        if (!c->depth) std::fprintf(stderr, " [%08x %ux%u fmt %d %s]", c->address, c->width, c->height, int(c->format), c->loaded ? "loaded" : "-");
      std::fprintf(stderr, "\n");
    }
    return false;
  }
  Surface& s = *found;

  const ColorFormat format = transfer_format(flags >> 12);
  const uint32_t bpp = bytes_per_pixel(format), bytes = copy_width * copy_height * bpp;
  const uint32_t words = (bytes + 3) / 4;

  submit({}); // the draws into the surface first
  wgpu::BufferDescriptor bd{};
  bd.size = uint64_t{words} * 4;
  bd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
  const wgpu::Buffer out = gpu_.device.CreateBuffer(&bd);
  bd.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
  const wgpu::Buffer back = gpu_.device.CreateBuffer(&bd);
  const uint32_t parameters[4] = {copy_width, copy_height,
                                  static_cast<uint32_t>(format) | (flip ? 1u : 0u) << 8 |
                                      static_cast<uint32_t>(input_format) << 9 | (sx - 1) << 12 | (sy - 1) << 13 |
                                      first << 16 | scale_ << 26,
                                  words};
  const wgpu::Buffer uniform = make_buffer(gpu_, parameters, sizeof parameters, wgpu::BufferUsage::Uniform);
  const wgpu::TextureView view = s.texture.CreateView();
  wgpu::BindGroupEntry e[3] = {};
  e[0].binding = 0, e[0].textureView = view;
  e[1].binding = 1, e[1].buffer = out, e[1].size = bd.size;
  e[2].binding = 2, e[2].buffer = uniform, e[2].size = sizeof parameters;
  wgpu::BindGroupDescriptor group{};
  group.layout = transfer_pipeline_.GetBindGroupLayout(0);
  group.entryCount = 3;
  group.entries = e;
  const wgpu::CommandEncoder encoder = gpu_.device.CreateCommandEncoder();
  const wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
  pass.SetPipeline(transfer_pipeline_);
  pass.SetBindGroup(0, gpu_.device.CreateBindGroup(&group));
  const uint32_t groups = (words + 63) / 64;
  pass.DispatchWorkgroups(std::min(groups, 65535u), (groups + 65534) / 65535);
  pass.End();
  encoder.CopyBufferToBuffer(out, 0, back, 0, bd.size);
  const wgpu::CommandBuffer commands = encoder.Finish();
  gpu_.queue.Submit(1, &commands);
  ++stats_.gpu_transfers;

  // write_back(output) delivered what overlapped: deliveries never overlap
  wgpu::TextureView scaled;
  if (scale_ > 1) { // what the window shows: the pixels transferred, at the scale (its filter averages a downscale)
    wgpu::TextureDescriptor td{};
    td.size = {copy_width * sx * scale_, copy_height * sy * scale_, 1};
    td.format = wgpu::TextureFormat::RGBA8Unorm;
    td.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
    const wgpu::Texture copy = gpu_.device.CreateTexture(&td);
    wgpu::TexelCopyTextureInfo from{}, to{};
    from.texture = s.texture;
    from.origin = {0, first * scale_, 0};
    to.texture = copy;
    const wgpu::CommandEncoder copier = gpu_.device.CreateCommandEncoder();
    copier.CopyTextureToTexture(&from, &to, &td.size);
    const wgpu::CommandBuffer copied = copier.Finish();
    gpu_.queue.Submit(1, &copied);
    scaled = copy.CreateView();
  }
  pictures_.push_back({output, bytes, out, scaled, flip});
  std::vector<uint8_t> before;
  if (const uint8_t* now = memory_.map(output, bytes)) before.assign(now, now + bytes);
  deliveries_.push_back({output, bytes, back,
                         back.MapAsync(wgpu::MapMode::Read, 0, bd.size, wgpu::CallbackMode::WaitAnyOnly,
                                       [](wgpu::MapAsyncStatus, wgpu::StringView) {}),
                         std::move(before)});
  if (on_pending) on_pending(output, bytes, true);
  // shortcut: a host that never reads its pictures back keeps up to 16 on the way
  if (deliveries_.size() > 16) deliver(deliveries_.front().address, 1);
  return true;
}

wgpu::TextureView Renderer::picture_view(uint32_t address, bool& flip) const {
  for (const Picture& p : pictures_)
    if (p.address == address && p.view) {
      flip = p.flip;
      return p.view;
    }
  return nullptr;
}

wgpu::Buffer Renderer::picture(uint32_t address, uint32_t size) const {
  for (const Picture& p : pictures_)
    if (p.address == address && size <= p.size) return p.buffer;
  return nullptr;
}

void Renderer::deliver(uint32_t address, uint32_t size) {
  for (size_t i = 0; i < deliveries_.size();) {
    Delivery& d = deliveries_[i];
    if (!(address < d.address + d.size && d.address < address + size)) {
      ++i;
      continue;
    }
    Timed waited{stats_.wait};
    gpu_.instance.WaitAny(d.mapped, UINT64_MAX);
    if (const auto* bytes = static_cast<const uint8_t*>(d.buffer.GetConstMappedRange(0, (d.size + 3) & ~3u))) {
      if (uint8_t* to = memory_.map(d.address, d.size)) {
        // the console's engine wrote when the transfer ran: what the game wrote there since is newer, kept
        if (d.before.size() != d.size || std::memcmp(to, d.before.data(), d.size) == 0) std::memcpy(to, bytes, d.size);
        else {
          ++stats_.stores_merged;
          for (uint32_t k = 0; k < d.size; ++k)
            if (to[k] == d.before[k]) to[k] = bytes[k];
        }
      } else
        for (uint32_t k = 0; k < d.size; ++k)
          if (uint8_t* b = memory_.map(d.address + k, 1)) *b = bytes[k];
    }
    d.buffer.Unmap();
    const uint32_t done_address = d.address, done_size = d.size;
    deliveries_.erase(deliveries_.begin() + static_cast<std::ptrdiff_t>(i));
    if (on_pending) on_pending(done_address, done_size, false);
  }
}

void Renderer::texture_copy(uint32_t input, uint32_t output, uint32_t size, uint32_t input_gap, uint32_t output_gap) {
  if (cached_) {
    write_back(input, copy_span(size, input_gap), false);
    write_back(output, copy_span(size, output_gap), true);
  }
  picanative::texture_copy(memory_, input, output, size, input_gap, output_gap);
  forget_textures(output, copy_span(size, output_gap));
}

void Renderer::sync(uint32_t address, uint32_t size) {
  if (cached_) write_back(address, size, false);
}

void Renderer::before_cpu_write(uint32_t address, uint32_t size) {
  if (!cached_) return;
  write_back(address, size, true);
  forget_textures(address, size);
}

} // namespace picanative
