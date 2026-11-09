#include "picanative/gpu.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace picanative::gpu {
namespace {

// Dawn's cache of compiled shaders and pipelines, one file per key, kept
// between runs: a pipeline compiles once per machine. PICANATIVE_CACHE_DIR, or
// %LOCALAPPDATA%/picanative/cache (./picanative-cache elsewhere).
std::filesystem::path cache_dir() {
  if (const char* dir = std::getenv("PICANATIVE_CACHE_DIR")) return dir;
  if (const char* local = std::getenv("LOCALAPPDATA")) return std::filesystem::path(local) / "picanative" / "cache";
  return "picanative-cache";
}

std::filesystem::path cache_file(const void* key, size_t size) {
  // two FNV-1a hashes of the key, for a name of 128 bits
  uint64_t a = 0xCBF29CE484222325ull, b = 0x84222325CBF29CE4ull;
  const auto* bytes = static_cast<const uint8_t*>(key);
  for (size_t i = 0; i < size; ++i) {
    a = (a ^ bytes[i]) * 0x100000001B3ull;
    b = (b ^ bytes[i]) * 0x100000001B3ull + 0x9E3779B97F4A7C15ull;
  }
  char name[40];
  std::snprintf(name, sizeof name, "%016llx%016llx", static_cast<unsigned long long>(a),
                static_cast<unsigned long long>(b));
  return cache_dir() / name;
}

size_t load_cached(const void* key, size_t key_size, void* value, size_t value_size, void*) {
  std::ifstream in(cache_file(key, key_size), std::ios::binary | std::ios::ate);
  if (!in) return 0;
  const size_t size = static_cast<size_t>(in.tellg());
  if (!value || value_size < size) return size; // Dawn asks the size first
  in.seekg(0);
  in.read(static_cast<char*>(value), static_cast<std::streamsize>(size));
  return in ? size : 0;
}

void store_cached(const void* key, size_t key_size, const void* value, size_t value_size, void*) {
  std::error_code error;
  std::filesystem::create_directories(cache_dir(), error);
  const std::filesystem::path path = cache_file(key, key_size), part = path.string() + ".part";
  {
    std::ofstream out(part, std::ios::binary);
    out.write(static_cast<const char*>(value), static_cast<std::streamsize>(value_size));
    if (!out) return;
  }
  std::filesystem::rename(part, path, error); // whole files only, even if two runs store at once
}

} // namespace

bool wait(Context& context, wgpu::Future future) {
  return context.instance.WaitAny(future, UINT64_MAX) == wgpu::WaitStatus::Success;
}

bool create(Context& context, std::string& error, wgpu::BackendType backend) {
  const std::array features{wgpu::InstanceFeatureName::TimedWaitAny};
  wgpu::InstanceDescriptor instance_descriptor{};
  instance_descriptor.requiredFeatureCount = features.size();
  instance_descriptor.requiredFeatures = features.data();
  context.instance = wgpu::CreateInstance(&instance_descriptor);
  if (!context.instance) {
    error = "cannot create a WebGPU instance";
    return false;
  }

  wgpu::RequestAdapterOptions options{};
  options.powerPreference = wgpu::PowerPreference::HighPerformance;
  options.backendType = backend;
  wait(context, context.instance.RequestAdapter(
                    &options, wgpu::CallbackMode::WaitAnyOnly,
                    [&](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, wgpu::StringView message) {
                      if (status == wgpu::RequestAdapterStatus::Success) {
                        context.adapter = std::move(adapter);
                      } else {
                        error = std::string(message.data, message.length);
                      }
                    }));
  if (!context.adapter) {
    if (error.empty()) error = "no GPU adapter";
    return false;
  }
  wgpu::AdapterInfo info;
  context.adapter.GetInfo(&info);
  context.adapter_name = std::string(info.device.data, info.device.length);
  context.backend = info.backendType;

  // depth and stencil the engine copies to and from the game's memory
  std::vector<wgpu::FeatureName> features_wanted;
  if (context.adapter.HasFeature(wgpu::FeatureName::Depth32FloatStencil8))
    features_wanted.push_back(wgpu::FeatureName::Depth32FloatStencil8);
  // as Aurora sets Dawn up: DXC rather than FXC, no bounds checks added to
  // every array access (our shaders index many arrays), and the shader cache
  wgpu::DawnCacheDeviceDescriptor cache{};
  cache.loadDataFunction = load_cached;
  cache.storeDataFunction = store_cached;
  const std::array toggles{"use_dxc", "disable_robustness", "disable_symbol_renaming"};
  wgpu::DawnTogglesDescriptor toggles_descriptor{};
  toggles_descriptor.nextInChain = &cache;
  toggles_descriptor.enabledToggleCount = toggles.size();
  toggles_descriptor.enabledToggles = toggles.data();
  wgpu::DeviceDescriptor device_descriptor{};
  device_descriptor.nextInChain = &toggles_descriptor;
  device_descriptor.requiredFeatureCount = features_wanted.size();
  device_descriptor.requiredFeatures = features_wanted.data();
  device_descriptor.SetUncapturedErrorCallback(
      [](const wgpu::Device&, wgpu::ErrorType type, wgpu::StringView message) {
        std::fprintf(stderr, "WebGPU error %d: %.*s\n", static_cast<int>(type), static_cast<int>(message.length),
                     message.data);
      });
  device_descriptor.SetDeviceLostCallback(
      wgpu::CallbackMode::AllowSpontaneous, [](const wgpu::Device&, wgpu::DeviceLostReason reason, wgpu::StringView message) {
        if (reason != wgpu::DeviceLostReason::Destroyed)
          std::fprintf(stderr, "WebGPU device lost: %.*s\n", static_cast<int>(message.length), message.data);
      });
  wait(context, context.adapter.RequestDevice(
                    &device_descriptor, wgpu::CallbackMode::WaitAnyOnly,
                    [&](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
                      if (status == wgpu::RequestDeviceStatus::Success) {
                        context.device = std::move(device);
                      } else {
                        error = std::string(message.data, message.length);
                      }
                    }));
  if (!context.device) {
    if (error.empty()) error = "cannot create a GPU device";
    return false;
  }
  context.queue = context.device.GetQueue();
  return true;
}

std::vector<uint8_t> read_back(Context& context, const wgpu::Buffer& buffer, uint64_t size) {
  wgpu::BufferDescriptor descriptor{};
  descriptor.size = (size + 3) & ~uint64_t{3};
  descriptor.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  const wgpu::Buffer staging = context.device.CreateBuffer(&descriptor);
  const wgpu::CommandEncoder encoder = context.device.CreateCommandEncoder();
  encoder.CopyBufferToBuffer(buffer, 0, staging, 0, descriptor.size);
  const wgpu::CommandBuffer commands = encoder.Finish();
  context.queue.Submit(1, &commands);
  std::vector<uint8_t> out(size);
  bool mapped = false;
  wait(context, staging.MapAsync(wgpu::MapMode::Read, 0, descriptor.size, wgpu::CallbackMode::WaitAnyOnly,
                                 [&](wgpu::MapAsyncStatus status, wgpu::StringView) {
                                   mapped = status == wgpu::MapAsyncStatus::Success;
                                 }));
  if (mapped) {
    std::memcpy(out.data(), staging.GetConstMappedRange(0, descriptor.size), size);
    staging.Unmap();
  }
  return out;
}

} // namespace picanative::gpu
