#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string>
#include <vector>

// The WebGPU device the engine draws with (Dawn: D3D12 or Vulkan on
// Windows, Vulkan on Android), set up as Aurora sets it up.
namespace picanative::gpu {

struct Context {
  wgpu::Instance instance;
  wgpu::Adapter adapter;
  wgpu::Device device;
  wgpu::Queue queue;
  std::string adapter_name;
  wgpu::BackendType backend = wgpu::BackendType::Undefined;
};

// A device without a window, for tools and tests. backend Undefined lets
// Dawn choose.
bool create(Context& context, std::string& error, wgpu::BackendType backend = wgpu::BackendType::Undefined);

// Waits for a future of the instance (map, request, work done).
bool wait(Context& context, wgpu::Future future);

// Copies a GPU buffer back to the CPU and waits for it.
std::vector<uint8_t> read_back(Context& context, const wgpu::Buffer& buffer, uint64_t size);

// Runs `scope` with an error scope around it, returning the validation
// error message, or an empty string.
template <typename F>
std::string checked(Context& context, F&& scope) {
  context.device.PushErrorScope(wgpu::ErrorFilter::Validation);
  scope();
  std::string message;
  wait(context, context.device.PopErrorScope(wgpu::CallbackMode::WaitAnyOnly,
                                             [&](wgpu::PopErrorScopeStatus, wgpu::ErrorType type, wgpu::StringView text) {
                                               if (type != wgpu::ErrorType::NoError)
                                                 message = std::string(text.data, text.length);
                                             }));
  return message;
}

} // namespace picanative::gpu
