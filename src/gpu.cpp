#include "picanative/gpu.hpp"

#include <array>
#include <cstdio>
#include <cstring>

namespace picanative::gpu {

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

  wgpu::DeviceDescriptor device_descriptor{};
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
