// The port's window, with SDL3 as Aurora has it: the engine's device draws
// the two screens straight into it, and it hands the host the player's
// inputs and plays the console's sound.
#define PICANATIVE_BUILDING
#include "picanative/api.h"

#include "engine.hpp"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace picanative;

namespace {

// The LCD as the console scans it: a framebuffer is column-major, each
// column a line of 240 pixels from the bottom up (zakuro's read_screen).
// p: LCD width, stride, format.
const char* kPresentWgsl = R"(
@group(0) @binding(0) var<storage, read> fb: array<u32>;
@group(0) @binding(1) var<uniform> p: vec4u;
struct V { @builtin(position) position: vec4f, @location(0) uv: vec2f, }
@vertex fn vs(@builtin(vertex_index) i: u32) -> V {
  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var v: V;
  v.position = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
  v.uv = uv;
  return v;
}
fn byte(at: u32) -> u32 { return (fb[at >> 2u] >> ((at & 3u) * 8u)) & 0xFFu; }
fn five(c: u32) -> f32 { return f32((c << 3u) | (c >> 2u)) / 255.0; }
@fragment fn fs(v: V) -> @location(0) vec4f {
  let x = min(u32(v.uv.x * f32(p.x)), p.x - 1u);
  let y = min(u32(v.uv.y * 240.0), 239u);
  let format = p.z;
  let bpp = select(2u, select(3u, 4u, format == 0u), format <= 1u);
  let at = x * p.y + (239u - y) * bpp;
  switch format {
    case 0u: { return vec4f(f32(byte(at + 3u)), f32(byte(at + 2u)), f32(byte(at + 1u)), 255.0) / 255.0; }
    case 1u: { return vec4f(f32(byte(at + 2u)), f32(byte(at + 1u)), f32(byte(at)), 255.0) / 255.0; }
    case 2u: {
      let c = byte(at) | (byte(at + 1u) << 8u);
      let g = (c >> 5u) & 63u;
      return vec4f(five(c >> 11u), f32((g << 2u) | (g >> 4u)) / 255.0, five(c & 31u), 1.0);
    }
    case 3u: {
      let c = byte(at) | (byte(at + 1u) << 8u);
      return vec4f(five(c >> 11u), five((c >> 6u) & 31u), five((c >> 1u) & 31u), 1.0);
    }
    default: {
      let c = byte(at) | (byte(at + 1u) << 8u);
      return vec4f(f32(c >> 12u), f32((c >> 8u) & 15u), f32((c >> 4u) & 15u), 15.0) / 15.0;
    }
  }
}
)";

// what the last poll read, for a platform layer reading inputs itself
pn_input last_input{};
bool polled = false;

constexpr int kWidth = 400, kHeight = 480; // the top screen over the bottom one
constexpr int kAudioRate = 32728;          // the DSP's, the CPU clock over 8192

} // namespace

struct pn_window {
  pn_engine* engine = nullptr;
  SDL_Window* window = nullptr;
  wgpu::Surface surface;
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  int width = 0, height = 0;
  wgpu::PresentMode mode = wgpu::PresentMode::Fifo;
  wgpu::RenderPipeline pipeline;
  wgpu::Buffer upload[2];
  uint32_t upload_size[2] = {};
  SDL_Gamepad* gamepad = nullptr;
  SDL_AudioStream* audio = nullptr;
  bool touching = false;
  float touch_x = 0, touch_y = 0; // window pixels

  // where the screens go in the window, letterboxed at its aspect
  void layout(float& x, float& y, float& scale) const {
    scale = std::min(width / float(kWidth), height / float(kHeight));
    x = (width - kWidth * scale) / 2, y = (height - kHeight * scale) / 2;
  }

  void configure() {
    SDL_GetWindowSizeInPixels(window, &width, &height);
    wgpu::SurfaceConfiguration config{};
    config.device = engine->gpu.device;
    config.format = format;
    config.width = static_cast<uint32_t>(std::max(width, 1));
    config.height = static_cast<uint32_t>(std::max(height, 1));
    config.presentMode = mode; // Fifo: paced by the display; or the host paces it
    surface.Configure(&config);
  }
};

extern "C" {

pn_window* pn_window_open(pn_engine* engine, const char* title, uint32_t scale, char* error, uint32_t error_size) {
  auto fail = [&](const std::string& message) -> pn_window* {
    if (error && error_size) std::snprintf(error, error_size, "%s", message.c_str());
    return nullptr;
  };
  if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD | SDL_INIT_AUDIO)) return fail(SDL_GetError());
  auto w = new pn_window;
  w->engine = engine;
  scale = std::max(scale, 1u);
  w->window = SDL_CreateWindow(title, kWidth * static_cast<int>(scale), kHeight * static_cast<int>(scale),
                               SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
  if (!w->window) {
    delete w;
    return fail(SDL_GetError());
  }
  gpu::Context& g = engine->gpu;
  const SDL_PropertiesID properties = SDL_GetWindowProperties(w->window);
#if defined(_WIN32)
  wgpu::SurfaceSourceWindowsHWND source{};
  source.hwnd = SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
  source.hinstance = SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_WIN32_INSTANCE_POINTER, nullptr);
#elif defined(__ANDROID__)
  wgpu::SurfaceSourceAndroidNativeWindow source{};
  source.window = SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_ANDROID_WINDOW_POINTER, nullptr);
#else
#error "picanative's window: no surface for this platform yet"
#endif
  wgpu::SurfaceDescriptor sd{};
  sd.nextInChain = &source;
  w->surface = g.instance.CreateSurface(&sd);
  wgpu::SurfaceCapabilities capabilities;
  if (!w->surface || !w->surface.GetCapabilities(g.adapter, &capabilities) || capabilities.formatCount == 0) {
    SDL_DestroyWindow(w->window);
    delete w;
    return fail("the GPU cannot draw into this window");
  }
  // the screens' colors as they are: a format that does not convert them to sRGB
  w->format = capabilities.formats[0];
  for (size_t i = 0; i < capabilities.formatCount; ++i)
    if (capabilities.formats[i] == wgpu::TextureFormat::BGRA8Unorm ||
        capabilities.formats[i] == wgpu::TextureFormat::RGBA8Unorm) {
      w->format = capabilities.formats[i];
      break;
    }
  w->configure();

  wgpu::ShaderSourceWGSL wgsl{};
  wgsl.code = kPresentWgsl;
  wgpu::ShaderModuleDescriptor md{};
  md.nextInChain = &wgsl;
  const wgpu::ShaderModule module = g.device.CreateShaderModule(&md);
  wgpu::ColorTargetState target{};
  target.format = w->format;
  wgpu::FragmentState fs{};
  fs.module = module;
  fs.entryPoint = "fs";
  fs.targetCount = 1;
  fs.targets = &target;
  wgpu::RenderPipelineDescriptor pd{};
  pd.vertex.module = module;
  pd.vertex.entryPoint = "vs";
  pd.fragment = &fs;
  w->pipeline = g.device.CreateRenderPipeline(&pd);

  SDL_AudioSpec spec{SDL_AUDIO_S16, 2, kAudioRate};
  w->audio = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
  if (w->audio) SDL_ResumeAudioStreamDevice(w->audio);
  else std::fprintf(stderr, "picanative: no sound (%s)\n", SDL_GetError());
  return w;
}

void pn_window_close(pn_window* w) {
  if (!w) return;
  if (w->audio) SDL_DestroyAudioStream(w->audio);
  if (w->gamepad) SDL_CloseGamepad(w->gamepad);
  w->surface = nullptr;
  SDL_DestroyWindow(w->window);
  delete w;
  SDL_Quit();
}

void pn_window_present(pn_window* w, const pn_screen screens[2]) {
  gpu::Context& g = w->engine->gpu;
  Renderer& renderer = *w->engine->renderer;
  wgpu::SurfaceTexture frame;
  w->surface.GetCurrentTexture(&frame);
  if (frame.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal &&
      frame.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
    w->configure(); // resized, or lost: the next frame shows
    return;
  }
  const wgpu::TextureView view = frame.texture.CreateView();
  const wgpu::CommandEncoder encoder = g.device.CreateCommandEncoder();
  wgpu::RenderPassColorAttachment ca{};
  ca.view = view;
  ca.loadOp = wgpu::LoadOp::Clear;
  ca.storeOp = wgpu::StoreOp::Store;
  ca.clearValue = {0.094, 0.094, 0.094, 1.0}; // the margins, dark grey
  wgpu::RenderPassDescriptor rp{};
  rp.colorAttachmentCount = 1;
  rp.colorAttachments = &ca;
  const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&rp);
  pass.SetPipeline(w->pipeline);
  float ox, oy, scale;
  w->layout(ox, oy, scale);
  for (int i = 0; i < 2; ++i) {
    const pn_screen& s = screens[i];
    const uint32_t lcd_width = i == 0 ? 400 : 320;
    const uint32_t format = s.format & 7, bpp = format == 0 ? 4 : format == 1 ? 3 : 2;
    const uint32_t stride = s.stride ? s.stride : 240 * bpp, bytes = stride * lcd_width;
    if (s.address == 0) continue; // the LCD is off: the margins' grey
    // drawn by a transfer on the GPU: shown from there; otherwise from memory
    wgpu::Buffer source = renderer.picture(s.address, bytes);
    if (!source) {
      renderer.sync(s.address, bytes);
      const uint32_t padded = (bytes + 3) & ~3u;
      if (w->upload_size[i] < padded) {
        wgpu::BufferDescriptor bd{};
        bd.size = padded;
        bd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        w->upload[i] = g.device.CreateBuffer(&bd);
        w->upload_size[i] = padded;
      }
      std::vector<uint8_t> copy(padded, 0);
      if (const uint8_t* from = w->engine->memory.map(s.address, bytes)) std::memcpy(copy.data(), from, bytes);
      else
        for (uint32_t k = 0; k < bytes; ++k)
          if (const uint8_t* b = w->engine->memory.map(s.address + k, 1)) copy[k] = *b;
      g.queue.WriteBuffer(w->upload[i], 0, copy.data(), padded);
      source = w->upload[i];
    }
    const uint32_t parameters[4] = {lcd_width, stride, format, 0};
    wgpu::BufferDescriptor ud{};
    ud.size = sizeof parameters;
    ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    const wgpu::Buffer uniform = g.device.CreateBuffer(&ud);
    g.queue.WriteBuffer(uniform, 0, parameters, sizeof parameters);
    wgpu::BindGroupEntry e[2] = {};
    e[0].binding = 0, e[0].buffer = source, e[0].size = (bytes + 3) & ~3u;
    e[1].binding = 1, e[1].buffer = uniform, e[1].size = sizeof parameters;
    wgpu::BindGroupDescriptor bd{};
    bd.layout = w->pipeline.GetBindGroupLayout(0);
    bd.entryCount = 2;
    bd.entries = e;
    pass.SetBindGroup(0, g.device.CreateBindGroup(&bd));
    const float x = ox + (i == 0 ? 0 : 40) * scale, y = oy + (i == 0 ? 0 : 240) * scale;
    pass.SetViewport(x, y, lcd_width * scale, 240 * scale, 0, 1);
    pass.Draw(3);
  }
  pass.End();
  const wgpu::CommandBuffer commands = encoder.Finish();
  g.queue.Submit(1, &commands);
  w->surface.Present();
}

int pn_window_poll(pn_window* w, pn_input* input) {
  bool open = true;
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    switch (event.type) {
    case SDL_EVENT_QUIT:
    case SDL_EVENT_WINDOW_CLOSE_REQUESTED: open = false; break;
    case SDL_EVENT_KEY_DOWN:
      if (event.key.key == SDLK_ESCAPE) open = false;
      break;
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: w->configure(); break;
    case SDL_EVENT_GAMEPAD_ADDED:
      if (!w->gamepad) w->gamepad = SDL_OpenGamepad(event.gdevice.which);
      break;
    case SDL_EVENT_GAMEPAD_REMOVED:
      if (w->gamepad && SDL_GetGamepadID(w->gamepad) == event.gdevice.which) {
        SDL_CloseGamepad(w->gamepad);
        w->gamepad = nullptr;
      }
      break;
    // the touch screen: the mouse, or a finger on a device's own screen
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP:
      if (event.button.button == SDL_BUTTON_LEFT && event.button.which != SDL_TOUCH_MOUSEID) {
        w->touching = event.type == SDL_EVENT_MOUSE_BUTTON_DOWN;
        w->touch_x = event.button.x * SDL_GetWindowPixelDensity(w->window);
        w->touch_y = event.button.y * SDL_GetWindowPixelDensity(w->window);
      }
      break;
    case SDL_EVENT_MOUSE_MOTION:
      if (event.motion.which != SDL_TOUCH_MOUSEID) {
        w->touch_x = event.motion.x * SDL_GetWindowPixelDensity(w->window);
        w->touch_y = event.motion.y * SDL_GetWindowPixelDensity(w->window);
      }
      break;
    case SDL_EVENT_FINGER_DOWN:
    case SDL_EVENT_FINGER_MOTION:
    case SDL_EVENT_FINGER_UP:
      w->touching = event.type != SDL_EVENT_FINGER_UP;
      w->touch_x = event.tfinger.x * w->width;
      w->touch_y = event.tfinger.y * w->height;
      break;
    default: break;
    }
  }

  pn_input in{};
  // the keyboard, as the scaffold's window had it
  const bool* key = SDL_GetKeyboardState(nullptr);
  auto down = [&](SDL_Scancode a, SDL_Scancode b = SDL_SCANCODE_UNKNOWN, SDL_Scancode c = SDL_SCANCODE_UNKNOWN) {
    return key[a] || (b != SDL_SCANCODE_UNKNOWN && key[b]) || (c != SDL_SCANCODE_UNKNOWN && key[c]);
  };
  float cx = 0, cy = 0;
  if (down(SDL_SCANCODE_UP, SDL_SCANCODE_W)) cy += 1, in.buttons |= PN_UP;
  if (down(SDL_SCANCODE_DOWN, SDL_SCANCODE_S)) cy -= 1, in.buttons |= PN_DOWN;
  if (down(SDL_SCANCODE_LEFT, SDL_SCANCODE_A)) cx -= 1, in.buttons |= PN_LEFT;
  if (down(SDL_SCANCODE_RIGHT, SDL_SCANCODE_D)) cx += 1, in.buttons |= PN_RIGHT;
  if (down(SDL_SCANCODE_J, SDL_SCANCODE_X, SDL_SCANCODE_SPACE)) in.buttons |= PN_A;
  if (down(SDL_SCANCODE_K, SDL_SCANCODE_C, SDL_SCANCODE_LSHIFT)) in.buttons |= PN_B;
  if (down(SDL_SCANCODE_U, SDL_SCANCODE_Z)) in.buttons |= PN_X;
  if (down(SDL_SCANCODE_I, SDL_SCANCODE_V)) in.buttons |= PN_Y;
  if (down(SDL_SCANCODE_E, SDL_SCANCODE_LCTRL)) in.buttons |= PN_R;
  if (down(SDL_SCANCODE_Q)) in.buttons |= PN_L;
  if (down(SDL_SCANCODE_RETURN)) in.buttons |= PN_START;
  if (down(SDL_SCANCODE_BACKSPACE)) in.buttons |= PN_SELECT;

  // a gamepad, by position: the 3DS has A on the right, B at the bottom
  if (SDL_Gamepad* p = w->gamepad) {
    auto button = [&](SDL_GamepadButton b, uint32_t bit) {
      if (SDL_GetGamepadButton(p, b)) in.buttons |= bit;
    };
    button(SDL_GAMEPAD_BUTTON_EAST, PN_A);
    button(SDL_GAMEPAD_BUTTON_SOUTH, PN_B);
    button(SDL_GAMEPAD_BUTTON_NORTH, PN_X);
    button(SDL_GAMEPAD_BUTTON_WEST, PN_Y);
    button(SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, PN_L);
    button(SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, PN_R);
    button(SDL_GAMEPAD_BUTTON_START, PN_START);
    button(SDL_GAMEPAD_BUTTON_BACK, PN_SELECT);
    button(SDL_GAMEPAD_BUTTON_DPAD_UP, PN_UP);
    button(SDL_GAMEPAD_BUTTON_DPAD_DOWN, PN_DOWN);
    button(SDL_GAMEPAD_BUTTON_DPAD_LEFT, PN_LEFT);
    button(SDL_GAMEPAD_BUTTON_DPAD_RIGHT, PN_RIGHT);
    if (SDL_GetGamepadAxis(p, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) > 16384) in.buttons |= PN_ZL;
    if (SDL_GetGamepadAxis(p, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) > 16384) in.buttons |= PN_ZR;
    auto axis = [&](SDL_GamepadAxis a) {
      const float v = SDL_GetGamepadAxis(p, a) / 32767.0f;
      return std::abs(v) < 0.15f ? 0.0f : std::clamp(v, -1.0f, 1.0f); // a dead zone
    };
    cx += axis(SDL_GAMEPAD_AXIS_LEFTX), cy -= axis(SDL_GAMEPAD_AXIS_LEFTY);
    in.c_stick_x = axis(SDL_GAMEPAD_AXIS_RIGHTX), in.c_stick_y = -axis(SDL_GAMEPAD_AXIS_RIGHTY);
  }
  in.circle_x = std::clamp(cx, -1.0f, 1.0f), in.circle_y = std::clamp(cy, -1.0f, 1.0f);

  if (w->touching) {
    float ox, oy, scale;
    w->layout(ox, oy, scale);
    const float x = (w->touch_x - ox) / scale - 40, y = (w->touch_y - oy) / scale - 240;
    if (x >= 0 && x < 320 && y >= 0 && y < 240) {
      in.touching = 1;
      in.touch_x = static_cast<uint16_t>(x), in.touch_y = static_cast<uint16_t>(y);
    }
  }
  *input = in;
  last_input = in, polled = true;
  return open ? 1 : 0;
}

int pn_input_now(pn_input* input) {
  if (!polled) return 0;
  *input = last_input;
  return 1;
}

void pn_window_audio(pn_window* w, const int16_t* samples, uint32_t frames) {
  if (!w->audio || frames == 0) return;
  // past a fifth of a second queued, the host runs ahead of the sound: drop
  // rather than let the delay grow
  if (SDL_GetAudioStreamQueued(w->audio) > kAudioRate / 5 * 4) return;
  SDL_PutAudioStreamData(w->audio, samples, static_cast<int>(frames * 4));
}

void pn_window_title(pn_window* w, const char* title) { SDL_SetWindowTitle(w->window, title); }

void pn_window_set_vsync(pn_window* w, int on) {
  w->mode = wgpu::PresentMode::Fifo;
  wgpu::SurfaceCapabilities capabilities;
  if (!on && w->surface.GetCapabilities(w->engine->gpu.adapter, &capabilities)) {
    // without waiting for the display: Mailbox (no tearing) when there is one
    for (size_t i = 0; i < capabilities.presentModeCount; ++i)
      if (capabilities.presentModes[i] == wgpu::PresentMode::Mailbox) w->mode = wgpu::PresentMode::Mailbox;
    if (w->mode == wgpu::PresentMode::Fifo)
      for (size_t i = 0; i < capabilities.presentModeCount; ++i)
        if (capabilities.presentModes[i] == wgpu::PresentMode::Immediate) w->mode = wgpu::PresentMode::Immediate;
  }
  w->configure();
}

} // extern "C"
