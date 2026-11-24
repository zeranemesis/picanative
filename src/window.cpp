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
#include <cstdlib>
#include <filesystem>
#include <ctime>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#if defined(__ANDROID__)
#include <android/native_window.h>
#endif

using namespace picanative;

namespace {

#if defined(__ANDROID__)
// A second display (the AYN Thor's lower screen): its native window, which
// the app's Java side hands over (pn_android_second_screen); the console's
// bottom screen goes there and the main window keeps the top one.
struct Second {
  std::mutex lock; // held while the window draws into it, or changes it
  ANativeWindow* given = nullptr;
  bool changed = false;
  bool touching = false;
  float u = 0, v = 0; // the finger, across the display (0..1)
} second;
#endif

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
fn pixel(xi: i32, yi: i32) -> vec3f {
  let x = u32(clamp(xi, 0, i32(p.x) - 1));
  let y = u32(clamp(yi, 0, 239));
  let format = p.z & 7u;
  let bpp = select(2u, select(3u, 4u, format == 0u), format <= 1u);
  let at = x * p.y + (239u - y) * bpp;
  switch format {
    case 0u: { return vec3f(f32(byte(at + 3u)), f32(byte(at + 2u)), f32(byte(at + 1u))) / 255.0; }
    case 1u: { return vec3f(f32(byte(at + 2u)), f32(byte(at + 1u)), f32(byte(at))) / 255.0; }
    case 2u: {
      let c = byte(at) | (byte(at + 1u) << 8u);
      let g = (c >> 5u) & 63u;
      return vec3f(five(c >> 11u), f32((g << 2u) | (g >> 4u)) / 255.0, five(c & 31u));
    }
    case 3u: {
      let c = byte(at) | (byte(at + 1u) << 8u);
      return vec3f(five(c >> 11u), five((c >> 6u) & 31u), five((c >> 1u) & 31u));
    }
    default: {
      let c = byte(at) | (byte(at + 1u) << 8u);
      return vec3f(f32(c >> 12u), f32((c >> 8u) & 15u), f32((c >> 4u) & 15u)) / 15.0;
    }
  }
}
fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.299, 0.587, 0.114)); }
// the scanlines: a dark line on the last window pixels of each of the console's 240 rows, one pixel wide (a fifth
// of a row at least, half at most), its edge smoothed at sizes between whole multiples. Not sin(fract(row) * pi),
// symmetric: at 2x both pixels of a row took the same value, no line seen
fn scanline(rows: f32) -> f32 {
  let fw = max(fwidth(rows), 1e-4);
  let width = clamp(fw, 0.2, 0.5);
  return 1.0 - 0.35 * clamp((fract(rows) - (1.0 - width)) / fw + 0.5, 0.0, 1.0);
}
// p.w: the effects, as the picture's shader has them (none by default: the pixels as they are)
@fragment fn fs(v: V) -> @location(0) vec4f {
  let fx = v.uv.x * f32(p.x);
  let fy = v.uv.y * 240.0;
  let x = i32(min(u32(fx), p.x - 1u));
  let y = i32(min(u32(fy), 239u));
  var c = pixel(x, y);
  let e = p.w;
  if ((e & 3u) != 0u) {
    let n = pixel(x, y - 1); let s = pixel(x, y + 1); let r = pixel(x + 1, y); let l = pixel(x - 1, y);
    if ((e & 1u) != 0u) {
      let ln = luma(n); let ls = luma(s); let lr = luma(r); let ll = luma(l); let lc = luma(c);
      let range = max(max(max(ln, ls), max(lr, ll)), lc) - min(min(min(ln, ls), min(lr, ll)), lc);
      if (range > max(0.0312, 0.125 * max(lc, max(max(ln, ls), max(lr, ll))))) {
        let along_x = abs(ln + ls - 2.0 * lc) >= abs(lr + ll - 2.0 * lc);
        c = mix(c, select((r + l) * 0.5, (n + s) * 0.5, along_x), clamp(range * 2.0, 0.0, 0.5));
      }
    }
    if ((e & 2u) != 0u) { c = clamp(c * 1.6 - (n + s + r + l) * 0.15, vec3f(0.0), vec3f(1.0)); }
  }
  if ((e & 4u) != 0u) {
    c = clamp(mat3x3f(0.86, 0.07, 0.04, 0.10, 0.86, 0.09, 0.04, 0.07, 0.87) * pow(c, vec3f(1.12)), vec3f(0.0), vec3f(1.0));
  }
  if ((e & 8u) != 0u) { c = c * scanline(fy); }
  return vec4f(c, 1.0);
}
)";

// what the last poll read, for a platform layer reading inputs itself
pn_input last_input{};
bool polled = false;

// A picture the engine drew above the console's resolution: the lines a
// transfer took from the surface, sampled straight (an LCD column is a line,
// its rows run up the surface's columns). p.x: flipped.
const char* kPictureWgsl = R"(
@group(0) @binding(0) var picture: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var<uniform> p: vec4u;
struct V { @builtin(position) position: vec4f, @location(0) uv: vec2f, }
@vertex fn vs(@builtin(vertex_index) i: u32) -> V {
  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var v: V;
  v.position = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
  v.uv = uv;
  return v;
}
// p.y: the filter (0 bilinear, 1 nearest, 2 sharp bilinear); p.z: effects (1 FXAA, 2 sharpen, 4 the 3DS LCD's
// colors, 8 scanlines); all 0 by default: the picture as it was
fn at(t: vec2f) -> vec3f { return textureSampleLevel(picture, linear_sampler, t, 0.0).rgb; }
fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.299, 0.587, 0.114)); }
// the scanlines: a dark line on the last window pixels of each of the console's 240 rows, one pixel wide (a fifth
// of a row at least, half at most), its edge smoothed at sizes between whole multiples. Not sin(fract(row) * pi),
// symmetric: at 2x both pixels of a row took the same value, no line seen
fn scanline(rows: f32) -> f32 {
  let fw = max(fwidth(rows), 1e-4);
  let width = clamp(fw, 0.2, 0.5);
  return 1.0 - 0.35 * clamp((fract(rows) - (1.0 - width)) / fw + 0.5, 0.0, 1.0);
}
@fragment fn fs(v: V) -> @location(0) vec4f {
  let t = vec2f(1.0 - v.uv.y, select(v.uv.x, 1.0 - v.uv.x, p.x != 0u));
  let size = vec2f(textureDimensions(picture));
  let texel = t * size;
  let fw = max(fwidth(texel), vec2f(1e-5));
  var uv = t;
  if (p.y == 1u) { uv = (floor(texel) + 0.5) / size; }
  if (p.y == 2u) { uv = (floor(texel) + 0.5 + clamp((fract(texel) - 0.5) / fw, vec2f(-0.5), vec2f(0.5))) / size; }
  var c = at(uv);
  let d = 1.0 / size;
  if ((p.z & 3u) != 0u) {
    let n = at(uv + vec2f(0.0, -d.y)); let s = at(uv + vec2f(0.0, d.y));
    let e = at(uv + vec2f(d.x, 0.0)); let w = at(uv + vec2f(-d.x, 0.0));
    if ((p.z & 1u) != 0u) { // FXAA, its light form: where the contrast is an edge, blend across it
      let ln = luma(n); let ls = luma(s); let le = luma(e); let lw = luma(w); let lc = luma(c);
      let range = max(max(max(ln, ls), max(le, lw)), lc) - min(min(min(ln, ls), min(le, lw)), lc);
      if (range > max(0.0312, 0.125 * max(lc, max(max(ln, ls), max(le, lw))))) {
        let along_x = abs(ln + ls - 2.0 * lc) >= abs(le + lw - 2.0 * lc);
        let blend = select((e + w) * 0.5, (n + s) * 0.5, along_x);
        c = mix(c, blend, clamp(range * 2.0, 0.0, 0.5));
      }
    }
    if ((p.z & 2u) != 0u) { c = clamp(c * 1.6 - (n + s + e + w) * 0.15, vec3f(0.0), vec3f(1.0)); }
  }
  if ((p.z & 4u) != 0u) { // as the console's screen shows it: a little darker, its colors bleeding a little
    let g = pow(c, vec3f(1.12));
    c = clamp(mat3x3f(0.86, 0.07, 0.04, 0.10, 0.86, 0.09, 0.04, 0.07, 0.87) * g, vec3f(0.0), vec3f(1.0));
  }
  if ((p.z & 8u) != 0u) { // lines between the screen's rows (an LCD row is a texel column of the picture)
    c = c * scanline(t.x * 240.0); // the console's rows, whatever the picture's resolution scale
  }
  return vec4f(c, 1.0);
}
)";

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
  wgpu::RenderPipeline pipeline, picture_pipeline;
  wgpu::Sampler smooth;
  wgpu::Buffer upload[4]; // per screen, and the top's right eye
  uint32_t upload_size[4] = {};
  SDL_Gamepad* gamepad = nullptr;
  SDL_AudioStream* audio = nullptr;
  bool touching = false;
  float touch_x = 0, touch_y = 0; // window pixels

  // where the screens go in the window, letterboxed at its aspect
  // where the screens go in the window: each layout's own arrangement (in
  // the console's pixels), as large as the window allows, centred; a hidden
  // screen has no size
  int layout = PN_LAYOUT_STACKED;
  uint32_t filter = 0, effects = 0; // pn_window_set_filter
  int stereo = 0;                    // pn_window_set_stereo
  bool shot_asked = false;           // F12
  wgpu::RenderPipeline picture_masked[2], memory_masked[2]; // red alone, green and blue: an anaglyph's eyes
  struct Rect {
    float x, y, w, h;
  };
#if defined(__ANDROID__)
  ANativeWindow* second_window = nullptr;
  wgpu::Surface second_surface;
  int second_width = 0, second_height = 0;
  bool second_on() const { return second_surface != nullptr; }
#else
  bool second_on() const { return false; }
#endif
  // the bottom screen on the second display, as large as it allows, centred
  Rect second_place() const {
#if defined(__ANDROID__)
    const float scale = std::min(second_width / 320.0f, second_height / 240.0f);
    return {(second_width - 320 * scale) / 2, (second_height - 240 * scale) / 2, 320 * scale, 240 * scale};
#else
    return {0, 0, 0, 0};
#endif
  }
  void screens(Rect out[2]) const {
    Rect top{0, 0, 400, 240}, bottom{40, 240, 320, 240};
    float cw = 400, ch = 480;
    // with a second display, the bottom screen is there
    switch (second_on() ? PN_LAYOUT_TOP_ONLY : layout) {
    case PN_LAYOUT_SIDE_BY_SIDE: bottom = {400, 0, 320, 240}, cw = 720, ch = 240; break;
    case PN_LAYOUT_LARGE_TOP: bottom = {400, 120, 160, 120}, cw = 560, ch = 240; break; // the bottom at half size
    case PN_LAYOUT_TOP_ONLY: bottom = {0, 0, 0, 0}, cw = 400, ch = 240; break;
    default: break;
    }
    const float scale = std::min(width / cw, height / ch);
    const float ox = (width - cw * scale) / 2, oy = (height - ch * scale) / 2;
    for (int i = 0; i < 2; ++i) {
      const Rect& r = i ? bottom : top;
      out[i] = {ox + r.x * scale, oy + r.y * scale, r.w * scale, r.h * scale};
    }
  }
  bool fullscreen = false;

  // a surface on the window's native window; Android takes it away in the
  // background and gives another back (attach again then)
  bool attach() {
    gpu::Context& g = engine->gpu;
    const SDL_PropertiesID properties = SDL_GetWindowProperties(window);
#if defined(_WIN32)
    wgpu::SurfaceSourceWindowsHWND source{};
    source.hwnd = SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    source.hinstance = SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_WIN32_INSTANCE_POINTER, nullptr);
#elif defined(__ANDROID__)
    wgpu::SurfaceSourceAndroidNativeWindow source{};
    source.window = SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_ANDROID_WINDOW_POINTER, nullptr);
    if (!source.window) return false;
    native = source.window;
#else
#error "picanative's window: no surface for this platform yet"
#endif
    wgpu::SurfaceDescriptor sd{};
    sd.nextInChain = &source;
    surface = g.instance.CreateSurface(&sd);
    wgpu::SurfaceCapabilities capabilities;
    if (!surface || !surface.GetCapabilities(g.adapter, &capabilities) || capabilities.formatCount == 0) {
      surface = nullptr;
      return false;
    }
    // the screens' colors as they are: a format that does not convert them to sRGB
    format = capabilities.formats[0];
    for (size_t i = 0; i < capabilities.formatCount; ++i)
      if (capabilities.formats[i] == wgpu::TextureFormat::BGRA8Unorm ||
          capabilities.formats[i] == wgpu::TextureFormat::RGBA8Unorm) {
        format = capabilities.formats[i];
        break;
      }
    configure();
    return true;
  }
  void detach() {
    if (surface) surface.Unconfigure();
    surface = nullptr;
  }

#if defined(__ANDROID__)
  void* native = nullptr; // the native window the surface is on
  // the focus back: a surface again if the old one was lost, or on the
  // window Android made meanwhile (one window takes one surface at a time)
  void reattach() {
    void* now = SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_ANDROID_WINDOW_POINTER, nullptr);
    if (surface && now == native) return;
    detach();
    attach();
  }
  // the second display's window changed (second.lock held): let the old one
  // go, without touching it (it may be gone), and draw into the new one
  void second_change() {
    if (second.given && second.given == second_window && second_surface) {
      // the same window, resized: one window takes one surface at a time
      ANativeWindow_release(second.given);
      second.given = nullptr;
      second.changed = false;
      second_width = ANativeWindow_getWidth(second_window);
      second_height = ANativeWindow_getHeight(second_window);
      second_configure();
      return;
    }
    if (second_surface) second_surface.Unconfigure();
    second_surface = nullptr;
    if (second_window) ANativeWindow_release(second_window);
    second_window = second.given;
    second.given = nullptr;
    second.changed = false;
    if (!second_window) return;
    gpu::Context& g = engine->gpu;
    wgpu::SurfaceSourceAndroidNativeWindow source{};
    source.window = second_window;
    wgpu::SurfaceDescriptor sd{};
    sd.nextInChain = &source;
    wgpu::Surface made = g.instance.CreateSurface(&sd);
    wgpu::SurfaceCapabilities capabilities;
    if (!made || !made.GetCapabilities(g.adapter, &capabilities)) return;
    // the same format as the main window's: its pipelines draw there too
    bool format_ok = false, mailbox = false;
    for (size_t i = 0; i < capabilities.formatCount; ++i) format_ok |= capabilities.formats[i] == format;
    for (size_t i = 0; i < capabilities.presentModeCount; ++i)
      mailbox |= capabilities.presentModes[i] == wgpu::PresentMode::Mailbox;
    if (!format_ok) {
      std::fprintf(stderr, "picanative: the second display does not take the main window's format\n");
      return;
    }
    second_width = ANativeWindow_getWidth(second_window);
    second_height = ANativeWindow_getHeight(second_window);
    // the main window paces the frames: this one does not wait a second time
    second_mode = mailbox ? wgpu::PresentMode::Mailbox : wgpu::PresentMode::Fifo;
    second_surface = made;
    second_configure();
  }
  wgpu::PresentMode second_mode = wgpu::PresentMode::Fifo;
  void second_configure() {
    wgpu::SurfaceConfiguration config{};
    config.device = engine->gpu.device;
    config.format = format;
    config.width = static_cast<uint32_t>(std::max(second_width, 1));
    config.height = static_cast<uint32_t>(std::max(second_height, 1));
    config.presentMode = second_mode;
    second_surface.Configure(&config);
  }
#endif

  void configure() {
    if (!surface) return;
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

namespace {
// the pipelines that draw the screens, for the window's format (or a picture's)
void make_pipelines(pn_window* w) {
  gpu::Context& g = w->engine->gpu;
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
  target.writeMask = wgpu::ColorWriteMask::Red;
  w->memory_masked[0] = g.device.CreateRenderPipeline(&pd);
  target.writeMask = wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue;
  w->memory_masked[1] = g.device.CreateRenderPipeline(&pd);
  target.writeMask = wgpu::ColorWriteMask::All;
  {
    wgpu::ShaderSourceWGSL picture_wgsl{};
    picture_wgsl.code = kPictureWgsl;
    wgpu::ShaderModuleDescriptor pmd{};
    pmd.nextInChain = &picture_wgsl;
    const wgpu::ShaderModule picture_module = g.device.CreateShaderModule(&pmd);
    fs.module = picture_module;
    pd.vertex.module = picture_module;
    w->picture_pipeline = g.device.CreateRenderPipeline(&pd);
    target.writeMask = wgpu::ColorWriteMask::Red;
    w->picture_masked[0] = g.device.CreateRenderPipeline(&pd);
    target.writeMask = wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue;
    w->picture_masked[1] = g.device.CreateRenderPipeline(&pd);
    target.writeMask = wgpu::ColorWriteMask::All;
    wgpu::SamplerDescriptor sd{};
    sd.magFilter = sd.minFilter = wgpu::FilterMode::Linear;
    w->smooth = g.device.CreateSampler(&sd);
  }
}
} // namespace

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
  // PN_WINDOW_AT=x,y: where the window opens (off the screen, for a test that captures it alone)
  if (const char* at = std::getenv("PN_WINDOW_AT")) {
    int x = 0, y = 0;
    if (std::sscanf(at, "%d,%d", &x, &y) == 2) SDL_SetWindowPosition(w->window, x, y);
  }
  gpu::Context& g = engine->gpu;
  if (!w->attach()) {
    SDL_DestroyWindow(w->window);
    delete w;
    return fail("the GPU cannot draw into this window");
  }

  make_pipelines(w);

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

static void compose(pn_window* w, const pn_screen screens[2], const wgpu::Texture* target);
int pn_window_shot(pn_window* w, const pn_screen screens[2], const char* path);

void pn_window_present(pn_window* w, const pn_screen screens[2]) {
  if (!w->surface) return; // in the background (Android): no window to draw into
  wgpu::SurfaceTexture frame;
  w->surface.GetCurrentTexture(&frame);
  if (frame.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal &&
      frame.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
    // lost: Android took the window away; configuring a lost surface loses
    // the GPU device, so let it go until the window has the focus again
    if (frame.status == wgpu::SurfaceGetCurrentTextureStatus::Lost) {
      w->detach();
      return;
    }
    w->configure(); // resized: the next frame shows
    return;
  }
  compose(w, screens, &frame.texture);
  w->surface.Present();
  // F12: what the window shows, at its size, into PN_SCREENSHOT_DIR (or Pictures/ctrnative), a BMP by the time
  if (w->shot_asked) {
    w->shot_asked = false;
    std::filesystem::path dir;
    if (const char* d = std::getenv("PN_SCREENSHOT_DIR")) dir = d;
    else if (const char* home = std::getenv("USERPROFILE")) dir = std::filesystem::path(home) / "Pictures" / "ctrnative";
    else dir = "screenshots";
    std::error_code ignored;
    std::filesystem::create_directories(dir, ignored);
    char name[64];
    const std::time_t now = std::time(nullptr);
    std::strftime(name, sizeof name, "%Y-%m-%d_%H-%M-%S.bmp", std::localtime(&now));
    pn_window_shot(w, screens, (dir / name).string().c_str());
  }
}

// the screens drawn into the window's frame, or into a picture (an offscreen window: no second display)
static void compose(pn_window* w, const pn_screen screens[2], const wgpu::Texture* target) {
  gpu::Context& g = w->engine->gpu;
  Renderer& renderer = *w->engine->renderer;
  // one screen into a pass, at a place in its target
  // mask: 0 all colors, 1 red alone, 2 green and blue (an anaglyph's eyes); right: the right eye's picture
  auto draw = [&](const wgpu::RenderPassEncoder& pass, int i, const pn_window::Rect& place, int mask = 0,
                  bool right = false) {
    if (place.w <= 0) return; // hidden in this layout
    pn_screen s = screens[i];
    if (right) s.address = s.right;
    const int slot = i + (right ? 2 : 0); // its own upload buffer: both eyes go in one pass
    const wgpu::RenderPipeline& picture_pipeline = mask ? w->picture_masked[mask - 1] : w->picture_pipeline;
    const wgpu::RenderPipeline& memory_pipeline = mask ? w->memory_masked[mask - 1] : w->pipeline;
    const uint32_t lcd_width = i == 0 ? 400 : 320;
    const uint32_t format = s.format & 7, bpp = format == 0 ? 4 : format == 1 ? 3 : 2;
    const uint32_t stride = s.stride ? s.stride : 240 * bpp, bytes = stride * lcd_width;
    if (s.address == 0) return; // the LCD is off: the margins' grey
    // drawn above the console's resolution: that picture, straight
    bool flipped = false;
    if (const wgpu::TextureView picture = renderer.picture_view(s.address, flipped)) {
      const uint32_t parameters[4] = {flipped ? 1u : 0u, w->filter, w->effects, 0};
      wgpu::BufferDescriptor ud{};
      ud.size = sizeof parameters;
      ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
      const wgpu::Buffer uniform = g.device.CreateBuffer(&ud);
      g.queue.WriteBuffer(uniform, 0, parameters, sizeof parameters);
      wgpu::BindGroupEntry e[3] = {};
      e[0].binding = 0, e[0].textureView = picture;
      e[1].binding = 1, e[1].sampler = w->smooth;
      e[2].binding = 2, e[2].buffer = uniform, e[2].size = sizeof parameters;
      wgpu::BindGroupDescriptor bd{};
      bd.layout = picture_pipeline.GetBindGroupLayout(0);
      bd.entryCount = 3;
      bd.entries = e;
      pass.SetPipeline(picture_pipeline);
      pass.SetBindGroup(0, g.device.CreateBindGroup(&bd));
      pass.SetViewport(place.x, place.y, place.w, place.h, 0, 1);
      pass.Draw(3);
      return;
    }
    // drawn by a transfer on the GPU: shown from there; otherwise from memory
    wgpu::Buffer source = renderer.picture(s.address, bytes);
    if (!source) {
      renderer.sync(s.address, bytes);
      const uint32_t padded = (bytes + 3) & ~3u;
      if (w->upload_size[slot] < padded) {
        wgpu::BufferDescriptor bd{};
        bd.size = padded;
        bd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        w->upload[slot] = g.device.CreateBuffer(&bd);
        w->upload_size[slot] = padded;
      }
      std::vector<uint8_t> copy(padded, 0);
      if (const uint8_t* from = w->engine->memory.map(s.address, bytes)) std::memcpy(copy.data(), from, bytes);
      else
        for (uint32_t k = 0; k < bytes; ++k)
          if (const uint8_t* b = w->engine->memory.map(s.address + k, 1)) copy[k] = *b;
      g.queue.WriteBuffer(w->upload[slot], 0, copy.data(), padded);
      source = w->upload[slot];
    }
    const uint32_t parameters[4] = {lcd_width, stride, format, w->effects};
    wgpu::BufferDescriptor ud{};
    ud.size = sizeof parameters;
    ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    const wgpu::Buffer uniform = g.device.CreateBuffer(&ud);
    g.queue.WriteBuffer(uniform, 0, parameters, sizeof parameters);
    wgpu::BindGroupEntry e[2] = {};
    e[0].binding = 0, e[0].buffer = source, e[0].size = (bytes + 3) & ~3u;
    e[1].binding = 1, e[1].buffer = uniform, e[1].size = sizeof parameters;
    wgpu::BindGroupDescriptor bd{};
    bd.layout = memory_pipeline.GetBindGroupLayout(0);
    bd.entryCount = 2;
    bd.entries = e;
    pass.SetPipeline(memory_pipeline);
    pass.SetBindGroup(0, g.device.CreateBindGroup(&bd));
    pass.SetViewport(place.x, place.y, place.w, place.h, 0, 1);
    pass.Draw(3);
  };
  // a frame's screens into a target, the margins dark grey
  auto frame_pass = [&](const wgpu::Texture& target, auto&& fill) {
    const wgpu::CommandEncoder encoder = g.device.CreateCommandEncoder();
    wgpu::RenderPassColorAttachment ca{};
    ca.view = target.CreateView();
    ca.loadOp = wgpu::LoadOp::Clear;
    ca.storeOp = wgpu::StoreOp::Store;
    ca.clearValue = {0.094, 0.094, 0.094, 1.0};
    wgpu::RenderPassDescriptor rp{};
    rp.colorAttachmentCount = 1;
    rp.colorAttachments = &ca;
    const wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&rp);
    fill(pass);
    pass.End();
    const wgpu::CommandBuffer commands = encoder.Finish();
    g.queue.Submit(1, &commands);
  };

#if defined(__ANDROID__)
  // the second display first: whether it is there decides the main layout
  {
    std::lock_guard<std::mutex> hold(second.lock);
    if (second.changed) w->second_change();
    if (w->second_surface) {
      wgpu::SurfaceTexture lower;
      w->second_surface.GetCurrentTexture(&lower);
      if (lower.status == wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal ||
          lower.status == wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
        frame_pass(lower.texture, [&](const wgpu::RenderPassEncoder& pass) { draw(pass, 1, w->second_place()); });
        w->second_surface.Present();
      } else if (lower.status == wgpu::SurfaceGetCurrentTextureStatus::Lost) {
        // its window went away: never configure it again (that loses the device)
        w->second_surface.Unconfigure();
        w->second_surface = nullptr;
      }
    }
  }
#endif

  frame_pass(*target, [&](const wgpu::RenderPassEncoder& pass) {
    pn_window::Rect places[2];
    w->screens(places);
    for (int i = 0; i < 2; ++i) {
      // 3D: the top screen's two eyes, when the game draws them
      if (i == 0 && w->stereo && screens[0].right && screens[0].right != screens[0].address) {
        if (w->stereo == 1) {
          draw(pass, 0, places[0], 1, false), draw(pass, 0, places[0], 2, true);
        } else {
          pn_window::Rect left = places[0], right = places[0];
          left.w /= 2, right.w /= 2, right.x += left.w;
          draw(pass, 0, left, 0, false), draw(pass, 0, right, 0, true);
        }
        continue;
      }
      draw(pass, i, places[i]);
    }
  });
}

pn_window* pn_offscreen_open(pn_engine* engine, uint32_t scale) {
  auto w = new pn_window;
  w->engine = engine;
  w->format = wgpu::TextureFormat::RGBA8Unorm;
  w->width = 400 * static_cast<int>(std::max(scale, 1u)), w->height = 480 * static_cast<int>(std::max(scale, 1u));
  make_pipelines(w);
  return w;
}

int pn_window_shot(pn_window* w, const pn_screen screens[2], const char* path) {
  gpu::Context& g = w->engine->gpu;
  wgpu::TextureDescriptor td{};
  td.size = {static_cast<uint32_t>(w->width), static_cast<uint32_t>(w->height), 1};
  td.format = w->format;
  td.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc;
  const wgpu::Texture picture = g.device.CreateTexture(&td);
  compose(w, screens, &picture);
  const uint32_t row = (td.size.width * 4 + 255) & ~255u;
  wgpu::BufferDescriptor bd{};
  bd.size = uint64_t{row} * td.size.height;
  bd.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  const wgpu::Buffer back = g.device.CreateBuffer(&bd);
  wgpu::TexelCopyTextureInfo from{};
  from.texture = picture;
  wgpu::TexelCopyBufferInfo to{};
  to.buffer = back;
  to.layout.bytesPerRow = row;
  to.layout.rowsPerImage = td.size.height;
  const wgpu::CommandEncoder encoder = g.device.CreateCommandEncoder();
  encoder.CopyTextureToBuffer(&from, &to, &td.size);
  const wgpu::CommandBuffer commands = encoder.Finish();
  g.queue.Submit(1, &commands);
  bool mapped = false;
  gpu::wait(g, back.MapAsync(wgpu::MapMode::Read, 0, bd.size, wgpu::CallbackMode::WaitAnyOnly,
                             [&](wgpu::MapAsyncStatus status, wgpu::StringView) { mapped = status == wgpu::MapAsyncStatus::Success; }));
  if (!mapped) return 0;
  const auto* bytes = static_cast<const uint8_t*>(back.GetConstMappedRange(0, bd.size));
  // the target's order: a window's surface is often BGRA
  const bool bgra = td.format == wgpu::TextureFormat::BGRA8Unorm || td.format == wgpu::TextureFormat::BGRA8UnormSrgb;
  const std::string name = path;
  const bool bmp = name.size() > 4 && (name.substr(name.size() - 4) == ".bmp" || name.substr(name.size() - 4) == ".BMP");
  std::FILE* f = std::fopen(path, "wb");
  if (f) {
    const uint32_t w_px = td.size.width, h_px = td.size.height;
    if (bmp) {
      const uint32_t line = (w_px * 3 + 3) & ~3u, data = line * h_px;
      uint8_t header[54] = {'B', 'M'};
      auto put = [&](int at, uint32_t v) { for (int k = 0; k < 4; ++k) header[at + k] = static_cast<uint8_t>(v >> (8 * k)); };
      put(2, 54 + data), put(10, 54), put(14, 40), put(18, w_px), put(22, h_px);
      header[26] = 1, header[28] = 24;
      put(34, data);
      std::fwrite(header, 1, 54, f);
      std::vector<uint8_t> out(line, 0);
      for (uint32_t y = h_px; y-- > 0;) { // bottom up
        for (uint32_t x = 0; x < w_px; ++x) {
          const uint8_t* p = bytes + size_t{y} * row + x * 4;
          out[x * 3] = bgra ? p[0] : p[2], out[x * 3 + 1] = p[1], out[x * 3 + 2] = bgra ? p[2] : p[0];
        }
        std::fwrite(out.data(), 1, line, f);
      }
    } else {
      std::fprintf(f, "P6\n%u %u\n255\n", w_px, h_px);
      for (uint32_t y = 0; y < h_px; ++y)
        for (uint32_t x = 0; x < w_px; ++x) {
          const uint8_t* p = bytes + size_t{y} * row + x * 4;
          const uint8_t rgb[3] = {bgra ? p[2] : p[0], p[1], bgra ? p[0] : p[2]};
          std::fwrite(rgb, 1, 3, f);
        }
    }
    std::fclose(f);
  }
  back.Unmap();
  return f != nullptr;
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
      // F10: the next layout; F11 or Alt+Enter: full screen; F12: a screenshot of the window
      if (event.key.key == SDLK_F10 && !event.key.repeat) w->layout = (w->layout + 1) % 4;
      if (event.key.key == SDLK_F12 && !event.key.repeat) w->shot_asked = true;
      if ((event.key.key == SDLK_F11 || (event.key.key == SDLK_RETURN && (event.key.mod & SDL_KMOD_ALT))) &&
          !event.key.repeat) {
        w->fullscreen = !w->fullscreen;
        SDL_SetWindowFullscreen(w->window, w->fullscreen);
      }
      break;
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: w->configure(); break;
#if defined(__ANDROID__)
    // Android destroys the native window in the background (present lets the
    // lost surface go) and makes another before the focus comes back: a
    // surface on that one. Its app events do not reach this loop.
    case SDL_EVENT_WINDOW_FOCUS_GAINED: w->reattach(); break;
#endif
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
    pn_window::Rect places[2];
    w->screens(places);
    const pn_window::Rect& b = places[1];
    const float x = b.w > 0 ? (w->touch_x - b.x) / b.w * 320 : -1, y = b.h > 0 ? (w->touch_y - b.y) / b.h * 240 : -1;
    if (x >= 0 && x < 320 && y >= 0 && y < 240) {
      in.touching = 1;
      in.touch_x = static_cast<uint16_t>(x), in.touch_y = static_cast<uint16_t>(y);
    }
  }
#if defined(__ANDROID__)
  // the bottom screen on the second display: its touches
  if (w->second_on()) {
    std::lock_guard<std::mutex> hold(second.lock);
    const pn_window::Rect b = w->second_place();
    const float x = (second.u * w->second_width - b.x) / b.w * 320, y = (second.v * w->second_height - b.y) / b.h * 240;
    if (second.touching && x >= 0 && x < 320 && y >= 0 && y < 240) {
      in.touching = 1;
      in.touch_x = static_cast<uint16_t>(x), in.touch_y = static_cast<uint16_t>(y);
    }
  }
#endif
  *input = in;
  last_input = in, polled = true;
  return open ? 1 : 0;
}

int pn_input_now(pn_input* input) {
  if (!polled) return 0;
  *input = last_input;
  return 1;
}

#if defined(__ANDROID__)
void pn_android_second_screen(void* native_window) {
  std::lock_guard<std::mutex> hold(second.lock);
  if (second.given) ANativeWindow_release(second.given);
  second.given = static_cast<ANativeWindow*>(native_window);
  second.changed = true;
  second.touching = false;
}

void pn_android_second_touch(int down, float u, float v) {
  std::lock_guard<std::mutex> hold(second.lock);
  second.touching = down != 0, second.u = u, second.v = v;
}
#endif

void pn_window_audio(pn_window* w, const int16_t* samples, uint32_t frames) {
  if (!w->audio || frames == 0) return;
  // past a fifth of a second queued, the host runs ahead of the sound: drop
  // rather than let the delay grow
  if (SDL_GetAudioStreamQueued(w->audio) > kAudioRate / 5 * 4) return;
  SDL_PutAudioStreamData(w->audio, samples, static_cast<int>(frames * 4));
}

void pn_window_title(pn_window* w, const char* title) { SDL_SetWindowTitle(w->window, title); }

void pn_window_set_stereo(pn_window* w, int mode) { w->stereo = std::clamp(mode, 0, 2); }

void pn_window_set_filter(pn_window* w, uint32_t filter, uint32_t effects) {
  w->filter = std::min(filter, 2u);
  w->effects = effects & 15u;
}

void pn_window_set_layout(pn_window* w, int layout) {
  w->layout = std::clamp(layout, 0, 3);
  if (w->fullscreen) return;
  // the window takes the layout's shape, at the scale it was opened with
  static const int sizes[4][2] = {{400, 480}, {720, 240}, {560, 240}, {400, 240}};
  int width = 0, height = 0;
  SDL_GetWindowSize(w->window, &width, &height);
  const int scale = std::max(1, (width + 200) / 400);
  SDL_SetWindowSize(w->window, sizes[w->layout][0] * scale, sizes[w->layout][1] * scale);
}

void pn_window_set_vsync(pn_window* w, int on) {
  w->mode = wgpu::PresentMode::Fifo;
  wgpu::SurfaceCapabilities capabilities;
  if (!on && w->surface && w->surface.GetCapabilities(w->engine->gpu.adapter, &capabilities)) {
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
