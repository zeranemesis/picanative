/* picanative's C interface, for a port's host written in any language.
   The host owns the game's memory and hands the engine a way to reach it by
   physical address; the engine draws with its own WebGPU device and leaves
   its results in that memory. */
#ifndef PICANATIVE_API_H
#define PICANATIVE_API_H

#include <stdint.h>

#ifdef _WIN32
#ifdef PICANATIVE_BUILDING
#define PN_API __declspec(dllexport)
#else
#define PN_API __declspec(dllimport)
#endif
#else
#define PN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* host bytes for [address, address + size) of the game's physical memory,
   contiguous, or null */
typedef uint8_t* (*pn_map_fn)(void* user, uint32_t address, uint32_t size);

typedef struct pn_engine pn_engine;

/* null, with the reason in error (when not null), if no GPU is usable */
PN_API pn_engine* pn_create(pn_map_fn map, void* user, char* error, uint32_t error_size);
PN_API void pn_destroy(pn_engine* engine);

/* the GPU operations, as GSP takes them; addresses are physical */
PN_API void pn_command_list(pn_engine* engine, uint32_t address, uint32_t size);
PN_API void pn_memory_fill(pn_engine* engine, uint32_t start, uint32_t end, uint32_t value, uint32_t width);
PN_API void pn_display_transfer(pn_engine* engine, uint32_t input, uint32_t output, uint32_t input_dimensions,
                                uint32_t output_dimensions, uint32_t flags);
PN_API void pn_texture_copy(pn_engine* engine, uint32_t input, uint32_t output, uint32_t size, uint32_t input_gap,
                            uint32_t output_gap);

/* memory the CPU is about to read gets what the GPU drew there; memory the
   CPU wrote is what the GPU reads next */
PN_API void pn_sync_for_cpu(pn_engine* engine, uint32_t address, uint32_t size);
PN_API void pn_cpu_wrote(pn_engine* engine, uint32_t address, uint32_t size);
/* memory drawn into on the GPU and not in memory yet (pending 1), then written there (0): the console's GPU writes
   at once, so a host that sees the CPU write there first calls pn_cpu_wrote (with surfaces cached) */
typedef void (*pn_pending_fn)(void* user, uint32_t address, uint32_t size, int pending);
PN_API void pn_set_pending_hook(pn_engine* engine, pn_pending_fn hook, void* user);

/* a host that does a game's graphics library natively puts marks in the game's command lists in place of the
   commands the library made (register 0x2F0, which the PICA200 has not): each one met calls this, in the list's
   order, with the mark's value, the host's own operation to run there */
typedef void (*pn_native_fn)(void* user, uint32_t operation);
PN_API void pn_set_native_operations(pn_engine* engine, pn_native_fn run, void* user);
/* what a native operation sets, in place of commands: a register (a draw register draws, with the state as it is);
   vertex shader float uniforms from `first`, four 32-bit floats each, w first as a command list carries them */
PN_API void pn_write_register(pn_engine* engine, uint32_t reg, uint32_t value, uint32_t mask);
/* a whole native operation in one call: steps of
     reg | mask << 16, value                                  a register
     0x80000000 | geometry << 15 | first << 8 | rows, 4 words a row (w first)   float uniforms
     0xC0000000 | mask << 16 | reg, count, count values       one register written count times */
PN_API void pn_run_program(pn_engine* engine, const uint32_t* words, uint32_t count);
/* PN_REG_STATS set: the command words written so far, by register, to stderr */
PN_API void pn_print_register_stats(pn_engine* engine);
PN_API void pn_vertex_uniforms(pn_engine* engine, uint32_t first, const uint32_t* words, uint32_t count);
/* the same for the geometry shader */
PN_API void pn_geometry_uniforms(pn_engine* engine, uint32_t first, const uint32_t* words, uint32_t count);

/* how far the engine may trust the host: PN_CACHE_SURFACES keeps what it
   draws on the GPU (the host routes every fill, transfer and copy through
   the engine, and calls pn_sync_for_cpu / pn_cpu_wrote when the CPU touches
   what the GPU drew); PN_TRUST_CPU_WRITES also keeps decoded textures (the
   host reports every CPU write). 0, the default, is safe with any host. */
#define PN_CACHE_SURFACES 1u
#define PN_TRUST_CPU_WRITES 2u
/* pipelines compile on other threads, draws waiting for one are skipped:
   no stall the first time a shader is met (the disk cache makes it once) */
#define PN_ASYNC_PIPELINES 4u
PN_API pn_engine* pn_create_ex(pn_map_fn map, void* user, uint32_t flags, char* error, uint32_t error_size);
/* draws at this many times the console's resolution (1 to 8; 1, the default,
   is the console's pixels exactly); during a game too, after a short reload. Memory
   always gets the console's pixels; the window shows the scaled pictures. */
PN_API void pn_set_resolution_scale(pn_engine* engine, uint32_t scale);
/* the texture improvements, off by default (PN_ANISOTROPY, PN_MIPMAPS set them at the start): anisotropic
   filtering up to that many samples (0 or 1: off), smaller levels made for textures with none. Between frames,
   during a game: the textures are decoded again and the pipelines made again as they are used */
PN_API void pn_set_texture_options(pn_engine* engine, int anisotropy, int mipmaps);

/* Vertex programs translated ahead of time (by pn_shaders, from the game's
   shader binaries): loaded before the first command list, they are not
   translated while the game runs. 0 with the reason in error, for a missing
   file or one another build of picanative made. */
PN_API int pn_load_shaders(pn_engine* engine, const char* path, char* error, uint32_t error_size);
/* the pipelines the game made in its earlier runs, kept in this file, made again now on Dawn's threads (from its
   disk cache); the new ones are added to it. Returns how many were started. */
PN_API uint32_t pn_warm_pipelines(pn_engine* engine, const char* path);
/* to stdout: pipelines made ahead, textures read from color buffers on the GPU, write-backs merged */
PN_API void pn_print_stats(pn_engine* engine);
/* vertex programs loaded ahead, translated while running (each logged), and
   the microseconds those translations took */
PN_API void pn_shader_stats(pn_engine* engine, uint64_t out[3]);

/* counters, for a status line: draws, pipelines, shader errors */
PN_API void pn_stats(pn_engine* engine, uint64_t out[3]);
/* where the engine's time went, in seconds: recording draws (of which
   textures, then pipelines), encoding, waiting for the GPU (of which storing
   to memory), loading from memory */
PN_API void pn_times(pn_engine* engine, double out[7]);
/* of the textures' time: write-backs, known ones found, hashing, decoding (seconds); calls */
PN_API void pn_texture_times(pn_engine* engine, double out[4], uint64_t calls[4]); /* lookups, decodes, submits, passes */

/* The port's window (built with SDL3): both screens drawn by the engine's
   device, the player's inputs, the console's sound. */
typedef struct pn_window pn_window;

/* a screen as the LCD controller scans it out: the framebuffer's physical
   address (0: off), its color format (the low 3 bits of the register) and
   the bytes from one column to the next (0: 240 pixels' worth) */
typedef struct pn_screen {
  uint32_t address, format, stride;
  uint32_t right; /* the right eye's framebuffer, when the game draws one (3D): 0 or the same otherwise */
} pn_screen;

/* the buttons as the console's HID reports them */
#define PN_A (1u << 0)
#define PN_B (1u << 1)
#define PN_SELECT (1u << 2)
#define PN_START (1u << 3)
#define PN_RIGHT (1u << 4)
#define PN_LEFT (1u << 5)
#define PN_UP (1u << 6)
#define PN_DOWN (1u << 7)
#define PN_R (1u << 8)
#define PN_L (1u << 9)
#define PN_X (1u << 10)
#define PN_Y (1u << 11)
#define PN_ZL (1u << 14)
#define PN_ZR (1u << 15)

typedef struct pn_input {
  uint32_t buttons;
  float circle_x, circle_y;   /* -1 to 1, up and right positive */
  float c_stick_x, c_stick_y; /* the same */
  uint16_t touch_x, touch_y;  /* in the bottom screen's pixels, when touching */
  uint8_t touching;
} pn_input;

/* opens the window, 400x480 times scale, or null with the reason */
PN_API pn_window* pn_window_open(pn_engine* engine, const char* title, uint32_t scale, char* error,
                                 uint32_t error_size);
PN_API void pn_window_close(pn_window* window);
/* shows the top and bottom screens; waits for the display (vsync) */
PN_API void pn_window_present(pn_window* window, const pn_screen screens[2]);
/* reads the inputs; 0 once the player closed the window (or pressed Escape) */
PN_API int pn_window_poll(pn_window* window, pn_input* input);
/* plays stereo samples at the DSP's 32728 Hz, interleaved left, right */
PN_API void pn_window_audio(pn_window* window, const int16_t* samples, uint32_t frames);
PN_API void pn_window_title(pn_window* window, const char* title);
/* 1 (the default): presenting waits for the display; 0: it does not, for a
   host that paces its frames itself */
PN_API void pn_window_set_vsync(pn_window* window, int on);
/* how the two screens share the window (F10 cycles them, F11 or Alt+Enter
   toggles full screen): the console's, one over the other; side by side; a
   large top screen with the bottom one at half size beside it; the top one
   alone (touch is then off) */
#define PN_LAYOUT_STACKED 0
#define PN_LAYOUT_SIDE_BY_SIDE 1
#define PN_LAYOUT_LARGE_TOP 2
#define PN_LAYOUT_TOP_ONLY 3
PN_API void pn_window_set_layout(pn_window* window, int layout);
/* how the screens are shown, improvements off by default (0, 0): filter 0 bilinear, 1 nearest, 2 sharp bilinear;
   effects, or'ed: PN_EFFECT_FXAA (edges smoothed), PN_EFFECT_SHARPEN, PN_EFFECT_LCD_COLORS (the 3DS screen's
   colors), PN_EFFECT_SCANLINES */
enum { PN_EFFECT_FXAA = 1, PN_EFFECT_SHARPEN = 2, PN_EFFECT_LCD_COLORS = 4, PN_EFFECT_SCANLINES = 8 };
PN_API void pn_window_set_filter(pn_window* window, uint32_t filter, uint32_t effects);
/* a window with no window: what one at that scale would show (layout, filter, effects), saved by
   pn_window_shot as a PPM picture, to check the screens' look with no window on the screen */
PN_API pn_window* pn_offscreen_open(pn_engine* engine, uint32_t scale);
PN_API int pn_window_shot(pn_window* window, const pn_screen screens[2], const char* path);
/* the top screen in 3D, when the game draws both eyes: 0 off (the left eye, as before), 1 anaglyph red and cyan,
   2 side by side (each eye half as wide, for 3D televisions and headsets) */
PN_API void pn_window_set_stereo(pn_window* window, int mode);
/* the inputs the last pn_window_poll read, for code that reads them on its
   own (a game's platform layer finding this export at run time); 0 before
   any window polled */
PN_API int pn_input_now(pn_input* input);

/* Android: a second display (a dual-screen handheld's lower screen). The
   app hands over the native window (an ANativeWindow* it acquired; picanative
   releases it) of a surface on that display, null when the surface goes away
   (call before returning from surfaceDestroyed): the bottom screen is drawn
   there and the main window shows the top one. Then its touches, across the
   surface (0..1). Any thread. */
PN_API void pn_android_second_screen(void* native_window);
PN_API void pn_android_second_touch(int down, float u, float v);

#ifdef __cplusplus
}
#endif

#endif
