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
   is the console's pixels exactly); before the first command list. Memory
   always gets the console's pixels; the window shows the scaled pictures. */
PN_API void pn_set_resolution_scale(pn_engine* engine, uint32_t scale);

/* counters, for a status line: draws, pipelines, shader errors */
PN_API void pn_stats(pn_engine* engine, uint64_t out[3]);
/* where the engine's time went, in seconds: recording draws (of which
   textures, then pipelines), encoding, waiting for the GPU (of which storing
   to memory), loading from memory */
PN_API void pn_times(pn_engine* engine, double out[7]);

/* The port's window (built with SDL3): both screens drawn by the engine's
   device, the player's inputs, the console's sound. */
typedef struct pn_window pn_window;

/* a screen as the LCD controller scans it out: the framebuffer's physical
   address (0: off), its color format (the low 3 bits of the register) and
   the bytes from one column to the next (0: 240 pixels' worth) */
typedef struct pn_screen {
  uint32_t address, format, stride;
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
/* the inputs the last pn_window_poll read, for code that reads them on its
   own (a game's platform layer finding this export at run time); 0 before
   any window polled */
PN_API int pn_input_now(pn_input* input);

#ifdef __cplusplus
}
#endif

#endif
