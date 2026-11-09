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

/* counters, for a status line: draws, pipelines, shader errors */
PN_API void pn_stats(pn_engine* engine, uint64_t out[3]);

#ifdef __cplusplus
}
#endif

#endif
