cmake_minimum_required(VERSION 3.25)
project(picanative LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Engine core: pure modules, no graphics API (see DESIGN.md).
add_library(picanative STATIC
  src/color.cpp
  src/pica.cpp
  src/trace.cpp
  src/shadergen_vertex.cpp
  src/texture.cpp
  src/transfer.cpp
  src/vertex.cpp
)
target_include_directories(picanative PUBLIC include)
if (MSVC)
  target_compile_options(picanative PRIVATE /W4 /permissive-)
else ()
  target_compile_options(picanative PRIVATE -Wall -Wextra)
endif ()

# GPU layer: Dawn (WebGPU), found as Aurora ships it (a prebuilt install tree,
# https://github.com/encounter/dawn-build), through CMAKE_PREFIX_PATH or Dawn_DIR.
if (ANDROID)
  # Dawn as Aurora ships it for Android: a static library (its CMake config
  # names a liblog path on the machine that built it, so the target is ours)
  set(PICANATIVE_DAWN_ANDROID "" CACHE PATH "Dawn prebuilt for Android (include/, lib/libwebgpu_dawn.a)")
  add_library(dawn::webgpu_dawn STATIC IMPORTED)
  set_target_properties(dawn::webgpu_dawn PROPERTIES
    IMPORTED_LOCATION "${PICANATIVE_DAWN_ANDROID}/lib/libwebgpu_dawn.a"
    INTERFACE_INCLUDE_DIRECTORIES "${PICANATIVE_DAWN_ANDROID}/include"
    INTERFACE_LINK_LIBRARIES "log;android;dl")
else ()
  find_package(Dawn CONFIG REQUIRED)
endif ()
add_library(picanative_gpu STATIC src/gpu.cpp src/draw.cpp src/renderer.cpp)
target_include_directories(picanative_gpu PUBLIC include)
target_link_libraries(picanative_gpu PUBLIC picanative dawn::webgpu_dawn)

# the engine as a shared library with a C interface (include/picanative/api.h)
add_library(picanative_api SHARED src/api.cpp)
target_link_libraries(picanative_api PRIVATE picanative_gpu)
target_include_directories(picanative_api PUBLIC include)
set_target_properties(picanative_api PROPERTIES OUTPUT_NAME picanative_engine)

# the window (src/window.cpp), with SDL3 as Aurora ships it: through
# CMAKE_PREFIX_PATH or SDL3_DIR
set(PICANATIVE_SDL3_SOURCE "" CACHE PATH "SDL3's sources, to build it with the engine (Android)")
if (PICANATIVE_SDL3_SOURCE)
  set(SDL_SHARED ON CACHE BOOL "" FORCE)
  set(SDL_STATIC OFF CACHE BOOL "" FORCE)
  set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
  add_subdirectory("${PICANATIVE_SDL3_SOURCE}" SDL3 EXCLUDE_FROM_ALL)
  set(SDL3_FOUND TRUE)
else ()
  find_package(SDL3 CONFIG)
endif ()
if (SDL3_FOUND)
  target_sources(picanative_api PRIVATE src/window.cpp)
  target_link_libraries(picanative_api PRIVATE SDL3::SDL3)
endif ()

# the tools run next to Dawn's libraries
function(picanative_copy_dawn target)
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
    COMMAND_EXPAND_LISTS)
endfunction()

if (WIN32)
  add_custom_command(TARGET picanative_api POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:picanative_api> $<TARGET_FILE_DIR:picanative_api>
    COMMAND_EXPAND_LISTS)
endif ()

option(PICANATIVE_TOOLS "Build the test tools" ON)
if (PICANATIVE_TOOLS)
  add_executable(pn_optest tools/optest.cpp)
  target_link_libraries(pn_optest PRIVATE picanative)
  add_executable(pn_state tools/state.cpp)
  add_executable(pn_texturetest tools/texturetest.cpp)
  target_link_libraries(pn_texturetest PRIVATE picanative)
  add_executable(pn_stats tools/stats.cpp)
  target_link_libraries(pn_stats PRIVATE picanative)
  target_link_libraries(pn_state PRIVATE picanative)
  add_executable(pn_render tools/render.cpp)
  target_link_libraries(pn_render PRIVATE picanative_gpu)
  add_executable(pn_shadertest tools/shadertest.cpp)
  target_link_libraries(pn_shadertest PRIVATE picanative_gpu)
  if (WIN32)
    picanative_copy_dawn(pn_shadertest)
    picanative_copy_dawn(pn_render)
  endif ()
endif ()
