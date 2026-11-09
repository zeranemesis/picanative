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
find_package(Dawn CONFIG REQUIRED)
add_library(picanative_gpu STATIC src/gpu.cpp)
target_include_directories(picanative_gpu PUBLIC include)
target_link_libraries(picanative_gpu PUBLIC picanative dawn::webgpu_dawn)

# the tools run next to Dawn's libraries
function(picanative_copy_dawn target)
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
    COMMAND_EXPAND_LISTS)
endfunction()

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
  add_executable(pn_shadertest tools/shadertest.cpp)
  target_link_libraries(pn_shadertest PRIVATE picanative_gpu)
  if (WIN32)
    picanative_copy_dawn(pn_shadertest)
  endif ()
endif ()
