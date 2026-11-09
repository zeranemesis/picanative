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
  src/transfer.cpp
  src/vertex.cpp
)
target_include_directories(picanative PUBLIC include)
if (MSVC)
  target_compile_options(picanative PRIVATE /W4 /permissive-)
else ()
  target_compile_options(picanative PRIVATE -Wall -Wextra)
endif ()

option(PICANATIVE_TOOLS "Build the test tools" ON)
if (PICANATIVE_TOOLS)
  add_executable(pn_optest tools/optest.cpp)
  target_link_libraries(pn_optest PRIVATE picanative)
  add_executable(pn_state tools/state.cpp)
  target_link_libraries(pn_state PRIVATE picanative)
endif ()
