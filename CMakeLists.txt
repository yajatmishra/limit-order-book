cmake_minimum_required(VERSION 3.20)

project(
  limit_order_book
  VERSION     0.3.0
  DESCRIPTION "C++17 limit order book — core + Python research stack"
  LANGUAGES   CXX
)

# ── Standard options ──────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # for clangd / IDEs

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── Compiler flags ────────────────────────────────────────────────────────────
add_compile_options(
  -Wall -Wextra -Wpedantic -Wshadow
  -Wno-unused-parameter        # suppress in test scaffolding
  # NOTE: each flag needs its own generator expression — multiple
  # space-separated flags inside one $<...> are not split correctly
  # and leak the literal $<...> text to the compiler.
  "$<$<CONFIG:Release>:-O3>"
  "$<$<CONFIG:Release>:-DNDEBUG>"
  "$<$<CONFIG:RelWithDebInfo>:-O2>"
  "$<$<CONFIG:RelWithDebInfo>:-g>"
  "$<$<CONFIG:RelWithDebInfo>:-DNDEBUG>"
  "$<$<CONFIG:Debug>:-g>"
  "$<$<CONFIG:Debug>:-O0>"
  "$<$<CONFIG:Debug>:-fno-omit-frame-pointer>"
)

# Address + Undefined Sanitizers in Debug (disable for CI speed if needed)
option(LOB_SANITIZE "Enable ASan + UBSan in Debug builds" ON)
if(LOB_SANITIZE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-fsanitize=address,undefined)
  add_link_options(   -fsanitize=address,undefined)
endif()

# ── Threads (required by ring-buffer / shm tests) ─────────────────────────────
find_package(Threads REQUIRED)

# ── Catch2 via FetchContent ───────────────────────────────────────────────────
# Offline CI: set FETCHCONTENT_FULLY_DISCONNECTED=ON and pre-populate the
# cache, or set LOB_USE_SYSTEM_CATCH2=ON to link a system-installed Catch2.
option(LOB_USE_SYSTEM_CATCH2 "Use system Catch2 instead of FetchContent" OFF)

if(LOB_USE_SYSTEM_CATCH2)
  find_package(Catch2 3 REQUIRED)
else()
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.4
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(Catch2)
endif()

# Convenience: include Catch CMake helpers (catch_discover_tests)
include(CTest)
include(Catch)

# ════════════════════════════════════════════════════════════════════════════════
# Shared libraries
# ════════════════════════════════════════════════════════════════════════════════

# ── core_lob ──────────────────────────────────────────────────────────────────
add_library(core_lob STATIC
  core/lob/price_level.cpp
  core/lob/limit_order_book.cpp
)
target_include_directories(core_lob PUBLIC ${CMAKE_SOURCE_DIR})
set_target_properties(core_lob PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ── core_feed ─────────────────────────────────────────────────────────────────
add_library(core_feed STATIC
  core/feed_handler/itch_parser.cpp
  core/feed_handler/pcap_replayer.cpp
  core/feed_handler/feed_handler.cpp
)
target_include_directories(core_feed PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(core_feed PUBLIC core_lob)
set_target_properties(core_feed PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ── core_shm ──────────────────────────────────────────────────────────────────
add_library(core_shm STATIC
  core/shared_memory/shm_writer.cpp
)
target_include_directories(core_shm PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(core_shm PUBLIC Threads::Threads)
set_target_properties(core_shm PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ── core_bus ──────────────────────────────────────────────────────────────────
add_library(core_bus STATIC
  core/event_bus/event_bus.cpp
)
target_include_directories(core_bus PUBLIC ${CMAKE_SOURCE_DIR})
set_target_properties(core_bus PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ── core_exec ─────────────────────────────────────────────────────────────────
add_library(core_exec STATIC
  core/execution/order_router.cpp
  core/execution/fill_simulator.cpp
)
target_include_directories(core_exec PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(core_exec PUBLIC core_lob core_bus)
set_target_properties(core_exec PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ── lob (umbrella interface lib, links everything) ────────────────────────────
add_library(lob INTERFACE)
target_link_libraries(lob INTERFACE
  core_lob
  core_feed
  core_shm
  core_bus
  core_exec
  Threads::Threads
)

# ════════════════════════════════════════════════════════════════════════════════
# Test executables
# ════════════════════════════════════════════════════════════════════════════════

enable_testing()

# Helper function: add one test exe, link it, register with CTest/Catch
function(lob_add_test target source)
  add_executable(${target} ${source})
  # Our test files include "../../third_party/catch.hpp" which is a v1/v2
  # compatibility shim.  For Catch2 v3 the real headers are needed.
  # We inject the real Catch2 include dir *before* the shim path so both work.
  target_include_directories(${target} SYSTEM BEFORE PRIVATE
    $<TARGET_PROPERTY:Catch2::Catch2,INTERFACE_INCLUDE_DIRECTORIES>
  )
  target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR})
  target_link_libraries(${target} PRIVATE ${ARGN} Catch2::Catch2WithMain)
  catch_discover_tests(${target})
endfunction()

# ── LOB tests ─────────────────────────────────────────────────────────────────
lob_add_test(test_lob
  tests/cpp/test_lob.cpp
  core_lob
)

# ── ITCH parser + pcap replayer tests ────────────────────────────────────────
lob_add_test(test_itch_parser
  tests/cpp/test_itch_parser.cpp
  core_feed
)

# ── Ring buffer / FeedHandler / ShmWriter tests ───────────────────────────────
lob_add_test(test_ring_buffer
  tests/cpp/test_ring_buffer.cpp
  core_feed core_shm Threads::Threads
)

# ── Event bus / OrderRouter / FillSimulator tests ─────────────────────────────
lob_add_test(test_event_bus
  tests/cpp/test_event_bus.cpp
  lob
)

# ════════════════════════════════════════════════════════════════════════════════
# Installation
# ════════════════════════════════════════════════════════════════════════════════

include(GNUInstallDirs)

install(
  TARGETS     core_lob core_feed core_shm core_bus core_exec
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  DIRECTORY   core/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lob
  FILES_MATCHING PATTERN "*.hpp"
)

# ════════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════════

message(STATUS "")
message(STATUS "  ── Limit Order Book ${PROJECT_VERSION} ──────────────────────")
message(STATUS "  Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler        : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Sanitizers      : ${LOB_SANITIZE}")
message(STATUS "  System Catch2   : ${LOB_USE_SYSTEM_CATCH2}")
message(STATUS "  Install prefix  : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "──────────────────────────────────────────────────────")
message(STATUS "")
