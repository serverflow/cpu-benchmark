# Release Build Configuration for Portable CPU Benchmark
# Requirements: 2.1, 3.1, 3.2 - Portable Binary Build and Static Linking
#
# This module configures the build for creating portable release binaries
# that can run on any compatible system without requiring specific CPU features
# or external runtime libraries.

# Ensure this is only included once
if(RELEASE_BUILD_CMAKE_INCLUDED)
    return()
endif()
set(RELEASE_BUILD_CMAKE_INCLUDED TRUE)

message(STATUS "=== Configuring Release Build for Portable Binary ===")

# Force BUILD_PORTABLE to ON for release builds
set(BUILD_PORTABLE ON CACHE BOOL "Build portable binary with runtime SIMD dispatch" FORCE)

# Disable native architecture optimizations
# Requirements: 2.1 - Do NOT use -march=native for release builds
set(ENABLE_NATIVE_ARCH OFF CACHE BOOL "Enable native CPU optimizations" FORCE)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Release optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS "Configuring release optimizations")
    
    if(MSVC)
        # MSVC release optimizations
        add_compile_options(
            /O2           # Maximum optimization
            /Ob2          # Inline function expansion
            /Oi           # Enable intrinsic functions
            /Ot           # Favor fast code
            /GL           # Whole program optimization
        )
        add_link_options(
            /LTCG         # Link-time code generation
            /OPT:REF      # Remove unreferenced functions
            /OPT:ICF      # Identical COMDAT folding
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # GCC/Clang release optimizations
        add_compile_options(
            -O3                    # Maximum optimization
            -fomit-frame-pointer   # Omit frame pointer for more registers
            -ffunction-sections    # Place each function in its own section
            -fdata-sections        # Place each data item in its own section
        )
        # Platform-specific linker flags
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            add_link_options(-Wl,--gc-sections)  # Remove unused sections (Linux only)
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
            add_link_options(-Wl,-dead_strip)    # Remove unused code (macOS)
        endif()
        
        # LTO (Link-Time Optimization) for smaller and faster binaries
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            add_compile_options(-flto)
            add_link_options(-flto)
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            add_compile_options(-flto=thin)
            add_link_options(-flto=thin)
        endif()
    endif()
endif()

# Static linking configuration
# Requirements: 3.1, 3.2, 3.3, 3.4 - Static Linking
message(STATUS "Configuring static linking for portable binary")

if(MSVC)
    # MSVC: Statically link the C++ runtime (/MT flag)
    # Requirements: 3.1 - Static linking for Windows
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)
    message(STATUS "  MSVC: Using static runtime (MultiThreaded)")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # GCC/Clang: Statically link libstdc++ and libgcc
    # Requirements: 3.2 - Static linking for Linux
    add_link_options(-static-libgcc -static-libstdc++)
    message(STATUS "  GCC/Clang: Using static libgcc and libstdc++")
    
    # On Linux, try to link statically with pthread if possible
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Note: Full static linking with -static may cause issues with glibc
        # We use partial static linking for better compatibility
        message(STATUS "  Linux: Using partial static linking for compatibility")
    endif()
endif()

# SIMD baseline configuration
# Requirements: 2.1, 2.2, 2.4 - Portable Binary Build
message(STATUS "Configuring SIMD baseline for portable binary")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    # x86-64: SSE2 is the baseline (guaranteed on all x86-64 CPUs)
    # Requirements: 2.4 - Target baseline x86-64 (SSE2) as minimum requirement
    if(NOT MSVC)
        # GCC/Clang: explicitly set SSE2 baseline
        add_compile_options(-msse2)
    endif()
    message(STATUS "  x86-64: Using SSE2 as baseline (runtime dispatch for higher levels)")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    # ARM64: NEON is always available
    message(STATUS "  ARM64: NEON is baseline (always available)")
endif()

# Disable tests for release builds by default (can be overridden)
if(NOT DEFINED BUILD_TESTS)
    set(BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
    message(STATUS "Tests disabled for release build")
endif()

# Strip symbols for smaller binary (optional, can be overridden)
option(STRIP_RELEASE_BINARY "Strip debug symbols from release binary" ON)
if(STRIP_RELEASE_BINARY AND NOT MSVC)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_link_options(-s)
        message(STATUS "Release binary will be stripped")
    endif()
endif()

# Version information for release
if(NOT DEFINED PROJECT_VERSION)
    set(PROJECT_VERSION "2.0.0" CACHE STRING "Project version")
endif()

message(STATUS "=== Release Build Configuration Complete ===")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Portable: ${BUILD_PORTABLE}")
message(STATUS "  Static Linking: Enabled")
message(STATUS "  SIMD Dispatch: Runtime")
