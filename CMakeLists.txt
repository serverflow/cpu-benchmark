cmake_minimum_required(VERSION 3.14)
project(cpu_benchmark VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Options
option(ENABLE_OPENMP "Enable OpenMP support" OFF)
option(BUILD_TESTS "Build tests" ON)
option(ENABLE_NATIVE_ARCH "Enable native CPU optimizations (-march=native)" ON)
option(BUILD_PORTABLE "Build portable binary with runtime SIMD dispatch" OFF)
option(USE_RELEASE_BUILD_CONFIG "Include release build configuration module" OFF)

# Include release build configuration if requested
# This provides optimized settings for portable release builds
if(USE_RELEASE_BUILD_CONFIG)
    include(release_build OPTIONAL)
endif()

# Static linking configuration for portable builds

if(BUILD_PORTABLE)
    message(STATUS "Building portable binary with static linking")
    
    if(MSVC)
        # MSVC: Statically link the C++ runtime (/MT flag)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        message(STATUS "MSVC: Using static runtime (MultiThreaded)")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # GCC/Clang: Statically link libstdc++ and libgcc
        add_link_options(-static-libgcc -static-libstdc++)
        message(STATUS "GCC/Clang: Using static libgcc and libstdc++")
    endif()
endif()

# SIMD optimization configuration
if(BUILD_PORTABLE)
    # Portable build: compile with baseline only, use runtime dispatch
    # Do NOT use -march=native for release builds
    message(STATUS "Building portable binary with runtime SIMD dispatch")
    
    if(MSVC)
        # MSVC: baseline is SSE2 for x64
        # No global arch flag needed - SSE2 is default for x64
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # GCC/Clang: compile main code with baseline
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
            add_compile_options(-msse2)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
            # ARM64: use ARMv8.2-a with FP16 as baseline (covers Apple Silicon, modern ARM servers)
            # This enables __ARM_FEATURE_FP16_VECTOR_ARITHMETIC for native FP16 support
            add_compile_options(-march=armv8.2-a+fp16+simd)
            message(STATUS "ARM64: Enabling FP16 vector arithmetic (-march=armv8.2-a+fp16+simd)")
        endif()
    endif()
else()
    # Development build: use native optimizations
    if(MSVC)
        # MSVC: Enable AVX2 for all targets
        add_compile_options(/arch:AVX2)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
            # ARM64 development build: always use armv8.2-a+fp16+simd
            # -march=native on Apple Clang may not enable FP16 vector arithmetic
            # Explicitly enable it for consistent behavior
            add_compile_options(-march=armv8.2-a+fp16+simd)
            message(STATUS "ARM64 dev build: Enabling FP16 vector arithmetic (-march=armv8.2-a+fp16+simd)")
        elseif(ENABLE_NATIVE_ARCH)
            include(CheckCXXCompilerFlag)
            check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
            if(COMPILER_SUPPORTS_MARCH_NATIVE)
                add_compile_options(-march=native)
            endif()
        else()
            # Fallback to explicit AVX2
            include(CheckCXXCompilerFlag)
            check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
            if(COMPILER_SUPPORTS_AVX2)
                add_compile_options(-mavx2 -mfma)
            endif()
        endif()
    endif()
endif()

# Source files
set(BENCHMARK_SOURCES
    src/main.cpp
    src/platform.cpp
    src/threading.cpp
    src/benchmark_core.cpp
    src/memory_validator.cpp
    src/runtime_dispatcher.cpp
    src/progress_reporter.cpp
    src/auto_size.cpp
    src/warmup.cpp
    src/score_calculator.cpp
    src/thread_affinity.cpp
    src/persistent_thread_pool.cpp
    src/result_submission.cpp
)

# Kernel source files for runtime SIMD dispatch
set(KERNEL_SOURCES
    src/kernels/kernel_scalar.cpp
    src/kernels/kernel_sse2.cpp
    src/kernels/kernel_avx.cpp
    src/kernels/kernel_avx2.cpp
    src/kernels/kernel_avx512.cpp
    src/kernels/kernel_neon.cpp
    src/kernels/kernel_compute.cpp
    src/kernels/kernel_compute_avx512.cpp
)

# Main executable
add_executable(cpu_benchmark ${BENCHMARK_SOURCES} ${KERNEL_SOURCES})
target_include_directories(cpu_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/kernels)

# Set per-file compile flags for SIMD kernels
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i686|x86")
    # x86/x64 platform
    if(MSVC)
        # MSVC compile flags
        set_source_files_properties(src/kernels/kernel_scalar.cpp PROPERTIES COMPILE_FLAGS "")
        set_source_files_properties(src/kernels/kernel_sse2.cpp PROPERTIES COMPILE_FLAGS "/arch:SSE2")
        set_source_files_properties(src/kernels/kernel_avx.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX")
        set_source_files_properties(src/kernels/kernel_avx2.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
        set_source_files_properties(src/kernels/kernel_avx512.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX512")
        # Compute kernel base needs AVX2 for FMA support
        set_source_files_properties(src/kernels/kernel_compute.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
        # AVX-512 compute kernel needs /arch:AVX512
        set_source_files_properties(src/kernels/kernel_compute_avx512.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX512")
        # NEON not applicable on x86
        set_source_files_properties(src/kernels/kernel_neon.cpp PROPERTIES COMPILE_FLAGS "")
    else()
        # GCC/Clang compile flags
        set_source_files_properties(src/kernels/kernel_scalar.cpp PROPERTIES COMPILE_FLAGS "")
        set_source_files_properties(src/kernels/kernel_sse2.cpp PROPERTIES COMPILE_FLAGS "-msse2")
        set_source_files_properties(src/kernels/kernel_avx.cpp PROPERTIES COMPILE_FLAGS "-mavx")
        set_source_files_properties(src/kernels/kernel_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
        set_source_files_properties(src/kernels/kernel_avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512dq")
        # Compute kernel base needs AVX2 and FMA for optimal performance
        set_source_files_properties(src/kernels/kernel_compute.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
        # AVX-512 compute kernel needs AVX-512 flags
        set_source_files_properties(src/kernels/kernel_compute_avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512dq -mfma")
        # NEON not applicable on x86
        set_source_files_properties(src/kernels/kernel_neon.cpp PROPERTIES COMPILE_FLAGS "")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    # ARM64 platform
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(src/kernels/kernel_scalar.cpp PROPERTIES COMPILE_FLAGS "")
        # NEON with FP16 support for Apple Silicon and ARMv8.2+ processors
        # -march=armv8.2-a+fp16+simd enables native FP16 vector arithmetic
        set_source_files_properties(src/kernels/kernel_neon.cpp PROPERTIES COMPILE_FLAGS "-march=armv8.2-a+fp16+simd -O3 -ffast-math")
        # Compute kernel uses NEON with FP16 on ARM - optimized for Apple Silicon
        # -O3: Maximum optimization
        # -ffast-math: Allow aggressive FP optimizations (safe for benchmarks)
        # -funroll-loops: Help compiler unroll remaining loops
        set_source_files_properties(src/kernels/kernel_compute.cpp PROPERTIES COMPILE_FLAGS "-march=armv8.2-a+fp16+simd -O3 -ffast-math -funroll-loops")
        # AVX-512 compute kernel not applicable on ARM
        set_source_files_properties(src/kernels/kernel_compute_avx512.cpp PROPERTIES COMPILE_FLAGS "")
        # x86 SIMD not applicable on ARM
        set_source_files_properties(src/kernels/kernel_sse2.cpp PROPERTIES COMPILE_FLAGS "")
        set_source_files_properties(src/kernels/kernel_avx.cpp PROPERTIES COMPILE_FLAGS "")
        set_source_files_properties(src/kernels/kernel_avx2.cpp PROPERTIES COMPILE_FLAGS "")
        set_source_files_properties(src/kernels/kernel_avx512.cpp PROPERTIES COMPILE_FLAGS "")
    endif()
endif()

# OpenMP support (optional)
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(cpu_benchmark PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(cpu_benchmark PRIVATE USE_OPENMP)
    endif()
endif()

# Platform-specific threading and networking
if(WIN32)
    # Windows threading and WinSock for result submission
    # - advapi32: registry access (RegOpenKeyEx, ...)
    # - pdh: real-time CPU frequency sampling (Processor Frequency counter)
    target_link_libraries(cpu_benchmark PRIVATE ws2_32 advapi32 pdh)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(cpu_benchmark PRIVATE Threads::Threads)
endif()

# Tests with RapidCheck
if(BUILD_TESTS)
    include(FetchContent)
    
    FetchContent_Declare(
        rapidcheck
        GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(rapidcheck)

    enable_testing()
    
    # Library sources (without main.cpp) for tests
    set(LIB_SOURCES
        src/platform.cpp
        src/threading.cpp
        src/benchmark_core.cpp
        src/memory_validator.cpp
        src/runtime_dispatcher.cpp
        src/progress_reporter.cpp
        src/auto_size.cpp
        src/warmup.cpp
        src/score_calculator.cpp
        src/thread_affinity.cpp
        src/persistent_thread_pool.cpp
        src/result_submission.cpp
        src/kernels/kernel_scalar.cpp
        src/kernels/kernel_sse2.cpp
        src/kernels/kernel_avx.cpp
        src/kernels/kernel_avx2.cpp
        src/kernels/kernel_avx512.cpp
        src/kernels/kernel_neon.cpp
        src/kernels/kernel_compute.cpp
        src/kernels/kernel_compute_avx512.cpp
    )
    
    # Common include directories for tests
    set(TEST_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/kernels)
    
    # Create separate test executables for each test file
    add_executable(test_threading tests/test_threading.cpp ${LIB_SOURCES})
    target_include_directories(test_threading PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_threading PRIVATE rapidcheck)
    
    add_executable(test_math_kernels tests/test_math_kernels.cpp ${LIB_SOURCES})
    target_include_directories(test_math_kernels PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_math_kernels PRIVATE rapidcheck)
    
    add_executable(test_benchmark_core tests/test_benchmark_core.cpp ${LIB_SOURCES})
    target_include_directories(test_benchmark_core PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_benchmark_core PRIVATE rapidcheck)
    
    add_executable(test_cli tests/test_cli.cpp ${LIB_SOURCES})
    target_include_directories(test_cli PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_cli PRIVATE rapidcheck)
    
    add_executable(test_json tests/test_json.cpp ${LIB_SOURCES})
    target_include_directories(test_json PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_json PRIVATE rapidcheck)
    
    add_executable(test_precision_types tests/test_precision_types.cpp ${LIB_SOURCES})
    target_include_directories(test_precision_types PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_precision_types PRIVATE rapidcheck)
    
    add_executable(test_cpu_capabilities tests/test_cpu_capabilities.cpp ${LIB_SOURCES})
    target_include_directories(test_cpu_capabilities PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_cpu_capabilities PRIVATE rapidcheck)
    
    add_executable(test_math_kernels_extended tests/test_math_kernels_extended.cpp ${LIB_SOURCES})
    target_include_directories(test_math_kernels_extended PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_math_kernels_extended PRIVATE rapidcheck)
    
    add_executable(test_comparison_output tests/test_comparison_output.cpp ${LIB_SOURCES})
    target_include_directories(test_comparison_output PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_comparison_output PRIVATE rapidcheck)
    
    add_executable(test_size_overflow tests/test_size_overflow.cpp ${LIB_SOURCES})
    target_include_directories(test_size_overflow PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_size_overflow PRIVATE rapidcheck)
    
    add_executable(test_memory_validator tests/test_memory_validator.cpp ${LIB_SOURCES})
    target_include_directories(test_memory_validator PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_memory_validator PRIVATE rapidcheck)
    
    add_executable(test_simd_kernels tests/test_simd_kernels.cpp ${LIB_SOURCES})
    target_include_directories(test_simd_kernels PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_simd_kernels PRIVATE rapidcheck)
    
    add_executable(test_kernel_dispatcher tests/test_kernel_dispatcher.cpp ${LIB_SOURCES})
    target_include_directories(test_kernel_dispatcher PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_kernel_dispatcher PRIVATE rapidcheck)
    
    add_executable(test_fp16_kernel_selection tests/test_fp16_kernel_selection.cpp ${LIB_SOURCES})
    target_include_directories(test_fp16_kernel_selection PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_fp16_kernel_selection PRIVATE rapidcheck)
    
    add_executable(test_neon_kernels tests/test_neon_kernels.cpp ${LIB_SOURCES})
    target_include_directories(test_neon_kernels PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_neon_kernels PRIVATE rapidcheck)
    
    add_executable(test_int8_simd_kernels tests/test_int8_simd_kernels.cpp ${LIB_SOURCES})
    target_include_directories(test_int8_simd_kernels PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_int8_simd_kernels PRIVATE rapidcheck)
    
    add_executable(test_benchmark_output tests/test_benchmark_output.cpp ${LIB_SOURCES})
    target_include_directories(test_benchmark_output PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_benchmark_output PRIVATE rapidcheck)
    
    add_executable(test_integration tests/test_integration.cpp ${LIB_SOURCES})
    target_include_directories(test_integration PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_integration PRIVATE rapidcheck)
    
    add_executable(test_runtime_dispatcher tests/test_runtime_dispatcher.cpp ${LIB_SOURCES})
    target_include_directories(test_runtime_dispatcher PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_runtime_dispatcher PRIVATE rapidcheck)
    
    add_executable(test_platform tests/test_platform.cpp ${LIB_SOURCES})
    target_include_directories(test_platform PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_platform PRIVATE rapidcheck)
    
    add_executable(test_version tests/test_version.cpp ${LIB_SOURCES})
    target_include_directories(test_version PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_version PRIVATE rapidcheck)
    
    add_executable(test_progress_reporter tests/test_progress_reporter.cpp ${LIB_SOURCES})
    target_include_directories(test_progress_reporter PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_progress_reporter PRIVATE rapidcheck)
    
    add_executable(test_auto_size tests/test_auto_size.cpp ${LIB_SOURCES})
    target_include_directories(test_auto_size PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_auto_size PRIVATE rapidcheck)
    
    add_executable(test_warmup tests/test_warmup.cpp ${LIB_SOURCES})
    target_include_directories(test_warmup PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_warmup PRIVATE rapidcheck)
    
    add_executable(test_score_calculator tests/test_score_calculator.cpp ${LIB_SOURCES})
    target_include_directories(test_score_calculator PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_score_calculator PRIVATE rapidcheck)
    
    add_executable(test_reference_comparison tests/test_reference_comparison.cpp ${LIB_SOURCES})
    target_include_directories(test_reference_comparison PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_reference_comparison PRIVATE rapidcheck)
    
    add_executable(test_thread_affinity tests/test_thread_affinity.cpp ${LIB_SOURCES})
    target_include_directories(test_thread_affinity PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_thread_affinity PRIVATE rapidcheck)
    
    add_executable(test_persistent_thread_pool tests/test_persistent_thread_pool.cpp ${LIB_SOURCES})
    target_include_directories(test_persistent_thread_pool PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_persistent_thread_pool PRIVATE rapidcheck)
    
    add_executable(test_compute_kernel tests/test_compute_kernel.cpp ${LIB_SOURCES})
    target_include_directories(test_compute_kernel PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_compute_kernel PRIVATE rapidcheck)
    
    add_executable(test_cache_level_sizes tests/test_cache_level_sizes.cpp ${LIB_SOURCES})
    target_include_directories(test_cache_level_sizes PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_cache_level_sizes PRIVATE rapidcheck)
    
    add_executable(test_result_submission tests/test_result_submission.cpp ${LIB_SOURCES})
    target_include_directories(test_result_submission PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_libraries(test_result_submission PRIVATE rapidcheck)
    
    # OpenMP support for tests (optional)
    if(ENABLE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(test_threading PRIVATE OpenMP::OpenMP_CXX)
        target_link_libraries(test_math_kernels PRIVATE OpenMP::OpenMP_CXX)
        target_link_libraries(test_benchmark_core PRIVATE OpenMP::OpenMP_CXX)
        target_link_libraries(test_cli PRIVATE OpenMP::OpenMP_CXX)
        target_link_libraries(test_json PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(test_threading PRIVATE USE_OPENMP)
        target_compile_definitions(test_math_kernels PRIVATE USE_OPENMP)
        target_compile_definitions(test_benchmark_core PRIVATE USE_OPENMP)
        target_compile_definitions(test_cli PRIVATE USE_OPENMP)
        target_compile_definitions(test_json PRIVATE USE_OPENMP)
        target_link_libraries(test_precision_types PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(test_precision_types PRIVATE USE_OPENMP)
        target_link_libraries(test_cpu_capabilities PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(test_cpu_capabilities PRIVATE USE_OPENMP)
        target_link_libraries(test_math_kernels_extended PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(test_math_kernels_extended PRIVATE USE_OPENMP)
        target_link_libraries(test_comparison_output PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(test_comparison_output PRIVATE USE_OPENMP)
        target_link_libraries(test_size_overflow PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(test_size_overflow PRIVATE USE_OPENMP)
        target_link_libraries(test_memory_validator PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(test_memory_validator PRIVATE USE_OPENMP)
        target_link_libraries(test_simd_kernels PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(test_simd_kernels PRIVATE USE_OPENMP)
        target_link_libraries(test_kernel_dispatcher PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(test_kernel_dispatcher PRIVATE USE_OPENMP)
        target_link_libraries(test_fp16_kernel_selection PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(test_fp16_kernel_selection PRIVATE USE_OPENMP)
        target_link_libraries(test_neon_kernels PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(test_neon_kernels PRIVATE USE_OPENMP)
        target_link_libraries(test_int8_simd_kernels PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(test_int8_simd_kernels PRIVATE USE_OPENMP)
        target_link_libraries(test_benchmark_output PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(test_benchmark_output PRIVATE USE_OPENMP)
        target_link_libraries(test_integration PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(test_integration PRIVATE USE_OPENMP)
    endif()
    
    if(NOT WIN32)
        find_package(Threads REQUIRED)
        target_link_libraries(test_threading PRIVATE Threads::Threads)
        target_link_libraries(test_math_kernels PRIVATE Threads::Threads)
        target_link_libraries(test_benchmark_core PRIVATE Threads::Threads)
        target_link_libraries(test_cli PRIVATE Threads::Threads)
        target_link_libraries(test_json PRIVATE Threads::Threads)
        target_link_libraries(test_precision_types PRIVATE Threads::Threads)
        target_link_libraries(test_cpu_capabilities PRIVATE Threads::Threads)
        target_link_libraries(test_math_kernels_extended PRIVATE Threads::Threads)
        target_link_libraries(test_comparison_output PRIVATE Threads::Threads)
        target_link_libraries(test_size_overflow PRIVATE Threads::Threads)
        target_link_libraries(test_memory_validator PRIVATE Threads::Threads)
        target_link_libraries(test_simd_kernels PRIVATE Threads::Threads)
        target_link_libraries(test_kernel_dispatcher PRIVATE Threads::Threads)
        target_link_libraries(test_fp16_kernel_selection PRIVATE Threads::Threads)
        target_link_libraries(test_neon_kernels PRIVATE Threads::Threads)
        target_link_libraries(test_int8_simd_kernels PRIVATE Threads::Threads)
        target_link_libraries(test_benchmark_output PRIVATE Threads::Threads)
        target_link_libraries(test_integration PRIVATE Threads::Threads)
        target_link_libraries(test_runtime_dispatcher PRIVATE Threads::Threads)
        target_link_libraries(test_platform PRIVATE Threads::Threads)
        target_link_libraries(test_version PRIVATE Threads::Threads)
        target_link_libraries(test_progress_reporter PRIVATE Threads::Threads)
        target_link_libraries(test_auto_size PRIVATE Threads::Threads)
        target_link_libraries(test_warmup PRIVATE Threads::Threads)
        target_link_libraries(test_score_calculator PRIVATE Threads::Threads)
        target_link_libraries(test_reference_comparison PRIVATE Threads::Threads)
        target_link_libraries(test_thread_affinity PRIVATE Threads::Threads)
        target_link_libraries(test_persistent_thread_pool PRIVATE Threads::Threads)
        target_link_libraries(test_compute_kernel PRIVATE Threads::Threads)
        target_link_libraries(test_cache_level_sizes PRIVATE Threads::Threads)
        target_link_libraries(test_result_submission PRIVATE Threads::Threads)
    endif()
    
    # Windows socket library for result submission test
    if(WIN32)
        target_link_libraries(test_result_submission PRIVATE ws2_32)
    endif()
    
    add_test(NAME test_threading COMMAND test_threading)
    add_test(NAME test_math_kernels COMMAND test_math_kernels)
    add_test(NAME test_benchmark_core COMMAND test_benchmark_core)
    add_test(NAME test_cli COMMAND test_cli)
    add_test(NAME test_json COMMAND test_json)
    add_test(NAME test_precision_types COMMAND test_precision_types)
    add_test(NAME test_cpu_capabilities COMMAND test_cpu_capabilities)
    add_test(NAME test_math_kernels_extended COMMAND test_math_kernels_extended)
    add_test(NAME test_comparison_output COMMAND test_comparison_output)
    add_test(NAME test_size_overflow COMMAND test_size_overflow)
    add_test(NAME test_memory_validator COMMAND test_memory_validator)
    add_test(NAME test_simd_kernels COMMAND test_simd_kernels)
    add_test(NAME test_kernel_dispatcher COMMAND test_kernel_dispatcher)
    add_test(NAME test_fp16_kernel_selection COMMAND test_fp16_kernel_selection)
    add_test(NAME test_neon_kernels COMMAND test_neon_kernels)
    add_test(NAME test_int8_simd_kernels COMMAND test_int8_simd_kernels)
    add_test(NAME test_benchmark_output COMMAND test_benchmark_output)
    add_test(NAME test_integration COMMAND test_integration)
    add_test(NAME test_runtime_dispatcher COMMAND test_runtime_dispatcher)
    add_test(NAME test_platform COMMAND test_platform)
    add_test(NAME test_version COMMAND test_version)
    add_test(NAME test_progress_reporter COMMAND test_progress_reporter)
    add_test(NAME test_auto_size COMMAND test_auto_size)
    add_test(NAME test_warmup COMMAND test_warmup)
    add_test(NAME test_score_calculator COMMAND test_score_calculator)
    add_test(NAME test_reference_comparison COMMAND test_reference_comparison)
    add_test(NAME test_thread_affinity COMMAND test_thread_affinity)
    add_test(NAME test_persistent_thread_pool COMMAND test_persistent_thread_pool)
    add_test(NAME test_compute_kernel COMMAND test_compute_kernel)
    add_test(NAME test_cache_level_sizes COMMAND test_cache_level_sizes)
    add_test(NAME test_result_submission COMMAND test_result_submission)
endif()
