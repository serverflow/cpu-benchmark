// CPU Benchmark - Compute-Intensive Kernel Implementations
// Pure compute kernels that stress CPU without memory bottlenecks
// Uses FMA (Fused Multiply-Add) for maximum FLOPS

#include "kernel_compute.hpp"

#include <cmath>

// Platform detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define PLATFORM_X86 1
    #ifdef _WIN32
        #include <intrin.h>
    #else
        #include <immintrin.h>
    #endif
    
    // MSVC doesn't define __AVX2__ and __FMA__ even with /arch:AVX2
    // Check for MSVC and enable based on _M_IX86_FP or explicit /arch flag
    // This file is compiled with /arch:AVX2 via CMakeLists.txt
    #if defined(_MSC_VER) && !defined(__clang__)
        // When this file is compiled with /arch:AVX2, MSVC enables AVX2 intrinsics
        // but doesn't define __AVX2__ or __FMA__. We define them here since
        // CMakeLists.txt sets /arch:AVX2 for this file.
        #ifndef __AVX__
            #define __AVX__ 1
        #endif
        #ifndef __AVX2__
            #define __AVX2__ 1
        #endif
        #ifndef __FMA__
            #define __FMA__ 1
        #endif
    #endif
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    #define PLATFORM_ARM64 1
    #include <arm_neon.h>
#endif

namespace kernels {
namespace compute {

// ============================================================================
// BASELINE KERNEL - Cross-architecture comparable
// ============================================================================
// This kernel is THE STANDARD for cross-architecture comparison.
// It uses pure scalar FP64 FMA operations with NO SIMD.
// Auto-vectorization is explicitly disabled to ensure identical code
// generation across x86 and ARM.
//
// Design principles:
// 1. Pure scalar FP64 - same precision on all architectures
// 2. 8 independent chains - saturate FMA unit pipeline (latency hiding)
// 3. No SIMD intrinsics - compiler generates scalar FMA instructions
// 4. Volatile barriers - prevent unwanted optimizations
// ============================================================================


// Compiler-specific controls to disable auto-vectorization
// - MSVC: per-loop pragma
// - Clang: per-loop pragma
// - GCC: use function-level optimize attribute (disables both loop and SLP vectorizers)
#if defined(_MSC_VER) && !defined(__clang__)
    #define NOVEC_LOOP __pragma(loop(no_vector))
#elif defined(__clang__)
    #define NOVEC_LOOP _Pragma("clang loop vectorize(disable) interleave(disable)")
#else
    #define NOVEC_LOOP
#endif

#if defined(__GNUC__) && !defined(__clang__)
    #define NO_VECTORIZE_ATTR __attribute__((optimize("no-tree-vectorize,no-tree-slp-vectorize")))
#else
    #define NO_VECTORIZE_ATTR
#endif

// Baseline scalar FP64 kernel - THE cross-architecture standard
// Uses 8 independent FMA chains for ILP, NO SIMD
// Each iteration: 8 FMAs = 16 FLOPs
#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
NO_VECTORIZE_ATTR
size_t scalar_fp64_baseline(double* result, size_t iterations) {
    // 8 independent accumulator chains for instruction-level parallelism
    // This saturates the FMA pipeline on most modern CPUs
    double a0 = 1.0, a1 = 1.1, a2 = 1.2, a3 = 1.3;
    double a4 = 1.4, a5 = 1.5, a6 = 1.6, a7 = 1.7;
    
    // Constants chosen to keep values stable (no overflow/underflow)
    const double mul = 1.0000001;
    const double add = 0.0000001;
    
    // Main compute loop - pure scalar FMA
    // The noinline attribute prevents inlining which could enable vectorization
    NOVEC_LOOP
    for (size_t i = 0; i < iterations; ++i) {
        // 8 independent FMA operations: a = a * mul + add
        // Each FMA = 2 FLOPs (multiply + add)
        a0 = std::fma(a0, mul, add);
        a1 = std::fma(a1, mul, add);
        a2 = std::fma(a2, mul, add);
        a3 = std::fma(a3, mul, add);
        a4 = std::fma(a4, mul, add);
        a5 = std::fma(a5, mul, add);
        a6 = std::fma(a6, mul, add);
        a7 = std::fma(a7, mul, add);
    }
        
    // Prevent dead code elimination
    *result = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    
    // Return total FLOPs: 8 FMAs * 2 FLOPs each * iterations
    return iterations * 8 * 2;
}

// ============================================================================
// Scalar implementations (may be auto-vectorized - legacy)
// ============================================================================

// Scalar double compute kernel
// Uses 8 independent accumulator chains to maximize instruction-level parallelism
// Each FMA = 2 FLOPs (multiply + add)
// Total: 8 FMAs * 2 FLOPs = 16 FLOPs per iteration
size_t scalar_double(double* result, size_t iterations) {
    // 8 independent accumulator chains
    double a0 = 1.0, a1 = 1.1, a2 = 1.2, a3 = 1.3;
    double a4 = 1.4, a5 = 1.5, a6 = 1.6, a7 = 1.7;
    
    // Multiplier and addend chosen to keep values stable
    const double mul = 1.0000001;
    const double add = 0.0000001;
    
    for (size_t i = 0; i < iterations; ++i) {
        // 8 independent FMA operations
        a0 = std::fma(a0, mul, add);
        a1 = std::fma(a1, mul, add);
        a2 = std::fma(a2, mul, add);
        a3 = std::fma(a3, mul, add);
        a4 = std::fma(a4, mul, add);
        a5 = std::fma(a5, mul, add);
        a6 = std::fma(a6, mul, add);
        a7 = std::fma(a7, mul, add);
    }
    
    // Prevent optimization by storing result
    *result = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    
    // Return total FLOPs: 8 FMAs * 2 FLOPs each * iterations
    return iterations * 8 * 2;
}

// Scalar float compute kernel
size_t scalar_float(float* result, size_t iterations) {
    float a0 = 1.0f, a1 = 1.1f, a2 = 1.2f, a3 = 1.3f;
    float a4 = 1.4f, a5 = 1.5f, a6 = 1.6f, a7 = 1.7f;
    
    const float mul = 1.0000001f;
    const float add = 0.0000001f;
    
    for (size_t i = 0; i < iterations; ++i) {
        a0 = std::fma(a0, mul, add);
        a1 = std::fma(a1, mul, add);
        a2 = std::fma(a2, mul, add);
        a3 = std::fma(a3, mul, add);
        a4 = std::fma(a4, mul, add);
        a5 = std::fma(a5, mul, add);
        a6 = std::fma(a6, mul, add);
        a7 = std::fma(a7, mul, add);
    }
    
    *result = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    return iterations * 8 * 2;
}

// ============================================================================
// x86 SIMD implementations
// ============================================================================

#ifdef PLATFORM_X86

// SSE2 double compute kernel
// 4 chains of __m128d (2 doubles each)
// Total: 4 FMAs * 2 doubles * 2 FLOPs = 16 FLOPs per iteration

#if defined(__SSE2__) || defined(_M_X64)
size_t sse2_double(double* result, size_t iterations) {
    __m128d a0 = _mm_set1_pd(1.0);
    __m128d a1 = _mm_set1_pd(1.1);
    __m128d a2 = _mm_set1_pd(1.2);
    __m128d a3 = _mm_set1_pd(1.3);
    
    __m128d mul = _mm_set1_pd(1.0000001);
    __m128d add = _mm_set1_pd(0.0000001);
    
    for (size_t i = 0; i < iterations; ++i) {
        // SSE2 doesn't have FMA, so we do mul + add separately
        a0 = _mm_add_pd(_mm_mul_pd(a0, mul), add);
        a1 = _mm_add_pd(_mm_mul_pd(a1, mul), add);
        a2 = _mm_add_pd(_mm_mul_pd(a2, mul), add);
        a3 = _mm_add_pd(_mm_mul_pd(a3, mul), add);
    }
    
    // Sum all results
    __m128d sum = _mm_add_pd(_mm_add_pd(a0, a1), _mm_add_pd(a2, a3));
    double temp[2];
    _mm_storeu_pd(temp, sum);
    *result = temp[0] + temp[1];
    
    // 4 chains * 2 doubles * 2 FLOPs (mul+add) = 16 FLOPs per iteration
    return iterations * 16;
}

size_t sse2_float(float* result, size_t iterations) {
    __m128 a0 = _mm_set1_ps(1.0f);
    __m128 a1 = _mm_set1_ps(1.1f);
    __m128 a2 = _mm_set1_ps(1.2f);
    __m128 a3 = _mm_set1_ps(1.3f);
    
    __m128 mul = _mm_set1_ps(1.0000001f);
    __m128 add = _mm_set1_ps(0.0000001f);
    
    for (size_t i = 0; i < iterations; ++i) {
        a0 = _mm_add_ps(_mm_mul_ps(a0, mul), add);
        a1 = _mm_add_ps(_mm_mul_ps(a1, mul), add);
        a2 = _mm_add_ps(_mm_mul_ps(a2, mul), add);
        a3 = _mm_add_ps(_mm_mul_ps(a3, mul), add);
    }
    
    __m128 sum = _mm_add_ps(_mm_add_ps(a0, a1), _mm_add_ps(a2, a3));
    float temp[4];
    _mm_storeu_ps(temp, sum);
    *result = temp[0] + temp[1] + temp[2] + temp[3];
    
    // 4 chains * 4 floats * 2 FLOPs = 32 FLOPs per iteration
    return iterations * 32;
}
#else
size_t sse2_double(double* result, size_t iterations) {
    return scalar_double(result, iterations);
}
size_t sse2_float(float* result, size_t iterations) {
    return scalar_float(result, iterations);
}
#endif


// AVX double compute kernel
// 4 chains of __m256d (4 doubles each)
// Total: 4 FMAs * 4 doubles * 2 FLOPs = 32 FLOPs per iteration
#if defined(__AVX__)
size_t avx_double(double* result, size_t iterations) {
    __m256d a0 = _mm256_set1_pd(1.0);
    __m256d a1 = _mm256_set1_pd(1.1);
    __m256d a2 = _mm256_set1_pd(1.2);
    __m256d a3 = _mm256_set1_pd(1.3);
    
    __m256d mul = _mm256_set1_pd(1.0000001);
    __m256d add = _mm256_set1_pd(0.0000001);
    
    for (size_t i = 0; i < iterations; ++i) {
        // AVX doesn't have FMA, so we do mul + add separately
        a0 = _mm256_add_pd(_mm256_mul_pd(a0, mul), add);
        a1 = _mm256_add_pd(_mm256_mul_pd(a1, mul), add);
        a2 = _mm256_add_pd(_mm256_mul_pd(a2, mul), add);
        a3 = _mm256_add_pd(_mm256_mul_pd(a3, mul), add);
    }
    
    __m256d sum = _mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3));
    double temp[4];
    _mm256_storeu_pd(temp, sum);
    *result = temp[0] + temp[1] + temp[2] + temp[3];
    
    // 4 chains * 4 doubles * 2 FLOPs = 32 FLOPs per iteration
    return iterations * 32;
}

size_t avx_float(float* result, size_t iterations) {
    __m256 a0 = _mm256_set1_ps(1.0f);
    __m256 a1 = _mm256_set1_ps(1.1f);
    __m256 a2 = _mm256_set1_ps(1.2f);
    __m256 a3 = _mm256_set1_ps(1.3f);
    
    __m256 mul = _mm256_set1_ps(1.0000001f);
    __m256 add = _mm256_set1_ps(0.0000001f);
    
    for (size_t i = 0; i < iterations; ++i) {
        a0 = _mm256_add_ps(_mm256_mul_ps(a0, mul), add);
        a1 = _mm256_add_ps(_mm256_mul_ps(a1, mul), add);
        a2 = _mm256_add_ps(_mm256_mul_ps(a2, mul), add);
        a3 = _mm256_add_ps(_mm256_mul_ps(a3, mul), add);
    }
    
    __m256 sum = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    float temp[8];
    _mm256_storeu_ps(temp, sum);
    *result = temp[0]+temp[1]+temp[2]+temp[3]+temp[4]+temp[5]+temp[6]+temp[7];
    return iterations * 64;
}
#else
size_t avx_double(double* result, size_t iterations) {
    return sse2_double(result, iterations);
}
size_t avx_float(float* result, size_t iterations) {
    return sse2_float(result, iterations);
}
#endif


// AVX2 double compute kernel with FMA
// 8 independent chains of __m256d (4 doubles each)
// Each FMA = 2 FLOPs (multiply + add fused)
// Total: 8 FMAs * 4 doubles * 2 FLOPs = 64 FLOPs per iteration
#if defined(__AVX2__) && defined(__FMA__)
size_t avx2_double(double* result, size_t iterations) {
    // 8 independent accumulator chains for maximum ILP
    __m256d a0 = _mm256_set1_pd(1.0);
    __m256d a1 = _mm256_set1_pd(1.1);
    __m256d a2 = _mm256_set1_pd(1.2);
    __m256d a3 = _mm256_set1_pd(1.3);
    __m256d a4 = _mm256_set1_pd(1.4);
    __m256d a5 = _mm256_set1_pd(1.5);
    __m256d a6 = _mm256_set1_pd(1.6);
    __m256d a7 = _mm256_set1_pd(1.7);
    
    __m256d mul = _mm256_set1_pd(1.0000001);
    __m256d add = _mm256_set1_pd(0.0000001);
    
    for (size_t i = 0; i < iterations; ++i) {
        // 8 independent FMA operations
        a0 = _mm256_fmadd_pd(a0, mul, add);
        a1 = _mm256_fmadd_pd(a1, mul, add);
        a2 = _mm256_fmadd_pd(a2, mul, add);
        a3 = _mm256_fmadd_pd(a3, mul, add);
        a4 = _mm256_fmadd_pd(a4, mul, add);
        a5 = _mm256_fmadd_pd(a5, mul, add);
        a6 = _mm256_fmadd_pd(a6, mul, add);
        a7 = _mm256_fmadd_pd(a7, mul, add);
    }
    
    // Sum all results to prevent optimization
    __m256d sum01 = _mm256_add_pd(a0, a1);
    __m256d sum23 = _mm256_add_pd(a2, a3);
    __m256d sum45 = _mm256_add_pd(a4, a5);
    __m256d sum67 = _mm256_add_pd(a6, a7);
    __m256d sum0123 = _mm256_add_pd(sum01, sum23);
    __m256d sum4567 = _mm256_add_pd(sum45, sum67);
    __m256d sum = _mm256_add_pd(sum0123, sum4567);
    
    double temp[4];
    _mm256_storeu_pd(temp, sum);
    *result = temp[0] + temp[1] + temp[2] + temp[3];
    
    // 8 chains * 4 doubles * 2 FLOPs = 64 FLOPs per iteration
    return iterations * 64;
}

size_t avx2_float(float* result, size_t iterations) {
    __m256 a0 = _mm256_set1_ps(1.0f);
    __m256 a1 = _mm256_set1_ps(1.1f);
    __m256 a2 = _mm256_set1_ps(1.2f);
    __m256 a3 = _mm256_set1_ps(1.3f);
    __m256 a4 = _mm256_set1_ps(1.4f);
    __m256 a5 = _mm256_set1_ps(1.5f);
    __m256 a6 = _mm256_set1_ps(1.6f);
    __m256 a7 = _mm256_set1_ps(1.7f);
    
    __m256 mul = _mm256_set1_ps(1.0000001f);
    __m256 add = _mm256_set1_ps(0.0000001f);
    
    for (size_t i = 0; i < iterations; ++i) {
        a0 = _mm256_fmadd_ps(a0, mul, add);
        a1 = _mm256_fmadd_ps(a1, mul, add);
        a2 = _mm256_fmadd_ps(a2, mul, add);
        a3 = _mm256_fmadd_ps(a3, mul, add);
        a4 = _mm256_fmadd_ps(a4, mul, add);
        a5 = _mm256_fmadd_ps(a5, mul, add);
        a6 = _mm256_fmadd_ps(a6, mul, add);
        a7 = _mm256_fmadd_ps(a7, mul, add);
    }
    
    __m256 sum01 = _mm256_add_ps(a0, a1);
    __m256 sum23 = _mm256_add_ps(a2, a3);
    __m256 sum45 = _mm256_add_ps(a4, a5);
    __m256 sum67 = _mm256_add_ps(a6, a7);
    __m256 sum0123 = _mm256_add_ps(sum01, sum23);
    __m256 sum4567 = _mm256_add_ps(sum45, sum67);
    __m256 sum = _mm256_add_ps(sum0123, sum4567);
    
    float temp[8];
    _mm256_storeu_ps(temp, sum);
    *result = temp[0]+temp[1]+temp[2]+temp[3]+temp[4]+temp[5]+temp[6]+temp[7];
    
    // 8 chains * 8 floats * 2 FLOPs = 128 FLOPs per iteration
    return iterations * 128;
}
#else
size_t avx2_double(double* result, size_t iterations) {
    return avx_double(result, iterations);
}
size_t avx2_float(float* result, size_t iterations) {
    return avx_float(result, iterations);
}
#endif


// AVX-512 compute kernels are implemented in kernel_compute_avx512.cpp
// which is compiled with /arch:AVX512 flag
// Forward declarations for the AVX-512 implementations
size_t avx512_double_impl(double* result, size_t iterations);
size_t avx512_float_impl(float* result, size_t iterations);

// Wrapper functions that call the AVX-512 implementations
size_t avx512_double(double* result, size_t iterations) {
    return avx512_double_impl(result, iterations);
}
size_t avx512_float(float* result, size_t iterations) {
    return avx512_float_impl(result, iterations);
}

#endif // PLATFORM_X86

// ============================================================================
// ARM NEON implementations - Optimized for Apple Silicon (M1/M2/M3/M4)
// ============================================================================
// Apple Silicon has 4 NEON FMA units per P-core, each can execute 128-bit ops
// Key optimizations:
// 1. Use enough independent chains to saturate all FMA units (latency hiding)
// 2. Manual loop unrolling to help compiler scheduling
// 3. Prevent compiler from reordering operations that break ILP
// ============================================================================

#ifdef PLATFORM_ARM64

// Prevent compiler from optimizing away the computation
#if defined(__clang__)
#define NEON_BARRIER() __asm__ __volatile__("" ::: "memory")
#else
#define NEON_BARRIER() 
#endif

size_t neon_double(double* result, size_t iterations) {
    // Apple M-series P-cores: 4 FMA units, each handles 128-bit (2 doubles)
    // FMA latency ~4 cycles, throughput 4/cycle
    // Need 4 * 4 = 16 independent chains minimum to hide latency
    // Using 20 chains for safety margin
    // 20 chains * 2 doubles * 2 FLOPs = 80 FLOPs per iteration
    float64x2_t a0 = vdupq_n_f64(1.0);
    float64x2_t a1 = vdupq_n_f64(1.1);
    float64x2_t a2 = vdupq_n_f64(1.2);
    float64x2_t a3 = vdupq_n_f64(1.3);
    float64x2_t a4 = vdupq_n_f64(1.4);
    float64x2_t a5 = vdupq_n_f64(1.5);
    float64x2_t a6 = vdupq_n_f64(1.6);
    float64x2_t a7 = vdupq_n_f64(1.7);
    float64x2_t a8 = vdupq_n_f64(1.8);
    float64x2_t a9 = vdupq_n_f64(1.9);
    float64x2_t a10 = vdupq_n_f64(2.0);
    float64x2_t a11 = vdupq_n_f64(2.1);
    float64x2_t a12 = vdupq_n_f64(2.2);
    float64x2_t a13 = vdupq_n_f64(2.3);
    float64x2_t a14 = vdupq_n_f64(2.4);
    float64x2_t a15 = vdupq_n_f64(2.5);
    float64x2_t a16 = vdupq_n_f64(2.6);
    float64x2_t a17 = vdupq_n_f64(2.7);
    float64x2_t a18 = vdupq_n_f64(2.8);
    float64x2_t a19 = vdupq_n_f64(2.9);
    float64x2_t a20 = vdupq_n_f64(3.0);
    float64x2_t a21 = vdupq_n_f64(3.1);
    float64x2_t a22 = vdupq_n_f64(3.2);
    float64x2_t a23 = vdupq_n_f64(3.3);
    float64x2_t a24 = vdupq_n_f64(3.4);
    float64x2_t a25 = vdupq_n_f64(3.5);
    float64x2_t a26 = vdupq_n_f64(3.6);
    float64x2_t a27 = vdupq_n_f64(3.7);
    float64x2_t a28 = vdupq_n_f64(3.8);
    float64x2_t a29 = vdupq_n_f64(3.9);
    float64x2_t a30 = vdupq_n_f64(4.0);
    float64x2_t a31 = vdupq_n_f64(4.1);
    
    float64x2_t mul = vdupq_n_f64(1.0000001);
    float64x2_t add = vdupq_n_f64(0.0000001);
    
    // Process iterations in chunks of 4 for better instruction scheduling
    size_t chunks = iterations / 4;
    size_t remainder = iterations % 4;
    
    for (size_t i = 0; i < chunks; ++i) {
        // Unroll 4x - each unroll does 32 FMAs
        // Iteration 1
        a0 = vfmaq_f64(add, a0, mul);
        a1 = vfmaq_f64(add, a1, mul);
        a2 = vfmaq_f64(add, a2, mul);
        a3 = vfmaq_f64(add, a3, mul);
        a4 = vfmaq_f64(add, a4, mul);
        a5 = vfmaq_f64(add, a5, mul);
        a6 = vfmaq_f64(add, a6, mul);
        a7 = vfmaq_f64(add, a7, mul);
        a8 = vfmaq_f64(add, a8, mul);
        a9 = vfmaq_f64(add, a9, mul);
        a10 = vfmaq_f64(add, a10, mul);
        a11 = vfmaq_f64(add, a11, mul);
        a12 = vfmaq_f64(add, a12, mul);
        a13 = vfmaq_f64(add, a13, mul);
        a14 = vfmaq_f64(add, a14, mul);
        a15 = vfmaq_f64(add, a15, mul);
        a16 = vfmaq_f64(add, a16, mul);
        a17 = vfmaq_f64(add, a17, mul);
        a18 = vfmaq_f64(add, a18, mul);
        a19 = vfmaq_f64(add, a19, mul);
        a20 = vfmaq_f64(add, a20, mul);
        a21 = vfmaq_f64(add, a21, mul);
        a22 = vfmaq_f64(add, a22, mul);
        a23 = vfmaq_f64(add, a23, mul);
        a24 = vfmaq_f64(add, a24, mul);
        a25 = vfmaq_f64(add, a25, mul);
        a26 = vfmaq_f64(add, a26, mul);
        a27 = vfmaq_f64(add, a27, mul);
        a28 = vfmaq_f64(add, a28, mul);
        a29 = vfmaq_f64(add, a29, mul);
        a30 = vfmaq_f64(add, a30, mul);
        a31 = vfmaq_f64(add, a31, mul);
        
        // Iteration 2
        a0 = vfmaq_f64(add, a0, mul);
        a1 = vfmaq_f64(add, a1, mul);
        a2 = vfmaq_f64(add, a2, mul);
        a3 = vfmaq_f64(add, a3, mul);
        a4 = vfmaq_f64(add, a4, mul);
        a5 = vfmaq_f64(add, a5, mul);
        a6 = vfmaq_f64(add, a6, mul);
        a7 = vfmaq_f64(add, a7, mul);
        a8 = vfmaq_f64(add, a8, mul);
        a9 = vfmaq_f64(add, a9, mul);
        a10 = vfmaq_f64(add, a10, mul);
        a11 = vfmaq_f64(add, a11, mul);
        a12 = vfmaq_f64(add, a12, mul);
        a13 = vfmaq_f64(add, a13, mul);
        a14 = vfmaq_f64(add, a14, mul);
        a15 = vfmaq_f64(add, a15, mul);
        a16 = vfmaq_f64(add, a16, mul);
        a17 = vfmaq_f64(add, a17, mul);
        a18 = vfmaq_f64(add, a18, mul);
        a19 = vfmaq_f64(add, a19, mul);
        a20 = vfmaq_f64(add, a20, mul);
        a21 = vfmaq_f64(add, a21, mul);
        a22 = vfmaq_f64(add, a22, mul);
        a23 = vfmaq_f64(add, a23, mul);
        a24 = vfmaq_f64(add, a24, mul);
        a25 = vfmaq_f64(add, a25, mul);
        a26 = vfmaq_f64(add, a26, mul);
        a27 = vfmaq_f64(add, a27, mul);
        a28 = vfmaq_f64(add, a28, mul);
        a29 = vfmaq_f64(add, a29, mul);
        a30 = vfmaq_f64(add, a30, mul);
        a31 = vfmaq_f64(add, a31, mul);
        
        // Iteration 3
        a0 = vfmaq_f64(add, a0, mul);
        a1 = vfmaq_f64(add, a1, mul);
        a2 = vfmaq_f64(add, a2, mul);
        a3 = vfmaq_f64(add, a3, mul);
        a4 = vfmaq_f64(add, a4, mul);
        a5 = vfmaq_f64(add, a5, mul);
        a6 = vfmaq_f64(add, a6, mul);
        a7 = vfmaq_f64(add, a7, mul);
        a8 = vfmaq_f64(add, a8, mul);
        a9 = vfmaq_f64(add, a9, mul);
        a10 = vfmaq_f64(add, a10, mul);
        a11 = vfmaq_f64(add, a11, mul);
        a12 = vfmaq_f64(add, a12, mul);
        a13 = vfmaq_f64(add, a13, mul);
        a14 = vfmaq_f64(add, a14, mul);
        a15 = vfmaq_f64(add, a15, mul);
        a16 = vfmaq_f64(add, a16, mul);
        a17 = vfmaq_f64(add, a17, mul);
        a18 = vfmaq_f64(add, a18, mul);
        a19 = vfmaq_f64(add, a19, mul);
        a20 = vfmaq_f64(add, a20, mul);
        a21 = vfmaq_f64(add, a21, mul);
        a22 = vfmaq_f64(add, a22, mul);
        a23 = vfmaq_f64(add, a23, mul);
        a24 = vfmaq_f64(add, a24, mul);
        a25 = vfmaq_f64(add, a25, mul);
        a26 = vfmaq_f64(add, a26, mul);
        a27 = vfmaq_f64(add, a27, mul);
        a28 = vfmaq_f64(add, a28, mul);
        a29 = vfmaq_f64(add, a29, mul);
        a30 = vfmaq_f64(add, a30, mul);
        a31 = vfmaq_f64(add, a31, mul);
        
        // Iteration 4
        a0 = vfmaq_f64(add, a0, mul);
        a1 = vfmaq_f64(add, a1, mul);
        a2 = vfmaq_f64(add, a2, mul);
        a3 = vfmaq_f64(add, a3, mul);
        a4 = vfmaq_f64(add, a4, mul);
        a5 = vfmaq_f64(add, a5, mul);
        a6 = vfmaq_f64(add, a6, mul);
        a7 = vfmaq_f64(add, a7, mul);
        a8 = vfmaq_f64(add, a8, mul);
        a9 = vfmaq_f64(add, a9, mul);
        a10 = vfmaq_f64(add, a10, mul);
        a11 = vfmaq_f64(add, a11, mul);
        a12 = vfmaq_f64(add, a12, mul);
        a13 = vfmaq_f64(add, a13, mul);
        a14 = vfmaq_f64(add, a14, mul);
        a15 = vfmaq_f64(add, a15, mul);
        a16 = vfmaq_f64(add, a16, mul);
        a17 = vfmaq_f64(add, a17, mul);
        a18 = vfmaq_f64(add, a18, mul);
        a19 = vfmaq_f64(add, a19, mul);
        a20 = vfmaq_f64(add, a20, mul);
        a21 = vfmaq_f64(add, a21, mul);
        a22 = vfmaq_f64(add, a22, mul);
        a23 = vfmaq_f64(add, a23, mul);
        a24 = vfmaq_f64(add, a24, mul);
        a25 = vfmaq_f64(add, a25, mul);
        a26 = vfmaq_f64(add, a26, mul);
        a27 = vfmaq_f64(add, a27, mul);
        a28 = vfmaq_f64(add, a28, mul);
        a29 = vfmaq_f64(add, a29, mul);
        a30 = vfmaq_f64(add, a30, mul);
        a31 = vfmaq_f64(add, a31, mul);
    }
    
    // Handle remainder
    for (size_t i = 0; i < remainder; ++i) {
        a0 = vfmaq_f64(add, a0, mul);
        a1 = vfmaq_f64(add, a1, mul);
        a2 = vfmaq_f64(add, a2, mul);
        a3 = vfmaq_f64(add, a3, mul);
        a4 = vfmaq_f64(add, a4, mul);
        a5 = vfmaq_f64(add, a5, mul);
        a6 = vfmaq_f64(add, a6, mul);
        a7 = vfmaq_f64(add, a7, mul);
        a8 = vfmaq_f64(add, a8, mul);
        a9 = vfmaq_f64(add, a9, mul);
        a10 = vfmaq_f64(add, a10, mul);
        a11 = vfmaq_f64(add, a11, mul);
        a12 = vfmaq_f64(add, a12, mul);
        a13 = vfmaq_f64(add, a13, mul);
        a14 = vfmaq_f64(add, a14, mul);
        a15 = vfmaq_f64(add, a15, mul);
        a16 = vfmaq_f64(add, a16, mul);
        a17 = vfmaq_f64(add, a17, mul);
        a18 = vfmaq_f64(add, a18, mul);
        a19 = vfmaq_f64(add, a19, mul);
        a20 = vfmaq_f64(add, a20, mul);
        a21 = vfmaq_f64(add, a21, mul);
        a22 = vfmaq_f64(add, a22, mul);
        a23 = vfmaq_f64(add, a23, mul);
        a24 = vfmaq_f64(add, a24, mul);
        a25 = vfmaq_f64(add, a25, mul);
        a26 = vfmaq_f64(add, a26, mul);
        a27 = vfmaq_f64(add, a27, mul);
        a28 = vfmaq_f64(add, a28, mul);
        a29 = vfmaq_f64(add, a29, mul);
        a30 = vfmaq_f64(add, a30, mul);
        a31 = vfmaq_f64(add, a31, mul);
    }
    
    // Sum all results
    float64x2_t sum01 = vaddq_f64(a0, a1);
    float64x2_t sum23 = vaddq_f64(a2, a3);
    float64x2_t sum45 = vaddq_f64(a4, a5);
    float64x2_t sum67 = vaddq_f64(a6, a7);
    float64x2_t sum89 = vaddq_f64(a8, a9);
    float64x2_t sum1011 = vaddq_f64(a10, a11);
    float64x2_t sum1213 = vaddq_f64(a12, a13);
    float64x2_t sum1415 = vaddq_f64(a14, a15);
    float64x2_t sum1617 = vaddq_f64(a16, a17);
    float64x2_t sum1819 = vaddq_f64(a18, a19);
    float64x2_t sum2021 = vaddq_f64(a20, a21);
    float64x2_t sum2223 = vaddq_f64(a22, a23);
    float64x2_t sum2425 = vaddq_f64(a24, a25);
    float64x2_t sum2627 = vaddq_f64(a26, a27);
    float64x2_t sum2829 = vaddq_f64(a28, a29);
    float64x2_t sum3031 = vaddq_f64(a30, a31);
    
    float64x2_t sumA = vaddq_f64(vaddq_f64(sum01, sum23), vaddq_f64(sum45, sum67));
    float64x2_t sumB = vaddq_f64(vaddq_f64(sum89, sum1011), vaddq_f64(sum1213, sum1415));
    float64x2_t sumC = vaddq_f64(vaddq_f64(sum1617, sum1819), vaddq_f64(sum2021, sum2223));
    float64x2_t sumD = vaddq_f64(vaddq_f64(sum2425, sum2627), vaddq_f64(sum2829, sum3031));
    float64x2_t sum = vaddq_f64(vaddq_f64(sumA, sumB), vaddq_f64(sumC, sumD));
    
    *result = vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1);
    
    // 32 chains * 2 doubles * 2 FLOPs = 128 FLOPs per iteration
    return iterations * 128;
}

size_t neon_float(float* result, size_t iterations) {
    // Apple M-series P-cores: 4 FMA units, each handles 128-bit (4 floats)
    // FMA latency ~4 cycles, throughput 4/cycle
    // Need 4 * 4 = 16 independent chains minimum to hide latency
    // Using 32 chains for maximum throughput
    // 32 chains * 4 floats * 2 FLOPs = 256 FLOPs per iteration
    float32x4_t a0 = vdupq_n_f32(1.0f);
    float32x4_t a1 = vdupq_n_f32(1.1f);
    float32x4_t a2 = vdupq_n_f32(1.2f);
    float32x4_t a3 = vdupq_n_f32(1.3f);
    float32x4_t a4 = vdupq_n_f32(1.4f);
    float32x4_t a5 = vdupq_n_f32(1.5f);
    float32x4_t a6 = vdupq_n_f32(1.6f);
    float32x4_t a7 = vdupq_n_f32(1.7f);
    float32x4_t a8 = vdupq_n_f32(1.8f);
    float32x4_t a9 = vdupq_n_f32(1.9f);
    float32x4_t a10 = vdupq_n_f32(2.0f);
    float32x4_t a11 = vdupq_n_f32(2.1f);
    float32x4_t a12 = vdupq_n_f32(2.2f);
    float32x4_t a13 = vdupq_n_f32(2.3f);
    float32x4_t a14 = vdupq_n_f32(2.4f);
    float32x4_t a15 = vdupq_n_f32(2.5f);
    float32x4_t a16 = vdupq_n_f32(2.6f);
    float32x4_t a17 = vdupq_n_f32(2.7f);
    float32x4_t a18 = vdupq_n_f32(2.8f);
    float32x4_t a19 = vdupq_n_f32(2.9f);
    float32x4_t a20 = vdupq_n_f32(3.0f);
    float32x4_t a21 = vdupq_n_f32(3.1f);
    float32x4_t a22 = vdupq_n_f32(3.2f);
    float32x4_t a23 = vdupq_n_f32(3.3f);
    float32x4_t a24 = vdupq_n_f32(3.4f);
    float32x4_t a25 = vdupq_n_f32(3.5f);
    float32x4_t a26 = vdupq_n_f32(3.6f);
    float32x4_t a27 = vdupq_n_f32(3.7f);
    float32x4_t a28 = vdupq_n_f32(3.8f);
    float32x4_t a29 = vdupq_n_f32(3.9f);
    float32x4_t a30 = vdupq_n_f32(4.0f);
    float32x4_t a31 = vdupq_n_f32(4.1f);
    
    float32x4_t mul = vdupq_n_f32(1.0000001f);
    float32x4_t add = vdupq_n_f32(0.0000001f);
    
    // Process iterations in chunks of 4 for better instruction scheduling
    size_t chunks = iterations / 4;
    size_t remainder = iterations % 4;
    
    for (size_t i = 0; i < chunks; ++i) {
        // Unroll 4x - each unroll does 32 FMAs
        // Iteration 1
        a0 = vfmaq_f32(add, a0, mul);
        a1 = vfmaq_f32(add, a1, mul);
        a2 = vfmaq_f32(add, a2, mul);
        a3 = vfmaq_f32(add, a3, mul);
        a4 = vfmaq_f32(add, a4, mul);
        a5 = vfmaq_f32(add, a5, mul);
        a6 = vfmaq_f32(add, a6, mul);
        a7 = vfmaq_f32(add, a7, mul);
        a8 = vfmaq_f32(add, a8, mul);
        a9 = vfmaq_f32(add, a9, mul);
        a10 = vfmaq_f32(add, a10, mul);
        a11 = vfmaq_f32(add, a11, mul);
        a12 = vfmaq_f32(add, a12, mul);
        a13 = vfmaq_f32(add, a13, mul);
        a14 = vfmaq_f32(add, a14, mul);
        a15 = vfmaq_f32(add, a15, mul);
        a16 = vfmaq_f32(add, a16, mul);
        a17 = vfmaq_f32(add, a17, mul);
        a18 = vfmaq_f32(add, a18, mul);
        a19 = vfmaq_f32(add, a19, mul);
        a20 = vfmaq_f32(add, a20, mul);
        a21 = vfmaq_f32(add, a21, mul);
        a22 = vfmaq_f32(add, a22, mul);
        a23 = vfmaq_f32(add, a23, mul);
        a24 = vfmaq_f32(add, a24, mul);
        a25 = vfmaq_f32(add, a25, mul);
        a26 = vfmaq_f32(add, a26, mul);
        a27 = vfmaq_f32(add, a27, mul);
        a28 = vfmaq_f32(add, a28, mul);
        a29 = vfmaq_f32(add, a29, mul);
        a30 = vfmaq_f32(add, a30, mul);
        a31 = vfmaq_f32(add, a31, mul);
        
        // Iteration 2
        a0 = vfmaq_f32(add, a0, mul);
        a1 = vfmaq_f32(add, a1, mul);
        a2 = vfmaq_f32(add, a2, mul);
        a3 = vfmaq_f32(add, a3, mul);
        a4 = vfmaq_f32(add, a4, mul);
        a5 = vfmaq_f32(add, a5, mul);
        a6 = vfmaq_f32(add, a6, mul);
        a7 = vfmaq_f32(add, a7, mul);
        a8 = vfmaq_f32(add, a8, mul);
        a9 = vfmaq_f32(add, a9, mul);
        a10 = vfmaq_f32(add, a10, mul);
        a11 = vfmaq_f32(add, a11, mul);
        a12 = vfmaq_f32(add, a12, mul);
        a13 = vfmaq_f32(add, a13, mul);
        a14 = vfmaq_f32(add, a14, mul);
        a15 = vfmaq_f32(add, a15, mul);
        a16 = vfmaq_f32(add, a16, mul);
        a17 = vfmaq_f32(add, a17, mul);
        a18 = vfmaq_f32(add, a18, mul);
        a19 = vfmaq_f32(add, a19, mul);
        a20 = vfmaq_f32(add, a20, mul);
        a21 = vfmaq_f32(add, a21, mul);
        a22 = vfmaq_f32(add, a22, mul);
        a23 = vfmaq_f32(add, a23, mul);
        a24 = vfmaq_f32(add, a24, mul);
        a25 = vfmaq_f32(add, a25, mul);
        a26 = vfmaq_f32(add, a26, mul);
        a27 = vfmaq_f32(add, a27, mul);
        a28 = vfmaq_f32(add, a28, mul);
        a29 = vfmaq_f32(add, a29, mul);
        a30 = vfmaq_f32(add, a30, mul);
        a31 = vfmaq_f32(add, a31, mul);
        
        // Iteration 3
        a0 = vfmaq_f32(add, a0, mul);
        a1 = vfmaq_f32(add, a1, mul);
        a2 = vfmaq_f32(add, a2, mul);
        a3 = vfmaq_f32(add, a3, mul);
        a4 = vfmaq_f32(add, a4, mul);
        a5 = vfmaq_f32(add, a5, mul);
        a6 = vfmaq_f32(add, a6, mul);
        a7 = vfmaq_f32(add, a7, mul);
        a8 = vfmaq_f32(add, a8, mul);
        a9 = vfmaq_f32(add, a9, mul);
        a10 = vfmaq_f32(add, a10, mul);
        a11 = vfmaq_f32(add, a11, mul);
        a12 = vfmaq_f32(add, a12, mul);
        a13 = vfmaq_f32(add, a13, mul);
        a14 = vfmaq_f32(add, a14, mul);
        a15 = vfmaq_f32(add, a15, mul);
        a16 = vfmaq_f32(add, a16, mul);
        a17 = vfmaq_f32(add, a17, mul);
        a18 = vfmaq_f32(add, a18, mul);
        a19 = vfmaq_f32(add, a19, mul);
        a20 = vfmaq_f32(add, a20, mul);
        a21 = vfmaq_f32(add, a21, mul);
        a22 = vfmaq_f32(add, a22, mul);
        a23 = vfmaq_f32(add, a23, mul);
        a24 = vfmaq_f32(add, a24, mul);
        a25 = vfmaq_f32(add, a25, mul);
        a26 = vfmaq_f32(add, a26, mul);
        a27 = vfmaq_f32(add, a27, mul);
        a28 = vfmaq_f32(add, a28, mul);
        a29 = vfmaq_f32(add, a29, mul);
        a30 = vfmaq_f32(add, a30, mul);
        a31 = vfmaq_f32(add, a31, mul);
        
        // Iteration 4
        a0 = vfmaq_f32(add, a0, mul);
        a1 = vfmaq_f32(add, a1, mul);
        a2 = vfmaq_f32(add, a2, mul);
        a3 = vfmaq_f32(add, a3, mul);
        a4 = vfmaq_f32(add, a4, mul);
        a5 = vfmaq_f32(add, a5, mul);
        a6 = vfmaq_f32(add, a6, mul);
        a7 = vfmaq_f32(add, a7, mul);
        a8 = vfmaq_f32(add, a8, mul);
        a9 = vfmaq_f32(add, a9, mul);
        a10 = vfmaq_f32(add, a10, mul);
        a11 = vfmaq_f32(add, a11, mul);
        a12 = vfmaq_f32(add, a12, mul);
        a13 = vfmaq_f32(add, a13, mul);
        a14 = vfmaq_f32(add, a14, mul);
        a15 = vfmaq_f32(add, a15, mul);
        a16 = vfmaq_f32(add, a16, mul);
        a17 = vfmaq_f32(add, a17, mul);
        a18 = vfmaq_f32(add, a18, mul);
        a19 = vfmaq_f32(add, a19, mul);
        a20 = vfmaq_f32(add, a20, mul);
        a21 = vfmaq_f32(add, a21, mul);
        a22 = vfmaq_f32(add, a22, mul);
        a23 = vfmaq_f32(add, a23, mul);
        a24 = vfmaq_f32(add, a24, mul);
        a25 = vfmaq_f32(add, a25, mul);
        a26 = vfmaq_f32(add, a26, mul);
        a27 = vfmaq_f32(add, a27, mul);
        a28 = vfmaq_f32(add, a28, mul);
        a29 = vfmaq_f32(add, a29, mul);
        a30 = vfmaq_f32(add, a30, mul);
        a31 = vfmaq_f32(add, a31, mul);
    }
    
    // Handle remainder
    for (size_t i = 0; i < remainder; ++i) {
        a0 = vfmaq_f32(add, a0, mul);
        a1 = vfmaq_f32(add, a1, mul);
        a2 = vfmaq_f32(add, a2, mul);
        a3 = vfmaq_f32(add, a3, mul);
        a4 = vfmaq_f32(add, a4, mul);
        a5 = vfmaq_f32(add, a5, mul);
        a6 = vfmaq_f32(add, a6, mul);
        a7 = vfmaq_f32(add, a7, mul);
        a8 = vfmaq_f32(add, a8, mul);
        a9 = vfmaq_f32(add, a9, mul);
        a10 = vfmaq_f32(add, a10, mul);
        a11 = vfmaq_f32(add, a11, mul);
        a12 = vfmaq_f32(add, a12, mul);
        a13 = vfmaq_f32(add, a13, mul);
        a14 = vfmaq_f32(add, a14, mul);
        a15 = vfmaq_f32(add, a15, mul);
        a16 = vfmaq_f32(add, a16, mul);
        a17 = vfmaq_f32(add, a17, mul);
        a18 = vfmaq_f32(add, a18, mul);
        a19 = vfmaq_f32(add, a19, mul);
        a20 = vfmaq_f32(add, a20, mul);
        a21 = vfmaq_f32(add, a21, mul);
        a22 = vfmaq_f32(add, a22, mul);
        a23 = vfmaq_f32(add, a23, mul);
        a24 = vfmaq_f32(add, a24, mul);
        a25 = vfmaq_f32(add, a25, mul);
        a26 = vfmaq_f32(add, a26, mul);
        a27 = vfmaq_f32(add, a27, mul);
        a28 = vfmaq_f32(add, a28, mul);
        a29 = vfmaq_f32(add, a29, mul);
        a30 = vfmaq_f32(add, a30, mul);
        a31 = vfmaq_f32(add, a31, mul);
    }
    
    // Sum all results using NEON horizontal add
    float32x4_t sum01 = vaddq_f32(a0, a1);
    float32x4_t sum23 = vaddq_f32(a2, a3);
    float32x4_t sum45 = vaddq_f32(a4, a5);
    float32x4_t sum67 = vaddq_f32(a6, a7);
    float32x4_t sum89 = vaddq_f32(a8, a9);
    float32x4_t sum1011 = vaddq_f32(a10, a11);
    float32x4_t sum1213 = vaddq_f32(a12, a13);
    float32x4_t sum1415 = vaddq_f32(a14, a15);
    float32x4_t sum1617 = vaddq_f32(a16, a17);
    float32x4_t sum1819 = vaddq_f32(a18, a19);
    float32x4_t sum2021 = vaddq_f32(a20, a21);
    float32x4_t sum2223 = vaddq_f32(a22, a23);
    float32x4_t sum2425 = vaddq_f32(a24, a25);
    float32x4_t sum2627 = vaddq_f32(a26, a27);
    float32x4_t sum2829 = vaddq_f32(a28, a29);
    float32x4_t sum3031 = vaddq_f32(a30, a31);
    
    float32x4_t sumA = vaddq_f32(vaddq_f32(sum01, sum23), vaddq_f32(sum45, sum67));
    float32x4_t sumB = vaddq_f32(vaddq_f32(sum89, sum1011), vaddq_f32(sum1213, sum1415));
    float32x4_t sumC = vaddq_f32(vaddq_f32(sum1617, sum1819), vaddq_f32(sum2021, sum2223));
    float32x4_t sumD = vaddq_f32(vaddq_f32(sum2425, sum2627), vaddq_f32(sum2829, sum3031));
    float32x4_t sum = vaddq_f32(vaddq_f32(sumA, sumB), vaddq_f32(sumC, sumD));
    
    // Use vaddvq_f32 for efficient horizontal sum on ARM64
    *result = vaddvq_f32(sum);
    
    // 32 chains * 4 floats * 2 FLOPs = 256 FLOPs per iteration
    return iterations * 256;
}

#else // Non-ARM platforms - provide stub implementations

#ifndef PLATFORM_ARM64
size_t neon_double(double* result, size_t iterations) {
    return scalar_double(result, iterations);
}
size_t neon_float(float* result, size_t iterations) {
    return scalar_float(result, iterations);
}
#endif

#endif // PLATFORM_ARM64

// Fallback implementations for non-x86 platforms
#ifndef PLATFORM_X86
size_t sse2_double(double* result, size_t iterations) {
    return scalar_double(result, iterations);
}
size_t sse2_float(float* result, size_t iterations) {
    return scalar_float(result, iterations);
}
size_t avx_double(double* result, size_t iterations) {
    return scalar_double(result, iterations);
}
size_t avx_float(float* result, size_t iterations) {
    return scalar_float(result, iterations);
}
size_t avx2_double(double* result, size_t iterations) {
    return scalar_double(result, iterations);
}
size_t avx2_float(float* result, size_t iterations) {
    return scalar_float(result, iterations);
}
size_t avx512_double(double* result, size_t iterations) {
    return scalar_double(result, iterations);
}
size_t avx512_float(float* result, size_t iterations) {
    return scalar_float(result, iterations);
}
#endif

} // namespace compute
} // namespace kernels
