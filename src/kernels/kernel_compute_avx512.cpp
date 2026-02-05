// CPU Benchmark - AVX-512 Compute Kernel Implementation
// Separate file for AVX-512 to enable proper /arch:AVX512 compilation

#include "kernel_compute.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define PLATFORM_X86 1
    #ifdef _WIN32
        #include <intrin.h>
    #else
        #include <immintrin.h>
    #endif
    
    // MSVC with /arch:AVX512 enables AVX-512 intrinsics but doesn't define __AVX512F__
    #if defined(_MSC_VER) && !defined(__clang__)
        #ifndef __AVX512F__
            #define __AVX512F__ 1
        #endif
    #endif
#endif

namespace kernels {
namespace compute {

#ifdef PLATFORM_X86

#if defined(__AVX512F__)

// AVX-512 double compute kernel with FMA
// 16 independent chains of __m512d (8 doubles each)
// Each FMA = 2 FLOPs (multiply + add fused)
// Total: 16 FMAs * 8 doubles * 2 FLOPs = 256 FLOPs per iteration
size_t avx512_double_impl(double* result, size_t iterations) {
    // 16 independent accumulator chains for maximum ILP
    __m512d a0 = _mm512_set1_pd(1.0);
    __m512d a1 = _mm512_set1_pd(1.1);
    __m512d a2 = _mm512_set1_pd(1.2);
    __m512d a3 = _mm512_set1_pd(1.3);
    __m512d a4 = _mm512_set1_pd(1.4);
    __m512d a5 = _mm512_set1_pd(1.5);
    __m512d a6 = _mm512_set1_pd(1.6);
    __m512d a7 = _mm512_set1_pd(1.7);
    __m512d a8 = _mm512_set1_pd(1.8);
    __m512d a9 = _mm512_set1_pd(1.9);
    __m512d a10 = _mm512_set1_pd(2.0);
    __m512d a11 = _mm512_set1_pd(2.1);
    __m512d a12 = _mm512_set1_pd(2.2);
    __m512d a13 = _mm512_set1_pd(2.3);
    __m512d a14 = _mm512_set1_pd(2.4);
    __m512d a15 = _mm512_set1_pd(2.5);
    
    __m512d mul = _mm512_set1_pd(1.0000001);
    __m512d add = _mm512_set1_pd(0.0000001);
    
    for (size_t i = 0; i < iterations; ++i) {
        // 16 independent FMA operations
        a0 = _mm512_fmadd_pd(a0, mul, add);
        a1 = _mm512_fmadd_pd(a1, mul, add);
        a2 = _mm512_fmadd_pd(a2, mul, add);
        a3 = _mm512_fmadd_pd(a3, mul, add);
        a4 = _mm512_fmadd_pd(a4, mul, add);
        a5 = _mm512_fmadd_pd(a5, mul, add);
        a6 = _mm512_fmadd_pd(a6, mul, add);
        a7 = _mm512_fmadd_pd(a7, mul, add);
        a8 = _mm512_fmadd_pd(a8, mul, add);
        a9 = _mm512_fmadd_pd(a9, mul, add);
        a10 = _mm512_fmadd_pd(a10, mul, add);
        a11 = _mm512_fmadd_pd(a11, mul, add);
        a12 = _mm512_fmadd_pd(a12, mul, add);
        a13 = _mm512_fmadd_pd(a13, mul, add);
        a14 = _mm512_fmadd_pd(a14, mul, add);
        a15 = _mm512_fmadd_pd(a15, mul, add);
    }
    
    // Sum all results
    __m512d sum0 = _mm512_add_pd(_mm512_add_pd(a0, a1), _mm512_add_pd(a2, a3));
    __m512d sum1 = _mm512_add_pd(_mm512_add_pd(a4, a5), _mm512_add_pd(a6, a7));
    __m512d sum2 = _mm512_add_pd(_mm512_add_pd(a8, a9), _mm512_add_pd(a10, a11));
    __m512d sum3 = _mm512_add_pd(_mm512_add_pd(a12, a13), _mm512_add_pd(a14, a15));
    __m512d sum = _mm512_add_pd(_mm512_add_pd(sum0, sum1), _mm512_add_pd(sum2, sum3));
    
    *result = _mm512_reduce_add_pd(sum);
    
    // 16 chains * 8 doubles * 2 FLOPs = 256 FLOPs per iteration
    return iterations * 256;
}

size_t avx512_float_impl(float* result, size_t iterations) {
    __m512 a0 = _mm512_set1_ps(1.0f);
    __m512 a1 = _mm512_set1_ps(1.1f);
    __m512 a2 = _mm512_set1_ps(1.2f);
    __m512 a3 = _mm512_set1_ps(1.3f);
    __m512 a4 = _mm512_set1_ps(1.4f);
    __m512 a5 = _mm512_set1_ps(1.5f);
    __m512 a6 = _mm512_set1_ps(1.6f);
    __m512 a7 = _mm512_set1_ps(1.7f);
    __m512 a8 = _mm512_set1_ps(1.8f);
    __m512 a9 = _mm512_set1_ps(1.9f);
    __m512 a10 = _mm512_set1_ps(2.0f);
    __m512 a11 = _mm512_set1_ps(2.1f);
    __m512 a12 = _mm512_set1_ps(2.2f);
    __m512 a13 = _mm512_set1_ps(2.3f);
    __m512 a14 = _mm512_set1_ps(2.4f);
    __m512 a15 = _mm512_set1_ps(2.5f);
    
    __m512 mul = _mm512_set1_ps(1.0000001f);
    __m512 add = _mm512_set1_ps(0.0000001f);
    
    for (size_t i = 0; i < iterations; ++i) {
        a0 = _mm512_fmadd_ps(a0, mul, add);
        a1 = _mm512_fmadd_ps(a1, mul, add);
        a2 = _mm512_fmadd_ps(a2, mul, add);
        a3 = _mm512_fmadd_ps(a3, mul, add);
        a4 = _mm512_fmadd_ps(a4, mul, add);
        a5 = _mm512_fmadd_ps(a5, mul, add);
        a6 = _mm512_fmadd_ps(a6, mul, add);
        a7 = _mm512_fmadd_ps(a7, mul, add);
        a8 = _mm512_fmadd_ps(a8, mul, add);
        a9 = _mm512_fmadd_ps(a9, mul, add);
        a10 = _mm512_fmadd_ps(a10, mul, add);
        a11 = _mm512_fmadd_ps(a11, mul, add);
        a12 = _mm512_fmadd_ps(a12, mul, add);
        a13 = _mm512_fmadd_ps(a13, mul, add);
        a14 = _mm512_fmadd_ps(a14, mul, add);
        a15 = _mm512_fmadd_ps(a15, mul, add);
    }
    
    __m512 sum0 = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
    __m512 sum1 = _mm512_add_ps(_mm512_add_ps(a4, a5), _mm512_add_ps(a6, a7));
    __m512 sum2 = _mm512_add_ps(_mm512_add_ps(a8, a9), _mm512_add_ps(a10, a11));
    __m512 sum3 = _mm512_add_ps(_mm512_add_ps(a12, a13), _mm512_add_ps(a14, a15));
    __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    
    *result = _mm512_reduce_add_ps(sum);
    
    // 16 chains * 16 floats * 2 FLOPs = 512 FLOPs per iteration
    return iterations * 512;
}

#endif // __AVX512F__

#endif // PLATFORM_X86

// Fallback implementations for non-x86 platforms or when AVX-512 not available
#if !defined(PLATFORM_X86) || !defined(__AVX512F__)

// Forward declare AVX2 fallback (defined in kernel_compute.cpp)
size_t avx2_double(double* result, size_t iterations);
size_t avx2_float(float* result, size_t iterations);

size_t avx512_double_impl(double* result, size_t iterations) {
    return avx2_double(result, iterations);
}

size_t avx512_float_impl(float* result, size_t iterations) {
    return avx2_float(result, iterations);
}

#endif

} // namespace compute
} // namespace kernels
