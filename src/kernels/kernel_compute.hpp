// CPU Benchmark - Compute-Intensive Kernel Header
// Pure compute kernels that stress CPU without memory bottlenecks
// Uses FMA (Fused Multiply-Add) for maximum FLOPS

#pragma once

#include <cstddef>
#include <cstdint>

// Function pointer types for compute kernels
// Parameters: result pointer, iterations count
// Returns: number of FLOPs performed
using ComputeKernelDoubleFn = size_t(*)(double* result, size_t iterations);
using ComputeKernelFloatFn = size_t(*)(float* result, size_t iterations);

// Generic type alias for backward compatibility
using ComputeKernelFn = ComputeKernelDoubleFn;

namespace kernels {
namespace compute {

// ============================================================================
// BASELINE KERNEL - Cross-architecture comparable (NO SIMD, NO auto-vectorization)
// ============================================================================

// Scalar FP64 compute kernel - THE BASELINE for cross-arch comparison
// - Pure scalar FP64 FMA operations
// - Compiler auto-vectorization DISABLED
// - 8 independent chains for ILP (instruction-level parallelism)
// - Each iteration: 8 FMAs = 16 FLOPs
// This is the ONLY kernel used for cross-architecture scoring
size_t scalar_fp64_baseline(double* result, size_t iterations);

// ============================================================================
// LEGACY SCALAR KERNELS (may be auto-vectorized by compiler)
// ============================================================================

// Scalar compute kernel - baseline implementation
// Uses 8 independent accumulator chains to maximize ILP
// Each iteration: 8 FMAs = 16 FLOPs
size_t scalar_double(double* result, size_t iterations);

// Scalar float compute kernel
size_t scalar_float(float* result, size_t iterations);

// ============================================================================
// SIMD KERNELS - Architecture-specific, NOT for cross-arch comparison
// ============================================================================

// SSE2 compute kernel
// Uses 4 independent __m128d chains (2 doubles each)
// Each iteration: 4 chains * 2 doubles * 2 FLOPs (mul+add) = 16 FLOPs
size_t sse2_double(double* result, size_t iterations);

// AVX compute kernel
// Uses 4 independent __m256d chains (4 doubles each)
// Each iteration: 4 chains * 4 doubles * 2 FLOPs (mul+add) = 32 FLOPs
size_t avx_double(double* result, size_t iterations);

// AVX2 compute kernel with FMA
// Uses 8 independent __m256d chains (4 doubles each)
// Each iteration: 8 FMAs * 4 doubles = 32 FLOPs per chain = 64 FLOPs total
// With FMA: each FMA = 2 FLOPs (multiply + add)
// Total: 8 chains * 4 doubles * 2 FLOPs = 64 FLOPs per iteration
size_t avx2_double(double* result, size_t iterations);

// AVX-512 compute kernel with FMA
// Uses 16 independent __m512d chains (8 doubles each)
// Each iteration: 16 FMAs * 8 doubles = 128 FLOPs per chain = 256 FLOPs total
// With FMA: each FMA = 2 FLOPs (multiply + add)
// Total: 16 chains * 8 doubles * 2 FLOPs = 256 FLOPs per iteration
size_t avx512_double(double* result, size_t iterations);

// ARM NEON compute kernel
// Uses 32 independent float64x2_t chains (2 doubles each)
// Each iteration: 32 FMAs * 2 doubles * 2 FLOPs = 128 FLOPs total
// Optimized for Apple M-series with 4 FMA units per core
size_t neon_double(double* result, size_t iterations);

// Float versions for single precision compute tests (SIMD informational only)
size_t sse2_float(float* result, size_t iterations);
size_t avx_float(float* result, size_t iterations);
size_t avx2_float(float* result, size_t iterations);
size_t avx512_float(float* result, size_t iterations);
size_t neon_float(float* result, size_t iterations);

} // namespace compute
} // namespace kernels

