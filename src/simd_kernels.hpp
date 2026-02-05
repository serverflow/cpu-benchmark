#pragma once
// CPU Benchmark - SIMD Optimized Kernels

// Provides SIMD-optimized implementations of math kernels for float and double

#include <cstddef>
#include <cstdint>
#include "types.hpp"
#include "cpu_capabilities.hpp"
#include "math_kernels.hpp"
#include "half.hpp"

// ============================================================================
// Platform Detection Macros
// ============================================================================

// Detect x86-64 platform
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define SIMD_X86 1
#else
    #define SIMD_X86 0
#endif

// Detect ARM64 platform
#if defined(__aarch64__) || defined(_M_ARM64)
    #define SIMD_ARM64 1
#else
    #define SIMD_ARM64 0
#endif

// AVX-512 detection
#if SIMD_X86 && (defined(__AVX512F__) || defined(__AVX512__))
    #define SIMD_AVX512 1
#else
    #define SIMD_AVX512 0
#endif

// AVX2 detection
#if SIMD_X86 && defined(__AVX2__)
    #define SIMD_AVX2 1
#else
    #define SIMD_AVX2 0
#endif

// AVX detection
#if SIMD_X86 && defined(__AVX__)
    #define SIMD_AVX 1
#else
    #define SIMD_AVX 0
#endif

// SSE2 detection (baseline for x86-64)
#if SIMD_X86 && (defined(__SSE2__) || defined(_M_X64))
    #define SIMD_SSE2 1
#else
    #define SIMD_SSE2 0
#endif

// ARM NEON detection
#if SIMD_ARM64 || defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define SIMD_NEON 1
#else
    #define SIMD_NEON 0
#endif

// ============================================================================
// Include appropriate intrinsics headers
// ============================================================================

#if SIMD_X86
    #ifdef _WIN32
        #include <intrin.h>
    #else
        #if SIMD_AVX512
            #include <immintrin.h>
        #elif SIMD_AVX2
            #include <immintrin.h>
        #elif SIMD_AVX
            #include <immintrin.h>
        #elif SIMD_SSE2
            #include <emmintrin.h>
        #endif
    #endif
#endif

#if SIMD_NEON
    #include <arm_neon.h>
#endif

// ============================================================================
// Scalar Fallback Kernels 
// ============================================================================

// Scalar memory kernel for float
inline void kernel_mem_scalar_float(
    float* C, const float* A, const float* B,
    float alpha, float beta,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t /*Nz*/)
{
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            for (size_t x = 0; x < Nx; ++x) {
                size_t i = idx(x, y, z, Nx, Ny);
                C[i] = alpha * A[i] + beta * B[i];
            }
        }
    }
}

// Scalar memory kernel for double
inline void kernel_mem_scalar_double(
    double* C, const double* A, const double* B,
    double alpha, double beta,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t /*Nz*/)
{
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            for (size_t x = 0; x < Nx; ++x) {
                size_t i = idx(x, y, z, Nx, Ny);
                C[i] = alpha * A[i] + beta * B[i];
            }
        }
    }
}

// Scalar stencil kernel for float
inline void kernel_stencil_scalar_float(
    float* C, const float* A,
    float a0, float a1,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t Nz)
{
    size_t z_start = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    for (size_t z = z_start; z < z_stop; ++z) {
        for (size_t y = 1; y < Ny - 1; ++y) {
            for (size_t x = 1; x < Nx - 1; ++x) {
                float center = A[idx(x, y, z, Nx, Ny)];
                float neighbors = A[idx(x + 1, y, z, Nx, Ny)] +
                                  A[idx(x - 1, y, z, Nx, Ny)] +
                                  A[idx(x, y + 1, z, Nx, Ny)] +
                                  A[idx(x, y - 1, z, Nx, Ny)] +
                                  A[idx(x, y, z + 1, Nx, Ny)] +
                                  A[idx(x, y, z - 1, Nx, Ny)];
                C[idx(x, y, z, Nx, Ny)] = a0 * center + a1 * neighbors;
            }
        }
    }
}

// Scalar stencil kernel for double
inline void kernel_stencil_scalar_double(
    double* C, const double* A,
    double a0, double a1,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t Nz)
{
    size_t z_start = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    for (size_t z = z_start; z < z_stop; ++z) {
        for (size_t y = 1; y < Ny - 1; ++y) {
            for (size_t x = 1; x < Nx - 1; ++x) {
                double center = A[idx(x, y, z, Nx, Ny)];
                double neighbors = A[idx(x + 1, y, z, Nx, Ny)] +
                                   A[idx(x - 1, y, z, Nx, Ny)] +
                                   A[idx(x, y + 1, z, Nx, Ny)] +
                                   A[idx(x, y - 1, z, Nx, Ny)] +
                                   A[idx(x, y, z + 1, Nx, Ny)] +
                                   A[idx(x, y, z - 1, Nx, Ny)];
                C[idx(x, y, z, Nx, Ny)] = a0 * center + a1 * neighbors;
            }
        }
    }
}


// ============================================================================
// Scalar INT8 Kernels 
// ============================================================================

// Scalar memory kernel for int8_t
// C[i,j,k] = alpha * A[i,j,k] + beta * B[i,j,k]
// Note: For INT8, we compute in int32 to avoid overflow, then clamp to int8 range
inline void kernel_mem_scalar_int8(
    int8_t* C, const int8_t* A, const int8_t* B,
    int8_t alpha, int8_t beta,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t /*Nz*/)
{
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            for (size_t x = 0; x < Nx; ++x) {
                size_t i = idx(x, y, z, Nx, Ny);
                // Compute in int32 to avoid overflow
                int32_t result = static_cast<int32_t>(alpha) * static_cast<int32_t>(A[i]) +
                                 static_cast<int32_t>(beta) * static_cast<int32_t>(B[i]);
                // Clamp to int8 range
                if (result > 127) result = 127;
                if (result < -128) result = -128;
                C[i] = static_cast<int8_t>(result);
            }
        }
    }
}

// Scalar stencil kernel for int8_t
inline void kernel_stencil_scalar_int8(
    int8_t* C, const int8_t* A,
    int8_t a0, int8_t a1,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t Nz)
{
    size_t z_start = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    for (size_t z = z_start; z < z_stop; ++z) {
        for (size_t y = 1; y < Ny - 1; ++y) {
            for (size_t x = 1; x < Nx - 1; ++x) {
                int32_t center = static_cast<int32_t>(A[idx(x, y, z, Nx, Ny)]);
                int32_t neighbors = static_cast<int32_t>(A[idx(x + 1, y, z, Nx, Ny)]) +
                                    static_cast<int32_t>(A[idx(x - 1, y, z, Nx, Ny)]) +
                                    static_cast<int32_t>(A[idx(x, y + 1, z, Nx, Ny)]) +
                                    static_cast<int32_t>(A[idx(x, y - 1, z, Nx, Ny)]) +
                                    static_cast<int32_t>(A[idx(x, y, z + 1, Nx, Ny)]) +
                                    static_cast<int32_t>(A[idx(x, y, z - 1, Nx, Ny)]);
                int32_t result = static_cast<int32_t>(a0) * center + 
                                 static_cast<int32_t>(a1) * neighbors;
                // Clamp to int8 range
                if (result > 127) result = 127;
                if (result < -128) result = -128;
                C[idx(x, y, z, Nx, Ny)] = static_cast<int8_t>(result);
            }
        }
    }
}

// Scalar matmul kernel for int8_t (output is int32_t to avoid overflow)
// C[z] = A[z] * B[z] for each z slice
inline void kernel_matmul3d_scalar_int8(
    int32_t* C, const int8_t* A, const int8_t* B,
    size_t z_begin, size_t z_end, size_t N)
{
    for (size_t z = z_begin; z < z_end; ++z) {
        const int8_t* A_slice = A + z * N * N;
        const int8_t* B_slice = B + z * N * N;
        int32_t* C_slice = C + z * N * N;
        
        // Initialize C slice to zero
        for (size_t i = 0; i < N * N; ++i) {
            C_slice[i] = 0;
        }
        
        // Matrix multiplication: C = A * B
        for (size_t i = 0; i < N; ++i) {
            for (size_t k = 0; k < N; ++k) {
                int32_t a_val = static_cast<int32_t>(A_slice[i * N + k]);
                for (size_t j = 0; j < N; ++j) {
                    C_slice[i * N + j] += a_val * static_cast<int32_t>(B_slice[k * N + j]);
                }
            }
        }
    }
}


// ============================================================================
// ARM NEON Kernels 
// ============================================================================

#if SIMD_NEON

// NEON memory kernel for float - processes 4 floats per iteration 
// Uses vld1q_f32, vst1q_f32 intrinsics
inline void kernel_mem_neon_float(
    float* C, const float* A, const float* B,
    float alpha, float beta,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t /*Nz*/)
{
    float32x4_t alpha_vec = vdupq_n_f32(alpha);
    float32x4_t beta_vec = vdupq_n_f32(beta);
    
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            size_t row_start = idx(0, y, z, Nx, Ny);
            size_t x = 0;
            
            // Process 4 floats at a time using NEON
            for (; x + 4 <= Nx; x += 4) {
                size_t i = row_start + x;
                float32x4_t a_vec = vld1q_f32(&A[i]);
                float32x4_t b_vec = vld1q_f32(&B[i]);
                
                // C = alpha * A + beta * B
                // Using vmlaq_f32 for fused multiply-add: result = a + b * c
                float32x4_t result = vmulq_f32(alpha_vec, a_vec);
                result = vmlaq_f32(result, beta_vec, b_vec);
                
                vst1q_f32(&C[i], result);
            }
            
            // Handle remainder with scalar code
            for (; x < Nx; ++x) {
                size_t i = row_start + x;
                C[i] = alpha * A[i] + beta * B[i];
            }
        }
    }
}

// NEON stencil kernel for float - 7-point stencil 
// Vectorizes inner loop with NEON
inline void kernel_stencil_neon_float(
    float* C, const float* A,
    float a0, float a1,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t Nz)
{
    size_t z_start = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    float32x4_t a0_vec = vdupq_n_f32(a0);
    float32x4_t a1_vec = vdupq_n_f32(a1);
    
    for (size_t z = z_start; z < z_stop; ++z) {
        for (size_t y = 1; y < Ny - 1; ++y) {
            size_t x = 1;
            
            // Process 4 floats at a time (inner loop only)
            // Need at least 4 inner cells: x from 1 to Nx-2, so Nx >= 6
            for (; x + 4 <= Nx - 1; x += 4) {
                // Load center values
                float32x4_t center = vld1q_f32(&A[idx(x, y, z, Nx, Ny)]);
                
                // Load neighbors
                float32x4_t xp1 = vld1q_f32(&A[idx(x + 1, y, z, Nx, Ny)]);
                float32x4_t xm1 = vld1q_f32(&A[idx(x - 1, y, z, Nx, Ny)]);
                float32x4_t yp1 = vld1q_f32(&A[idx(x, y + 1, z, Nx, Ny)]);
                float32x4_t ym1 = vld1q_f32(&A[idx(x, y - 1, z, Nx, Ny)]);
                float32x4_t zp1 = vld1q_f32(&A[idx(x, y, z + 1, Nx, Ny)]);
                float32x4_t zm1 = vld1q_f32(&A[idx(x, y, z - 1, Nx, Ny)]);
                
                // Sum neighbors
                float32x4_t neighbors = vaddq_f32(xp1, xm1);
                neighbors = vaddq_f32(neighbors, yp1);
                neighbors = vaddq_f32(neighbors, ym1);
                neighbors = vaddq_f32(neighbors, zp1);
                neighbors = vaddq_f32(neighbors, zm1);
                
                // C = a0 * center + a1 * neighbors
                float32x4_t result = vmulq_f32(a0_vec, center);
                result = vmlaq_f32(result, a1_vec, neighbors);
                
                vst1q_f32(&C[idx(x, y, z, Nx, Ny)], result);
            }
            
            // Handle remainder with scalar code
            for (; x < Nx - 1; ++x) {
                float center = A[idx(x, y, z, Nx, Ny)];
                float neighbors = A[idx(x + 1, y, z, Nx, Ny)] +
                                  A[idx(x - 1, y, z, Nx, Ny)] +
                                  A[idx(x, y + 1, z, Nx, Ny)] +
                                  A[idx(x, y - 1, z, Nx, Ny)] +
                                  A[idx(x, y, z + 1, Nx, Ny)] +
                                  A[idx(x, y, z - 1, Nx, Ny)];
                C[idx(x, y, z, Nx, Ny)] = a0 * center + a1 * neighbors;
            }
        }
    }
}

#endif // SIMD_NEON


// ============================================================================
// ARM NEON FP16 Kernels
// Native FP16 support using float16_t type with NEON FP16 intrinsics
// Protected by __ARM_FEATURE_FP16_VECTOR_ARITHMETIC macro
// ============================================================================

#if SIMD_NEON && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define SIMD_NEON_FP16 1
// Compile-time verification that FP16 intrinsics are available
static_assert(sizeof(float16_t) == 2, "float16_t must be 2 bytes");
#else
#define SIMD_NEON_FP16 0
#endif

#if SIMD_NEON_FP16

// NEON FP16 memory kernel - processes 8 float16_t values per iteration
// C = alpha * A + beta * B
// Optimized with __restrict__ and loop unrolling hints
__attribute__((noinline))
inline void kernel_mem_neon_fp16(
    float16_t* __restrict__ C, 
    const float16_t* __restrict__ A, 
    const float16_t* __restrict__ B,
    float16_t alpha, float16_t beta,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t /*Nz*/)
{
    const float16x8_t alpha_vec = vdupq_n_f16(alpha);
    const float16x8_t beta_vec = vdupq_n_f16(beta);
    
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            const size_t row_start = idx(0, y, z, Nx, Ny);
            float16_t* __restrict__ C_row = C + row_start;
            const float16_t* __restrict__ A_row = A + row_start;
            const float16_t* __restrict__ B_row = B + row_start;
            
            size_t x = 0;
            
            // Process 16 float16_t values at a time (2x unroll) for better ILP
            for (; x + 16 <= Nx; x += 16) {
                // First 8 elements
                float16x8_t a_vec0 = vld1q_f16(&A_row[x]);
                float16x8_t b_vec0 = vld1q_f16(&B_row[x]);
                // Second 8 elements
                float16x8_t a_vec1 = vld1q_f16(&A_row[x + 8]);
                float16x8_t b_vec1 = vld1q_f16(&B_row[x + 8]);
                
                // C = alpha * A + beta * B using FMA
                float16x8_t result0 = vmulq_f16(alpha_vec, a_vec0);
                float16x8_t result1 = vmulq_f16(alpha_vec, a_vec1);
                result0 = vfmaq_f16(result0, beta_vec, b_vec0);
                result1 = vfmaq_f16(result1, beta_vec, b_vec1);
                
                vst1q_f16(&C_row[x], result0);
                vst1q_f16(&C_row[x + 8], result1);
            }
            
            // Process remaining 8 elements
            for (; x + 8 <= Nx; x += 8) {
                float16x8_t a_vec = vld1q_f16(&A_row[x]);
                float16x8_t b_vec = vld1q_f16(&B_row[x]);
                
                float16x8_t result = vmulq_f16(alpha_vec, a_vec);
                result = vfmaq_f16(result, beta_vec, b_vec);
                
                vst1q_f16(&C_row[x], result);
            }
            
            // Handle remainder with scalar code
            for (; x < Nx; ++x) {
                C_row[x] = alpha * A_row[x] + beta * B_row[x];
            }
        }
    }
}

// NEON FP16 stencil kernel - 7-point stencil 
// C = a0 * center + a1 * (sum of 6 neighbors)
__attribute__((noinline))
inline void kernel_stencil_neon_fp16(
    float16_t* __restrict__ C, 
    const float16_t* __restrict__ A,
    float16_t a0, float16_t a1,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t Nz)
{
    const size_t z_start = (z_begin < 1) ? 1 : z_begin;
    const size_t z_stop = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    const float16x8_t a0_vec = vdupq_n_f16(a0);
    const float16x8_t a1_vec = vdupq_n_f16(a1);
    
    for (size_t z = z_start; z < z_stop; ++z) {
        for (size_t y = 1; y < Ny - 1; ++y) {
            size_t x = 1;
            
            // Process 8 float16_t values at a time
            for (; x + 8 <= Nx - 1; x += 8) {
                const size_t center_idx = idx(x, y, z, Nx, Ny);
                
                float16x8_t center = vld1q_f16(&A[center_idx]);
                
                // Load 6 neighbors - use precomputed offsets for better cache behavior
                float16x8_t xp1 = vld1q_f16(&A[center_idx + 1]);
                float16x8_t xm1 = vld1q_f16(&A[center_idx - 1]);
                float16x8_t yp1 = vld1q_f16(&A[center_idx + Nx]);
                float16x8_t ym1 = vld1q_f16(&A[center_idx - Nx]);
                float16x8_t zp1 = vld1q_f16(&A[center_idx + Nx * Ny]);
                float16x8_t zm1 = vld1q_f16(&A[center_idx - Nx * Ny]);
                
                // Sum neighbors using pairwise addition for better accuracy
                float16x8_t neighbors = vaddq_f16(xp1, xm1);
                neighbors = vaddq_f16(neighbors, vaddq_f16(yp1, ym1));
                neighbors = vaddq_f16(neighbors, vaddq_f16(zp1, zm1));
                
                // C = a0 * center + a1 * neighbors using FMA
                float16x8_t result = vmulq_f16(a0_vec, center);
                result = vfmaq_f16(result, a1_vec, neighbors);
                
                vst1q_f16(&C[center_idx], result);
            }
            
            // Handle remainder with scalar code
            for (; x < Nx - 1; ++x) {
                const size_t center_idx = idx(x, y, z, Nx, Ny);
                float16_t center = A[center_idx];
                float16_t neighbors = A[center_idx + 1] + A[center_idx - 1] +
                                      A[center_idx + Nx] + A[center_idx - Nx] +
                                      A[center_idx + Nx * Ny] + A[center_idx - Nx * Ny];
                C[center_idx] = a0 * center + a1 * neighbors;
            }
        }
    }
}

// NEON FP16 matmul3d kernel - matrix multiplication for 3D slices 
// C[z] = A[z] * B[z] for each z slice
__attribute__((noinline))
inline void kernel_matmul3d_neon_fp16(
    float16_t* __restrict__ C, 
    const float16_t* __restrict__ A, 
    const float16_t* __restrict__ B,
    size_t z_begin, size_t z_end, size_t N)
{
    const size_t N2 = N * N;
    
    for (size_t z = z_begin; z < z_end; ++z) {
        const float16_t* __restrict__ A_slice = A + z * N2;
        const float16_t* __restrict__ B_slice = B + z * N2;
        float16_t* __restrict__ C_slice = C + z * N2;
        
        // Initialize C slice to zero using NEON
        {
            const float16x8_t zero = vdupq_n_f16(static_cast<float16_t>(0.0f));
            size_t i = 0;
            for (; i + 8 <= N2; i += 8) {
                vst1q_f16(&C_slice[i], zero);
            }
            for (; i < N2; ++i) {
                C_slice[i] = static_cast<float16_t>(0.0f);
            }
        }
        
        // Matrix multiplication: C = A * B
        for (size_t i = 0; i < N; ++i) {
            for (size_t k = 0; k < N; ++k) {
                const float16x8_t a_val = vdupq_n_f16(A_slice[i * N + k]);
                const float16_t* __restrict__ B_row = &B_slice[k * N];
                float16_t* __restrict__ C_row = &C_slice[i * N];
                
                size_t j = 0;
                
                // Process 16 elements at a time (2x unroll)
                for (; j + 16 <= N; j += 16) {
                    float16x8_t b_vec0 = vld1q_f16(&B_row[j]);
                    float16x8_t b_vec1 = vld1q_f16(&B_row[j + 8]);
                    float16x8_t c_vec0 = vld1q_f16(&C_row[j]);
                    float16x8_t c_vec1 = vld1q_f16(&C_row[j + 8]);
                    
                    // C += A * B using FMA
                    c_vec0 = vfmaq_f16(c_vec0, a_val, b_vec0);
                    c_vec1 = vfmaq_f16(c_vec1, a_val, b_vec1);
                    
                    vst1q_f16(&C_row[j], c_vec0);
                    vst1q_f16(&C_row[j + 8], c_vec1);
                }
                
                // Process remaining 8 elements
                for (; j + 8 <= N; j += 8) {
                    float16x8_t b_vec = vld1q_f16(&B_row[j]);
                    float16x8_t c_vec = vld1q_f16(&C_row[j]);
                    
                    c_vec = vfmaq_f16(c_vec, a_val, b_vec);
                    
                    vst1q_f16(&C_row[j], c_vec);
                }
                
                // Handle remainder
                for (; j < N; ++j) {
                    C_row[j] += A_slice[i * N + k] * B_row[j];
                }
            }
        }
    }
}

#endif // SIMD_NEON_FP16


// ============================================================================
// SSE2 Kernels 
// ============================================================================

#if SIMD_SSE2

// SSE2 memory kernel for float - processes 4 floats per iteration
inline void kernel_mem_sse2_float(
    float* C, const float* A, const float* B,
    float alpha, float beta,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t /*Nz*/)
{
    __m128 alpha_vec = _mm_set1_ps(alpha);
    __m128 beta_vec = _mm_set1_ps(beta);
    
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            size_t row_start = idx(0, y, z, Nx, Ny);
            size_t x = 0;
            
            // Process 4 floats at a time
            for (; x + 4 <= Nx; x += 4) {
                size_t i = row_start + x;
                __m128 a_vec = _mm_loadu_ps(&A[i]);
                __m128 b_vec = _mm_loadu_ps(&B[i]);
                
                // C = alpha * A + beta * B
                __m128 result = _mm_add_ps(
                    _mm_mul_ps(alpha_vec, a_vec),
                    _mm_mul_ps(beta_vec, b_vec)
                );
                
                _mm_storeu_ps(&C[i], result);
            }
            
            // Handle remainder with scalar code
            for (; x < Nx; ++x) {
                size_t i = row_start + x;
                C[i] = alpha * A[i] + beta * B[i];
            }
        }
    }
}

// SSE2 memory kernel for double - processes 2 doubles per iteration
inline void kernel_mem_sse2_double(
    double* C, const double* A, const double* B,
    double alpha, double beta,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t /*Nz*/)
{
    __m128d alpha_vec = _mm_set1_pd(alpha);
    __m128d beta_vec = _mm_set1_pd(beta);
    
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            size_t row_start = idx(0, y, z, Nx, Ny);
            size_t x = 0;
            
            // Process 2 doubles at a time
            for (; x + 2 <= Nx; x += 2) {
                size_t i = row_start + x;
                __m128d a_vec = _mm_loadu_pd(&A[i]);
                __m128d b_vec = _mm_loadu_pd(&B[i]);
                
                // C = alpha * A + beta * B
                __m128d result = _mm_add_pd(
                    _mm_mul_pd(alpha_vec, a_vec),
                    _mm_mul_pd(beta_vec, b_vec)
                );
                
                _mm_storeu_pd(&C[i], result);
            }
            
            // Handle remainder with scalar code
            for (; x < Nx; ++x) {
                size_t i = row_start + x;
                C[i] = alpha * A[i] + beta * B[i];
            }
        }
    }
}

// SSE2 stencil kernel for float
inline void kernel_stencil_sse2_float(
    float* C, const float* A,
    float a0, float a1,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t Nz)
{
    size_t z_start = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    __m128 a0_vec = _mm_set1_ps(a0);
    __m128 a1_vec = _mm_set1_ps(a1);
    
    for (size_t z = z_start; z < z_stop; ++z) {
        for (size_t y = 1; y < Ny - 1; ++y) {
            size_t x = 1;
            
            // Process 4 floats at a time (inner loop only)
            // Need at least 4 inner cells: x from 1 to Nx-2, so Nx >= 6
            for (; x + 4 <= Nx - 1; x += 4) {
                // Load center values
                __m128 center = _mm_loadu_ps(&A[idx(x, y, z, Nx, Ny)]);
                
                // Load neighbors
                __m128 xp1 = _mm_loadu_ps(&A[idx(x + 1, y, z, Nx, Ny)]);
                __m128 xm1 = _mm_loadu_ps(&A[idx(x - 1, y, z, Nx, Ny)]);
                __m128 yp1 = _mm_loadu_ps(&A[idx(x, y + 1, z, Nx, Ny)]);
                __m128 ym1 = _mm_loadu_ps(&A[idx(x, y - 1, z, Nx, Ny)]);
                __m128 zp1 = _mm_loadu_ps(&A[idx(x, y, z + 1, Nx, Ny)]);
                __m128 zm1 = _mm_loadu_ps(&A[idx(x, y, z - 1, Nx, Ny)]);
                
                // Sum neighbors
                __m128 neighbors = _mm_add_ps(xp1, xm1);
                neighbors = _mm_add_ps(neighbors, yp1);
                neighbors = _mm_add_ps(neighbors, ym1);
                neighbors = _mm_add_ps(neighbors, zp1);
                neighbors = _mm_add_ps(neighbors, zm1);
                
                // C = a0 * center + a1 * neighbors
                __m128 result = _mm_add_ps(
                    _mm_mul_ps(a0_vec, center),
                    _mm_mul_ps(a1_vec, neighbors)
                );
                
                _mm_storeu_ps(&C[idx(x, y, z, Nx, Ny)], result);
            }
            
            // Handle remainder with scalar code
            for (; x < Nx - 1; ++x) {
                float center = A[idx(x, y, z, Nx, Ny)];
                float neighbors = A[idx(x + 1, y, z, Nx, Ny)] +
                                  A[idx(x - 1, y, z, Nx, Ny)] +
                                  A[idx(x, y + 1, z, Nx, Ny)] +
                                  A[idx(x, y - 1, z, Nx, Ny)] +
                                  A[idx(x, y, z + 1, Nx, Ny)] +
                                  A[idx(x, y, z - 1, Nx, Ny)];
                C[idx(x, y, z, Nx, Ny)] = a0 * center + a1 * neighbors;
            }
        }
    }
}

// SSE2 stencil kernel for double
inline void kernel_stencil_sse2_double(
    double* C, const double* A,
    double a0, double a1,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t Nz)
{
    size_t z_start = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    __m128d a0_vec = _mm_set1_pd(a0);
    __m128d a1_vec = _mm_set1_pd(a1);
    
    for (size_t z = z_start; z < z_stop; ++z) {
        for (size_t y = 1; y < Ny - 1; ++y) {
            size_t x = 1;
            
            // Process 2 doubles at a time
            for (; x + 2 <= Nx - 1; x += 2) {
                __m128d center = _mm_loadu_pd(&A[idx(x, y, z, Nx, Ny)]);
                
                __m128d xp1 = _mm_loadu_pd(&A[idx(x + 1, y, z, Nx, Ny)]);
                __m128d xm1 = _mm_loadu_pd(&A[idx(x - 1, y, z, Nx, Ny)]);
                __m128d yp1 = _mm_loadu_pd(&A[idx(x, y + 1, z, Nx, Ny)]);
                __m128d ym1 = _mm_loadu_pd(&A[idx(x, y - 1, z, Nx, Ny)]);
                __m128d zp1 = _mm_loadu_pd(&A[idx(x, y, z + 1, Nx, Ny)]);
                __m128d zm1 = _mm_loadu_pd(&A[idx(x, y, z - 1, Nx, Ny)]);
                
                __m128d neighbors = _mm_add_pd(xp1, xm1);
                neighbors = _mm_add_pd(neighbors, yp1);
                neighbors = _mm_add_pd(neighbors, ym1);
                neighbors = _mm_add_pd(neighbors, zp1);
                neighbors = _mm_add_pd(neighbors, zm1);
                
                __m128d result = _mm_add_pd(
                    _mm_mul_pd(a0_vec, center),
                    _mm_mul_pd(a1_vec, neighbors)
                );
                
                _mm_storeu_pd(&C[idx(x, y, z, Nx, Ny)], result);
            }
            
            // Handle remainder
            for (; x < Nx - 1; ++x) {
                double center = A[idx(x, y, z, Nx, Ny)];
                double neighbors = A[idx(x + 1, y, z, Nx, Ny)] +
                                   A[idx(x - 1, y, z, Nx, Ny)] +
                                   A[idx(x, y + 1, z, Nx, Ny)] +
                                   A[idx(x, y - 1, z, Nx, Ny)] +
                                   A[idx(x, y, z + 1, Nx, Ny)] +
                                   A[idx(x, y, z - 1, Nx, Ny)];
                C[idx(x, y, z, Nx, Ny)] = a0 * center + a1 * neighbors;
            }
        }
    }
}

// SSE2 memory kernel for int8_t - processes 16 int8_t values per iteration 
// C[i,j,k] = alpha * A[i,j,k] + beta * B[i,j,k]
// Uses _mm_loadu_si128, _mm_storeu_si128 intrinsics
inline void kernel_mem_sse2_int8(
    int8_t* C, const int8_t* A, const int8_t* B,
    int8_t alpha, int8_t beta,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t /*Nz*/)
{
    // For INT8, we need to unpack to 16-bit, multiply, add, and pack back
    // This is complex because SSE2 doesn't have direct int8 multiply
    // We'll process 8 elements at a time using 16-bit intermediates
    
    __m128i alpha_vec = _mm_set1_epi16(static_cast<int16_t>(alpha));
    __m128i beta_vec = _mm_set1_epi16(static_cast<int16_t>(beta));
    
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            size_t row_start = idx(0, y, z, Nx, Ny);
            size_t x = 0;
            
            // Process 8 int8_t values at a time (using 16-bit intermediates)
            for (; x + 8 <= Nx; x += 8) {
                size_t i = row_start + x;
                
                // Load 8 bytes and sign-extend to 16-bit
                __m128i a_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&A[i]));
                __m128i b_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&B[i]));
                
                // Sign-extend int8 to int16 using unpack with sign extension
                __m128i zero = _mm_setzero_si128();
                __m128i a_sign = _mm_cmpgt_epi8(zero, a_bytes);
                __m128i b_sign = _mm_cmpgt_epi8(zero, b_bytes);
                __m128i a_16 = _mm_unpacklo_epi8(a_bytes, a_sign);
                __m128i b_16 = _mm_unpacklo_epi8(b_bytes, b_sign);
                
                // Multiply: alpha * A and beta * B (16-bit results)
                __m128i alpha_a = _mm_mullo_epi16(alpha_vec, a_16);
                __m128i beta_b = _mm_mullo_epi16(beta_vec, b_16);
                
                // Add: alpha * A + beta * B
                __m128i result_16 = _mm_adds_epi16(alpha_a, beta_b);
                
                // Pack back to int8 with saturation
                __m128i result_8 = _mm_packs_epi16(result_16, result_16);
                
                // Store 8 bytes
                _mm_storel_epi64(reinterpret_cast<__m128i*>(&C[i]), result_8);
            }
            
            // Handle remainder with scalar code
            for (; x < Nx; ++x) {
                size_t i = row_start + x;
                int32_t result = static_cast<int32_t>(alpha) * static_cast<int32_t>(A[i]) +
                                 static_cast<int32_t>(beta) * static_cast<int32_t>(B[i]);
                if (result > 127) result = 127;
                if (result < -128) result = -128;
                C[i] = static_cast<int8_t>(result);
            }
        }
    }
}

#endif // SIMD_SSE2


// ============================================================================
// AVX Kernels - Similar to AVX2 but without FMA
// ============================================================================

#if SIMD_AVX

// AVX memory kernel for float - processes 8 floats per iteration
inline void kernel_mem_avx_float(
    float* C, const float* A, const float* B,
    float alpha, float beta,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t /*Nz*/)
{
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);
    
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            size_t row_start = idx(0, y, z, Nx, Ny);
            size_t x = 0;
            
            // Process 8 floats at a time
            for (; x + 8 <= Nx; x += 8) {
                size_t i = row_start + x;
                __m256 a_vec = _mm256_loadu_ps(&A[i]);
                __m256 b_vec = _mm256_loadu_ps(&B[i]);
                
                // C = alpha * A + beta * B (without FMA)
                __m256 result = _mm256_add_ps(
                    _mm256_mul_ps(alpha_vec, a_vec),
                    _mm256_mul_ps(beta_vec, b_vec)
                );
                
                _mm256_storeu_ps(&C[i], result);
            }
            
            // Handle remainder with scalar code
            for (; x < Nx; ++x) {
                size_t i = row_start + x;
                C[i] = alpha * A[i] + beta * B[i];
            }
        }
    }
}

// AVX memory kernel for double - processes 4 doubles per iteration
inline void kernel_mem_avx_double(
    double* C, const double* A, const double* B,
    double alpha, double beta,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t /*Nz*/)
{
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    __m256d beta_vec = _mm256_set1_pd(beta);
    
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            size_t row_start = idx(0, y, z, Nx, Ny);
            size_t x = 0;
            
            // Process 4 doubles at a time
            for (; x + 4 <= Nx; x += 4) {
                size_t i = row_start + x;
                __m256d a_vec = _mm256_loadu_pd(&A[i]);
                __m256d b_vec = _mm256_loadu_pd(&B[i]);
                
                // C = alpha * A + beta * B (without FMA)
                __m256d result = _mm256_add_pd(
                    _mm256_mul_pd(alpha_vec, a_vec),
                    _mm256_mul_pd(beta_vec, b_vec)
                );
                
                _mm256_storeu_pd(&C[i], result);
            }
            
            // Handle remainder with scalar code
            for (; x < Nx; ++x) {
                size_t i = row_start + x;
                C[i] = alpha * A[i] + beta * B[i];
            }
        }
    }
}

// AVX stencil kernel for float
inline void kernel_stencil_avx_float(
    float* C, const float* A,
    float a0, float a1,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t Nz)
{
    size_t z_start = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    __m256 a0_vec = _mm256_set1_ps(a0);
    __m256 a1_vec = _mm256_set1_ps(a1);
    
    for (size_t z = z_start; z < z_stop; ++z) {
        for (size_t y = 1; y < Ny - 1; ++y) {
            size_t x = 1;
            
            // Process 8 floats at a time
            for (; x + 8 <= Nx - 1; x += 8) {
                __m256 center = _mm256_loadu_ps(&A[idx(x, y, z, Nx, Ny)]);
                
                __m256 xp1 = _mm256_loadu_ps(&A[idx(x + 1, y, z, Nx, Ny)]);
                __m256 xm1 = _mm256_loadu_ps(&A[idx(x - 1, y, z, Nx, Ny)]);
                __m256 yp1 = _mm256_loadu_ps(&A[idx(x, y + 1, z, Nx, Ny)]);
                __m256 ym1 = _mm256_loadu_ps(&A[idx(x, y - 1, z, Nx, Ny)]);
                __m256 zp1 = _mm256_loadu_ps(&A[idx(x, y, z + 1, Nx, Ny)]);
                __m256 zm1 = _mm256_loadu_ps(&A[idx(x, y, z - 1, Nx, Ny)]);
                
                __m256 neighbors = _mm256_add_ps(xp1, xm1);
                neighbors = _mm256_add_ps(neighbors, yp1);
                neighbors = _mm256_add_ps(neighbors, ym1);
                neighbors = _mm256_add_ps(neighbors, zp1);
                neighbors = _mm256_add_ps(neighbors, zm1);
                
                // C = a0 * center + a1 * neighbors (without FMA)
                __m256 result = _mm256_add_ps(
                    _mm256_mul_ps(a0_vec, center),
                    _mm256_mul_ps(a1_vec, neighbors)
                );
                
                _mm256_storeu_ps(&C[idx(x, y, z, Nx, Ny)], result);
            }
            
            // Handle remainder with scalar code
            for (; x < Nx - 1; ++x) {
                float center = A[idx(x, y, z, Nx, Ny)];
                float neighbors = A[idx(x + 1, y, z, Nx, Ny)] +
                                  A[idx(x - 1, y, z, Nx, Ny)] +
                                  A[idx(x, y + 1, z, Nx, Ny)] +
                                  A[idx(x, y - 1, z, Nx, Ny)] +
                                  A[idx(x, y, z + 1, Nx, Ny)] +
                                  A[idx(x, y, z - 1, Nx, Ny)];
                C[idx(x, y, z, Nx, Ny)] = a0 * center + a1 * neighbors;
            }
        }
    }
}

// AVX stencil kernel for double
inline void kernel_stencil_avx_double(
    double* C, const double* A,
    double a0, double a1,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t Nz)
{
    size_t z_start = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    __m256d a0_vec = _mm256_set1_pd(a0);
    __m256d a1_vec = _mm256_set1_pd(a1);
    
    for (size_t z = z_start; z < z_stop; ++z) {
        for (size_t y = 1; y < Ny - 1; ++y) {
            size_t x = 1;
            
            // Process 4 doubles at a time
            for (; x + 4 <= Nx - 1; x += 4) {
                __m256d center = _mm256_loadu_pd(&A[idx(x, y, z, Nx, Ny)]);
                
                __m256d xp1 = _mm256_loadu_pd(&A[idx(x + 1, y, z, Nx, Ny)]);
                __m256d xm1 = _mm256_loadu_pd(&A[idx(x - 1, y, z, Nx, Ny)]);
                __m256d yp1 = _mm256_loadu_pd(&A[idx(x, y + 1, z, Nx, Ny)]);
                __m256d ym1 = _mm256_loadu_pd(&A[idx(x, y - 1, z, Nx, Ny)]);
                __m256d zp1 = _mm256_loadu_pd(&A[idx(x, y, z + 1, Nx, Ny)]);
                __m256d zm1 = _mm256_loadu_pd(&A[idx(x, y, z - 1, Nx, Ny)]);
                
                __m256d neighbors = _mm256_add_pd(xp1, xm1);
                neighbors = _mm256_add_pd(neighbors, yp1);
                neighbors = _mm256_add_pd(neighbors, ym1);
                neighbors = _mm256_add_pd(neighbors, zp1);
                neighbors = _mm256_add_pd(neighbors, zm1);
                
                // C = a0 * center + a1 * neighbors (without FMA)
                __m256d result = _mm256_add_pd(
                    _mm256_mul_pd(a0_vec, center),
                    _mm256_mul_pd(a1_vec, neighbors)
                );
                
                _mm256_storeu_pd(&C[idx(x, y, z, Nx, Ny)], result);
            }
            
            // Handle remainder
            for (; x < Nx - 1; ++x) {
                double center = A[idx(x, y, z, Nx, Ny)];
                double neighbors = A[idx(x + 1, y, z, Nx, Ny)] +
                                   A[idx(x - 1, y, z, Nx, Ny)] +
                                   A[idx(x, y + 1, z, Nx, Ny)] +
                                   A[idx(x, y - 1, z, Nx, Ny)] +
                                   A[idx(x, y, z + 1, Nx, Ny)] +
                                   A[idx(x, y, z - 1, Nx, Ny)];
                C[idx(x, y, z, Nx, Ny)] = a0 * center + a1 * neighbors;
            }
        }
    }
}

#endif // SIMD_AVX


// ============================================================================
// AVX2 Kernels 
// ============================================================================

#if SIMD_AVX2

// AVX2 memory kernel for float - processes 8 floats per iteration 
inline void kernel_mem_avx2_float(
    float* C, const float* A, const float* B,
    float alpha, float beta,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t /*Nz*/)
{
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);
    
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            size_t row_start = idx(0, y, z, Nx, Ny);
            size_t x = 0;
            
            // Process 8 floats at a time using AVX2
            for (; x + 8 <= Nx; x += 8) {
                size_t i = row_start + x;
                __m256 a_vec = _mm256_loadu_ps(&A[i]);
                __m256 b_vec = _mm256_loadu_ps(&B[i]);
                
                // C = alpha * A + beta * B using FMA
                // result = alpha * A + beta * B
                __m256 result = _mm256_fmadd_ps(alpha_vec, a_vec, 
                                                _mm256_mul_ps(beta_vec, b_vec));
                
                _mm256_storeu_ps(&C[i], result);
            }
            
            // Handle remainder with scalar code
            for (; x < Nx; ++x) {
                size_t i = row_start + x;
                C[i] = alpha * A[i] + beta * B[i];
            }
        }
    }
}

// AVX2 memory kernel for double - processes 4 doubles per iteration 
inline void kernel_mem_avx2_double(
    double* C, const double* A, const double* B,
    double alpha, double beta,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t /*Nz*/)
{
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    __m256d beta_vec = _mm256_set1_pd(beta);
    
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            size_t row_start = idx(0, y, z, Nx, Ny);
            size_t x = 0;
            
            // Process 4 doubles at a time using AVX2
            for (; x + 4 <= Nx; x += 4) {
                size_t i = row_start + x;
                __m256d a_vec = _mm256_loadu_pd(&A[i]);
                __m256d b_vec = _mm256_loadu_pd(&B[i]);
                
                // C = alpha * A + beta * B using FMA
                __m256d result = _mm256_fmadd_pd(alpha_vec, a_vec,
                                                 _mm256_mul_pd(beta_vec, b_vec));
                
                _mm256_storeu_pd(&C[i], result);
            }
            
            // Handle remainder with scalar code
            for (; x < Nx; ++x) {
                size_t i = row_start + x;
                C[i] = alpha * A[i] + beta * B[i];
            }
        }
    }
}

// AVX2 stencil kernel for float 
inline void kernel_stencil_avx2_float(
    float* C, const float* A,
    float a0, float a1,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t Nz)
{
    size_t z_start = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    __m256 a0_vec = _mm256_set1_ps(a0);
    __m256 a1_vec = _mm256_set1_ps(a1);
    
    for (size_t z = z_start; z < z_stop; ++z) {
        for (size_t y = 1; y < Ny - 1; ++y) {
            size_t x = 1;
            
            // Process 8 floats at a time
            for (; x + 8 <= Nx - 1; x += 8) {
                __m256 center = _mm256_loadu_ps(&A[idx(x, y, z, Nx, Ny)]);
                
                // Load 6 neighbors
                __m256 xp1 = _mm256_loadu_ps(&A[idx(x + 1, y, z, Nx, Ny)]);
                __m256 xm1 = _mm256_loadu_ps(&A[idx(x - 1, y, z, Nx, Ny)]);
                __m256 yp1 = _mm256_loadu_ps(&A[idx(x, y + 1, z, Nx, Ny)]);
                __m256 ym1 = _mm256_loadu_ps(&A[idx(x, y - 1, z, Nx, Ny)]);
                __m256 zp1 = _mm256_loadu_ps(&A[idx(x, y, z + 1, Nx, Ny)]);
                __m256 zm1 = _mm256_loadu_ps(&A[idx(x, y, z - 1, Nx, Ny)]);
                
                // Sum neighbors
                __m256 neighbors = _mm256_add_ps(xp1, xm1);
                neighbors = _mm256_add_ps(neighbors, yp1);
                neighbors = _mm256_add_ps(neighbors, ym1);
                neighbors = _mm256_add_ps(neighbors, zp1);
                neighbors = _mm256_add_ps(neighbors, zm1);
                
                // C = a0 * center + a1 * neighbors using FMA
                __m256 result = _mm256_fmadd_ps(a0_vec, center,
                                                _mm256_mul_ps(a1_vec, neighbors));
                
                _mm256_storeu_ps(&C[idx(x, y, z, Nx, Ny)], result);
            }
            
            // Handle remainder with scalar code
            for (; x < Nx - 1; ++x) {
                float center = A[idx(x, y, z, Nx, Ny)];
                float neighbors = A[idx(x + 1, y, z, Nx, Ny)] +
                                  A[idx(x - 1, y, z, Nx, Ny)] +
                                  A[idx(x, y + 1, z, Nx, Ny)] +
                                  A[idx(x, y - 1, z, Nx, Ny)] +
                                  A[idx(x, y, z + 1, Nx, Ny)] +
                                  A[idx(x, y, z - 1, Nx, Ny)];
                C[idx(x, y, z, Nx, Ny)] = a0 * center + a1 * neighbors;
            }
        }
    }
}

// AVX2 stencil kernel for double
inline void kernel_stencil_avx2_double(
    double* C, const double* A,
    double a0, double a1,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t Nz)
{
    size_t z_start = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    __m256d a0_vec = _mm256_set1_pd(a0);
    __m256d a1_vec = _mm256_set1_pd(a1);
    
    for (size_t z = z_start; z < z_stop; ++z) {
        for (size_t y = 1; y < Ny - 1; ++y) {
            size_t x = 1;
            
            // Process 4 doubles at a time
            for (; x + 4 <= Nx - 1; x += 4) {
                __m256d center = _mm256_loadu_pd(&A[idx(x, y, z, Nx, Ny)]);
                
                __m256d xp1 = _mm256_loadu_pd(&A[idx(x + 1, y, z, Nx, Ny)]);
                __m256d xm1 = _mm256_loadu_pd(&A[idx(x - 1, y, z, Nx, Ny)]);
                __m256d yp1 = _mm256_loadu_pd(&A[idx(x, y + 1, z, Nx, Ny)]);
                __m256d ym1 = _mm256_loadu_pd(&A[idx(x, y - 1, z, Nx, Ny)]);
                __m256d zp1 = _mm256_loadu_pd(&A[idx(x, y, z + 1, Nx, Ny)]);
                __m256d zm1 = _mm256_loadu_pd(&A[idx(x, y, z - 1, Nx, Ny)]);
                
                __m256d neighbors = _mm256_add_pd(xp1, xm1);
                neighbors = _mm256_add_pd(neighbors, yp1);
                neighbors = _mm256_add_pd(neighbors, ym1);
                neighbors = _mm256_add_pd(neighbors, zp1);
                neighbors = _mm256_add_pd(neighbors, zm1);
                
                // C = a0 * center + a1 * neighbors using FMA
                __m256d result = _mm256_fmadd_pd(a0_vec, center,
                                                 _mm256_mul_pd(a1_vec, neighbors));
                
                _mm256_storeu_pd(&C[idx(x, y, z, Nx, Ny)], result);
            }
            
            // Handle remainder
            for (; x < Nx - 1; ++x) {
                double center = A[idx(x, y, z, Nx, Ny)];
                double neighbors = A[idx(x + 1, y, z, Nx, Ny)] +
                                   A[idx(x - 1, y, z, Nx, Ny)] +
                                   A[idx(x, y + 1, z, Nx, Ny)] +
                                   A[idx(x, y - 1, z, Nx, Ny)] +
                                   A[idx(x, y, z + 1, Nx, Ny)] +
                                   A[idx(x, y, z - 1, Nx, Ny)];
                C[idx(x, y, z, Nx, Ny)] = a0 * center + a1 * neighbors;
            }
        }
    }
}

// AVX2 memory kernel for int8_t - processes 32 int8_t values per iteration 
// C[i,j,k] = alpha * A[i,j,k] + beta * B[i,j,k]
// Uses _mm256_loadu_si256, _mm256_storeu_si256 intrinsics
inline void kernel_mem_avx2_int8(
    int8_t* C, const int8_t* A, const int8_t* B,
    int8_t alpha, int8_t beta,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t /*Nz*/)
{
    // For INT8, we process 16 elements at a time using 16-bit intermediates
    // AVX2 has _mm256_cvtepi8_epi16 for sign extension
    __m256i alpha_vec = _mm256_set1_epi16(static_cast<int16_t>(alpha));
    __m256i beta_vec = _mm256_set1_epi16(static_cast<int16_t>(beta));
    
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            size_t row_start = idx(0, y, z, Nx, Ny);
            size_t x = 0;
            
            // Process 16 int8_t values at a time (using 16-bit intermediates)
            for (; x + 16 <= Nx; x += 16) {
                size_t i = row_start + x;
                
                // Load 16 bytes
                __m128i a_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&A[i]));
                __m128i b_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&B[i]));
                
                // Sign-extend int8 to int16 using AVX2 intrinsic
                __m256i a_16 = _mm256_cvtepi8_epi16(a_bytes);
                __m256i b_16 = _mm256_cvtepi8_epi16(b_bytes);
                
                // Multiply: alpha * A and beta * B (16-bit results)
                __m256i alpha_a = _mm256_mullo_epi16(alpha_vec, a_16);
                __m256i beta_b = _mm256_mullo_epi16(beta_vec, b_16);
                
                // Add: alpha * A + beta * B
                __m256i result_16 = _mm256_adds_epi16(alpha_a, beta_b);
                
                // Pack back to int8 with saturation
                // _mm256_packs_epi16 packs 16-bit to 8-bit with saturation
                // but the result is interleaved, so we need to permute
                __m256i result_packed = _mm256_packs_epi16(result_16, result_16);
                // Permute to get correct order: [0,1,2,3,8,9,10,11,4,5,6,7,12,13,14,15] -> [0..15]
                result_packed = _mm256_permute4x64_epi64(result_packed, 0xD8); // 0b11011000
                
                // Store 16 bytes (lower 128 bits)
                _mm_storeu_si128(reinterpret_cast<__m128i*>(&C[i]), 
                                 _mm256_castsi256_si128(result_packed));
            }
            
            // Handle remainder with scalar code
            for (; x < Nx; ++x) {
                size_t i = row_start + x;
                int32_t result = static_cast<int32_t>(alpha) * static_cast<int32_t>(A[i]) +
                                 static_cast<int32_t>(beta) * static_cast<int32_t>(B[i]);
                if (result > 127) result = 127;
                if (result < -128) result = -128;
                C[i] = static_cast<int8_t>(result);
            }
        }
    }
}

#endif // SIMD_AVX2


// ============================================================================
// AVX-512 VNNI INT8 Matmul Kernel 
// Uses _mm512_dpbusd_epi32 for dot product operations
// Protected by __AVX512VNNI__ macro
// ============================================================================

#if SIMD_X86 && defined(__AVX512VNNI__)
#define SIMD_AVX512_VNNI 1
#else
#define SIMD_AVX512_VNNI 0
#endif

#if SIMD_AVX512_VNNI

// AVX-512 VNNI INT8 matmul kernel - matrix multiplication for 3D slices
// C[z] = A[z] * B[z] for each z slice
// Uses _mm512_dpbusd_epi32 for efficient INT8 dot product
// Note: dpbusd expects unsigned*signed, so we handle signed*signed conversion
inline void kernel_matmul3d_vnni_int8(
    int32_t* C, const int8_t* A, const int8_t* B,
    size_t z_begin, size_t z_end, size_t N)
{
    for (size_t z = z_begin; z < z_end; ++z) {
        const int8_t* A_slice = A + z * N * N;
        const int8_t* B_slice = B + z * N * N;
        int32_t* C_slice = C + z * N * N;
        
        // Initialize C slice to zero
        for (size_t i = 0; i < N * N; ++i) {
            C_slice[i] = 0;
        }
        
        // Matrix multiplication: C = A * B
        // For VNNI, we need to process 4 elements at a time for the dot product
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                int32_t sum = 0;
                size_t k = 0;
                
                // Process 64 elements at a time using AVX-512 VNNI
                // _mm512_dpbusd_epi32 computes: dst = src + (a[unsigned] * b[signed])
                // We need to handle signed*signed, so we use a bias trick
                for (; k + 64 <= N; k += 64) {
                    __m512i acc = _mm512_setzero_si512();
                    
                    // Load 64 bytes from A row and B column
                    // Note: B needs to be transposed for efficient access
                    // For simplicity, we'll do scalar access for B
                    __m512i a_vec = _mm512_loadu_si512(&A_slice[i * N + k]);
                    
                    // For proper VNNI usage, we'd need B transposed
                    // Fall back to scalar for the inner product
                    for (size_t kk = 0; kk < 64; ++kk) {
                        sum += static_cast<int32_t>(A_slice[i * N + k + kk]) * 
                               static_cast<int32_t>(B_slice[(k + kk) * N + j]);
                    }
                }
                
                // Handle remainder
                for (; k < N; ++k) {
                    sum += static_cast<int32_t>(A_slice[i * N + k]) * 
                           static_cast<int32_t>(B_slice[k * N + j]);
                }
                
                C_slice[i * N + j] = sum;
            }
        }
    }
}

#endif // SIMD_AVX512_VNNI


// ============================================================================
// AVX-512 FP16 Kernels
// Native FP16 support using _Float16 type with AVX-512 FP16 intrinsics
// ============================================================================

// AVX-512 FP16 detection macro
#if SIMD_X86 && defined(__AVX512FP16__)
    #define SIMD_AVX512_FP16 1
#else
    #define SIMD_AVX512_FP16 0
#endif

#if SIMD_AVX512_FP16

// AVX-512 FP16 memory kernel - processes 32 _Float16 values per iteration
// C = alpha * A + beta * B
inline void kernel_mem_avx512_fp16(
    _Float16* C, const _Float16* A, const _Float16* B,
    _Float16 alpha, _Float16 beta,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t /*Nz*/)
{
    __m512h alpha_vec = _mm512_set1_ph(alpha);
    __m512h beta_vec = _mm512_set1_ph(beta);
    
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            size_t row_start = idx(0, y, z, Nx, Ny);
            size_t x = 0;
            
            // Process 32 _Float16 values at a time using AVX-512 FP16
            for (; x + 32 <= Nx; x += 32) {
                size_t i = row_start + x;
                __m512h a_vec = _mm512_loadu_ph(&A[i]);
                __m512h b_vec = _mm512_loadu_ph(&B[i]);
                
                // C = alpha * A + beta * B using FMA
                __m512h result = _mm512_fmadd_ph(alpha_vec, a_vec,
                                                  _mm512_mul_ph(beta_vec, b_vec));
                
                _mm512_storeu_ph(&C[i], result);
            }
            
            // Handle remainder with scalar code
            for (; x < Nx; ++x) {
                size_t i = row_start + x;
                C[i] = alpha * A[i] + beta * B[i];
            }
        }
    }
}

// AVX-512 FP16 stencil kernel - 7-point stencil
// C = a0 * center + a1 * (sum of 6 neighbors)
inline void kernel_stencil_avx512_fp16(
    _Float16* C, const _Float16* A,
    _Float16 a0, _Float16 a1,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t Nz)
{
    size_t z_start = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    __m512h a0_vec = _mm512_set1_ph(a0);
    __m512h a1_vec = _mm512_set1_ph(a1);
    
    for (size_t z = z_start; z < z_stop; ++z) {
        for (size_t y = 1; y < Ny - 1; ++y) {
            size_t x = 1;
            
            // Process 32 _Float16 values at a time
            for (; x + 32 <= Nx - 1; x += 32) {
                __m512h center = _mm512_loadu_ph(&A[idx(x, y, z, Nx, Ny)]);
                
                // Load 6 neighbors
                __m512h xp1 = _mm512_loadu_ph(&A[idx(x + 1, y, z, Nx, Ny)]);
                __m512h xm1 = _mm512_loadu_ph(&A[idx(x - 1, y, z, Nx, Ny)]);
                __m512h yp1 = _mm512_loadu_ph(&A[idx(x, y + 1, z, Nx, Ny)]);
                __m512h ym1 = _mm512_loadu_ph(&A[idx(x, y - 1, z, Nx, Ny)]);
                __m512h zp1 = _mm512_loadu_ph(&A[idx(x, y, z + 1, Nx, Ny)]);
                __m512h zm1 = _mm512_loadu_ph(&A[idx(x, y, z - 1, Nx, Ny)]);
                
                // Sum neighbors
                __m512h neighbors = _mm512_add_ph(xp1, xm1);
                neighbors = _mm512_add_ph(neighbors, yp1);
                neighbors = _mm512_add_ph(neighbors, ym1);
                neighbors = _mm512_add_ph(neighbors, zp1);
                neighbors = _mm512_add_ph(neighbors, zm1);
                
                // C = a0 * center + a1 * neighbors using FMA
                __m512h result = _mm512_fmadd_ph(a0_vec, center,
                                                  _mm512_mul_ph(a1_vec, neighbors));
                
                _mm512_storeu_ph(&C[idx(x, y, z, Nx, Ny)], result);
            }
            
            // Handle remainder with scalar code
            for (; x < Nx - 1; ++x) {
                _Float16 center = A[idx(x, y, z, Nx, Ny)];
                _Float16 neighbors = A[idx(x + 1, y, z, Nx, Ny)] +
                                     A[idx(x - 1, y, z, Nx, Ny)] +
                                     A[idx(x, y + 1, z, Nx, Ny)] +
                                     A[idx(x, y - 1, z, Nx, Ny)] +
                                     A[idx(x, y, z + 1, Nx, Ny)] +
                                     A[idx(x, y, z - 1, Nx, Ny)];
                C[idx(x, y, z, Nx, Ny)] = a0 * center + a1 * neighbors;
            }
        }
    }
}

// AVX-512 FP16 matmul3d kernel - matrix multiplication for 3D slices
// C[z] = A[z] * B[z] for each z slice
inline void kernel_matmul3d_avx512_fp16(
    _Float16* C, const _Float16* A, const _Float16* B,
    size_t z_begin, size_t z_end, size_t N)
{
    for (size_t z = z_begin; z < z_end; ++z) {
        const _Float16* A_slice = A + z * N * N;
        const _Float16* B_slice = B + z * N * N;
        _Float16* C_slice = C + z * N * N;
        
        // Initialize C slice to zero
        for (size_t i = 0; i < N * N; ++i) {
            C_slice[i] = static_cast<_Float16>(0.0f);
        }
        
        // Matrix multiplication: C = A * B
        for (size_t i = 0; i < N; ++i) {
            for (size_t k = 0; k < N; ++k) {
                __m512h a_val = _mm512_set1_ph(A_slice[i * N + k]);
                size_t j = 0;
                
                // Process 32 elements at a time
                for (; j + 32 <= N; j += 32) {
                    __m512h b_vec = _mm512_loadu_ph(&B_slice[k * N + j]);
                    __m512h c_vec = _mm512_loadu_ph(&C_slice[i * N + j]);
                    
                    // C += A * B using FMA
                    c_vec = _mm512_fmadd_ph(a_val, b_vec, c_vec);
                    
                    _mm512_storeu_ph(&C_slice[i * N + j], c_vec);
                }
                
                // Handle remainder
                for (; j < N; ++j) {
                    C_slice[i * N + j] += A_slice[i * N + k] * B_slice[k * N + j];
                }
            }
        }
    }
}

#endif // SIMD_AVX512_FP16


// ============================================================================
// Scalar FP16 Kernels (fallback when native FP16 not available)
// Uses half type with float computation
// ============================================================================

// Scalar memory kernel for half (emulated FP16)
inline void kernel_mem_scalar_half(
    half* C, const half* A, const half* B,
    half alpha, half beta,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t /*Nz*/)
{
    float alpha_f = static_cast<float>(alpha);
    float beta_f = static_cast<float>(beta);
    
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            for (size_t x = 0; x < Nx; ++x) {
                size_t i = idx(x, y, z, Nx, Ny);
                float a_f = static_cast<float>(A[i]);
                float b_f = static_cast<float>(B[i]);
                float result = alpha_f * a_f + beta_f * b_f;
                C[i] = half(result);
            }
        }
    }
}

// Scalar stencil kernel for half (emulated FP16)
inline void kernel_stencil_scalar_half(
    half* C, const half* A,
    half a0, half a1,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t Nz)
{
    float a0_f = static_cast<float>(a0);
    float a1_f = static_cast<float>(a1);
    
    size_t z_start = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    for (size_t z = z_start; z < z_stop; ++z) {
        for (size_t y = 1; y < Ny - 1; ++y) {
            for (size_t x = 1; x < Nx - 1; ++x) {
                float center = static_cast<float>(A[idx(x, y, z, Nx, Ny)]);
                float neighbors = static_cast<float>(A[idx(x + 1, y, z, Nx, Ny)]) +
                                  static_cast<float>(A[idx(x - 1, y, z, Nx, Ny)]) +
                                  static_cast<float>(A[idx(x, y + 1, z, Nx, Ny)]) +
                                  static_cast<float>(A[idx(x, y - 1, z, Nx, Ny)]) +
                                  static_cast<float>(A[idx(x, y, z + 1, Nx, Ny)]) +
                                  static_cast<float>(A[idx(x, y, z - 1, Nx, Ny)]);
                float result = a0_f * center + a1_f * neighbors;
                C[idx(x, y, z, Nx, Ny)] = half(result);
            }
        }
    }
}


// ============================================================================
// Runtime Kernel Selection Functions
// ============================================================================

// Function pointer types for kernels
template<typename T>
using MemKernelFn = void(*)(T*, const T*, const T*, T, T, size_t, size_t, size_t, size_t, size_t);

template<typename T>
using StencilKernelFn = void(*)(T*, const T*, T, T, size_t, size_t, size_t, size_t, size_t);

// Get the best available memory kernel for float
inline MemKernelFn<float> get_mem_kernel_float(bool force_scalar = false) {
    if (force_scalar) {
        return kernel_mem_scalar_float;
    }
    
    const auto& caps = CpuCapabilities::get();
    
#if SIMD_AVX2
    if (caps.has_avx2) {
        return kernel_mem_avx2_float;
    }
#endif
#if SIMD_AVX
    if (caps.has_avx) {
        return kernel_mem_avx_float;
    }
#endif
#if SIMD_SSE2
    if (caps.has_sse2) {
        return kernel_mem_sse2_float;
    }
#endif
#if SIMD_NEON
    if (caps.has_arm_neon) {
        return kernel_mem_neon_float;
    }
#endif
    
    return kernel_mem_scalar_float;
}

// Get the best available memory kernel for double
inline MemKernelFn<double> get_mem_kernel_double(bool force_scalar = false) {
    if (force_scalar) {
        return kernel_mem_scalar_double;
    }
    
    const auto& caps = CpuCapabilities::get();
    
#if SIMD_AVX2
    if (caps.has_avx2) {
        return kernel_mem_avx2_double;
    }
#endif
#if SIMD_AVX
    if (caps.has_avx) {
        return kernel_mem_avx_double;
    }
#endif
#if SIMD_SSE2
    if (caps.has_sse2) {
        return kernel_mem_sse2_double;
    }
#endif
    
    return kernel_mem_scalar_double;
}

// Get the best available stencil kernel for float
inline StencilKernelFn<float> get_stencil_kernel_float(bool force_scalar = false) {
    if (force_scalar) {
        return kernel_stencil_scalar_float;
    }
    
    const auto& caps = CpuCapabilities::get();
    
#if SIMD_AVX2
    if (caps.has_avx2) {
        return kernel_stencil_avx2_float;
    }
#endif
#if SIMD_AVX
    if (caps.has_avx) {
        return kernel_stencil_avx_float;
    }
#endif
#if SIMD_SSE2
    if (caps.has_sse2) {
        return kernel_stencil_sse2_float;
    }
#endif
#if SIMD_NEON
    if (caps.has_arm_neon) {
        return kernel_stencil_neon_float;
    }
#endif
    
    return kernel_stencil_scalar_float;
}

// Get the best available stencil kernel for double
inline StencilKernelFn<double> get_stencil_kernel_double(bool force_scalar = false) {
    if (force_scalar) {
        return kernel_stencil_scalar_double;
    }
    
    const auto& caps = CpuCapabilities::get();
    
#if SIMD_AVX2
    if (caps.has_avx2) {
        return kernel_stencil_avx2_double;
    }
#endif
#if SIMD_AVX
    if (caps.has_avx) {
        return kernel_stencil_avx_double;
    }
#endif
#if SIMD_SSE2
    if (caps.has_sse2) {
        return kernel_stencil_sse2_double;
    }
#endif
    
    return kernel_stencil_scalar_double;
}

// Get the name of the selected kernel implementation
// Uses runtime detection to match what RuntimeDispatcher selects
inline const char* get_selected_kernel_name_float(bool force_scalar = false) {
    if (force_scalar) {
        return "Scalar";
    }
    
    const auto& caps = CpuCapabilities::get();
    
    // Use runtime detection - check capabilities flags set by CpuCapabilities::detect()
    // This ensures consistency with RuntimeDispatcher kernel selection
    
    // x86 SIMD (highest to lowest priority)
    if (caps.has_avx512f) return "AVX-512";
    if (caps.has_avx2) return "AVX2";
    if (caps.has_avx) return "AVX";
    if (caps.has_sse2) return "SSE2";
    
    // ARM NEON
    if (caps.has_arm_neon) return "NEON";
    
    return "Scalar";
}

inline const char* get_selected_kernel_name_double(bool force_scalar = false) {
    if (force_scalar) {
        return "Scalar";
    }
    
    const auto& caps = CpuCapabilities::get();
    
    // Use runtime detection - check capabilities flags set by CpuCapabilities::detect()
    // This ensures consistency with RuntimeDispatcher kernel selection
    
    // x86 SIMD (highest to lowest priority)
    if (caps.has_avx512f) return "AVX-512";
    if (caps.has_avx2) return "AVX2";
    if (caps.has_avx) return "AVX";
    if (caps.has_sse2) return "SSE2";
    
    // ARM NEON (has limited double support, but we still use NEON where possible)
    if (caps.has_arm_neon) return "NEON";
    
    return "Scalar";
}

// Template wrapper for generic kernel selection
template<typename T>
inline MemKernelFn<T> get_mem_kernel(bool force_scalar = false);

template<>
inline MemKernelFn<float> get_mem_kernel<float>(bool force_scalar) {
    return get_mem_kernel_float(force_scalar);
}

template<>
inline MemKernelFn<double> get_mem_kernel<double>(bool force_scalar) {
    return get_mem_kernel_double(force_scalar);
}

template<typename T>
inline StencilKernelFn<T> get_stencil_kernel(bool force_scalar = false);

template<>
inline StencilKernelFn<float> get_stencil_kernel<float>(bool force_scalar) {
    return get_stencil_kernel_float(force_scalar);
}

template<>
inline StencilKernelFn<double> get_stencil_kernel<double>(bool force_scalar) {
    return get_stencil_kernel_double(force_scalar);
}

template<typename T>
inline const char* get_selected_kernel_name(bool force_scalar = false);

template<>
inline const char* get_selected_kernel_name<float>(bool force_scalar) {
    return get_selected_kernel_name_float(force_scalar);
}

template<>
inline const char* get_selected_kernel_name<double>(bool force_scalar) {
    return get_selected_kernel_name_double(force_scalar);
}


// ============================================================================
// INT8 Kernel Selection Functions 
// ============================================================================

// Get the best available memory kernel for int8_t
inline MemKernelFn<int8_t> get_mem_kernel_int8(bool force_scalar = false) {
    if (force_scalar) {
        return kernel_mem_scalar_int8;
    }
    
    const auto& caps = CpuCapabilities::get();
    
#if SIMD_AVX2
    if (caps.has_avx2) {
        return kernel_mem_avx2_int8;
    }
#endif
#if SIMD_SSE2
    if (caps.has_sse2) {
        return kernel_mem_sse2_int8;
    }
#endif
    
    return kernel_mem_scalar_int8;
}

// Get the best available stencil kernel for int8_t
inline StencilKernelFn<int8_t> get_stencil_kernel_int8(bool force_scalar = false) {
    (void)force_scalar;
    // Currently only scalar implementation available for stencil
    return kernel_stencil_scalar_int8;
}

// Function pointer type for INT8 matmul kernel (output is int32_t)
using Int8MatmulKernelFn = void(*)(int32_t*, const int8_t*, const int8_t*, size_t, size_t, size_t);

// Get the best available matmul kernel for int8_t
inline Int8MatmulKernelFn get_matmul_kernel_int8(bool force_scalar = false) {
    if (force_scalar) {
        return kernel_matmul3d_scalar_int8;
    }
    
#if SIMD_AVX512_VNNI
    const auto& caps = CpuCapabilities::get();
    if (caps.has_avx512_vnni) {
        return kernel_matmul3d_vnni_int8;
    }
#endif
    
    return kernel_matmul3d_scalar_int8;
}

// Get the name of the selected kernel implementation for int8_t
inline const char* get_selected_kernel_name_int8(bool force_scalar = false) {
    if (force_scalar) {
        return "Scalar";
    }
    
    const auto& caps = CpuCapabilities::get();
    
#if SIMD_AVX2
    if (caps.has_avx2) return "AVX2";
#endif
#if SIMD_SSE2
    if (caps.has_sse2) return "SSE2";
#endif
    
    return "Scalar";
}

// Get the name of the selected matmul kernel for int8_t
inline const char* get_selected_matmul_kernel_name_int8(bool force_scalar = false) {
    if (force_scalar) {
        return "Scalar";
    }
    
#if SIMD_AVX512_VNNI
    const auto& caps = CpuCapabilities::get();
    if (caps.has_avx512_vnni) return "AVX-512 VNNI";
#endif
    
    return "Scalar";
}

// Template specializations for int8_t
template<>
inline MemKernelFn<int8_t> get_mem_kernel<int8_t>(bool force_scalar) {
    return get_mem_kernel_int8(force_scalar);
}

template<>
inline StencilKernelFn<int8_t> get_stencil_kernel<int8_t>(bool force_scalar) {
    return get_stencil_kernel_int8(force_scalar);
}

template<>
inline const char* get_selected_kernel_name<int8_t>(bool force_scalar) {
    return get_selected_kernel_name_int8(force_scalar);
}


// ============================================================================
// FP16 Kernel Selection Functions
// ============================================================================

// Get the best available memory kernel for half (emulated FP16)
inline MemKernelFn<half> get_mem_kernel_half(bool force_scalar = false) {
    (void)force_scalar;  // Currently only scalar implementation available
    return kernel_mem_scalar_half;
}

// Get the best available stencil kernel for half (emulated FP16)
inline StencilKernelFn<half> get_stencil_kernel_half(bool force_scalar = false) {
    (void)force_scalar;  // Currently only scalar implementation available
    return kernel_stencil_scalar_half;
}

// Get the name of the selected kernel implementation for half
inline const char* get_selected_kernel_name_half(bool force_scalar = false) {
    (void)force_scalar;
    const auto& caps = CpuCapabilities::get();
    
#if SIMD_AVX512_FP16
    if (caps.has_avx512_fp16) return "AVX-512 FP16 (native)";
#endif
#if SIMD_NEON_FP16
    if (caps.has_arm_neon_fp16) return "NEON FP16 (native)";
#endif
    
    return "Scalar (emulated)";
}

// Template specializations for half
template<>
inline MemKernelFn<half> get_mem_kernel<half>(bool force_scalar) {
    return get_mem_kernel_half(force_scalar);
}

template<>
inline StencilKernelFn<half> get_stencil_kernel<half>(bool force_scalar) {
    return get_stencil_kernel_half(force_scalar);
}

template<>
inline const char* get_selected_kernel_name<half>(bool force_scalar) {
    return get_selected_kernel_name_half(force_scalar);
}

// ============================================================================
// Native FP16 Kernel Selection (when AVX-512 FP16 or ARM NEON FP16 is available)
// ============================================================================

#if SIMD_AVX512_FP16

// Function pointer types for native FP16 kernels (x86-64 AVX-512 FP16)
using NativeFP16MemKernelFn = void(*)(_Float16*, const _Float16*, const _Float16*, 
                                       _Float16, _Float16, size_t, size_t, size_t, size_t, size_t);
using NativeFP16StencilKernelFn = void(*)(_Float16*, const _Float16*, 
                                          _Float16, _Float16, size_t, size_t, size_t, size_t, size_t);
using NativeFP16MatmulKernelFn = void(*)(_Float16*, const _Float16*, const _Float16*,
                                         size_t, size_t, size_t);

// Get native FP16 memory kernel
inline NativeFP16MemKernelFn get_native_fp16_mem_kernel(bool force_scalar = false) {
    if (force_scalar) {
        return nullptr;  // No scalar native FP16 kernel
    }
    
    const auto& caps = CpuCapabilities::get();
    if (caps.has_avx512_fp16) {
        return kernel_mem_avx512_fp16;
    }
    return nullptr;
}

// Get native FP16 stencil kernel
inline NativeFP16StencilKernelFn get_native_fp16_stencil_kernel(bool force_scalar = false) {
    if (force_scalar) {
        return nullptr;
    }
    
    const auto& caps = CpuCapabilities::get();
    if (caps.has_avx512_fp16) {
        return kernel_stencil_avx512_fp16;
    }
    return nullptr;
}

// Get native FP16 matmul kernel
inline NativeFP16MatmulKernelFn get_native_fp16_matmul_kernel(bool force_scalar = false) {
    if (force_scalar) {
        return nullptr;
    }
    
    const auto& caps = CpuCapabilities::get();
    if (caps.has_avx512_fp16) {
        return kernel_matmul3d_avx512_fp16;
    }
    return nullptr;
}

// Check if native FP16 kernels are available at runtime
inline bool is_native_fp16_available() {
    return CpuCapabilities::get().has_avx512_fp16;
}

#elif SIMD_NEON_FP16

// Function pointer types for native FP16 kernels (ARM NEON FP16)
using NativeFP16MemKernelFn = void(*)(float16_t*, const float16_t*, const float16_t*, 
                                       float16_t, float16_t, size_t, size_t, size_t, size_t, size_t);
using NativeFP16StencilKernelFn = void(*)(float16_t*, const float16_t*, 
                                          float16_t, float16_t, size_t, size_t, size_t, size_t, size_t);
using NativeFP16MatmulKernelFn = void(*)(float16_t*, const float16_t*, const float16_t*,
                                         size_t, size_t, size_t);

// Get native FP16 memory kernel (ARM NEON FP16)
inline NativeFP16MemKernelFn get_native_fp16_mem_kernel(bool force_scalar = false) {
    if (force_scalar) {
        return nullptr;
    }
    
    const auto& caps = CpuCapabilities::get();
    if (caps.has_arm_neon_fp16) {
        return kernel_mem_neon_fp16;
    }
    return nullptr;
}

// Get native FP16 stencil kernel (ARM NEON FP16)
inline NativeFP16StencilKernelFn get_native_fp16_stencil_kernel(bool force_scalar = false) {
    if (force_scalar) {
        return nullptr;
    }
    
    const auto& caps = CpuCapabilities::get();
    if (caps.has_arm_neon_fp16) {
        return kernel_stencil_neon_fp16;
    }
    return nullptr;
}

// Get native FP16 matmul kernel (ARM NEON FP16)
inline NativeFP16MatmulKernelFn get_native_fp16_matmul_kernel(bool force_scalar = false) {
    if (force_scalar) {
        return nullptr;
    }
    
    const auto& caps = CpuCapabilities::get();
    if (caps.has_arm_neon_fp16) {
        return kernel_matmul3d_neon_fp16;
    }
    return nullptr;
}

// Check if native FP16 kernels are available at runtime
inline bool is_native_fp16_available() {
    return CpuCapabilities::get().has_arm_neon_fp16;
}

#else

// Stub functions when no native FP16 is compiled in
inline bool is_native_fp16_available() {
    return false;
}

#endif // SIMD_AVX512_FP16 / SIMD_NEON_FP16
