// CPU Benchmark - ARM NEON Kernel Implementations
// ARM NEON implementations
// This file must be compiled with -march=armv8-a+simd (GCC/Clang)

#include "kernel_common.hpp"

// Only compile NEON code on ARM platforms
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)

#include <arm_neon.h>

namespace kernels {
namespace neon {

// NEON memory kernel for float - processes 4 floats per iteration
void mem_float(float* C, const float* A, const float* B,
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
void stencil_float(float* C, const float* A,
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

} // namespace neon
} // namespace kernels

#else // Non-ARM platforms - provide stub implementations

namespace kernels {
namespace neon {

void mem_float(float* C, const float* A, const float* B,
               float alpha, float beta,
               size_t z_begin, size_t z_end,
               size_t Nx, size_t Ny, size_t Nz) {
    kernels::scalar::mem_float(C, A, B, alpha, beta, z_begin, z_end, Nx, Ny, Nz);
}

void stencil_float(float* C, const float* A,
                   float a0, float a1,
                   size_t z_begin, size_t z_end,
                   size_t Nx, size_t Ny, size_t Nz) {
    kernels::scalar::stencil_float(C, A, a0, a1, z_begin, z_end, Nx, Ny, Nz);
}

} // namespace neon
} // namespace kernels

#endif // ARM platform check
