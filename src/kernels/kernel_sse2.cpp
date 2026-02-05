// CPU Benchmark - SSE2 Kernel Implementations
// SSE2 implementations - baseline for x86-64
// This file must be compiled with -msse2 (GCC/Clang) or /arch:SSE2 (MSVC)

#include "kernel_common.hpp"

// Only compile SSE2 code on x86 platforms
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#ifdef _WIN32
    #include <intrin.h>
#else
    #include <emmintrin.h>
#endif

namespace kernels {
namespace sse2 {

// SSE2 memory kernel for float - processes 4 floats per iteration
void mem_float(float* C, const float* A, const float* B,
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
void mem_double(double* C, const double* A, const double* B,
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
void stencil_float(float* C, const float* A,
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
            
            // Process 4 floats at a time
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
void stencil_double(double* C, const double* A,
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

// SSE2 memory kernel for int8_t - processes 8 int8_t values per iteration
void mem_int8(int8_t* C, const int8_t* A, const int8_t* B,
              int8_t alpha, int8_t beta,
              size_t z_begin, size_t z_end,
              size_t Nx, size_t Ny, size_t /*Nz*/)
{
    // For INT8, we process 8 elements at a time using 16-bit intermediates
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

} // namespace sse2
} // namespace kernels

#else // Non-x86 platforms - provide stub implementations

namespace kernels {
namespace sse2 {

void mem_float(float* C, const float* A, const float* B,
               float alpha, float beta,
               size_t z_begin, size_t z_end,
               size_t Nx, size_t Ny, size_t Nz) {
    kernels::scalar::mem_float(C, A, B, alpha, beta, z_begin, z_end, Nx, Ny, Nz);
}

void mem_double(double* C, const double* A, const double* B,
                double alpha, double beta,
                size_t z_begin, size_t z_end,
                size_t Nx, size_t Ny, size_t Nz) {
    kernels::scalar::mem_double(C, A, B, alpha, beta, z_begin, z_end, Nx, Ny, Nz);
}

void stencil_float(float* C, const float* A,
                   float a0, float a1,
                   size_t z_begin, size_t z_end,
                   size_t Nx, size_t Ny, size_t Nz) {
    kernels::scalar::stencil_float(C, A, a0, a1, z_begin, z_end, Nx, Ny, Nz);
}

void stencil_double(double* C, const double* A,
                    double a0, double a1,
                    size_t z_begin, size_t z_end,
                    size_t Nx, size_t Ny, size_t Nz) {
    kernels::scalar::stencil_double(C, A, a0, a1, z_begin, z_end, Nx, Ny, Nz);
}

void mem_int8(int8_t* C, const int8_t* A, const int8_t* B,
              int8_t alpha, int8_t beta,
              size_t z_begin, size_t z_end,
              size_t Nx, size_t Ny, size_t Nz) {
    kernels::scalar::mem_int8(C, A, B, alpha, beta, z_begin, z_end, Nx, Ny, Nz);
}

} // namespace sse2
} // namespace kernels

#endif // x86 platform check
