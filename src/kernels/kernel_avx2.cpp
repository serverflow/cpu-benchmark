// CPU Benchmark - AVX2 Kernel Implementations
// AVX2 implementations with FMA support
// This file must be compiled with -mavx2 -mfma (GCC/Clang) or /arch:AVX2 (MSVC)

#include "kernel_common.hpp"

// Only compile AVX2 code on x86 platforms
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#ifdef _WIN32
    #include <intrin.h>
#else
    #include <immintrin.h>
#endif

namespace kernels {
namespace avx2 {

// AVX2 memory kernel for float - processes 8 floats per iteration with FMA
void mem_float(float* C, const float* A, const float* B,
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

// AVX2 memory kernel for double - processes 4 doubles per iteration with FMA
void mem_double(double* C, const double* A, const double* B,
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

// AVX2 stencil kernel for float with FMA
void stencil_float(float* C, const float* A,
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

// AVX2 stencil kernel for double with FMA
void stencil_double(double* C, const double* A,
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

// AVX2 memory kernel for int8_t - processes 16 int8_t values per iteration
void mem_int8(int8_t* C, const int8_t* A, const int8_t* B,
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
                // Permute to get correct order
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

} // namespace avx2
} // namespace kernels

#else // Non-x86 platforms - provide stub implementations

namespace kernels {
namespace avx2 {

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

} // namespace avx2
} // namespace kernels

#endif // x86 platform check
