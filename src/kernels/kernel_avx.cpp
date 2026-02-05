// CPU Benchmark - AVX Kernel Implementations
// AVX implementations - without FMA
// This file must be compiled with -mavx (GCC/Clang) or /arch:AVX (MSVC)

#include "kernel_common.hpp"

// Only compile AVX code on x86 platforms
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#ifdef _WIN32
    #include <intrin.h>
#else
    #include <immintrin.h>
#endif

namespace kernels {
namespace avx {

// AVX memory kernel for float - processes 8 floats per iteration
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

} // namespace avx
} // namespace kernels

#else // Non-x86 platforms - provide stub implementations that call scalar

#include "kernel_common.hpp"

namespace kernels {
namespace avx {

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

} // namespace avx
} // namespace kernels

#endif // x86 platform check
