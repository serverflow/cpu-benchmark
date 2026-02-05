// CPU Benchmark - AVX-512 Kernel Implementations
// AVX-512 implementations
// This file must be compiled with -mavx512f (GCC/Clang) or /arch:AVX512 (MSVC)

#include "kernel_common.hpp"

// Only compile AVX-512 code on x86 platforms
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#ifdef _WIN32
    #include <intrin.h>
#else
    #include <immintrin.h>
#endif

namespace kernels {
namespace avx512 {

// AVX-512 memory kernel for float - processes 16 floats per iteration
void mem_float(float* C, const float* A, const float* B,
               float alpha, float beta,
               size_t z_begin, size_t z_end,
               size_t Nx, size_t Ny, size_t /*Nz*/)
{
    __m512 alpha_vec = _mm512_set1_ps(alpha);
    __m512 beta_vec = _mm512_set1_ps(beta);
    
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            size_t row_start = idx(0, y, z, Nx, Ny);
            size_t x = 0;
            
            // Process 16 floats at a time using AVX-512
            for (; x + 16 <= Nx; x += 16) {
                size_t i = row_start + x;
                __m512 a_vec = _mm512_loadu_ps(&A[i]);
                __m512 b_vec = _mm512_loadu_ps(&B[i]);
                
                // C = alpha * A + beta * B using FMA
                __m512 result = _mm512_fmadd_ps(alpha_vec, a_vec, 
                                                _mm512_mul_ps(beta_vec, b_vec));
                
                _mm512_storeu_ps(&C[i], result);
            }
            
            // Handle remainder with scalar code
            for (; x < Nx; ++x) {
                size_t i = row_start + x;
                C[i] = alpha * A[i] + beta * B[i];
            }
        }
    }
}

// AVX-512 memory kernel for double - processes 8 doubles per iteration
void mem_double(double* C, const double* A, const double* B,
                double alpha, double beta,
                size_t z_begin, size_t z_end,
                size_t Nx, size_t Ny, size_t /*Nz*/)
{
    __m512d alpha_vec = _mm512_set1_pd(alpha);
    __m512d beta_vec = _mm512_set1_pd(beta);
    
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            size_t row_start = idx(0, y, z, Nx, Ny);
            size_t x = 0;
            
            // Process 8 doubles at a time using AVX-512
            for (; x + 8 <= Nx; x += 8) {
                size_t i = row_start + x;
                __m512d a_vec = _mm512_loadu_pd(&A[i]);
                __m512d b_vec = _mm512_loadu_pd(&B[i]);
                
                // C = alpha * A + beta * B using FMA
                __m512d result = _mm512_fmadd_pd(alpha_vec, a_vec,
                                                 _mm512_mul_pd(beta_vec, b_vec));
                
                _mm512_storeu_pd(&C[i], result);
            }
            
            // Handle remainder with scalar code
            for (; x < Nx; ++x) {
                size_t i = row_start + x;
                C[i] = alpha * A[i] + beta * B[i];
            }
        }
    }
}

// AVX-512 stencil kernel for float
void stencil_float(float* C, const float* A,
                   float a0, float a1,
                   size_t z_begin, size_t z_end,
                   size_t Nx, size_t Ny, size_t Nz)
{
    size_t z_start = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    __m512 a0_vec = _mm512_set1_ps(a0);
    __m512 a1_vec = _mm512_set1_ps(a1);
    
    for (size_t z = z_start; z < z_stop; ++z) {
        for (size_t y = 1; y < Ny - 1; ++y) {
            size_t x = 1;
            
            // Process 16 floats at a time
            for (; x + 16 <= Nx - 1; x += 16) {
                __m512 center = _mm512_loadu_ps(&A[idx(x, y, z, Nx, Ny)]);
                
                // Load 6 neighbors
                __m512 xp1 = _mm512_loadu_ps(&A[idx(x + 1, y, z, Nx, Ny)]);
                __m512 xm1 = _mm512_loadu_ps(&A[idx(x - 1, y, z, Nx, Ny)]);
                __m512 yp1 = _mm512_loadu_ps(&A[idx(x, y + 1, z, Nx, Ny)]);
                __m512 ym1 = _mm512_loadu_ps(&A[idx(x, y - 1, z, Nx, Ny)]);
                __m512 zp1 = _mm512_loadu_ps(&A[idx(x, y, z + 1, Nx, Ny)]);
                __m512 zm1 = _mm512_loadu_ps(&A[idx(x, y, z - 1, Nx, Ny)]);
                
                // Sum neighbors
                __m512 neighbors = _mm512_add_ps(xp1, xm1);
                neighbors = _mm512_add_ps(neighbors, yp1);
                neighbors = _mm512_add_ps(neighbors, ym1);
                neighbors = _mm512_add_ps(neighbors, zp1);
                neighbors = _mm512_add_ps(neighbors, zm1);
                
                // C = a0 * center + a1 * neighbors using FMA
                __m512 result = _mm512_fmadd_ps(a0_vec, center,
                                                _mm512_mul_ps(a1_vec, neighbors));
                
                _mm512_storeu_ps(&C[idx(x, y, z, Nx, Ny)], result);
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

// AVX-512 stencil kernel for double
void stencil_double(double* C, const double* A,
                    double a0, double a1,
                    size_t z_begin, size_t z_end,
                    size_t Nx, size_t Ny, size_t Nz)
{
    size_t z_start = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    __m512d a0_vec = _mm512_set1_pd(a0);
    __m512d a1_vec = _mm512_set1_pd(a1);
    
    for (size_t z = z_start; z < z_stop; ++z) {
        for (size_t y = 1; y < Ny - 1; ++y) {
            size_t x = 1;
            
            // Process 8 doubles at a time
            for (; x + 8 <= Nx - 1; x += 8) {
                __m512d center = _mm512_loadu_pd(&A[idx(x, y, z, Nx, Ny)]);
                
                __m512d xp1 = _mm512_loadu_pd(&A[idx(x + 1, y, z, Nx, Ny)]);
                __m512d xm1 = _mm512_loadu_pd(&A[idx(x - 1, y, z, Nx, Ny)]);
                __m512d yp1 = _mm512_loadu_pd(&A[idx(x, y + 1, z, Nx, Ny)]);
                __m512d ym1 = _mm512_loadu_pd(&A[idx(x, y - 1, z, Nx, Ny)]);
                __m512d zp1 = _mm512_loadu_pd(&A[idx(x, y, z + 1, Nx, Ny)]);
                __m512d zm1 = _mm512_loadu_pd(&A[idx(x, y, z - 1, Nx, Ny)]);
                
                __m512d neighbors = _mm512_add_pd(xp1, xm1);
                neighbors = _mm512_add_pd(neighbors, yp1);
                neighbors = _mm512_add_pd(neighbors, ym1);
                neighbors = _mm512_add_pd(neighbors, zp1);
                neighbors = _mm512_add_pd(neighbors, zm1);
                
                // C = a0 * center + a1 * neighbors using FMA
                __m512d result = _mm512_fmadd_pd(a0_vec, center,
                                                 _mm512_mul_pd(a1_vec, neighbors));
                
                _mm512_storeu_pd(&C[idx(x, y, z, Nx, Ny)], result);
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

} // namespace avx512
} // namespace kernels

#else // Non-x86 platforms - provide stub implementations

namespace kernels {
namespace avx512 {

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

} // namespace avx512
} // namespace kernels

#endif // x86 platform check
