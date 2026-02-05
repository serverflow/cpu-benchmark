#pragma once
// CPU Benchmark - Common Kernel Definitions
// This header provides common types and the idx() function for all kernel implementations

#include <cstddef>
#include <cstdint>

// Index calculation for 3D array
// Computes linear index as (z * Ny * Nx) + (y * Nx) + x
// Guard to prevent redefinition if math_kernels.hpp is also included
#ifndef IDX_FUNCTION_DEFINED
#define IDX_FUNCTION_DEFINED
inline size_t idx(size_t x, size_t y, size_t z, size_t Nx, size_t Ny) {
    return (z * Ny * Nx) + (y * Nx) + x;
}
#endif

// Function pointer types for kernels
template<typename T>
using MemKernelFn = void(*)(T*, const T*, const T*, T, T, size_t, size_t, size_t, size_t, size_t);

template<typename T>
using StencilKernelFn = void(*)(T*, const T*, T, T, size_t, size_t, size_t, size_t, size_t);

// Forward declarations for kernel functions from each SIMD level
// These are implemented in separate .cpp files compiled with different flags

namespace kernels {

// Scalar kernels (always available)
namespace scalar {
    void mem_float(float* C, const float* A, const float* B,
                   float alpha, float beta,
                   size_t z_begin, size_t z_end,
                   size_t Nx, size_t Ny, size_t Nz);
    
    void mem_double(double* C, const double* A, const double* B,
                    double alpha, double beta,
                    size_t z_begin, size_t z_end,
                    size_t Nx, size_t Ny, size_t Nz);
    
    void stencil_float(float* C, const float* A,
                       float a0, float a1,
                       size_t z_begin, size_t z_end,
                       size_t Nx, size_t Ny, size_t Nz);
    
    void stencil_double(double* C, const double* A,
                        double a0, double a1,
                        size_t z_begin, size_t z_end,
                        size_t Nx, size_t Ny, size_t Nz);
    
    void mem_int8(int8_t* C, const int8_t* A, const int8_t* B,
                  int8_t alpha, int8_t beta,
                  size_t z_begin, size_t z_end,
                  size_t Nx, size_t Ny, size_t Nz);
    
    void stencil_int8(int8_t* C, const int8_t* A,
                      int8_t a0, int8_t a1,
                      size_t z_begin, size_t z_end,
                      size_t Nx, size_t Ny, size_t Nz);
}

// SSE2 kernels (x86-64 baseline)
namespace sse2 {
    void mem_float(float* C, const float* A, const float* B,
                   float alpha, float beta,
                   size_t z_begin, size_t z_end,
                   size_t Nx, size_t Ny, size_t Nz);
    
    void mem_double(double* C, const double* A, const double* B,
                    double alpha, double beta,
                    size_t z_begin, size_t z_end,
                    size_t Nx, size_t Ny, size_t Nz);
    
    void stencil_float(float* C, const float* A,
                       float a0, float a1,
                       size_t z_begin, size_t z_end,
                       size_t Nx, size_t Ny, size_t Nz);
    
    void stencil_double(double* C, const double* A,
                        double a0, double a1,
                        size_t z_begin, size_t z_end,
                        size_t Nx, size_t Ny, size_t Nz);
    
    void mem_int8(int8_t* C, const int8_t* A, const int8_t* B,
                  int8_t alpha, int8_t beta,
                  size_t z_begin, size_t z_end,
                  size_t Nx, size_t Ny, size_t Nz);
}

// AVX kernels
namespace avx {
    void mem_float(float* C, const float* A, const float* B,
                   float alpha, float beta,
                   size_t z_begin, size_t z_end,
                   size_t Nx, size_t Ny, size_t Nz);
    
    void mem_double(double* C, const double* A, const double* B,
                    double alpha, double beta,
                    size_t z_begin, size_t z_end,
                    size_t Nx, size_t Ny, size_t Nz);
    
    void stencil_float(float* C, const float* A,
                       float a0, float a1,
                       size_t z_begin, size_t z_end,
                       size_t Nx, size_t Ny, size_t Nz);
    
    void stencil_double(double* C, const double* A,
                        double a0, double a1,
                        size_t z_begin, size_t z_end,
                        size_t Nx, size_t Ny, size_t Nz);
}

// AVX2 kernels (with FMA)
namespace avx2 {
    void mem_float(float* C, const float* A, const float* B,
                   float alpha, float beta,
                   size_t z_begin, size_t z_end,
                   size_t Nx, size_t Ny, size_t Nz);
    
    void mem_double(double* C, const double* A, const double* B,
                    double alpha, double beta,
                    size_t z_begin, size_t z_end,
                    size_t Nx, size_t Ny, size_t Nz);
    
    void stencil_float(float* C, const float* A,
                       float a0, float a1,
                       size_t z_begin, size_t z_end,
                       size_t Nx, size_t Ny, size_t Nz);
    
    void stencil_double(double* C, const double* A,
                        double a0, double a1,
                        size_t z_begin, size_t z_end,
                        size_t Nx, size_t Ny, size_t Nz);
    
    void mem_int8(int8_t* C, const int8_t* A, const int8_t* B,
                  int8_t alpha, int8_t beta,
                  size_t z_begin, size_t z_end,
                  size_t Nx, size_t Ny, size_t Nz);
}

// AVX-512 kernels
namespace avx512 {
    void mem_float(float* C, const float* A, const float* B,
                   float alpha, float beta,
                   size_t z_begin, size_t z_end,
                   size_t Nx, size_t Ny, size_t Nz);
    
    void mem_double(double* C, const double* A, const double* B,
                    double alpha, double beta,
                    size_t z_begin, size_t z_end,
                    size_t Nx, size_t Ny, size_t Nz);
    
    void stencil_float(float* C, const float* A,
                       float a0, float a1,
                       size_t z_begin, size_t z_end,
                       size_t Nx, size_t Ny, size_t Nz);
    
    void stencil_double(double* C, const double* A,
                        double a0, double a1,
                        size_t z_begin, size_t z_end,
                        size_t Nx, size_t Ny, size_t Nz);
}

// ARM NEON kernels
namespace neon {
    void mem_float(float* C, const float* A, const float* B,
                   float alpha, float beta,
                   size_t z_begin, size_t z_end,
                   size_t Nx, size_t Ny, size_t Nz);
    
    void stencil_float(float* C, const float* A,
                       float a0, float a1,
                       size_t z_begin, size_t z_end,
                       size_t Nx, size_t Ny, size_t Nz);
}

} // namespace kernels
