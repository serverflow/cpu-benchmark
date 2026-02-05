// CPU Benchmark - Scalar Kernel Implementations
// Scalar fallback implementations - always available on all platforms

#include "kernel_common.hpp"

namespace kernels {
namespace scalar {

// Scalar memory kernel for float
// C[i,j,k] = alpha * A[i,j,k] + beta * B[i,j,k]
void mem_float(float* C, const float* A, const float* B,
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
void mem_double(double* C, const double* A, const double* B,
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
// 7-point stencil: C[x,y,z] = a0 * A[x,y,z] + a1 * (neighbors sum)
void stencil_float(float* C, const float* A,
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
void stencil_double(double* C, const double* A,
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

// Scalar memory kernel for int8_t
// Note: For INT8, we compute in int32 to avoid overflow, then clamp to int8 range
void mem_int8(int8_t* C, const int8_t* A, const int8_t* B,
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
void stencil_int8(int8_t* C, const int8_t* A,
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

} // namespace scalar
} // namespace kernels
