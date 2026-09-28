// CPU Benchmark - RISC-V Vector Extension (RVV 1.0) kernels
// Compiled as a separate TU with -march=rv64gcv so the rest of the binary
// stays at baseline; selected at runtime via CpuCapabilities::has_riscv_vector.
// Uses LMUL=1 (m1) types, so it adapts to any VLEN at runtime.

#if !defined(__riscv) || !defined(__riscv_v_intrinsic)
#error "kernel_rvv.cpp must be compiled with RVV support (e.g. -march=rv64gcv)"
#endif

#include <cstddef>
#include <riscv_vector.h>

#include "math_kernels.hpp"

// ============================================================================
// Memory kernel (fused AXPY): C[i] = alpha * A[i] + beta * B[i]
// ============================================================================

void kernel_mem_rvv_float(
    float* C, const float* A, const float* B,
    float alpha, float beta,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t /*Nz*/)
{
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            const size_t base = idx(0, y, z, Nx, Ny);
            size_t i = 0;
            while (i < Nx) {
                size_t vl = __riscv_vsetvl_e32m1(Nx - i);
                vfloat32m1_t va = __riscv_vle32_v_f32m1(A + base + i, vl);
                vfloat32m1_t vb = __riscv_vle32_v_f32m1(B + base + i, vl);
                vfloat32m1_t vc = __riscv_vfmul_vf_f32m1(va, alpha, vl);
                vc = __riscv_vfmacc_vf_f32m1(vc, beta, vb, vl);
                __riscv_vse32_v_f32m1(C + base + i, vc, vl);
                i += vl;
            }
        }
    }
}

void kernel_mem_rvv_double(
    double* C, const double* A, const double* B,
    double alpha, double beta,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t /*Nz*/)
{
    for (size_t z = z_begin; z < z_end; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            const size_t base = idx(0, y, z, Nx, Ny);
            size_t i = 0;
            while (i < Nx) {
                size_t vl = __riscv_vsetvl_e64m1(Nx - i);
                vfloat64m1_t va = __riscv_vle64_v_f64m1(A + base + i, vl);
                vfloat64m1_t vb = __riscv_vle64_v_f64m1(B + base + i, vl);
                vfloat64m1_t vc = __riscv_vfmul_vf_f64m1(va, alpha, vl);
                vc = __riscv_vfmacc_vf_f64m1(vc, beta, vb, vl);
                __riscv_vse64_v_f64m1(C + base + i, vc, vl);
                i += vl;
            }
        }
    }
}

// ============================================================================
// 7-point stencil: C = a0 * center + a1 * (6 neighbors)
// ============================================================================

void kernel_stencil_rvv_float(
    float* C, const float* A,
    float a0, float a1,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t Nz)
{
    size_t z_start = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop = (z_end > Nz - 1) ? Nz - 1 : z_end;
    const size_t count = (Nx >= 2) ? Nx - 2 : 0;
    const size_t stride_y = Nx;
    const size_t stride_z = Nx * Ny;

    for (size_t z = z_start; z < z_stop; ++z) {
        for (size_t y = 1; y + 1 < Ny; ++y) {
            size_t i = 0;
            while (i < count) {
                size_t vl = __riscv_vsetvl_e32m1(count - i);
                const size_t c = idx(1 + i, y, z, Nx, Ny);
                vfloat32m1_t vsum =
                    __riscv_vfadd_vv_f32m1(__riscv_vle32_v_f32m1(A + c - 1, vl),
                                           __riscv_vle32_v_f32m1(A + c + 1, vl), vl);
                vsum = __riscv_vfadd_vv_f32m1(vsum,
                        __riscv_vle32_v_f32m1(A + c - stride_y, vl), vl);
                vsum = __riscv_vfadd_vv_f32m1(vsum,
                        __riscv_vle32_v_f32m1(A + c + stride_y, vl), vl);
                vsum = __riscv_vfadd_vv_f32m1(vsum,
                        __riscv_vle32_v_f32m1(A + c - stride_z, vl), vl);
                vsum = __riscv_vfadd_vv_f32m1(vsum,
                        __riscv_vle32_v_f32m1(A + c + stride_z, vl), vl);
                vfloat32m1_t vc =
                    __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(A + c, vl), a0, vl);
                vfloat32m1_t vres = __riscv_vfmacc_vf_f32m1(vc, a1, vsum, vl);
                __riscv_vse32_v_f32m1(C + c, vres, vl);
                i += vl;
            }
        }
    }
}

void kernel_stencil_rvv_double(
    double* C, const double* A,
    double a0, double a1,
    size_t z_begin, size_t z_end,
    size_t Nx, size_t Ny, size_t Nz)
{
    size_t z_start = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop = (z_end > Nz - 1) ? Nz - 1 : z_end;
    const size_t count = (Nx >= 2) ? Nx - 2 : 0;
    const size_t stride_y = Nx;
    const size_t stride_z = Nx * Ny;

    for (size_t z = z_start; z < z_stop; ++z) {
        for (size_t y = 1; y + 1 < Ny; ++y) {
            size_t i = 0;
            while (i < count) {
                size_t vl = __riscv_vsetvl_e64m1(count - i);
                const size_t c = idx(1 + i, y, z, Nx, Ny);
                vfloat64m1_t vsum =
                    __riscv_vfadd_vv_f64m1(__riscv_vle64_v_f64m1(A + c - 1, vl),
                                           __riscv_vle64_v_f64m1(A + c + 1, vl), vl);
                vsum = __riscv_vfadd_vv_f64m1(vsum,
                        __riscv_vle64_v_f64m1(A + c - stride_y, vl), vl);
                vsum = __riscv_vfadd_vv_f64m1(vsum,
                        __riscv_vle64_v_f64m1(A + c + stride_y, vl), vl);
                vsum = __riscv_vfadd_vv_f64m1(vsum,
                        __riscv_vle64_v_f64m1(A + c - stride_z, vl), vl);
                vsum = __riscv_vfadd_vv_f64m1(vsum,
                        __riscv_vle64_v_f64m1(A + c + stride_z, vl), vl);
                vfloat64m1_t vc =
                    __riscv_vfmul_vf_f64m1(__riscv_vle64_v_f64m1(A + c, vl), a0, vl);
                vfloat64m1_t vres = __riscv_vfmacc_vf_f64m1(vc, a1, vsum, vl);
                __riscv_vse64_v_f64m1(C + c, vres, vl);
                i += vl;
            }
        }
    }
}
