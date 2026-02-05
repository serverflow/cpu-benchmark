#pragma once
// CPU Benchmark - Extended Math Kernels for Multi-Precision Support

//
// This file provides template-based math kernels that support multiple precision types:
// - FP64 (double): native double precision
// - FP16 (half): 16-bit half precision (emulated via float32)
// - INT8 (int8_t): 8-bit integer with shift-based normalization
// - FP4: 4-bit emulated float (packed, 2 values per byte)

#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>
#include "half.hpp"
#include "fp4.hpp"
#include "math_kernels.hpp"

// OpenMP support
#ifdef USE_OPENMP
#include <omp.h>
#endif

// ============================================================================
// Memory Kernel: C = alpha * A + beta * B
// Requirements: 7.1, 7.4
// ============================================================================

// Primary template for memory kernel (works for double, float)
// Operations per element: 2 multiplications + 1 addition = 3 FLOP/OPS
template<typename T>
void kernel_mem_typed(T* C, const T* A, const T* B,
                      float alpha, float beta,
                      size_t z_begin, size_t z_end,
                      size_t Nx, size_t Ny, size_t /*Nz*/) {
    const T alpha_t = static_cast<T>(alpha);
    const T beta_t = static_cast<T>(beta);
    
    const int z_start = static_cast<int>(z_begin);
    const int z_stop = static_cast<int>(z_end);
    const int ny = static_cast<int>(Ny);
    const int nx = static_cast<int>(Nx);
    
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int z = z_start; z < z_stop; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                size_t i = idx(static_cast<size_t>(x), static_cast<size_t>(y),
                               static_cast<size_t>(z), Nx, Ny);
                C[i] = alpha_t * A[i] + beta_t * B[i];
            }
        }
    }
}


// Specialization for half type (FP16 emulated mode)
// Convert to float, compute, convert back to half
template<>
inline void kernel_mem_typed<half>(half* C, const half* A, const half* B,
                                   float alpha, float beta,
                                   size_t z_begin, size_t z_end,
                                   size_t Nx, size_t Ny, size_t /*Nz*/) {
    const int z_start = static_cast<int>(z_begin);
    const int z_stop = static_cast<int>(z_end);
    const int ny = static_cast<int>(Ny);
    const int nx = static_cast<int>(Nx);
    
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int z = z_start; z < z_stop; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                size_t i = idx(static_cast<size_t>(x), static_cast<size_t>(y),
                               static_cast<size_t>(z), Nx, Ny);
                // Convert half to float, compute, convert back
                float a_f = static_cast<float>(A[i]);
                float b_f = static_cast<float>(B[i]);
                float result = alpha * a_f + beta * b_f;
                C[i] = half(result);
            }
        }
    }
}

// Specialization for int8_t (INT8 mode)
// Integer arithmetic with shift-based normalization
// alpha and beta are scaled to fixed-point representation
template<>
inline void kernel_mem_typed<int8_t>(int8_t* C, const int8_t* A, const int8_t* B,
                                     float alpha, float beta,
                                     size_t z_begin, size_t z_end,
                                     size_t Nx, size_t Ny, size_t /*Nz*/) {
    // Scale factors to fixed-point (Q7 format: 7 fractional bits)
    const int SHIFT = 7;
    const int SCALE = 1 << SHIFT;  // 128
    
    // Convert alpha and beta to fixed-point
    int32_t alpha_fp = static_cast<int32_t>(alpha * SCALE);
    int32_t beta_fp = static_cast<int32_t>(beta * SCALE);
    
    const int z_start = static_cast<int>(z_begin);
    const int z_stop = static_cast<int>(z_end);
    const int ny = static_cast<int>(Ny);
    const int nx = static_cast<int>(Nx);
    
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int z = z_start; z < z_stop; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                size_t i = idx(static_cast<size_t>(x), static_cast<size_t>(y),
                               static_cast<size_t>(z), Nx, Ny);
                // Integer multiply-accumulate with shift normalization
                int32_t a_val = static_cast<int32_t>(A[i]);
                int32_t b_val = static_cast<int32_t>(B[i]);
                int32_t result = (alpha_fp * a_val + beta_fp * b_val) >> SHIFT;
                
                // Clamp to int8_t range [-128, 127]
                if (result > 127) result = 127;
                if (result < -128) result = -128;
                
                C[i] = static_cast<int8_t>(result);
            }
        }
    }
}

// FP4 memory kernel (separate interface due to packed storage)
// Extract FP4 to float, compute in float, pack back to FP4
inline void kernel_mem_fp4(FP4Array& C, const FP4Array& A, const FP4Array& B,
                           float alpha, float beta,
                           size_t z_begin, size_t z_end,
                           size_t Nx, size_t Ny, size_t /*Nz*/) {
    const size_t slice_size = Nx * Ny;
    const size_t start_idx = z_begin * slice_size;
    const size_t end_idx = z_end * slice_size;
    const size_t total_elements = end_idx - start_idx;
    
    constexpr size_t CHUNK_SIZE = FP4_SIMD_CHUNK_SIZE * 8;  // Process multiple SIMD widths
    
    // Temporary buffers for batch operations
    alignas(32) float a_buf[CHUNK_SIZE];
    alignas(32) float b_buf[CHUNK_SIZE];
    alignas(32) float c_buf[CHUNK_SIZE];
    
#ifdef USE_OPENMP
    #pragma omp parallel
    {
        // Thread-local buffers
        alignas(32) float a_local[CHUNK_SIZE];
        alignas(32) float b_local[CHUNK_SIZE];
        alignas(32) float c_local[CHUNK_SIZE];
        
        #pragma omp for
        for (size_t chunk_start = 0; chunk_start < total_elements; chunk_start += CHUNK_SIZE) {
            size_t chunk_count = (chunk_start + CHUNK_SIZE <= total_elements) 
                                 ? CHUNK_SIZE : (total_elements - chunk_start);
            size_t global_start = start_idx + chunk_start;
            

            A.batch_unpack_simd(a_local, global_start, chunk_count);
            B.batch_unpack_simd(b_local, global_start, chunk_count);
            
            // Compute in float
            for (size_t i = 0; i < chunk_count; ++i) {
                c_local[i] = alpha * a_local[i] + beta * b_local[i];
            }
            
            C.batch_pack_simd(c_local, global_start, chunk_count);
        }
    }
#else
    // Non-OpenMP path: process in chunks
    for (size_t chunk_start = 0; chunk_start < total_elements; chunk_start += CHUNK_SIZE) {
        size_t chunk_count = (chunk_start + CHUNK_SIZE <= total_elements) 
                             ? CHUNK_SIZE : (total_elements - chunk_start);
        size_t global_start = start_idx + chunk_start;
        
        A.batch_unpack_simd(a_buf, global_start, chunk_count);
        B.batch_unpack_simd(b_buf, global_start, chunk_count);
        
        // Compute in float
        for (size_t i = 0; i < chunk_count; ++i) {
            c_buf[i] = alpha * a_buf[i] + beta * b_buf[i];
        }

        C.batch_pack_simd(c_buf, global_start, chunk_count);
    }
#endif
}


// ============================================================================
// Stencil Kernel: 7-point stencil
// C[x,y,z] = a0 * A[x,y,z] + a1 * (sum of 6 neighbors)
// Requirements: 7.2, 7.4
// ============================================================================

// Primary template for stencil kernel (works for double, float)
// Operations per inner cell: 6 additions (neighbors) + 2 multiplications = 8 FLOP/OPS
template<typename T>
void kernel_stencil_typed(T* C, const T* A,
                          float a0, float a1,
                          size_t z_begin, size_t z_end,
                          size_t Nx, size_t Ny, size_t Nz) {
    const T a0_t = static_cast<T>(a0);
    const T a1_t = static_cast<T>(a1);
    
    // Clamp z range to valid inner cells
    size_t z_start_u = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop_u = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    const int z_start = static_cast<int>(z_start_u);
    const int z_stop = static_cast<int>(z_stop_u);
    const int ny = static_cast<int>(Ny);
    const int nx = static_cast<int>(Nx);
    
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int z = z_start; z < z_stop; ++z) {
        for (int y = 1; y < ny - 1; ++y) {
            for (int x = 1; x < nx - 1; ++x) {
                size_t sx = static_cast<size_t>(x);
                size_t sy = static_cast<size_t>(y);
                size_t sz = static_cast<size_t>(z);
                
                T center = A[idx(sx, sy, sz, Nx, Ny)];
                T neighbors = A[idx(sx + 1, sy, sz, Nx, Ny)] +
                              A[idx(sx - 1, sy, sz, Nx, Ny)] +
                              A[idx(sx, sy + 1, sz, Nx, Ny)] +
                              A[idx(sx, sy - 1, sz, Nx, Ny)] +
                              A[idx(sx, sy, sz + 1, Nx, Ny)] +
                              A[idx(sx, sy, sz - 1, Nx, Ny)];
                
                C[idx(sx, sy, sz, Nx, Ny)] = a0_t * center + a1_t * neighbors;
            }
        }
    }
}

// Specialization for half type (FP16 emulated mode)
template<>
inline void kernel_stencil_typed<half>(half* C, const half* A,
                                       float a0, float a1,
                                       size_t z_begin, size_t z_end,
                                       size_t Nx, size_t Ny, size_t Nz) {
    // Clamp z range to valid inner cells
    size_t z_start_u = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop_u = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    const int z_start = static_cast<int>(z_start_u);
    const int z_stop = static_cast<int>(z_stop_u);
    const int ny = static_cast<int>(Ny);
    const int nx = static_cast<int>(Nx);
    
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int z = z_start; z < z_stop; ++z) {
        for (int y = 1; y < ny - 1; ++y) {
            for (int x = 1; x < nx - 1; ++x) {
                size_t sx = static_cast<size_t>(x);
                size_t sy = static_cast<size_t>(y);
                size_t sz = static_cast<size_t>(z);
                
                // Convert to float for computation
                float center = static_cast<float>(A[idx(sx, sy, sz, Nx, Ny)]);
                float neighbors = static_cast<float>(A[idx(sx + 1, sy, sz, Nx, Ny)]) +
                                  static_cast<float>(A[idx(sx - 1, sy, sz, Nx, Ny)]) +
                                  static_cast<float>(A[idx(sx, sy + 1, sz, Nx, Ny)]) +
                                  static_cast<float>(A[idx(sx, sy - 1, sz, Nx, Ny)]) +
                                  static_cast<float>(A[idx(sx, sy, sz + 1, Nx, Ny)]) +
                                  static_cast<float>(A[idx(sx, sy, sz - 1, Nx, Ny)]);
                
                float result = a0 * center + a1 * neighbors;
                C[idx(sx, sy, sz, Nx, Ny)] = half(result);
            }
        }
    }
}

// Specialization for int8_t (INT8 mode)
template<>
inline void kernel_stencil_typed<int8_t>(int8_t* C, const int8_t* A,
                                         float a0, float a1,
                                         size_t z_begin, size_t z_end,
                                         size_t Nx, size_t Ny, size_t Nz) {
    // Scale factors to fixed-point (Q7 format)
    const int SHIFT = 7;
    const int SCALE = 1 << SHIFT;
    
    int32_t a0_fp = static_cast<int32_t>(a0 * SCALE);
    int32_t a1_fp = static_cast<int32_t>(a1 * SCALE);
    
    // Clamp z range to valid inner cells
    size_t z_start_u = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop_u = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    const int z_start = static_cast<int>(z_start_u);
    const int z_stop = static_cast<int>(z_stop_u);
    const int ny = static_cast<int>(Ny);
    const int nx = static_cast<int>(Nx);
    
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int z = z_start; z < z_stop; ++z) {
        for (int y = 1; y < ny - 1; ++y) {
            for (int x = 1; x < nx - 1; ++x) {
                size_t sx = static_cast<size_t>(x);
                size_t sy = static_cast<size_t>(y);
                size_t sz = static_cast<size_t>(z);
                
                int32_t center = static_cast<int32_t>(A[idx(sx, sy, sz, Nx, Ny)]);
                int32_t neighbors = static_cast<int32_t>(A[idx(sx + 1, sy, sz, Nx, Ny)]) +
                                    static_cast<int32_t>(A[idx(sx - 1, sy, sz, Nx, Ny)]) +
                                    static_cast<int32_t>(A[idx(sx, sy + 1, sz, Nx, Ny)]) +
                                    static_cast<int32_t>(A[idx(sx, sy - 1, sz, Nx, Ny)]) +
                                    static_cast<int32_t>(A[idx(sx, sy, sz + 1, Nx, Ny)]) +
                                    static_cast<int32_t>(A[idx(sx, sy, sz - 1, Nx, Ny)]);
                
                int32_t result = (a0_fp * center + a1_fp * neighbors) >> SHIFT;
                
                // Clamp to int8_t range
                if (result > 127) result = 127;
                if (result < -128) result = -128;
                
                C[idx(sx, sy, sz, Nx, Ny)] = static_cast<int8_t>(result);
            }
        }
    }
}

// FP4 stencil kernel (separate interface due to packed storage)
inline void kernel_stencil_fp4(FP4Array& C, const FP4Array& A,
                               float a0, float a1,
                               size_t z_begin, size_t z_end,
                               size_t Nx, size_t Ny, size_t Nz) {
    // Clamp z range to valid inner cells
    size_t z_start_u = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop_u = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    const int z_start = static_cast<int>(z_start_u);
    const int z_stop = static_cast<int>(z_stop_u);
    const int ny = static_cast<int>(Ny);
    const int nx = static_cast<int>(Nx);
    

    constexpr size_t CHUNK_SIZE = FP4_SIMD_CHUNK_SIZE * 4;
    
#ifdef USE_OPENMP
    #pragma omp parallel
    {
        // Thread-local buffers for row processing
        alignas(32) float row_center[CHUNK_SIZE];
        alignas(32) float row_xp1[CHUNK_SIZE];
        alignas(32) float row_xm1[CHUNK_SIZE];
        alignas(32) float row_yp1[CHUNK_SIZE];
        alignas(32) float row_ym1[CHUNK_SIZE];
        alignas(32) float row_zp1[CHUNK_SIZE];
        alignas(32) float row_zm1[CHUNK_SIZE];
        alignas(32) float row_result[CHUNK_SIZE];
        
        #pragma omp for
        for (int z = z_start; z < z_stop; ++z) {
            for (int y = 1; y < ny - 1; ++y) {
                // Process row in chunks
                for (int x_start = 1; x_start < nx - 1; x_start += static_cast<int>(CHUNK_SIZE)) {
                    int x_end = (x_start + static_cast<int>(CHUNK_SIZE) < nx - 1) 
                                ? x_start + static_cast<int>(CHUNK_SIZE) : nx - 1;
                    size_t chunk_count = static_cast<size_t>(x_end - x_start);
                    
                    // Batch extract values for this chunk
                    for (size_t i = 0; i < chunk_count; ++i) {
                        size_t sx = static_cast<size_t>(x_start) + i;
                        size_t sy = static_cast<size_t>(y);
                        size_t sz = static_cast<size_t>(z);
                        
                        row_center[i] = A.get(idx(sx, sy, sz, Nx, Ny));
                        row_xp1[i] = A.get(idx(sx + 1, sy, sz, Nx, Ny));
                        row_xm1[i] = A.get(idx(sx - 1, sy, sz, Nx, Ny));
                        row_yp1[i] = A.get(idx(sx, sy + 1, sz, Nx, Ny));
                        row_ym1[i] = A.get(idx(sx, sy - 1, sz, Nx, Ny));
                        row_zp1[i] = A.get(idx(sx, sy, sz + 1, Nx, Ny));
                        row_zm1[i] = A.get(idx(sx, sy, sz - 1, Nx, Ny));
                    }
                    
                    // Compute stencil for chunk
                    for (size_t i = 0; i < chunk_count; ++i) {
                        float neighbors = row_xp1[i] + row_xm1[i] + row_yp1[i] + 
                                          row_ym1[i] + row_zp1[i] + row_zm1[i];
                        row_result[i] = a0 * row_center[i] + a1 * neighbors;
                    }
                    
                    // Batch pack results
                    for (size_t i = 0; i < chunk_count; ++i) {
                        size_t sx = static_cast<size_t>(x_start) + i;
                        size_t sy = static_cast<size_t>(y);
                        size_t sz = static_cast<size_t>(z);
                        C.set(idx(sx, sy, sz, Nx, Ny), row_result[i]);
                    }
                }
            }
        }
    }
#else
    // Non-OpenMP path
    alignas(32) float row_center[CHUNK_SIZE];
    alignas(32) float row_xp1[CHUNK_SIZE];
    alignas(32) float row_xm1[CHUNK_SIZE];
    alignas(32) float row_yp1[CHUNK_SIZE];
    alignas(32) float row_ym1[CHUNK_SIZE];
    alignas(32) float row_zp1[CHUNK_SIZE];
    alignas(32) float row_zm1[CHUNK_SIZE];
    alignas(32) float row_result[CHUNK_SIZE];
    
    for (int z = z_start; z < z_stop; ++z) {
        for (int y = 1; y < ny - 1; ++y) {
            for (int x_start = 1; x_start < nx - 1; x_start += static_cast<int>(CHUNK_SIZE)) {
                int x_end = (x_start + static_cast<int>(CHUNK_SIZE) < nx - 1) 
                            ? x_start + static_cast<int>(CHUNK_SIZE) : nx - 1;
                size_t chunk_count = static_cast<size_t>(x_end - x_start);
                
                // Batch extract values for this chunk
                for (size_t i = 0; i < chunk_count; ++i) {
                    size_t sx = static_cast<size_t>(x_start) + i;
                    size_t sy = static_cast<size_t>(y);
                    size_t sz = static_cast<size_t>(z);
                    
                    row_center[i] = A.get(idx(sx, sy, sz, Nx, Ny));
                    row_xp1[i] = A.get(idx(sx + 1, sy, sz, Nx, Ny));
                    row_xm1[i] = A.get(idx(sx - 1, sy, sz, Nx, Ny));
                    row_yp1[i] = A.get(idx(sx, sy + 1, sz, Nx, Ny));
                    row_ym1[i] = A.get(idx(sx, sy - 1, sz, Nx, Ny));
                    row_zp1[i] = A.get(idx(sx, sy, sz + 1, Nx, Ny));
                    row_zm1[i] = A.get(idx(sx, sy, sz - 1, Nx, Ny));
                }
                
                // Compute stencil for chunk
                for (size_t i = 0; i < chunk_count; ++i) {
                    float neighbors = row_xp1[i] + row_xm1[i] + row_yp1[i] + 
                                      row_ym1[i] + row_zp1[i] + row_zm1[i];
                    row_result[i] = a0 * row_center[i] + a1 * neighbors;
                }
                
                // Batch pack results
                for (size_t i = 0; i < chunk_count; ++i) {
                    size_t sx = static_cast<size_t>(x_start) + i;
                    size_t sy = static_cast<size_t>(y);
                    size_t sz = static_cast<size_t>(z);
                    C.set(idx(sx, sy, sz, Nx, Ny), row_result[i]);
                }
            }
        }
    }
#endif
}


// ============================================================================
// Matrix Multiplication Kernel: C[z] = A[z] * B[z] for each slice
// Treats 3D array as Z slices of N×N matrices
// Requirements: 7.3, 7.4
// ============================================================================

// Primary template for matmul3d kernel (works for double, float)
// Operations per slice: N³ multiplications + N³ additions = 2 * N³ FLOP/OPS
template<typename T>
void kernel_matmul3d_typed(T* C, const T* A, const T* B,
                           size_t z_begin, size_t z_end,
                           size_t N) {
    const int z_start = static_cast<int>(z_begin);
    const int z_stop = static_cast<int>(z_end);
    const int n = static_cast<int>(N);
    
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int z = z_start; z < z_stop; ++z) {
        size_t slice_offset = static_cast<size_t>(z) * N * N;
        const T* A_slice = A + slice_offset;
        const T* B_slice = B + slice_offset;
        T* C_slice = C + slice_offset;
        
        // Classical i-j-k matrix multiplication
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                T sum = T(0);
                for (int k = 0; k < n; ++k) {
                    sum += A_slice[i * n + k] * B_slice[k * n + j];
                }
                C_slice[i * n + j] = sum;
            }
        }
    }
}

// Specialization for half type (FP16 emulated mode)
template<>
inline void kernel_matmul3d_typed<half>(half* C, const half* A, const half* B,
                                        size_t z_begin, size_t z_end,
                                        size_t N) {
    const int z_start = static_cast<int>(z_begin);
    const int z_stop = static_cast<int>(z_end);
    const int n = static_cast<int>(N);
    
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int z = z_start; z < z_stop; ++z) {
        size_t slice_offset = static_cast<size_t>(z) * N * N;
        const half* A_slice = A + slice_offset;
        const half* B_slice = B + slice_offset;
        half* C_slice = C + slice_offset;
        
        // Compute in float, store as half
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < n; ++k) {
                    float a_f = static_cast<float>(A_slice[i * n + k]);
                    float b_f = static_cast<float>(B_slice[k * n + j]);
                    sum += a_f * b_f;
                }
                C_slice[i * n + j] = half(sum);
            }
        }
    }
}

// Specialization for int8_t (INT8 mode)
template<>
inline void kernel_matmul3d_typed<int8_t>(int8_t* C, const int8_t* A, const int8_t* B,
                                          size_t z_begin, size_t z_end,
                                          size_t N) {
    const int z_start = static_cast<int>(z_begin);
    const int z_stop = static_cast<int>(z_end);
    const int n = static_cast<int>(N);
    
    // For INT8 matmul, we accumulate in int32 and then scale back
    // This mimics quantized neural network inference patterns
    const int SHIFT = 7;  // Scale factor for normalization
    
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int z = z_start; z < z_stop; ++z) {
        size_t slice_offset = static_cast<size_t>(z) * N * N;
        const int8_t* A_slice = A + slice_offset;
        const int8_t* B_slice = B + slice_offset;
        int8_t* C_slice = C + slice_offset;
        
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                int32_t sum = 0;
                for (int k = 0; k < n; ++k) {
                    sum += static_cast<int32_t>(A_slice[i * n + k]) *
                           static_cast<int32_t>(B_slice[k * n + j]);
                }
                
                // Scale down and clamp to int8_t range
                int32_t result = sum >> SHIFT;
                if (result > 127) result = 127;
                if (result < -128) result = -128;
                
                C_slice[i * n + j] = static_cast<int8_t>(result);
            }
        }
    }
}

// FP4 matmul3d kernel (separate interface due to packed storage)

inline void kernel_matmul3d_fp4(FP4Array& C, const FP4Array& A, const FP4Array& B,
                                size_t z_begin, size_t z_end,
                                size_t N) {
    const int z_start = static_cast<int>(z_begin);
    const int z_stop = static_cast<int>(z_end);
    const int n = static_cast<int>(N);

    constexpr size_t CHUNK_SIZE = FP4_SIMD_CHUNK_SIZE * 4;
    
#ifdef USE_OPENMP
    #pragma omp parallel
    {
        // Thread-local buffers for row processing
        alignas(32) float b_col[CHUNK_SIZE];
        alignas(32) float c_row[CHUNK_SIZE];
        
        #pragma omp for
        for (int z = z_start; z < z_stop; ++z) {
            size_t slice_offset = static_cast<size_t>(z) * N * N;
            
            // Compute in float, store as FP4
            for (int i = 0; i < n; ++i) {
                // Process output row in chunks
                for (int j_start = 0; j_start < n; j_start += static_cast<int>(CHUNK_SIZE)) {
                    int j_end = (j_start + static_cast<int>(CHUNK_SIZE) < n) 
                                ? j_start + static_cast<int>(CHUNK_SIZE) : n;
                    size_t chunk_count = static_cast<size_t>(j_end - j_start);
                    
                    // Initialize output chunk to zero
                    for (size_t jj = 0; jj < chunk_count; ++jj) {
                        c_row[jj] = 0.0f;
                    }
                    
                    // Accumulate dot products
                    for (int k = 0; k < n; ++k) {
                        float a_val = A.get(slice_offset + i * N + k);
                        
                        // Batch extract B column values
                        for (size_t jj = 0; jj < chunk_count; ++jj) {
                            b_col[jj] = B.get(slice_offset + k * N + (j_start + jj));
                        }
                        
                        // Accumulate
                        for (size_t jj = 0; jj < chunk_count; ++jj) {
                            c_row[jj] += a_val * b_col[jj];
                        }
                    }
                    
                    // Batch pack results
                    for (size_t jj = 0; jj < chunk_count; ++jj) {
                        C.set(slice_offset + i * N + (j_start + jj), c_row[jj]);
                    }
                }
            }
        }
    }
#else
    // Non-OpenMP path
    alignas(32) float b_col[CHUNK_SIZE];
    alignas(32) float c_row[CHUNK_SIZE];
    
    for (int z = z_start; z < z_stop; ++z) {
        size_t slice_offset = static_cast<size_t>(z) * N * N;
        
        for (int i = 0; i < n; ++i) {
            for (int j_start = 0; j_start < n; j_start += static_cast<int>(CHUNK_SIZE)) {
                int j_end = (j_start + static_cast<int>(CHUNK_SIZE) < n) 
                            ? j_start + static_cast<int>(CHUNK_SIZE) : n;
                size_t chunk_count = static_cast<size_t>(j_end - j_start);
                
                // Initialize output chunk to zero
                for (size_t jj = 0; jj < chunk_count; ++jj) {
                    c_row[jj] = 0.0f;
                }
                
                // Accumulate dot products
                for (int k = 0; k < n; ++k) {
                    float a_val = A.get(slice_offset + i * N + k);
                    
                    // Batch extract B column values
                    for (size_t jj = 0; jj < chunk_count; ++jj) {
                        b_col[jj] = B.get(slice_offset + k * N + (j_start + jj));
                    }
                    
                    // Accumulate
                    for (size_t jj = 0; jj < chunk_count; ++jj) {
                        c_row[jj] += a_val * b_col[jj];
                    }
                }
                
                // Batch pack results
                for (size_t jj = 0; jj < chunk_count; ++jj) {
                    C.set(slice_offset + i * N + (j_start + jj), c_row[jj]);
                }
            }
        }
    }
#endif
}

// ============================================================================
// Helper: Matrix3D for extended types
// ============================================================================

// Matrix3D specialization for half type
template<>
class Matrix3D<half> {
public:
    Matrix3D(size_t Nx, size_t Ny, size_t Nz)
        : Nx_(Nx), Ny_(Ny), Nz_(Nz), buffer_(Nx * Ny * Nz) {}
    
    half& at(size_t x, size_t y, size_t z) {
        return buffer_[idx(x, y, z, Nx_, Ny_)];
    }
    
    const half& at(size_t x, size_t y, size_t z) const {
        return buffer_[idx(x, y, z, Nx_, Ny_)];
    }
    
    half* data() { return buffer_.data(); }
    const half* data() const { return buffer_.data(); }
    
    size_t size() const { return buffer_.size(); }
    Size3D dimensions() const { return {Nx_, Ny_, Nz_}; }
    
    size_t Nx() const { return Nx_; }
    size_t Ny() const { return Ny_; }
    size_t Nz() const { return Nz_; }
    
    void fill_random(float min_val, float max_val) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (auto& val : buffer_) {
            val = half(dist(gen));
        }
    }
    
    void fill_random_parallel(float min_val, float max_val, unsigned num_threads) {
        if (num_threads <= 1) {
            fill_random(min_val, max_val);
            return;
        }
        
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        
        size_t total = buffer_.size();
        size_t chunk_size = total / num_threads;
        size_t remainder = total % num_threads;
        
        size_t start = 0;
        for (unsigned t = 0; t < num_threads; ++t) {
            size_t this_chunk = chunk_size + (t < remainder ? 1 : 0);
            size_t end = start + this_chunk;
            
            threads.emplace_back([this, start, end, min_val, max_val, t]() {
                std::mt19937 gen(static_cast<unsigned>(t * 12345 + 67890));
                std::uniform_real_distribution<float> dist(min_val, max_val);
                for (size_t i = start; i < end; ++i) {
                    buffer_[i] = half(dist(gen));
                }
            });
            
            start = end;
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    void fill_zero() {
        for (auto& val : buffer_) {
            val = half(0.0f);
        }
    }
    
    void fill_zero_parallel(unsigned num_threads) {
        if (num_threads <= 1) {
            fill_zero();
            return;
        }
        
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        
        size_t total = buffer_.size();
        size_t chunk_size = total / num_threads;
        size_t remainder = total % num_threads;
        
        size_t start = 0;
        for (unsigned t = 0; t < num_threads; ++t) {
            size_t this_chunk = chunk_size + (t < remainder ? 1 : 0);
            size_t end = start + this_chunk;
            
            threads.emplace_back([this, start, end]() {
                for (size_t i = start; i < end; ++i) {
                    buffer_[i] = half(0.0f);
                }
            });
            
            start = end;
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
private:
    size_t Nx_, Ny_, Nz_;
    std::vector<half> buffer_;
};

// Matrix3D specialization for int8_t type
template<>
class Matrix3D<int8_t> {
public:
    Matrix3D(size_t Nx, size_t Ny, size_t Nz)
        : Nx_(Nx), Ny_(Ny), Nz_(Nz), buffer_(Nx * Ny * Nz) {}
    
    int8_t& at(size_t x, size_t y, size_t z) {
        return buffer_[idx(x, y, z, Nx_, Ny_)];
    }
    
    const int8_t& at(size_t x, size_t y, size_t z) const {
        return buffer_[idx(x, y, z, Nx_, Ny_)];
    }
    
    int8_t* data() { return buffer_.data(); }
    const int8_t* data() const { return buffer_.data(); }
    
    size_t size() const { return buffer_.size(); }
    Size3D dimensions() const { return {Nx_, Ny_, Nz_}; }
    
    size_t Nx() const { return Nx_; }
    size_t Ny() const { return Ny_; }
    size_t Nz() const { return Nz_; }
    
    void fill_random(int8_t min_val, int8_t max_val) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(min_val, max_val);
        for (auto& val : buffer_) {
            val = static_cast<int8_t>(dist(gen));
        }
    }
    
    void fill_random_parallel(int8_t min_val, int8_t max_val, unsigned num_threads) {
        if (num_threads <= 1) {
            fill_random(min_val, max_val);
            return;
        }
        
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        
        size_t total = buffer_.size();
        size_t chunk_size = total / num_threads;
        size_t remainder = total % num_threads;
        
        size_t start = 0;
        for (unsigned t = 0; t < num_threads; ++t) {
            size_t this_chunk = chunk_size + (t < remainder ? 1 : 0);
            size_t end = start + this_chunk;
            
            threads.emplace_back([this, start, end, min_val, max_val, t]() {
                std::mt19937 gen(static_cast<unsigned>(t * 12345 + 67890));
                std::uniform_int_distribution<int> dist(min_val, max_val);
                for (size_t i = start; i < end; ++i) {
                    buffer_[i] = static_cast<int8_t>(dist(gen));
                }
            });
            
            start = end;
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    void fill_zero() {
        std::fill(buffer_.begin(), buffer_.end(), int8_t(0));
    }
    
    void fill_zero_parallel(unsigned num_threads) {
        if (num_threads <= 1) {
            fill_zero();
            return;
        }
        
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        
        size_t total = buffer_.size();
        size_t chunk_size = total / num_threads;
        size_t remainder = total % num_threads;
        
        size_t start = 0;
        for (unsigned t = 0; t < num_threads; ++t) {
            size_t this_chunk = chunk_size + (t < remainder ? 1 : 0);
            size_t end = start + this_chunk;
            
            threads.emplace_back([this, start, end]() {
                for (size_t i = start; i < end; ++i) {
                    buffer_[i] = int8_t(0);
                }
            });
            
            start = end;
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
private:
    size_t Nx_, Ny_, Nz_;
    std::vector<int8_t> buffer_;
};

// ============================================================================
// FP4 Matrix3D wrapper
// ============================================================================

class Matrix3D_FP4 {
public:
    Matrix3D_FP4(size_t Nx, size_t Ny, size_t Nz)
        : Nx_(Nx), Ny_(Ny), Nz_(Nz), array_(Nx * Ny * Nz) {}
    
    float at(size_t x, size_t y, size_t z) const {
        return array_.get(idx(x, y, z, Nx_, Ny_));
    }
    
    void set(size_t x, size_t y, size_t z, float value) {
        array_.set(idx(x, y, z, Nx_, Ny_), value);
    }
    
    FP4Array& data() { return array_; }
    const FP4Array& data() const { return array_; }
    
    size_t size() const { return array_.size(); }
    size_t byte_size() const { return array_.byte_size(); }
    Size3D dimensions() const { return {Nx_, Ny_, Nz_}; }
    
    size_t Nx() const { return Nx_; }
    size_t Ny() const { return Ny_; }
    size_t Nz() const { return Nz_; }
    
    void fill_random() {
        array_.fill_random();
    }
    
    void fill_zero() {
        array_.fill_zero();
    }
    
private:
    size_t Nx_, Ny_, Nz_;
    FP4Array array_;
};
