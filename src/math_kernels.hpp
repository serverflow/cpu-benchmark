#pragma once
// CPU Benchmark - Math kernels header


#include <cstddef>
#include <vector>
#include <random>
#include <algorithm>
#include <thread>
#include "types.hpp"
#include "platform.hpp"

// OpenMP support (Requirements 10.1-10.4)
#ifdef USE_OPENMP
#include <omp.h>
#endif

// is_openmp_enabled() is defined in platform.hpp


// Computes linear index as (z * Ny * Nx) + (y * Nx) + x
// Guard to prevent redefinition if kernel_common.hpp is also included
#ifndef IDX_FUNCTION_DEFINED
#define IDX_FUNCTION_DEFINED
inline size_t idx(size_t x, size_t y, size_t z, size_t Nx, size_t Ny) {
    return (z * Ny * Nx) + (y * Nx) + x;
}
#endif

// Matrix3D class (Requirements 1.1, 1.3)
// Stores 3D matrix data in a contiguous linear buffer using std::vector<T>
template<typename T>
class Matrix3D {
public:
    Matrix3D(size_t Nx, size_t Ny, size_t Nz)
        : Nx_(Nx), Ny_(Ny), Nz_(Nz), buffer_(Nx * Ny * Nz) {}
    
    // Element access using idx() formula
    T& at(size_t x, size_t y, size_t z) {
        return buffer_[idx(x, y, z, Nx_, Ny_)];
    }
    
    const T& at(size_t x, size_t y, size_t z) const {
        return buffer_[idx(x, y, z, Nx_, Ny_)];
    }
    
    // Raw data access
    T* data() { return buffer_.data(); }
    const T* data() const { return buffer_.data(); }
    
    // Size information
    size_t size() const { return buffer_.size(); }
    Size3D dimensions() const { return {Nx_, Ny_, Nz_}; }
    
    size_t Nx() const { return Nx_; }
    size_t Ny() const { return Ny_; }
    size_t Nz() const { return Nz_; }
    
    // Fill with random values (single-threaded, for backward compatibility)
    void fill_random(T min_val, T max_val) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(min_val, max_val);
        for (auto& val : buffer_) {
            val = dist(gen);
        }
    }
    
    // Fill with random values using parallel initialization (NUMA-aware)
    // Each thread initializes its own portion for first-touch NUMA placement
    void fill_random_parallel(T min_val, T max_val, unsigned num_threads) {
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
                // Each thread uses its own RNG seeded with thread ID
                std::mt19937 gen(static_cast<unsigned>(t * 12345 + 67890));
                std::uniform_real_distribution<T> dist(min_val, max_val);
                for (size_t i = start; i < end; ++i) {
                    buffer_[i] = dist(gen);
                }
            });
            
            start = end;
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    // Fill with zeros
    void fill_zero() {
        std::fill(buffer_.begin(), buffer_.end(), T(0));
    }
    
    // Fill with zeros using parallel initialization (NUMA-aware)
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
                    buffer_[i] = T(0);
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
    std::vector<T> buffer_;
};

// Memory kernel (Requirements 2.1, 2.4, 10.3)
// C[i,j,k] = alpha * A[i,j,k] + beta * B[i,j,k]
// z_begin and z_end define the range for multi-threading
template<typename T>
void kernel_mem(T* C, const T* A, const T* B,
                T alpha, T beta,
                size_t z_begin, size_t z_end,
                size_t Nx, size_t Ny, size_t /*Nz*/) {
    // Use signed integers for OpenMP compatibility with MSVC
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
                C[i] = alpha * A[i] + beta * B[i];
            }
        }
    }
}

// Stencil kernel (Requirements 3.1, 3.2, 3.3, 10.3)
// 7-point stencil: C[x,y,z] = a0 * A[x,y,z] + a1 * (neighbors sum)
// Skips boundary cells to avoid out-of-bounds access
// z_begin and z_end define the range for multi-threading
template<typename T>
void kernel_stencil(T* C, const T* A,
                    T a0, T a1,
                    size_t z_begin, size_t z_end,
                    size_t Nx, size_t Ny, size_t Nz) {
    // Clamp z range to valid inner cells
    size_t z_start_u = (z_begin < 1) ? 1 : z_begin;
    size_t z_stop_u = (z_end > Nz - 1) ? Nz - 1 : z_end;
    
    // Use signed integers for OpenMP compatibility with MSVC
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
                C[idx(sx, sy, sz, Nx, Ny)] = a0 * center + a1 * neighbors;
            }
        }
    }
}

// Matrix multiplication kernel (Requirements 4.1, 4.2, 4.4, 10.3)
// Treats 3D array as Z slices of N×N matrices
// Computes C[z] = A[z] * B[z] for each slice using classical i-j-k loop
// z_begin and z_end define the range for multi-threading
template<typename T>
void kernel_matmul3d(T* C, const T* A, const T* B,
                     size_t z_begin, size_t z_end,
                     size_t N) {
    // For matmul3d, we treat the 3D array as Z slices of N×N matrices
    // Each slice is stored contiguously: slice z starts at offset z * N * N
    
    // Use signed integers for OpenMP compatibility with MSVC
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
                    // A[i,k] * B[k,j] where matrix is stored row-major
                    sum += A_slice[i * n + k] * B_slice[k * n + j];
                }
                C_slice[i * n + j] = sum;
            }
        }
    }
}

// FLOP counting functions (Requirements 2.2, 3.4, 4.3)

// Memory kernel: C[i,j,k] = alpha * A[i,j,k] + beta * B[i,j,k]
// Operations per element: 2 multiplications + 1 addition = 3 FLOP
// (Design document specifies 3 * Nx * Ny * Nz)
inline size_t flops_mem(size_t Nx, size_t Ny, size_t Nz) {
    return 3 * Nx * Ny * Nz;
}

// Stencil kernel: 7-point stencil
// Operations per inner cell: 6 additions (neighbors) + 2 multiplications = 8 FLOP
// Only inner cells are computed: (Nx-2) * (Ny-2) * (Nz-2)
inline size_t flops_stencil(size_t Nx, size_t Ny, size_t Nz) {
    if (Nx < 3 || Ny < 3 || Nz < 3) return 0;
    return 8 * (Nx - 2) * (Ny - 2) * (Nz - 2);
}

// Matrix multiplication kernel: C[z] = A[z] * B[z] for Z slices
// Operations per slice: N³ multiplications + N³ additions = 2 * N³
// Total: 2 * N³ * Z
inline size_t flops_matmul3d(size_t N, size_t Z) {
    return 2 * N * N * N * Z;
}

// Compute kernel: Pure FMA operations (Requirements 15.1, 15.4)
// FLOPS are returned by the kernel itself since they depend on SIMD level
// This function is a placeholder for consistency
inline size_t flops_compute(size_t iterations) {
    // Actual FLOPS depend on SIMD level and are returned by the kernel
    // This is a minimum estimate based on scalar implementation
    return iterations * 16;  // 8 FMAs * 2 FLOPs each
}
