#pragma once
// CPU Benchmark - Precision Dispatcher

#include "types.hpp"
#include "platform.hpp"
#include "cli.hpp"
#include "benchmark_core.hpp"
#include "half.hpp"
#include "fp4.hpp"
#include "math_kernels_extended.hpp"
#include "cpu_capabilities.hpp"
#include "comparison_output.hpp"
#include "threading.hpp"
#include "simd_kernels.hpp"
#include "score_calculator.hpp"
#include "runtime_dispatcher.hpp"

#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <cstdlib>
#include <utility>
#include <atomic>
#include <thread>

#if defined(__AVX2__) || defined(_M_AVX2)
    #include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
#endif

namespace {
inline bool precision_debug_enabled() {
    const char* env = std::getenv("SFBENCH_DEBUG");
    return env && env[0] != '\0' && env[0] != '0';
}

inline void precision_debug_log(const std::string& msg) {
    if (precision_debug_enabled()) {
        std::cerr << msg << "\n";
    }
}

inline bool should_use_persistent_pool() {
    const char* env = std::getenv("SFBENCH_NO_PERSISTENT_POOL");
    if (env && env[0] != '\0' && env[0] != '0') {
        return false;
    }
#ifdef _WIN32
    unsigned logical = get_logical_core_count();
    unsigned sockets = get_socket_count();
    if (logical > 64 || sockets > 1) {
        return false;
    }
#endif
    return true;
}
} // namespace

// Precision-thread pool wrapper: avoid PersistentThreadPool on large Windows systems
class PrecisionThreadPool {
public:
    PrecisionThreadPool(unsigned num_threads, const std::vector<unsigned>& core_ids)
        : use_persistent_(should_use_persistent_pool())
    {
        if (use_persistent_) {
            persistent_ = std::make_unique<PersistentThreadPool>(num_threads, core_ids);
        } else {
            simple_ = std::make_unique<ThreadPool>(num_threads, core_ids);
            precision_debug_log("[precision] using ThreadPool (persistent disabled)");
        }
    }

    unsigned thread_count() const {
        return use_persistent_ ? persistent_->thread_count() : simple_->thread_count();
    }

    template<typename Func>
    void parallel_for_z(size_t Nz, Func&& func) {
        if (use_persistent_) {
            persistent_->parallel_for_z(Nz, std::forward<Func>(func));
        } else {
            simple_->parallel_for_z(Nz, std::forward<Func>(func));
        }
    }

private:
    bool use_persistent_;
    std::unique_ptr<PersistentThreadPool> persistent_;
    std::unique_ptr<ThreadPool> simple_;
};

// ============================================================================
// Helper function to create ThreadPool with optional socket affinity
// ============================================================================
inline ThreadPool create_simple_thread_pool_for_config(const Config& config) {
    auto thread_config = get_thread_config_for_config(config);
    return ThreadPool(thread_config.first, thread_config.second);
}

// ============================================================================
// Native FP16 Benchmark Class (for ARM NEON FP16)
// Uses float16_t directly with NEON FP16 intrinsics for true native FP16 performance
// Note: AVX-512 FP16 support can be added later when kernel functions are available
// ============================================================================

// Compile-time check for native FP16 support
// This macro indicates whether the NativeFP16Benchmark class is compiled
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #define HAS_NATIVE_FP16_BENCHMARK 1
    // Verify that float16_t is available and has correct size
    static_assert(sizeof(float16_t) == 2, "float16_t must be 2 bytes for native FP16");
#elif defined(__AVX512FP16__)
    #define HAS_NATIVE_FP16_BENCHMARK 1
#else
    #define HAS_NATIVE_FP16_BENCHMARK 0
#endif

// ============================================================================
// FP16 Path Diagnostics
// These functions help diagnose which FP16 path is actually being used
// ============================================================================

// Returns true if native FP16 benchmark code was compiled into this binary
inline bool is_native_fp16_compiled() {
#if HAS_NATIVE_FP16_BENCHMARK
    return true;
#else
    return false;
#endif
}

// Returns a diagnostic string describing the FP16 execution path
inline std::string get_fp16_diagnostic_info() {
    std::ostringstream oss;
    const auto& caps = CpuCapabilities::get();
    
    oss << "FP16 Diagnostic Info:\n";
    oss << "  Compile-time:\n";
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    oss << "    __ARM_FEATURE_FP16_VECTOR_ARITHMETIC: DEFINED\n";
#else
    oss << "    __ARM_FEATURE_FP16_VECTOR_ARITHMETIC: NOT DEFINED\n";
#endif
#if defined(__AVX512FP16__)
    oss << "    __AVX512FP16__: DEFINED\n";
#else
    oss << "    __AVX512FP16__: NOT DEFINED\n";
#endif
    oss << "    HAS_NATIVE_FP16_BENCHMARK: " << HAS_NATIVE_FP16_BENCHMARK << "\n";
    
    oss << "  Runtime:\n";
    oss << "    CPU has ARM NEON FP16: " << (caps.has_arm_neon_fp16 ? "YES" : "NO") << "\n";
    oss << "    CPU has AVX-512 FP16: " << (caps.has_avx512_fp16 ? "YES" : "NO") << "\n";
    oss << "    is_native_fp16_available(): " << (is_native_fp16_available() ? "YES" : "NO") << "\n";
    oss << "    is_native_fp16_compiled(): " << (is_native_fp16_compiled() ? "YES" : "NO") << "\n";
    
    // Determine actual execution path
    bool will_use_native = is_native_fp16_compiled() && is_native_fp16_available();
    oss << "  Actual FP16 path: " << (will_use_native ? "NATIVE (SIMD)" : "EMULATED (half.hpp)") << "\n";
    
    if (caps.has_arm_neon_fp16 && !is_native_fp16_compiled()) {
        oss << "  WARNING: CPU supports native FP16 but binary was not compiled with FP16 support!\n";
        oss << "           Rebuild with -march=armv8.2-a+fp16+simd for optimal performance.\n";
    }
    
    return oss.str();
}

// Determines the actual FP16 mode that will be used (not just what CPU supports)
inline FP16Mode get_actual_fp16_mode() {
    // Native mode requires BOTH:
    // 1. Code compiled with FP16 support (HAS_NATIVE_FP16_BENCHMARK)
    // 2. CPU runtime support (is_native_fp16_available)
    if (is_native_fp16_compiled() && is_native_fp16_available()) {
        return FP16Mode::Native;
    }
    return FP16Mode::Emulated;
}

#if HAS_NATIVE_FP16_BENCHMARK

// Native FP16 3D Matrix class
class Matrix3D_NativeFP16 {
public:
    Matrix3D_NativeFP16(size_t nx, size_t ny, size_t nz)
        : nx_(nx), ny_(ny), nz_(nz), data_(nx * ny * nz) {}
    
    void fill_random() {
        for (size_t i = 0; i < data_.size(); ++i) {
            float val = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
            data_[i] = static_cast<float16_t>(val);
        }
    }
    
    void fill_zero() {
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] = static_cast<float16_t>(0.0f);
        }
    }
    
    float16_t* data() { return data_.data(); }
    const float16_t* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }
    
private:
    size_t nx_, ny_, nz_;
    std::vector<float16_t> data_;
};

class NativeFP16Benchmark {
public:
    NativeFP16Benchmark(const Config& config, const CpuInfo& cpu_info)
        : config_(config)
        , cpu_info_(cpu_info)
        , thread_pool_(create_simple_thread_pool_for_config(config))
        , A_(config.size.Nx, config.size.Ny, config.size.Nz)
        , B_(config.size.Nx, config.size.Ny, config.size.Nz)
        , C_(config.size.Nx, config.size.Ny, config.size.Nz)
        , progress_callback_(nullptr)
    {
        A_.fill_random();
        B_.fill_random();
        C_.fill_zero();
        
        alpha_ = static_cast<float16_t>(1.5f);
        beta_ = static_cast<float16_t>(0.5f);
        a0_ = static_cast<float16_t>(0.5f);
        a1_ = static_cast<float16_t>(1.0f / 6.0f);
    }
    
    void set_progress_callback(BenchmarkProgressCallback callback) {
        progress_callback_ = callback;
    }
    
    BenchmarkResult run() {
        warmup();
        measure();
        compute_statistics();
        return result_;
    }
    
    const BenchmarkResult& result() const { return result_; }
    
    unsigned thread_count() const { return thread_pool_.thread_count(); }
    
private:
    void warmup() {
        const int warmup_iterations = 3;
        for (int i = 0; i < warmup_iterations; ++i) {
            execute_kernel();
        }
    }

    void measure() {
        result_.times_sec.clear();
        result_.times_sec.reserve(config_.repeats);
        
        size_t iterations_per_repeat = 1;
        {
            auto start = std::chrono::steady_clock::now();
            execute_kernel();
            auto end = std::chrono::steady_clock::now();
            double single_time = std::chrono::duration<double>(end - start).count();
            
            if (single_time > 0 && single_time < config_.min_time) {
                iterations_per_repeat = static_cast<size_t>(
                    std::ceil(config_.min_time / single_time));
            }
        }
        
        result_.iterations = iterations_per_repeat;
        
        auto measure_start = std::chrono::steady_clock::now();
        auto last_progress_update = measure_start;
        constexpr double MIN_PROGRESS_INTERVAL_SEC = 2.0;
        
        for (unsigned r = 0; r < config_.repeats; ++r) {
            auto start = std::chrono::steady_clock::now();
            for (size_t i = 0; i < iterations_per_repeat; ++i) {
                execute_kernel();
            }
            auto end = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            result_.times_sec.push_back(elapsed / static_cast<double>(iterations_per_repeat));
            
            if (progress_callback_) {
                auto now = std::chrono::steady_clock::now();
                double since_last_update = std::chrono::duration<double>(now - last_progress_update).count();
                
                if (since_last_update >= MIN_PROGRESS_INTERVAL_SEC || r == config_.repeats - 1) {
                    last_progress_update = now;
                    double percentage = (static_cast<double>(r + 1) / static_cast<double>(config_.repeats)) * 100.0;
                    double total_elapsed = std::chrono::duration<double>(now - measure_start).count();
                    double eta = 0.0;
                    if (r + 1 < config_.repeats && r > 0) {
                        double rate = static_cast<double>(r + 1) / total_elapsed;
                        size_t remaining = config_.repeats - (r + 1);
                        eta = static_cast<double>(remaining) / rate;
                    }
                    progress_callback_(r + 1, config_.repeats, percentage, eta);
                }
            }
        }
    }
    
    void compute_statistics() {
        result_.time_avg_sec = compute_average(result_.times_sec);
        result_.time_min_sec = compute_minimum(result_.times_sec);
        result_.time_stddev_sec = compute_stddev(result_.times_sec);
        
        size_t Nx = config_.size.Nx;
        size_t Ny = config_.size.Ny;
        size_t Nz = config_.size.Nz;
        
        switch (config_.mode) {
            case BenchmarkMode::Mem:
                result_.total_flops = flops_mem(Nx, Ny, Nz);
                break;
            case BenchmarkMode::Stencil:
                result_.total_flops = flops_stencil(Nx, Ny, Nz);
                break;
            case BenchmarkMode::Matmul3D:
                result_.total_flops = flops_matmul3d(Nx, Nz);
                break;
            case BenchmarkMode::Compute:
            case BenchmarkMode::CacheLevel:
                result_.total_flops = flops_mem(Nx, Ny, Nz);
                break;
        }
        
        result_.gflops_avg = compute_gflops(result_.total_flops, result_.time_avg_sec);
        result_.gflops_max = compute_gflops(result_.total_flops, result_.time_min_sec);
        
        if (config_.mode == BenchmarkMode::Stencil) {
            size_t inner_cells = (Nx > 2 && Ny > 2 && Nz > 2) 
                ? (Nx - 2) * (Ny - 2) * (Nz - 2) : 0;
            result_.mlups_avg = (result_.time_avg_sec > 0) 
                ? static_cast<double>(inner_cells) / result_.time_avg_sec / 1e6 : 0.0;
        }
        
        if (config_.mode == BenchmarkMode::Mem) {
            // FP16: 2 bytes per element, 3 arrays
            size_t bytes = 3 * Nx * Ny * Nz * sizeof(float16_t);
            result_.bandwidth_gbs = (result_.time_avg_sec > 0)
                ? static_cast<double>(bytes) / result_.time_avg_sec / 1e9 : 0.0;
        }
    }
    
    void execute_kernel() {
        float16_t* C_data = C_.data();
        const float16_t* A_data = A_.data();
        const float16_t* B_data = B_.data();
        size_t Nx = config_.size.Nx;
        size_t Ny = config_.size.Ny;
        size_t Nz = config_.size.Nz;
        
        switch (config_.mode) {
            case BenchmarkMode::Mem:
            case BenchmarkMode::CacheLevel:
                thread_pool_.parallel_for_z(Nz, [&, this](size_t z_begin, size_t z_end) {
                    kernel_mem_neon_fp16(C_data, A_data, B_data, alpha_, beta_,
                                         z_begin, z_end, Nx, Ny, Nz);
                });
                break;
                
            case BenchmarkMode::Stencil:
                thread_pool_.parallel_for_z(Nz, [&, this](size_t z_begin, size_t z_end) {
                    kernel_stencil_neon_fp16(C_data, A_data, a0_, a1_,
                                             z_begin, z_end, Nx, Ny, Nz);
                });
                break;
                
            case BenchmarkMode::Matmul3D:
                thread_pool_.parallel_for_z(Nz, [&, this](size_t z_begin, size_t z_end) {
                    kernel_matmul3d_neon_fp16(C_data, A_data, B_data,
                                              z_begin, z_end, Nx);
                });
                break;
            
            case BenchmarkMode::Compute:
                // Compute mode uses mem kernel for FP16
                thread_pool_.parallel_for_z(Nz, [&, this](size_t z_begin, size_t z_end) {
                    kernel_mem_neon_fp16(C_data, A_data, B_data, alpha_, beta_,
                                         z_begin, z_end, Nx, Ny, Nz);
                });
                break;
        }
    }
    
    Config config_;
    CpuInfo cpu_info_;
    ThreadPool thread_pool_;
    Matrix3D_NativeFP16 A_, B_, C_;
    float16_t alpha_, beta_, a0_, a1_;
    BenchmarkResult result_;
    BenchmarkProgressCallback progress_callback_;
};

#endif // HAS_NATIVE_FP16_BENCHMARK

// ============================================================================
// FP4 Benchmark Class (specialized for packed FP4 storage)
// ============================================================================

class FP4Benchmark {
public:
    FP4Benchmark(const Config& config, const CpuInfo& cpu_info)
        : config_(config)
        , cpu_info_(cpu_info)
        , thread_pool_(create_simple_thread_pool_for_config(config))
        , A_(config.size.Nx, config.size.Ny, config.size.Nz)
        , B_(config.size.Nx, config.size.Ny, config.size.Nz)
        , C_(config.size.Nx, config.size.Ny, config.size.Nz)
        , progress_callback_(nullptr)
    {
        // Initialize matrices with random values
        A_.fill_random();
        B_.fill_random();
        C_.fill_zero();
        
        // Set kernel coefficients
        alpha_ = 1.5f;
        beta_ = 0.5f;
        a0_ = 0.5f;
        a1_ = 1.0f / 6.0f;
    }
    
    // Set progress callback 
    void set_progress_callback(BenchmarkProgressCallback callback) {
        progress_callback_ = callback;
    }
    
    BenchmarkResult run() {
        warmup();
        measure();
        compute_statistics();
        return result_;
    }
    
    const BenchmarkResult& result() const { return result_; }
    
    unsigned thread_count() const { return thread_pool_.thread_count(); }
    
private:
    void warmup() {
        const int warmup_iterations = 3;
        for (int i = 0; i < warmup_iterations; ++i) {
            execute_kernel();
        }
    }

    void measure() {
        result_.times_sec.clear();
        result_.times_sec.reserve(config_.repeats);
        
        // Determine iterations per repeat
        size_t iterations_per_repeat = 1;
        {
            auto start = std::chrono::steady_clock::now();
            execute_kernel();
            auto end = std::chrono::steady_clock::now();
            double single_time = std::chrono::duration<double>(end - start).count();
            
            if (single_time > 0 && single_time < config_.min_time) {
                iterations_per_repeat = static_cast<size_t>(
                    std::ceil(config_.min_time / single_time));
            }
        }
        
        result_.iterations = iterations_per_repeat;
        
        // Track progress timing 
        auto measure_start = std::chrono::steady_clock::now();
        auto last_progress_update = measure_start;
        constexpr double MIN_PROGRESS_INTERVAL_SEC = 2.0;
        
        for (unsigned r = 0; r < config_.repeats; ++r) {
            auto start = std::chrono::steady_clock::now();
            for (size_t i = 0; i < iterations_per_repeat; ++i) {
                execute_kernel();
            }
            auto end = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            result_.times_sec.push_back(elapsed / static_cast<double>(iterations_per_repeat));
            
            // Update progress callback 
            if (progress_callback_) {
                auto now = std::chrono::steady_clock::now();
                double since_last_update = std::chrono::duration<double>(now - last_progress_update).count();
                
                if (since_last_update >= MIN_PROGRESS_INTERVAL_SEC || r == config_.repeats - 1) {
                    last_progress_update = now;
                    
                    // Calculate percentage and ETA
                    double percentage = (static_cast<double>(r + 1) / static_cast<double>(config_.repeats)) * 100.0;
                    double total_elapsed = std::chrono::duration<double>(now - measure_start).count();
                    double eta = 0.0;
                    if (r + 1 < config_.repeats && r > 0) {
                        double rate = static_cast<double>(r + 1) / total_elapsed;
                        size_t remaining = config_.repeats - (r + 1);
                        eta = static_cast<double>(remaining) / rate;
                    }
                    
                    progress_callback_(r + 1, config_.repeats, percentage, eta);
                }
            }
        }
    }
    
    void compute_statistics() {
        result_.time_avg_sec = compute_average(result_.times_sec);
        result_.time_min_sec = compute_minimum(result_.times_sec);
        result_.time_stddev_sec = compute_stddev(result_.times_sec);
        
        size_t Nx = config_.size.Nx;
        size_t Ny = config_.size.Ny;
        size_t Nz = config_.size.Nz;
        
        switch (config_.mode) {
            case BenchmarkMode::Mem:
                result_.total_flops = flops_mem(Nx, Ny, Nz);
                break;
            case BenchmarkMode::Stencil:
                result_.total_flops = flops_stencil(Nx, Ny, Nz);
                break;
            case BenchmarkMode::Matmul3D:
                result_.total_flops = flops_matmul3d(Nx, Nz);
                break;
            case BenchmarkMode::Compute:
            case BenchmarkMode::CacheLevel:
                // These modes are handled separately
                result_.total_flops = flops_mem(Nx, Ny, Nz);
                break;
        }
        
        result_.gflops_avg = compute_gflops(result_.total_flops, result_.time_avg_sec);
        result_.gflops_max = compute_gflops(result_.total_flops, result_.time_min_sec);
        
        if (config_.mode == BenchmarkMode::Stencil) {
            size_t inner_cells = (Nx > 2 && Ny > 2 && Nz > 2) 
                ? (Nx - 2) * (Ny - 2) * (Nz - 2) : 0;
            result_.mlups_avg = (result_.time_avg_sec > 0) 
                ? static_cast<double>(inner_cells) / result_.time_avg_sec / 1e6 : 0.0;
        }
        
        if (config_.mode == BenchmarkMode::Mem) {
            // FP4: 0.5 bytes per element, 3 arrays
            size_t bytes = 3 * ((Nx * Ny * Nz + 1) / 2);  // ceil(N/2) bytes per array
            result_.bandwidth_gbs = (result_.time_avg_sec > 0)
                ? static_cast<double>(bytes) / result_.time_avg_sec / 1e9 : 0.0;
        }
    }
    
    void execute_kernel() {
        FP4Array& C_data = C_.data();
        const FP4Array& A_data = A_.data();
        const FP4Array& B_data = B_.data();
        size_t Nx = config_.size.Nx;
        size_t Ny = config_.size.Ny;
        size_t Nz = config_.size.Nz;
        
        switch (config_.mode) {
            case BenchmarkMode::Mem:
            case BenchmarkMode::CacheLevel:
                thread_pool_.parallel_for_z(Nz, [&, this](size_t z_begin, size_t z_end) {
                    kernel_mem_fp4(C_data, A_data, B_data, alpha_, beta_,
                                   z_begin, z_end, Nx, Ny, Nz);
                });
                break;
                
            case BenchmarkMode::Stencil:
                thread_pool_.parallel_for_z(Nz, [&, this](size_t z_begin, size_t z_end) {
                    kernel_stencil_fp4(C_data, A_data, a0_, a1_,
                                       z_begin, z_end, Nx, Ny, Nz);
                });
                break;
                
            case BenchmarkMode::Matmul3D:
                thread_pool_.parallel_for_z(Nz, [&, this](size_t z_begin, size_t z_end) {
                    kernel_matmul3d_fp4(C_data, A_data, B_data,
                                        z_begin, z_end, Nx);
                });
                break;
            
            case BenchmarkMode::Compute:
                // Compute mode not supported for FP4, fall back to mem
                thread_pool_.parallel_for_z(Nz, [&, this](size_t z_begin, size_t z_end) {
                    kernel_mem_fp4(C_data, A_data, B_data, alpha_, beta_,
                                   z_begin, z_end, Nx, Ny, Nz);
                });
                break;
        }
    }
    
    Config config_;
    CpuInfo cpu_info_;
    ThreadPool thread_pool_;
    Matrix3D_FP4 A_, B_, C_;
    float alpha_, beta_, a0_, a1_;
    BenchmarkResult result_;
    BenchmarkProgressCallback progress_callback_;
};


// ============================================================================
// Compute-Bound Precision Benchmark
// ============================================================================
// This benchmark measures pure computational performance for different data types.
// Design principles:
// 1. Data fits in L2 cache (~256KB) to eliminate memory bandwidth bottleneck
// 2. Results accumulated in registers (no store bottleneck)
// 3. Multiple iterations over same warmed data
// 4. Volatile sink prevents compiler optimization
// 5. Shows true computational differences between precision types

// Configuration for compute-bound precision test
struct ComputeBoundConfig {
    static constexpr size_t L2_TARGET_SIZE = 256 * 1024;  // 256KB target (fits in L2)
    static constexpr double MIN_TARGET_SECONDS = 0.2;
    static constexpr double MAX_TARGET_SECONDS = 1.0;
    static constexpr double MIN_WARMUP_SECONDS = 0.1;
    static constexpr double MAX_WARMUP_SECONDS = 0.5;
    static constexpr double CHUNK_TARGET_SECONDS = 0.002;
    static constexpr size_t MAX_CHUNK_ITERS = 1 << 20;
};

// Volatile sink to prevent compiler from optimizing away results
template<typename T>
inline void volatile_sink(T value) {
    volatile T sink = value;
    (void)sink;
}

// Compute-bound kernel for FP64 (double)
// FMA: result = a * b + c, accumulate in register
inline double compute_bound_fp64(const double* __restrict A, 
                                  const double* __restrict B,
                                  size_t count, size_t iterations) {
    double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
    
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i + 3 < count; i += 4) {
            acc0 += A[i]     * B[i]     + 0.5;
            acc1 += A[i + 1] * B[i + 1] + 0.5;
            acc2 += A[i + 2] * B[i + 2] + 0.5;
            acc3 += A[i + 3] * B[i + 3] + 0.5;
        }
    }
    return acc0 + acc1 + acc2 + acc3;
}

// Compute-bound kernel for FP32 (float)
inline float compute_bound_fp32(const float* __restrict A,
                                 const float* __restrict B,
                                 size_t count, size_t iterations) {
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i + 7 < count; i += 8) {
            acc0 += A[i]     * B[i]     + 0.5f;
            acc1 += A[i + 1] * B[i + 1] + 0.5f;
            acc2 += A[i + 2] * B[i + 2] + 0.5f;
            acc3 += A[i + 3] * B[i + 3] + 0.5f;
            acc4 += A[i + 4] * B[i + 4] + 0.5f;
            acc5 += A[i + 5] * B[i + 5] + 0.5f;
            acc6 += A[i + 6] * B[i + 6] + 0.5f;
            acc7 += A[i + 7] * B[i + 7] + 0.5f;
        }
    }
    return acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
}

// Compute-bound kernel for FP16 (emulated via half)
// Load as half, convert to float, compute, accumulate
inline float compute_bound_fp16_emulated(const half* __restrict A,
                                          const half* __restrict B,
                                          size_t count, size_t iterations) {
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i + 3 < count; i += 4) {
            // Load and convert from half to float (implicit conversion)
            float a0 = static_cast<float>(A[i]);
            float a1 = static_cast<float>(A[i + 1]);
            float a2 = static_cast<float>(A[i + 2]);
            float a3 = static_cast<float>(A[i + 3]);
            float b0 = static_cast<float>(B[i]);
            float b1 = static_cast<float>(B[i + 1]);
            float b2 = static_cast<float>(B[i + 2]);
            float b3 = static_cast<float>(B[i + 3]);
            
            // Compute as float
            acc0 += a0 * b0 + 0.5f;
            acc1 += a1 * b1 + 0.5f;
            acc2 += a2 * b2 + 0.5f;
            acc3 += a3 * b3 + 0.5f;
        }
    }
    return acc0 + acc1 + acc2 + acc3;
}

// Compute-bound kernel for INT8
// Integer multiply-add operations (int8 * int8, int32 accumulation)
inline int64_t compute_bound_int8(const int8_t* __restrict A,
                                   const int8_t* __restrict B,
                                   size_t count, size_t iterations) {
    int64_t total = 0;
    constexpr size_t kBlockSize = 65536;  // Keep int32 sums within range

    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t base = 0; base < count; base += kBlockSize) {
            size_t end = base + kBlockSize;
            if (end > count) end = count;

            int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            int32_t acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;

            for (size_t i = base; i + 7 < end; i += 8) {
                acc0 += static_cast<int32_t>(A[i])     * static_cast<int32_t>(B[i]);
                acc1 += static_cast<int32_t>(A[i + 1]) * static_cast<int32_t>(B[i + 1]);
                acc2 += static_cast<int32_t>(A[i + 2]) * static_cast<int32_t>(B[i + 2]);
                acc3 += static_cast<int32_t>(A[i + 3]) * static_cast<int32_t>(B[i + 3]);
                acc4 += static_cast<int32_t>(A[i + 4]) * static_cast<int32_t>(B[i + 4]);
                acc5 += static_cast<int32_t>(A[i + 5]) * static_cast<int32_t>(B[i + 5]);
                acc6 += static_cast<int32_t>(A[i + 6]) * static_cast<int32_t>(B[i + 6]);
                acc7 += static_cast<int32_t>(A[i + 7]) * static_cast<int32_t>(B[i + 7]);
            }

            total += static_cast<int64_t>(acc0) + acc1 + acc2 + acc3 +
                     acc4 + acc5 + acc6 + acc7;
        }
    }
    return total;
}

#if defined(__AVX2__) || defined(_M_AVX2)
inline int64_t compute_bound_int8_avx2(const int8_t* __restrict A,
                                       const int8_t* __restrict B,
                                       size_t count, size_t iterations) {
    int64_t total = 0;
    constexpr size_t kBlockSize = 65536;  // Keep int32 sums within range

    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t base = 0; base < count; base += kBlockSize) {
            size_t end = base + kBlockSize;
            if (end > count) end = count;

            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();

            size_t i = base;
            for (; i + 31 < end; i += 32) {
                __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(A + i));
                __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(B + i));
                __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(A + i + 16));
                __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(B + i + 16));

                __m256i a16_0 = _mm256_cvtepi8_epi16(a0);
                __m256i b16_0 = _mm256_cvtepi8_epi16(b0);
                __m256i a16_1 = _mm256_cvtepi8_epi16(a1);
                __m256i b16_1 = _mm256_cvtepi8_epi16(b1);

                __m256i prod0 = _mm256_madd_epi16(a16_0, b16_0);
                __m256i prod1 = _mm256_madd_epi16(a16_1, b16_1);

                acc0 = _mm256_add_epi32(acc0, prod0);
                acc1 = _mm256_add_epi32(acc1, prod1);
            }

            __m256i sum = _mm256_add_epi32(acc0, acc1);
            alignas(32) int32_t tmp[8];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), sum);

            int64_t block = 0;
            for (int lane = 0; lane < 8; ++lane) {
                block += tmp[lane];
            }
            for (; i < end; ++i) {
                block += static_cast<int32_t>(A[i]) * static_cast<int32_t>(B[i]);
            }

            total += block;
        }
    }
    return total;
}
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
inline int64_t compute_bound_int8_neon(const int8_t* __restrict A,
                                       const int8_t* __restrict B,
                                       size_t count, size_t iterations) {
    int64_t total = 0;
    constexpr size_t kBlockSize = 65536;  // Keep int32 sums within range

    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t base = 0; base < count; base += kBlockSize) {
            size_t end = base + kBlockSize;
            if (end > count) end = count;

            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);

            size_t i = base;
            for (; i + 15 < end; i += 16) {
                int8x16_t a = vld1q_s8(A + i);
                int8x16_t b = vld1q_s8(B + i);

                int16x8_t prod_lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
                int16x8_t prod_hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));

                acc0 = vpadalq_s16(acc0, prod_lo);
                acc1 = vpadalq_s16(acc1, prod_hi);
            }

            int32x4_t sum = vaddq_s32(acc0, acc1);
            int64x2_t sum64 = vpaddlq_s32(sum);
            int64_t block = vgetq_lane_s64(sum64, 0) + vgetq_lane_s64(sum64, 1);

            for (; i < end; ++i) {
                block += static_cast<int32_t>(A[i]) * static_cast<int32_t>(B[i]);
            }

            total += block;
        }
    }
    return total;
}
#endif

// Compute-bound kernel for FP4 (emulated)
// Unpack from FP4, convert to float, compute
inline float compute_bound_fp4(const FP4Array& A,
                                const FP4Array& B,
                                size_t count, size_t iterations) {
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i + 3 < count; i += 4) {
            // Unpack FP4 to float (significant overhead)
            float a0 = A.get(i);
            float a1 = A.get(i + 1);
            float a2 = A.get(i + 2);
            float a3 = A.get(i + 3);
            float b0 = B.get(i);
            float b1 = B.get(i + 1);
            float b2 = B.get(i + 2);
            float b3 = B.get(i + 3);
            
            acc0 += a0 * b0 + 0.5f;
            acc1 += a1 * b1 + 0.5f;
            acc2 += a2 * b2 + 0.5f;
            acc3 += a3 * b3 + 0.5f;
        }
    }
    return acc0 + acc1 + acc2 + acc3;
}

// Compute-bound precision benchmark class
class ComputeBoundPrecisionBenchmark {
public:
    ComputeBoundPrecisionBenchmark(const Config& config, const CpuInfo& cpu_info)
        : config_(config)
        , cpu_info_(cpu_info)
        , thread_config_(get_thread_config_for_config(config))
        , thread_pool_(thread_config_.first, thread_config_.second)
    {}
    
    // Run benchmark for a specific precision type
    PrecisionResult run(Precision precision) {
        FP16Mode fp16_mode = get_actual_fp16_mode();
        BenchmarkResult result;
        
        switch (precision) {
            case Precision::FP64:
            case Precision::Double:
                result = run_fp64();
                break;
            case Precision::Float:
                result = run_fp32();
                break;
            case Precision::FP16:
                result = run_fp16(fp16_mode);
                break;
            case Precision::INT8:
                result = run_int8();
                break;
            case Precision::FP4:
                result = run_fp4();
                break;
        }
        
        return make_precision_result(precision, fp16_mode, result, BenchmarkMode::Compute);
    }
    
    unsigned thread_count() const { return thread_pool_.thread_count(); }
    
private:
    // Calculate element count to fit in L2 cache
    template<typename T>
    size_t get_element_count() const {
        // Two arrays (A and B), target L2 size
        size_t bytes_per_array = ComputeBoundConfig::L2_TARGET_SIZE / 2;
        return bytes_per_array / sizeof(T);
    }
    
    double get_target_seconds() const {
        double target = config_.min_time / 10.0;
        if (target < ComputeBoundConfig::MIN_TARGET_SECONDS) {
            target = ComputeBoundConfig::MIN_TARGET_SECONDS;
        }
        if (target > ComputeBoundConfig::MAX_TARGET_SECONDS) {
            target = ComputeBoundConfig::MAX_TARGET_SECONDS;
        }
        return target;
    }

    double get_warmup_seconds() const {
        double warmup = config_.min_time / 20.0;
        if (warmup < ComputeBoundConfig::MIN_WARMUP_SECONDS) {
            warmup = ComputeBoundConfig::MIN_WARMUP_SECONDS;
        }
        if (warmup > ComputeBoundConfig::MAX_WARMUP_SECONDS) {
            warmup = ComputeBoundConfig::MAX_WARMUP_SECONDS;
        }
        return warmup;
    }

    template<typename RunFn>
    size_t calibrate_chunk_iters(RunFn run) const {
        size_t iters = 1;
        const double target = ComputeBoundConfig::CHUNK_TARGET_SECONDS;

        for (int attempt = 0; attempt < 6; ++attempt) {
            auto start = std::chrono::steady_clock::now();
            run(iters);
            auto end = std::chrono::steady_clock::now();

            double elapsed = std::chrono::duration<double>(end - start).count();
            if (elapsed <= 0.0) {
                iters = (std::min)(iters * 2, ComputeBoundConfig::MAX_CHUNK_ITERS);
                continue;
            }

            double scale = target / elapsed;
            size_t scaled = static_cast<size_t>(static_cast<double>(iters) * scale);
            if (scaled < 1) {
                scaled = 1;
            }
            if (scaled > ComputeBoundConfig::MAX_CHUNK_ITERS) {
                scaled = ComputeBoundConfig::MAX_CHUNK_ITERS;
            }

            iters = scaled;

            if (elapsed >= target * 0.5 && elapsed <= target * 2.0) {
                break;
            }
        }

        if (iters == 0) {
            iters = 1;
        }
        return iters;
    }

    template<typename WorkerFn>
    void run_parallel_for_duration(WorkerFn worker, size_t chunk_iters, double seconds) {
        unsigned num_threads = thread_pool_.thread_count();
        if (num_threads == 0) {
            return;
        }

        std::atomic<unsigned> ready{0};
        std::atomic<bool> start{false};
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point end_time;

        thread_pool_.parallel_for_z(num_threads, [&](size_t t_begin, size_t t_end) {
            for (size_t t = t_begin; t < t_end; ++t) {
                ready.fetch_add(1, std::memory_order_acq_rel);
                if (t == 0) {
                    while (ready.load(std::memory_order_acquire) < num_threads) {
                        std::this_thread::yield();
                    }
                    start_time = std::chrono::steady_clock::now();
                    end_time = start_time + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(seconds));
                    start.store(true, std::memory_order_release);
                } else {
                    while (!start.load(std::memory_order_acquire)) {
                        std::this_thread::yield();
                    }
                }

                auto local_end = end_time;
                while (std::chrono::steady_clock::now() < local_end) {
                    worker(t, chunk_iters);
                }
            }
        });
    }

    template<typename WorkerFn, typename FinalizeFn>
    BenchmarkResult measure_timed_kernel(WorkerFn worker,
                                         FinalizeFn finalize,
                                         size_t ops_per_iter,
                                         size_t chunk_iters,
                                         double seconds) {
        BenchmarkResult result;
        result.times_sec.reserve(config_.repeats);

        unsigned num_threads = thread_pool_.thread_count();
        if (num_threads == 0 || ops_per_iter == 0 || seconds <= 0.0) {
            return result;
        }

        std::vector<double> gflops_samples;
        std::vector<double> total_ops_samples;
        std::vector<double> iter_samples;
        gflops_samples.reserve(config_.repeats);
        total_ops_samples.reserve(config_.repeats);
        iter_samples.reserve(config_.repeats);

        for (unsigned r = 0; r < config_.repeats; ++r) {
            std::atomic<unsigned> ready{0};
            std::atomic<bool> start{false};
            std::chrono::steady_clock::time_point start_time;
            std::chrono::steady_clock::time_point end_time;
            std::vector<double> thread_times(num_threads, 0.0);
            std::vector<size_t> thread_iters(num_threads, 0);

            thread_pool_.parallel_for_z(num_threads, [&](size_t t_begin, size_t t_end) {
                for (size_t t = t_begin; t < t_end; ++t) {
                    ready.fetch_add(1, std::memory_order_acq_rel);
                    if (t == 0) {
                        while (ready.load(std::memory_order_acquire) < num_threads) {
                            std::this_thread::yield();
                        }
                        start_time = std::chrono::steady_clock::now();
                        end_time = start_time + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                            std::chrono::duration<double>(seconds));
                        start.store(true, std::memory_order_release);
                    } else {
                        while (!start.load(std::memory_order_acquire)) {
                            std::this_thread::yield();
                        }
                    }

                    size_t iters = 0;
                    auto local_end = end_time;
                    while (std::chrono::steady_clock::now() < local_end) {
                        worker(t, chunk_iters);
                        iters += chunk_iters;
                    }

                    auto stop_time = std::chrono::steady_clock::now();
                    finalize(t);
                    thread_times[t] = std::chrono::duration<double>(stop_time - start_time).count();
                    thread_iters[t] = iters;
                }
            });

            double max_time = *std::max_element(thread_times.begin(), thread_times.end());
            size_t total_iters = std::accumulate(thread_iters.begin(), thread_iters.end(), size_t{0});
            double total_ops = static_cast<double>(total_iters) * static_cast<double>(ops_per_iter);
            double gflops = (max_time > 0.0) ? (total_ops / max_time / 1e9) : 0.0;

            result.times_sec.push_back(max_time);
            gflops_samples.push_back(gflops);
            total_ops_samples.push_back(total_ops);
            iter_samples.push_back(static_cast<double>(total_iters) / static_cast<double>(num_threads));
        }

        result.time_avg_sec = compute_average(result.times_sec);
        result.time_min_sec = compute_minimum(result.times_sec);
        result.time_stddev_sec = compute_stddev(result.times_sec);

        result.gflops_avg = compute_average(gflops_samples);
        result.gflops_max = gflops_samples.empty() ? 0.0 : *std::max_element(gflops_samples.begin(), gflops_samples.end());
        result.total_flops = static_cast<size_t>(compute_average(total_ops_samples));
        result.iterations = static_cast<size_t>(compute_average(iter_samples));

        return result;
    }
    
    // FP64 benchmark
    BenchmarkResult run_fp64() {
        ComputeKernelDoubleFn kernel = RuntimeDispatcher::get_compute_kernel_double();
        double tmp = 0.0;
        size_t flops_per_iter = kernel(&tmp, 1);
        volatile_sink(tmp);
        if (flops_per_iter == 0) {
            flops_per_iter = 1;
        }

        size_t chunk_iters = calibrate_chunk_iters([&](size_t iters) {
            size_t flops = kernel(&tmp, iters);
            (void)flops;
            volatile_sink(tmp);
        });

        unsigned num_threads = thread_pool_.thread_count();
        std::vector<double> acc(num_threads, 0.0);

        auto worker = [&](size_t t, size_t iters) {
            size_t flops = kernel(&acc[t], iters);
            (void)flops;
        };
        auto finalize = [&](size_t t) {
            volatile_sink(acc[t]);
        };

        run_parallel_for_duration(worker, chunk_iters, get_warmup_seconds());
        return measure_timed_kernel(worker, finalize, flops_per_iter, chunk_iters, get_target_seconds());
    }
    
    // FP32 benchmark
    BenchmarkResult run_fp32() {
        ComputeKernelFloatFn kernel = RuntimeDispatcher::get_compute_kernel_float();
        float tmp = 0.0f;
        size_t flops_per_iter = kernel(&tmp, 1);
        volatile_sink(tmp);
        if (flops_per_iter == 0) {
            flops_per_iter = 1;
        }

        size_t chunk_iters = calibrate_chunk_iters([&](size_t iters) {
            size_t flops = kernel(&tmp, iters);
            (void)flops;
            volatile_sink(tmp);
        });

        unsigned num_threads = thread_pool_.thread_count();
        std::vector<float> acc(num_threads, 0.0f);

        auto worker = [&](size_t t, size_t iters) {
            size_t flops = kernel(&acc[t], iters);
            (void)flops;
        };
        auto finalize = [&](size_t t) {
            volatile_sink(acc[t]);
        };

        run_parallel_for_duration(worker, chunk_iters, get_warmup_seconds());
        return measure_timed_kernel(worker, finalize, flops_per_iter, chunk_iters, get_target_seconds());
    }
    
    // FP16 benchmark (emulated)
    BenchmarkResult run_fp16(FP16Mode fp16_mode) {
        (void)fp16_mode;
        size_t count = get_element_count<half>();
        if (count == 0) {
            count = 1;
        }
        std::vector<half> A(count), B(count);
        
        for (size_t i = 0; i < count; ++i) {
            float val_a = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
            float val_b = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
            A[i] = half(val_a);
            B[i] = half(val_b);
        }

        auto run_chunk = [&](size_t iters) {
            return compute_bound_fp16_emulated(A.data(), B.data(), count, iters);
        };

        size_t chunk_iters = calibrate_chunk_iters([&](size_t iters) {
            float res = run_chunk(iters);
            volatile_sink(res);
        });

        unsigned num_threads = thread_pool_.thread_count();
        std::vector<float> acc(num_threads, 0.0f);

        auto worker = [&](size_t t, size_t iters) {
            acc[t] = run_chunk(iters);
        };
        auto finalize = [&](size_t t) {
            volatile_sink(acc[t]);
        };

        run_parallel_for_duration(worker, chunk_iters, get_warmup_seconds());

        size_t ops_per_iter = count * 2;
        return measure_timed_kernel(worker, finalize, ops_per_iter, chunk_iters, get_target_seconds());
    }
    
    // INT8 benchmark
    BenchmarkResult run_int8() {
        size_t count = get_element_count<int8_t>();
        if (count == 0) count = 1;

        std::vector<int8_t> A(count), B(count);
        for (size_t i = 0; i < count; ++i) {
            A[i] = static_cast<int8_t>((rand() % 256) - 128);
            B[i] = static_cast<int8_t>((rand() % 256) - 128);
        }

        using Int8KernelFn = int64_t(*)(const int8_t*, const int8_t*, size_t, size_t);
        Int8KernelFn kernel = compute_bound_int8;
        SimdLevel level = RuntimeDispatcher::get_active_level();

#if defined(__AVX2__) || defined(_M_AVX2)
        if (level == SimdLevel::AVX2 || level == SimdLevel::AVX512) {
            kernel = compute_bound_int8_avx2;
        }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        if (level == SimdLevel::NEON || level == SimdLevel::NEON_FP16) {
            kernel = compute_bound_int8_neon;
        }
#endif

        size_t chunk_iters = calibrate_chunk_iters([&](size_t iters) {
            int64_t res = kernel(A.data(), B.data(), count, iters);
            volatile_sink(res);
        });

        unsigned num_threads = thread_pool_.thread_count();
        std::vector<int64_t> acc(num_threads, 0);

        auto worker = [&](size_t t, size_t iters) {
            acc[t] = kernel(A.data(), B.data(), count, iters);
        };
        auto finalize = [&](size_t t) {
            volatile_sink(acc[t]);
        };

        run_parallel_for_duration(worker, chunk_iters, get_warmup_seconds());

        size_t ops_per_iter = count * 2;
        return measure_timed_kernel(worker, finalize, ops_per_iter, chunk_iters, get_target_seconds());
    }
    
    // FP4 benchmark
    BenchmarkResult run_fp4() {
        // FP4: 0.5 bytes per element, so we can fit 2x more elements
        size_t count = ComputeBoundConfig::L2_TARGET_SIZE;  // 2 values per byte, 2 arrays
        FP4Array A(count), B(count);
        
        for (size_t i = 0; i < count; ++i) {
            float val_a = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
            float val_b = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
            A.set(i, val_a);
            B.set(i, val_b);
        }

        auto run_chunk = [&](size_t iters) {
            return compute_bound_fp4(A, B, count, iters);
        };

        size_t chunk_iters = calibrate_chunk_iters([&](size_t iters) {
            float res = run_chunk(iters);
            volatile_sink(res);
        });

        unsigned num_threads = thread_pool_.thread_count();
        std::vector<float> acc(num_threads, 0.0f);

        auto worker = [&](size_t t, size_t iters) {
            acc[t] = run_chunk(iters);
        };
        auto finalize = [&](size_t t) {
            volatile_sink(acc[t]);
        };

        run_parallel_for_duration(worker, chunk_iters, get_warmup_seconds());

        size_t ops_per_iter = count * 2;
        return measure_timed_kernel(worker, finalize, ops_per_iter, chunk_iters, get_target_seconds());
    }
    
    Config config_;
    CpuInfo cpu_info_;
    std::pair<unsigned, std::vector<unsigned>> thread_config_;
    PrecisionThreadPool thread_pool_;
};

// ============================================================================
// Precision Dispatcher Functions
// ============================================================================

// Run benchmark for a specific precision type and return result
// Returns PrecisionResult with benchmark data
// Uses compute-bound approach: data in L2 cache, accumulate in registers
inline PrecisionResult run_benchmark_for_precision(
    Precision precision,
    const Config& config,
    const CpuInfo& cpu_info
) {
    ComputeBoundPrecisionBenchmark benchmark(config, cpu_info);
    return benchmark.run(precision);
}

// Run benchmarks for all precision types and collect results
// Uses compute-bound approach for fair comparison of computational performance
inline ComparisonTable run_all_precision_benchmarks(
    const Config& base_config,
    const CpuInfo& cpu_info,
    bool verbose = true
) {
    ComparisonTable table;

    RuntimeDispatcher::initialize();
    if (verbose) {
        std::cout << "Precision SIMD level: "
                  << RuntimeDispatcher::get_active_level_name()
                  << "\n";
    }
    
    // Get list of all precision types to test
    std::vector<Precision> precisions = get_all_precisions();
    
    // Use actual FP16 mode for display (considers both compile-time and runtime)
    FP16Mode actual_fp16_mode = get_actual_fp16_mode();
    
    // Create single benchmark instance for all precision types
    // This ensures consistent thread pool and configuration
    ComputeBoundPrecisionBenchmark benchmark(base_config, cpu_info);
    
    for (const auto& precision : precisions) {
        if (verbose) {
            std::cout << "Running benchmark for " 
                      << precision_to_string_extended(precision, actual_fp16_mode) 
                      << "..." << std::endl;
        }
        
        // Run compute-bound benchmark for this precision
        PrecisionResult result = benchmark.run(precision);
        table.add_result(result);
    }
    
    return table;
}

// Format single precision benchmark output based on output format
inline std::string format_single_precision_output(
    const PrecisionResult& result,
    const Config& config,
    const CpuInfo& cpu_info,
    unsigned thread_count
) {
    std::ostringstream oss;
    
    switch (config.output) {
        case OutputFormat::Text: {
            // Use existing Benchmark format_text style but with precision info
            const int label_w = 22;
            const int value_w = 20;
            const int total_w = label_w + value_w + 6;
            
            std::string h_line_top = "+" + std::string(total_w - 2, '-') + "+";
            std::string h_line_mid = "+" + std::string(label_w + 1, '-') + "+" + std::string(value_w + 2, '-') + "+";
            std::string h_line_bot = "+" + std::string(total_w - 2, '-') + "+";
            std::string h_line_sec = "+" + std::string(total_w - 2, '=') + "+";
            
            std::string title = " CPU BENCHMARK RESULTS ";
            int pad_left = (total_w - 2 - static_cast<int>(title.size())) / 2;
            int pad_right = total_w - 2 - static_cast<int>(title.size()) - pad_left;
            
            oss << "\n";
            oss << h_line_top << "\n";
            oss << "|" << std::string(pad_left, ' ') << title << std::string(pad_right, ' ') << "|\n";
            oss << h_line_sec << "\n";
            
            // Configuration section
            std::string cfg_title = "    CONFIGURATION     ";
            oss << "|" << std::string(pad_left, ' ') << cfg_title << std::string(pad_right, ' ') << "|\n";
            oss << h_line_mid << "\n";
            oss << table_row("Mode", mode_to_string(config.mode), label_w, value_w) << "\n";
            oss << table_row("Size", config.size.to_string(), label_w, value_w) << "\n";
            oss << table_row("Precision", precision_to_string_extended(result.precision, result.fp16_mode), label_w, value_w) << "\n";
            oss << table_row("Threads", std::to_string(thread_count), label_w, value_w) << "\n";
            oss << table_row("OpenMP", is_openmp_enabled() ? "enabled" : "disabled", label_w, value_w) << "\n";
            oss << table_row("Repeats", std::to_string(config.repeats), label_w, value_w) << "\n";
            oss << table_row("Iterations/repeat", std::to_string(result.result.iterations), label_w, value_w) << "\n";
            
            // Timing section
            oss << h_line_sec << "\n";
            std::string time_title = "       TIMING         ";
            oss << "|" << std::string(pad_left, ' ') << time_title << std::string(pad_right, ' ') << "|\n";
            oss << h_line_mid << "\n";
            oss << table_row("Average", format_time_ms(result.result.time_avg_sec), label_w, value_w) << "\n";
            oss << table_row("Minimum", format_time_ms(result.result.time_min_sec), label_w, value_w) << "\n";
            oss << table_row("Std Dev", format_time_ms(result.result.time_stddev_sec), label_w, value_w) << "\n";
            
            // Performance section
            oss << h_line_sec << "\n";
            std::string perf_title = "     PERFORMANCE      ";
            oss << "|" << std::string(pad_left, ' ') << perf_title << std::string(pad_right, ' ') << "|\n";
            oss << h_line_mid << "\n";
            
            // Show GOPS for INT8, GFLOPS for others
            std::string perf_label = result.config.is_integer ? "GOPS (avg)" : "GFLOPS (avg)";
            std::string perf_label_max = result.config.is_integer ? "GOPS (max)" : "GFLOPS (max)";
            oss << table_row(perf_label, format_gflops(result.result.gflops_avg), label_w, value_w) << "\n";
            oss << table_row(perf_label_max, format_gflops(result.result.gflops_max), label_w, value_w) << "\n";
            
            if (config.mode == BenchmarkMode::Stencil) {
                oss << table_row("Throughput", format_mlups(result.result.mlups_avg), label_w, value_w) << "\n";
            }
            if (config.mode == BenchmarkMode::Mem) {
                oss << table_row("Bandwidth", format_bandwidth(result.result.bandwidth_gbs), label_w, value_w) << "\n";
            }
            
            oss << h_line_bot << "\n";
            
            // Score section 
            {
                ScoreCalculator calculator;
                
                // Determine test type based on benchmark mode
                TestResultEntry::TestType test_type;
                switch (config.mode) {
                    case BenchmarkMode::Mem:
                    case BenchmarkMode::CacheLevel:
                        test_type = TestResultEntry::TestType::Memory;
                        break;
                    case BenchmarkMode::Stencil:
                        test_type = TestResultEntry::TestType::Mixed;
                        break;
                    case BenchmarkMode::Matmul3D:
                    case BenchmarkMode::Compute:
                        test_type = TestResultEntry::TestType::Compute;
                        break;
                }
                
            }
            
            // Add notes for emulated modes 
            if (result.precision == Precision::FP16 && result.fp16_mode == FP16Mode::Emulated) {
                oss << "\nNote: FP16 running in emulated mode (store as half, compute as float)\n";
            }
            if (result.precision == Precision::FP4) {
                oss << "\nNote: FP4 is CPU emulation with significant conversion overhead\n";
            }
            break;
        }
        
        case OutputFormat::Json: {
            // Single result JSON with full system info 
            std::vector<PrecisionResult> results = {result};
            oss << format_json_with_system_info(results, cpu_info);
            break;
        }
        
        case OutputFormat::Csv: {
            ComparisonTable single_table;
            single_table.add_result(result);
            oss << single_table.format_csv();
            break;
        }
    }
    
    return oss.str();
}

