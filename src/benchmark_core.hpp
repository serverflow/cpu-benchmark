#pragma once
// CPU Benchmark - Benchmark core header


#include "types.hpp"
#include "platform.hpp"
#include "math_kernels.hpp"
#include "threading.hpp"
#include "persistent_thread_pool.hpp"
#include "cpu_capabilities.hpp"
#include "kernel_dispatcher.hpp"
#include "warmup.hpp"
#include "runtime_dispatcher.hpp"

#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <functional>

// Progress callback type for benchmark progress updates 
// Parameters: current_repeat, total_repeats, percentage, eta_seconds
using BenchmarkProgressCallback = std::function<void(size_t, size_t, double, double)>;

// Forward declarations for extended types
struct half;
template<typename T> void kernel_mem_typed(T*, const T*, const T*, float, float, size_t, size_t, size_t, size_t, size_t);
template<typename T> void kernel_stencil_typed(T*, const T*, float, float, size_t, size_t, size_t, size_t, size_t);
template<typename T> void kernel_matmul3d_typed(T*, const T*, const T*, size_t, size_t, size_t);

// Statistics calculation functions
// Computes average of a vector of values
inline double compute_average(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

// Computes minimum of a vector of values
inline double compute_minimum(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return *std::min_element(values.begin(), values.end());
}

// Computes standard deviation of a vector of values
inline double compute_stddev(const std::vector<double>& values) {
    if (values.size() < 2) return 0.0;
    double avg = compute_average(values);
    double sum_sq = 0.0;
    for (const auto& v : values) {
        double diff = v - avg;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / static_cast<double>(values.size()));
}

// GFLOPS calculation 
// GFLOPS = total_flops / time_seconds / 1e9
inline double compute_gflops(size_t total_flops, double time_sec) {
    if (time_sec <= 0.0) return 0.0;
    return static_cast<double>(total_flops) / time_sec / 1e9;
}

// Helper function to create thread pool with optional socket affinity
// Returns number of threads and optionally core_ids for socket affinity

// Helper function to create thread pool with optional socket affinity
// Returns number of threads and optionally core_ids for socket affinity or hybrid-aware pinning
inline std::pair<unsigned, std::vector<unsigned>> get_thread_config_for_config(const Config& config) {
    if (config.selected_socket >= 0) {
        // Get cores for the selected socket
        std::vector<unsigned> socket_cores = get_cores_for_socket(static_cast<unsigned>(config.selected_socket));
        if (!socket_cores.empty()) {
            // Use socket core count (or user-specified if smaller)
            unsigned socket_thread_count = static_cast<unsigned>(socket_cores.size());
            unsigned num_threads = (config.threads == 0 || config.threads > socket_thread_count)
                ? socket_thread_count : config.threads;
            return {num_threads, socket_cores};
        }
    }

    // Fallback: single-socket / no NUMA preference.
    unsigned hw = get_logical_core_count();

    unsigned num_threads = (config.threads == 0) ? hw : config.threads;

    // Hybrid-aware pinning: prefer P-cores first, then E-cores.
    // This matters when the OS enumerates E-cores with low CPU indices.
    std::vector<unsigned> core_ids;
    std::vector<unsigned> perf = get_performance_cores();
    if (!perf.empty() && perf.size() < hw) {
        std::vector<uint8_t> is_perf(hw, 0);
        for (unsigned c : perf) {
            if (c < hw) is_perf[c] = 1;
        }

        core_ids.reserve(hw);
        for (unsigned c : perf) {
            if (c < hw) core_ids.push_back(c);
        }
        for (unsigned i = 0; i < hw; ++i) {
            if (!is_perf[i]) core_ids.push_back(i);
        }
    }

    if (core_ids.empty()) {
        core_ids.reserve(num_threads);
        unsigned limit = hw > 0 ? hw : num_threads;
        if (limit == 0) limit = 1;
        for (unsigned i = 0; i < num_threads; ++i) {
            core_ids.push_back(i % limit);
        }
    }

    return {num_threads, core_ids};
}

// Benchmark class template 
template<typename T>
class Benchmark {
public:
    Benchmark(const Config& config, const CpuInfo& cpu_info)
        : config_(config)
        , cpu_info_(cpu_info)
        , thread_config_(get_thread_config_for_config(config))
        , thread_pool_(thread_config_.first, thread_config_.second)
        , A_(config.size.Nx, config.size.Ny, config.size.Nz)
        , B_(config.size.Nx, config.size.Ny, config.size.Nz)
        , C_(config.size.Nx, config.size.Ny, config.size.Nz)
        , progress_callback_(nullptr)
    {
        // Initialize matrices with random values using parallel initialization
        // This ensures NUMA-aware memory placement (first-touch policy)
        unsigned num_threads = thread_pool_.thread_count();
        A_.fill_random_parallel(T(-1.0), T(1.0), num_threads);
        B_.fill_random_parallel(T(-1.0), T(1.0), num_threads);
        C_.fill_zero_parallel(num_threads);
        
        // Set kernel coefficients
        alpha_ = T(1.5);
        beta_ = T(0.5);
        a0_ = T(0.5);
        a1_ = T(1.0 / 6.0);
    }
    
    // Set progress callback 
    void set_progress_callback(BenchmarkProgressCallback callback) {
        progress_callback_ = callback;
    }
    
    // Run the benchmark and return results
    BenchmarkResult run() {
        warmup();
        measure();
        compute_statistics();
        return result_;
    }
    
    // Format output as text 
    std::string format_text() const;
    
    // Format output as JSON
    std::string format_json() const;
    
    // Format output as CSV 
    std::string format_csv() const;
    
    // Get result
    const BenchmarkResult& result() const { return result_; }
    
    // Get thread count
    unsigned thread_count() const { return thread_pool_.thread_count(); }
    
private:
    // Warmup phase 
    void warmup() {
        const int warmup_iterations = 3;
        for (int i = 0; i < warmup_iterations; ++i) {
            execute_kernel();
        }
        
        // For compute mode, determine FLOPS per kernel call 
        if (config_.mode == BenchmarkMode::Compute) {
            compute_flops_per_call_ = execute_compute_kernel<T>();
        }
    }
    
    // Measurement phase 
    void measure() {
        result_.times_sec.clear();
        result_.times_sec.reserve(config_.repeats);
        
        // First, determine how many iterations needed to reach min_time
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
        
        // Execute repeats
        for (unsigned r = 0; r < config_.repeats; ++r) {
            auto start = std::chrono::steady_clock::now();
            for (size_t i = 0; i < iterations_per_repeat; ++i) {
                execute_kernel();
            }
            auto end = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            // Store time per iteration
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

    
    // Compute statistics 
    void compute_statistics() {
        result_.time_avg_sec = compute_average(result_.times_sec);
        result_.time_min_sec = compute_minimum(result_.times_sec);
        result_.time_stddev_sec = compute_stddev(result_.times_sec);
        
        // Calculate total FLOPS based on mode
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
                // For compute mode, FLOPS are calculated per kernel call
                // Total = FLOPS per call * number of threads
                result_.total_flops = compute_flops_per_call_ * thread_pool_.thread_count();
                break;
            case BenchmarkMode::CacheLevel:
                // Cache-level mode is handled separately in main.cpp
                result_.total_flops = flops_mem(Nx, Ny, Nz);
                break;
        }
        
        // GFLOPS calculations 
        result_.gflops_avg = compute_gflops(result_.total_flops, result_.time_avg_sec);
        result_.gflops_max = compute_gflops(result_.total_flops, result_.time_min_sec);
        
        // Mode-specific metrics
        if (config_.mode == BenchmarkMode::Stencil) {
            // MLUP/s = million lattice updates per second
            size_t inner_cells = (Nx > 2 && Ny > 2 && Nz > 2) 
                ? (Nx - 2) * (Ny - 2) * (Nz - 2) : 0;
            result_.mlups_avg = (result_.time_avg_sec > 0) 
                ? static_cast<double>(inner_cells) / result_.time_avg_sec / 1e6 : 0.0;
        }
        
        if (config_.mode == BenchmarkMode::Mem) {
            // Bandwidth: 3 arrays * size * sizeof(T) bytes read/written
            size_t bytes = 3 * Nx * Ny * Nz * sizeof(T);
            result_.bandwidth_gbs = (result_.time_avg_sec > 0)
                ? static_cast<double>(bytes) / result_.time_avg_sec / 1e9 : 0.0;
        }
    }
    
    // Execute the appropriate kernel based on mode
    // Uses dispatched SIMD kernels for float/double 
    // Uses typed kernels for extended precision types (half, int8_t)
    void execute_kernel() {
        T* C_data = C_.data();
        const T* A_data = A_.data();
        const T* B_data = B_.data();
        size_t Nx = config_.size.Nx;
        size_t Ny = config_.size.Ny;
        size_t Nz = config_.size.Nz;
        
        // Store coefficients as float for typed kernels
        float alpha_f = static_cast<float>(alpha_);
        float beta_f = static_cast<float>(beta_);
        float a0_f = static_cast<float>(a0_);
        float a1_f = static_cast<float>(a1_);
        
        switch (config_.mode) {
            case BenchmarkMode::Mem:
                thread_pool_.parallel_for_z(Nz, [=](size_t z_begin, size_t z_end) {
                    execute_mem_kernel(C_data, A_data, B_data, alpha_f, beta_f,
                                       z_begin, z_end, Nx, Ny, Nz);
                });
                break;
                
            case BenchmarkMode::Stencil:
                thread_pool_.parallel_for_z(Nz, [=](size_t z_begin, size_t z_end) {
                    execute_stencil_kernel(C_data, A_data, a0_f, a1_f,
                                           z_begin, z_end, Nx, Ny, Nz);
                });
                break;
                
            case BenchmarkMode::Matmul3D:
                thread_pool_.parallel_for_z(Nz, [=](size_t z_begin, size_t z_end) {
                    execute_matmul_kernel(C_data, A_data, B_data,
                                          z_begin, z_end, Nx);
                });
                break;
                
            case BenchmarkMode::Compute:
                // Compute mode: pure FMA operations 
                // Each thread runs compute kernel independently
                thread_pool_.parallel_for_z(thread_pool_.thread_count(), [this](size_t /*z_begin*/, size_t /*z_end*/) {
                    execute_compute_kernel();
                });
                break;
            case BenchmarkMode::CacheLevel:
                // Cache-level mode is handled separately by CacheBenchmark class
                // Fall back to mem kernel if called directly
                thread_pool_.parallel_for_z(Nz, [=](size_t z_begin, size_t z_end) {
                    execute_mem_kernel(C_data, A_data, B_data, alpha_f, beta_f,
                                       z_begin, z_end, Nx, Ny, Nz);
                });
                break;
        }
    }
    
    // Helper methods for kernel dispatch
    // These use SFINAE to select dispatched kernels for float/double
    // and typed kernels for other types
    
    // Memory kernel dispatch
    template<typename U = T>
    typename std::enable_if<std::is_same<U, float>::value>::type
    execute_mem_kernel(U* C, const U* A, const U* B, float alpha, float beta,
                       size_t z_begin, size_t z_end, size_t Nx, size_t Ny, size_t Nz) {
        auto kernel = KernelDispatcher::instance().get_mem_kernel<float>();
        kernel(C, A, B, alpha, beta, z_begin, z_end, Nx, Ny, Nz);
    }
    
    template<typename U = T>
    typename std::enable_if<std::is_same<U, double>::value>::type
    execute_mem_kernel(U* C, const U* A, const U* B, float alpha, float beta,
                       size_t z_begin, size_t z_end, size_t Nx, size_t Ny, size_t Nz) {
        auto kernel = KernelDispatcher::instance().get_mem_kernel<double>();
        kernel(C, A, B, static_cast<double>(alpha), static_cast<double>(beta), 
               z_begin, z_end, Nx, Ny, Nz);
    }
    
    template<typename U = T>
    typename std::enable_if<!std::is_same<U, float>::value && !std::is_same<U, double>::value>::type
    execute_mem_kernel(U* C, const U* A, const U* B, float alpha, float beta,
                       size_t z_begin, size_t z_end, size_t Nx, size_t Ny, size_t Nz) {
        kernel_mem_typed<U>(C, A, B, alpha, beta, z_begin, z_end, Nx, Ny, Nz);
    }
    
    // Stencil kernel dispatch
    template<typename U = T>
    typename std::enable_if<std::is_same<U, float>::value>::type
    execute_stencil_kernel(U* C, const U* A, float a0, float a1,
                           size_t z_begin, size_t z_end, size_t Nx, size_t Ny, size_t Nz) {
        auto kernel = KernelDispatcher::instance().get_stencil_kernel<float>();
        kernel(C, A, a0, a1, z_begin, z_end, Nx, Ny, Nz);
    }
    
    template<typename U = T>
    typename std::enable_if<std::is_same<U, double>::value>::type
    execute_stencil_kernel(U* C, const U* A, float a0, float a1,
                           size_t z_begin, size_t z_end, size_t Nx, size_t Ny, size_t Nz) {
        auto kernel = KernelDispatcher::instance().get_stencil_kernel<double>();
        kernel(C, A, static_cast<double>(a0), static_cast<double>(a1), 
               z_begin, z_end, Nx, Ny, Nz);
    }
    
    template<typename U = T>
    typename std::enable_if<!std::is_same<U, float>::value && !std::is_same<U, double>::value>::type
    execute_stencil_kernel(U* C, const U* A, float a0, float a1,
                           size_t z_begin, size_t z_end, size_t Nx, size_t Ny, size_t Nz) {
        kernel_stencil_typed<U>(C, A, a0, a1, z_begin, z_end, Nx, Ny, Nz);
    }
    
    // Matmul kernel dispatch
    template<typename U = T>
    typename std::enable_if<std::is_same<U, float>::value>::type
    execute_matmul_kernel(U* C, const U* A, const U* B,
                          size_t z_begin, size_t z_end, size_t N) {
        auto kernel = KernelDispatcher::instance().get_matmul_kernel<float>();
        kernel(C, A, B, z_begin, z_end, N);
    }
    
    template<typename U = T>
    typename std::enable_if<std::is_same<U, double>::value>::type
    execute_matmul_kernel(U* C, const U* A, const U* B,
                          size_t z_begin, size_t z_end, size_t N) {
        auto kernel = KernelDispatcher::instance().get_matmul_kernel<double>();
        kernel(C, A, B, z_begin, z_end, N);
    }
    
    template<typename U = T>
    typename std::enable_if<!std::is_same<U, float>::value && !std::is_same<U, double>::value>::type
    execute_matmul_kernel(U* C, const U* A, const U* B,
                          size_t z_begin, size_t z_end, size_t N) {
        kernel_matmul3d_typed<U>(C, A, B, z_begin, z_end, N);
    }
    

    // Returns the number of FLOPs performed
    template<typename U = T>
    typename std::enable_if<std::is_same<U, float>::value, size_t>::type
    execute_compute_kernel() {
        ComputeKernelFloatFn kernel = RuntimeDispatcher::get_compute_kernel_float();
        float result;
        return kernel(&result, compute_iterations_);
    }
    
    template<typename U = T>
    typename std::enable_if<std::is_same<U, double>::value, size_t>::type
    execute_compute_kernel() {
        ComputeKernelDoubleFn kernel = RuntimeDispatcher::get_compute_kernel_double();
        double result;
        return kernel(&result, compute_iterations_);
    }
    
    template<typename U = T>
    typename std::enable_if<!std::is_same<U, float>::value && !std::is_same<U, double>::value, size_t>::type
    execute_compute_kernel() {
        // Fallback for non-float/double types - use scalar double
        double result;
        return kernels::compute::scalar_double(&result, compute_iterations_);
    }
    
    Config config_;
    CpuInfo cpu_info_;
    std::pair<unsigned, std::vector<unsigned>> thread_config_;
    PersistentThreadPool thread_pool_;
    Matrix3D<T> A_, B_, C_;
    T alpha_, beta_, a0_, a1_;
    BenchmarkResult result_;
    BenchmarkProgressCallback progress_callback_;
  
    size_t compute_iterations_ = 10000000;  // 10M iterations per thread
    size_t compute_flops_per_call_ = 0;     // Cached FLOPS per kernel call
};


// Helper function to create a row in a table
inline std::string table_row(const std::string& label, const std::string& value, int label_width = 22, int value_width = 20) {
    std::ostringstream oss;
    oss << "| " << std::left << std::setw(label_width) << label 
        << "| " << std::right << std::setw(value_width) << value << " |";
    return oss.str();
}

// Helper to format numbers with units
inline std::string format_time_ms(double sec) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << sec * 1000.0 << " ms";
    return oss.str();
}

inline std::string format_gflops(double gflops) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << gflops << " GFLOPS";
    return oss.str();
}

inline std::string format_bandwidth(double gbs) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << gbs << " GB/s";
    return oss.str();
}

inline std::string format_mlups(double mlups) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << mlups << " MLUP/s";
    return oss.str();
}


template<typename T>
std::string Benchmark<T>::format_text() const {
    std::ostringstream oss;
    
    const int label_w = 22;
    const int value_w = 20;
    const int total_w = label_w + value_w + 6; // borders and spaces
    
    std::string h_line_top = "+" + std::string(total_w - 2, '-') + "+";
    std::string h_line_mid = "+" + std::string(label_w + 1, '-') + "+" + std::string(value_w + 2, '-') + "+";
    std::string h_line_bot = "+" + std::string(total_w - 2, '-') + "+";
    std::string h_line_sec = "+" + std::string(total_w - 2, '=') + "+";
    
    // Title
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
    oss << table_row("Mode", mode_to_string(config_.mode), label_w, value_w) << "\n";
    oss << table_row("Size", config_.size.to_string(), label_w, value_w) << "\n";
    oss << table_row("Precision", precision_to_string(config_.precision), label_w, value_w) << "\n";
    oss << table_row("Threads", std::to_string(thread_pool_.thread_count()), label_w, value_w) << "\n";
    oss << table_row("OpenMP", is_openmp_enabled() ? "enabled" : "disabled", label_w, value_w) << "\n";
    oss << table_row("Repeats", std::to_string(config_.repeats), label_w, value_w) << "\n";
    oss << table_row("Iterations/repeat", std::to_string(result_.iterations), label_w, value_w) << "\n";
    
    // Timing section
    oss << h_line_sec << "\n";
    std::string time_title = "       TIMING         ";
    oss << "|" << std::string(pad_left, ' ') << time_title << std::string(pad_right, ' ') << "|\n";
    oss << h_line_mid << "\n";
    oss << table_row("Average", format_time_ms(result_.time_avg_sec), label_w, value_w) << "\n";
    oss << table_row("Minimum", format_time_ms(result_.time_min_sec), label_w, value_w) << "\n";
    oss << table_row("Std Dev", format_time_ms(result_.time_stddev_sec), label_w, value_w) << "\n";
    
    // Performance section
    oss << h_line_sec << "\n";
    std::string perf_title = "     PERFORMANCE      ";
    oss << "|" << std::string(pad_left, ' ') << perf_title << std::string(pad_right, ' ') << "|\n";
    oss << h_line_mid << "\n";
    oss << table_row("GFLOPS (avg)", format_gflops(result_.gflops_avg), label_w, value_w) << "\n";
    oss << table_row("GFLOPS (max)", format_gflops(result_.gflops_max), label_w, value_w) << "\n";
    
    if (config_.mode == BenchmarkMode::Stencil) {
        oss << table_row("Throughput", format_mlups(result_.mlups_avg), label_w, value_w) << "\n";
    }
    if (config_.mode == BenchmarkMode::Mem) {
        oss << table_row("Bandwidth", format_bandwidth(result_.bandwidth_gbs), label_w, value_w) << "\n";
    }
    
    oss << h_line_bot << "\n";
    
    return oss.str();
}

// Format output as JSON 
template<typename T>
std::string Benchmark<T>::format_json() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(9);
    
    oss << "{";
    oss << "\"mode\":\"" << mode_to_string(config_.mode) << "\",";
    oss << "\"size\":{\"Nx\":" << config_.size.Nx 
        << ",\"Ny\":" << config_.size.Ny 
        << ",\"Nz\":" << config_.size.Nz << "},";
    oss << "\"precision\":\"" << precision_to_string(config_.precision) << "\",";
    oss << "\"threads\":" << thread_pool_.thread_count() << ",";
    oss << "\"openmp\":" << (is_openmp_enabled() ? "true" : "false") << ",";
    oss << "\"repeats\":" << config_.repeats << ",";
    oss << "\"iterations\":" << result_.iterations << ",";
    
    // Times array
    oss << "\"times\":[";
    for (size_t i = 0; i < result_.times_sec.size(); ++i) {
        if (i > 0) oss << ",";
        oss << result_.times_sec[i];
    }
    oss << "],";
    
    oss << "\"time_avg\":" << result_.time_avg_sec << ",";
    oss << "\"time_min\":" << result_.time_min_sec << ",";
    oss << "\"time_stddev\":" << result_.time_stddev_sec << ",";
    oss << "\"gflops_avg\":" << result_.gflops_avg << ",";
    oss << "\"gflops_max\":" << result_.gflops_max << ",";
    oss << "\"total_flops\":" << result_.total_flops << ",";
    
    if (config_.mode == BenchmarkMode::Stencil) {
        oss << "\"mlups_avg\":" << result_.mlups_avg << ",";
    }
    if (config_.mode == BenchmarkMode::Mem) {
        oss << "\"bandwidth_gbs\":" << result_.bandwidth_gbs << ",";
    }
    
    // CPU info with cache 
    oss << "\"cpu\":{";
    oss << "\"arch\":\"" << cpu_info_.arch << "\",";
    oss << "\"logical_cores\":" << cpu_info_.logical_cores << ",";
    oss << "\"physical_cores\":" << cpu_info_.physical_cores << ",";
    oss << "\"vendor\":\"" << cpu_info_.vendor << "\",";
    oss << "\"model\":\"" << cpu_info_.model << "\",";
    
    // Cache information 
    oss << "\"cache\":{";
    oss << "\"l1_data_kb\":" << (cpu_info_.cache.l1_available ? std::to_string(cpu_info_.cache.l1_data_size / 1024) : "null") << ",";
    oss << "\"l1_inst_kb\":" << (cpu_info_.cache.l1_available ? std::to_string(cpu_info_.cache.l1_inst_size / 1024) : "null") << ",";
    oss << "\"l2_kb\":" << (cpu_info_.cache.l2_available ? std::to_string(cpu_info_.cache.l2_size / 1024) : "null") << ",";
    oss << "\"l3_kb\":" << (cpu_info_.cache.l3_available ? std::to_string(cpu_info_.cache.l3_size / 1024) : "null") << ",";
    oss << "\"line_size\":" << cpu_info_.cache.cache_line_size;
    oss << "},";
    
    // SIMD capabilities 
    const auto& caps = CpuCapabilities::get();
    oss << "\"simd_capabilities\":{";
    oss << "\"sse2\":" << (caps.has_sse2 ? "true" : "false") << ",";
    oss << "\"sse4_2\":" << (caps.has_sse4_2 ? "true" : "false") << ",";
    oss << "\"avx\":" << (caps.has_avx ? "true" : "false") << ",";
    oss << "\"avx2\":" << (caps.has_avx2 ? "true" : "false") << ",";
    oss << "\"avx512f\":" << (caps.has_avx512f ? "true" : "false") << ",";
    oss << "\"avx512_fp16\":" << (caps.has_avx512_fp16 ? "true" : "false") << ",";
    oss << "\"avx512_vnni\":" << (caps.has_avx512_vnni ? "true" : "false") << ",";
    oss << "\"arm_neon\":" << (caps.has_arm_neon ? "true" : "false") << ",";
    oss << "\"arm_neon_fp16\":" << (caps.has_arm_neon_fp16 ? "true" : "false");
    oss << "}";
    
    oss << "},";
    
    // Benchmark SIMD info 
    oss << "\"benchmark\":{";
    oss << "\"simd_level\":\"" << simd_level_to_string(caps.get_simd_level()) << "\",";
    
    // FP16 mode 
    std::string fp16_mode;
    if (caps.has_avx512_fp16) {
        fp16_mode = "native (AVX-512 FP16)";
    } else if (caps.has_arm_neon_fp16) {
        fp16_mode = "native (ARM NEON FP16)";
    } else {
        fp16_mode = "emulated";
    }
    oss << "\"fp16_mode\":\"" << fp16_mode << "\",";
    
    // Warmup status 
    oss << "\"warmup_performed\":" << (WarmupManager::was_warmup_performed() ? "true" : "false");
    oss << "}";
    
    oss << "}";
    
    return oss.str();
}

// Format output as CSV 
template<typename T>
std::string Benchmark<T>::format_csv() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    
    // Header
    oss << "mode,size,precision,threads,openmp,repeats,time_avg_ms,time_min_ms,";
    oss << "time_stddev_ms,gflops_avg,gflops_max";
    if (config_.mode == BenchmarkMode::Stencil) {
        oss << ",mlups_avg";
    }
    if (config_.mode == BenchmarkMode::Mem) {
        oss << ",bandwidth_gbs";
    }
    oss << std::endl;
    
    // Data row
    oss << mode_to_string(config_.mode) << ",";
    oss << config_.size.to_string() << ",";
    oss << precision_to_string(config_.precision) << ",";
    oss << thread_pool_.thread_count() << ",";
    oss << (is_openmp_enabled() ? "true" : "false") << ",";
    oss << config_.repeats << ",";
    oss << result_.time_avg_sec * 1000.0 << ",";
    oss << result_.time_min_sec * 1000.0 << ",";
    oss << result_.time_stddev_sec * 1000.0 << ",";
    oss << result_.gflops_avg << ",";
    oss << result_.gflops_max;
    if (config_.mode == BenchmarkMode::Stencil) {
        oss << "," << result_.mlups_avg;
    }
    if (config_.mode == BenchmarkMode::Mem) {
        oss << "," << result_.bandwidth_gbs;
    }
    oss << std::endl;
    
    return oss.str();
}
