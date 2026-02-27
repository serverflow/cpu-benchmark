// CPU Benchmark - Cache-Level Benchmark
// Runs benchmarks at different cache levels (L1, L2, L3, RAM)

#pragma once

#include "types.hpp"
#include "platform.hpp"
#include "cache_level_sizes.hpp"
#include "persistent_thread_pool.hpp"
#include "kernel_dispatcher.hpp"

#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <functional>

// Result for a single cache-level test
struct CacheLevelResult {
    CacheLevel level;           // Which cache level was targeted
    size_t data_size_bytes;     // Actual data size used
    size_t cache_size_bytes;    // Target cache size
    double time_avg_sec;        // Average time per iteration
    double time_min_sec;        // Minimum time
    double bandwidth_gbs;       // Memory bandwidth in GB/s
    double gflops;              // GFLOPS achieved
    bool fits_in_cache;         // Whether data fits in target cache
};

// Combined results for all cache levels
struct CacheBenchmarkResults {
    std::vector<CacheLevelResult> level_results;
    CacheInfo cache_info;       // Detected cache information
    size_t threads_used;        // Number of threads used
    bool warmup_performed;      // Whether warmup was done
};

// Cache-Level Benchmark class 
template<typename T>
class CacheBenchmark {
public:
    CacheBenchmark(unsigned num_threads, unsigned repeats)
        : num_threads_(num_threads == 0 ? get_logical_core_count() : num_threads)
        , repeats_(repeats)
        , thread_pool_(num_threads_)
    {
        // Get cache information
        cache_info_ = get_cache_info();
        
        // Calculate sizes for each cache level
        cache_sizes_ = CacheLevelSizes::calculate(cache_info_, sizeof(T));
    }
    
    // Run benchmarks for all cache levels
    CacheBenchmarkResults run_all_levels() {
        CacheBenchmarkResults results;
        results.cache_info = cache_info_;
        results.threads_used = num_threads_;
        results.warmup_performed = true;  // Assume warmup was done externally
        
        // Run L1-fit test
        results.level_results.push_back(run_level(CacheLevel::L1));
        
        // Run L2-fit test
        results.level_results.push_back(run_level(CacheLevel::L2));
        
        // Run L3-fit test
        results.level_results.push_back(run_level(CacheLevel::L3));
        
        // Run RAM-bound test
        results.level_results.push_back(run_level(CacheLevel::RAM));
        
        return results;
    }
    
    // Run benchmark for a specific cache level
    CacheLevelResult run_level(CacheLevel level) {
        CacheLevelResult result;
        result.level = level;
        result.cache_size_bytes = cache_sizes_.get_cache_size_for_level(level);
        
        // Get element count for this level
        size_t elements = cache_sizes_.get_elements_for_level(level);
        
        // Allocate cache-aligned buffers 
        AlignedBuffer<T> A(elements);
        AlignedBuffer<T> B(elements);
        AlignedBuffer<T> C(elements);
        
        if (!A.valid() || !B.valid() || !C.valid()) {
            // Allocation failed - return empty result
            result.data_size_bytes = 0;
            result.time_avg_sec = 0;
            result.time_min_sec = 0;
            result.bandwidth_gbs = 0;
            result.gflops = 0;
            result.fits_in_cache = false;
            return result;
        }
        
        result.data_size_bytes = elements * sizeof(T) * 3;  // 3 arrays
        // For cache-fit tests, we target 75% utilization, so data should fit within cache
        // The CacheLevelSizes calculator already accounts for this
        result.fits_in_cache = (result.data_size_bytes <= result.cache_size_bytes * 0.80);
        
        // Initialize data
        initialize_data(A.data(), B.data(), C.data(), elements);
        
        // Warmup
        run_kernel(A.data(), B.data(), C.data(), elements);
        run_kernel(A.data(), B.data(), C.data(), elements);
        
        // Measure
        std::vector<double> times;
        times.reserve(repeats_);
        
        for (unsigned r = 0; r < repeats_; ++r) {
            auto start = std::chrono::steady_clock::now();
            run_kernel(A.data(), B.data(), C.data(), elements);
            auto end = std::chrono::steady_clock::now();
            
            double elapsed = std::chrono::duration<double>(end - start).count();
            times.push_back(elapsed);
        }
        
        // Calculate statistics
        result.time_avg_sec = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        result.time_min_sec = *std::min_element(times.begin(), times.end());
        
        // Calculate bandwidth: 3 arrays * elements * sizeof(T) bytes
        size_t bytes_accessed = elements * sizeof(T) * 3;
        result.bandwidth_gbs = (result.time_min_sec > 0) 
            ? static_cast<double>(bytes_accessed) / result.time_min_sec / 1e9 
            : 0.0;
        
        // Calculate GFLOPS: 3 FLOPs per element (alpha*A + beta*B)
        size_t flops = elements * 3;
        result.gflops = (result.time_min_sec > 0)
            ? static_cast<double>(flops) / result.time_min_sec / 1e9
            : 0.0;
        
        return result;
    }
    
    // Format results as text 
    static std::string format_text(const CacheBenchmarkResults& results) {
        std::ostringstream oss;
        
        const int label_w = 12;
        const int value_w = 15;
        const int total_w = 80;
        
        std::string h_line = "+" + std::string(total_w - 2, '-') + "+";
        std::string h_line_sec = "+" + std::string(total_w - 2, '=') + "+";
        
        // Title
        std::string title = " CACHE-LEVEL BENCHMARK RESULTS ";
        int pad_left = (total_w - 2 - static_cast<int>(title.size())) / 2;
        int pad_right = total_w - 2 - static_cast<int>(title.size()) - pad_left;
        
        oss << "\n" << h_line << "\n";
        oss << "|" << std::string(pad_left, ' ') << title << std::string(pad_right, ' ') << "|\n";
        oss << h_line_sec << "\n";
        
        // Cache info
        oss << "| Cache Info: L1=" << (results.cache_info.l1_data_size / 1024) << "KB"
            << ", L2=" << (results.cache_info.l2_size / 1024) << "KB"
            << ", L3=" << (results.cache_info.l3_size / (1024*1024)) << "MB"
            << std::string(total_w - 60, ' ') << "|\n";
        oss << h_line << "\n";
        
        // Header
        oss << "| " << std::left << std::setw(label_w) << "Level"
            << "| " << std::right << std::setw(value_w) << "Data Size"
            << "| " << std::right << std::setw(value_w) << "Time (ms)"
            << "| " << std::right << std::setw(value_w) << "BW (GB/s)"
            << "| " << std::right << std::setw(value_w) << "GFLOPS"
            << " |\n";
        oss << h_line << "\n";
        
        // Results for each level 
        for (const auto& r : results.level_results) {
            std::string size_str;
            if (r.data_size_bytes >= 1024 * 1024) {
                size_str = std::to_string(r.data_size_bytes / (1024 * 1024)) + " MB";
            } else {
                size_str = std::to_string(r.data_size_bytes / 1024) + " KB";
            }
            
            std::ostringstream time_ss, bw_ss, gflops_ss;
            time_ss << std::fixed << std::setprecision(3) << r.time_min_sec * 1000.0;
            bw_ss << std::fixed << std::setprecision(2) << r.bandwidth_gbs;
            gflops_ss << std::fixed << std::setprecision(2) << r.gflops;
            
            // Show which cache level was targeted 
            std::string level_str = cache_level_to_string(r.level);
            if (!r.fits_in_cache && r.level != CacheLevel::RAM) {
                level_str += "*";  // Mark if data doesn't fit
            }
            
            oss << "| " << std::left << std::setw(label_w) << level_str
                << "| " << std::right << std::setw(value_w) << size_str
                << "| " << std::right << std::setw(value_w) << time_ss.str()
                << "| " << std::right << std::setw(value_w) << bw_ss.str()
                << "| " << std::right << std::setw(value_w) << gflops_ss.str()
                << " |\n";
        }
        
        oss << h_line << "\n";
        oss << "| * = Data size exceeds target cache level" << std::string(total_w - 47, ' ') << "|\n";
        oss << h_line << "\n";
        
        return oss.str();
    }
    
    // Format results as JSON
    static std::string format_json(const CacheBenchmarkResults& results) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(9);
        
        oss << "{";
        oss << "\"mode\":\"cache\",";
        oss << "\"threads\":" << results.threads_used << ",";
        oss << "\"warmup_performed\":" << (results.warmup_performed ? "true" : "false") << ",";
        
        // Cache info
        oss << "\"cache_info\":{";
        oss << "\"l1_data_kb\":" << (results.cache_info.l1_data_size / 1024) << ",";
        oss << "\"l2_kb\":" << (results.cache_info.l2_size / 1024) << ",";
        oss << "\"l3_mb\":" << (results.cache_info.l3_size / (1024*1024)) << ",";
        oss << "\"line_size\":" << results.cache_info.cache_line_size;
        oss << "},";
        
        // Level results
        oss << "\"levels\":[";
        for (size_t i = 0; i < results.level_results.size(); ++i) {
            if (i > 0) oss << ",";
            const auto& r = results.level_results[i];
            oss << "{";
            oss << "\"level\":\"" << cache_level_to_string(r.level) << "\",";
            oss << "\"data_size_bytes\":" << r.data_size_bytes << ",";
            oss << "\"cache_size_bytes\":" << r.cache_size_bytes << ",";
            oss << "\"time_avg_sec\":" << r.time_avg_sec << ",";
            oss << "\"time_min_sec\":" << r.time_min_sec << ",";
            oss << "\"bandwidth_gbs\":" << r.bandwidth_gbs << ",";
            oss << "\"gflops\":" << r.gflops << ",";
            oss << "\"fits_in_cache\":" << (r.fits_in_cache ? "true" : "false");
            oss << "}";
        }
        oss << "]";
        
        oss << "}";
        return oss.str();
    }
    
private:
    void initialize_data(T* A, T* B, T* C, size_t elements) {
        // Initialize with simple pattern for reproducibility
        for (size_t i = 0; i < elements; ++i) {
            A[i] = static_cast<T>(1.0 + (i % 100) * 0.01);
            B[i] = static_cast<T>(2.0 - (i % 100) * 0.01);
            C[i] = T(0);
        }
    }
    
    // Run memory kernel: C = alpha*A + beta*B (sequential access pattern)
    
    void run_kernel(T* A, T* B, T* C, size_t elements) {
        const T alpha = T(1.5);
        const T beta = T(0.5);
        
        // Use thread pool for parallel execution
        thread_pool_.parallel_for_z(elements, [=](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                C[i] = alpha * A[i] + beta * B[i];
            }
        });
    }
    
    unsigned num_threads_;
    unsigned repeats_;
    PersistentThreadPool thread_pool_;
    CacheInfo cache_info_;
    CacheLevelSizes cache_sizes_;
};

