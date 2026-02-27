// CPU Benchmark - Compute Benchmark with ST/MT tests
// Cross-architecture comparable benchmark using Scalar FP64 as baseline
//
// Design principles:
// 1. Scalar FP64 FMA is THE baseline for cross-arch comparison (x86 vs ARM)
// 2. Fixed time measurement (not iterations) - count FLOPs done in N seconds
// 3. No best-of-N repeats - single timed run after warmup
// 4. SIMD results are informational only, NOT used for scoring
// 5. Score is architecture-neutral (based on scalar FP64 GFLOPS)

#pragma once

#include "types.hpp"
#include "platform.hpp"
#include "cpu_capabilities.hpp"
#include "runtime_dispatcher.hpp"
#include "kernels/kernel_compute.hpp"
#include "persistent_thread_pool.hpp"
#include "thread_affinity.hpp"

#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>
#include <thread>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {
inline bool compute_debug_enabled() {
    const char* env = std::getenv("SFBENCH_DEBUG");
    return env && env[0] != '\0' && env[0] != '0';
}

inline bool env_flag_enabled(const char* name) {
    const char* env = std::getenv(name);
    return env && env[0] != '\0' && env[0] != '0';
}

inline void compute_debug_log(const std::string& msg) {
    if (compute_debug_enabled()) {
        std::cerr << msg << "\n";
    }
}
} // namespace

// Compute benchmark result for a single test
struct ComputeTestResult {
    std::string test_name;
    std::string test_type;      // "scalar_fp64" or "simd_fp32"
    unsigned threads;
    double time_sec;
    double gflops;
    size_t total_flops;
    size_t iterations;
};

// Full compute benchmark results
struct ComputeBenchmarkResults {
    // Primary results (Scalar FP64 - cross-arch comparable)
    ComputeTestResult single_thread;    // ST scalar FP64
    ComputeTestResult multi_thread;     // MT scalar FP64
    
    // Informational SIMD results (NOT for cross-arch comparison)
    ComputeTestResult simd_single_thread;
    ComputeTestResult simd_multi_thread;
    bool simd_available;
    
    // Derived metrics (from scalar FP64 only)
    double mt_speedup;          // MT GFLOPS / ST GFLOPS
    
    // Scores (based on scalar FP64 only - architecture neutral)
    int st_score;
    int mt_score;
    int overall_score;
    
    // Metadata
    std::string arch;           // "x86_64" or "arm64"
    std::string simd_level;     // Informational only
    unsigned physical_cores;
    unsigned logical_cores;
    bool warmup_performed;
    double warmup_duration_sec;
    double test_duration_sec;
    
    // CPU frequency during test
    FrequencyStats frequency;
};

// Compute Benchmark class
class ComputeBenchmark {
public:
    ComputeBenchmark(unsigned threads = 0, bool high_priority = false, int selected_socket = -1)
        : high_priority_(high_priority)
        , selected_socket_(selected_socket)
    {
        // Determine thread count based on socket selection
        if (selected_socket >= 0) {
            // Get cores for the selected socket
            socket_cores_ = get_cores_for_socket(static_cast<unsigned>(selected_socket));
            if (socket_cores_.empty()) {
                // Fallback to all cores if socket info unavailable
                num_threads_ = threads == 0 ? get_logical_core_count() : threads;
            } else {
                // Use socket core count (or user-specified if smaller)
                unsigned socket_thread_count = static_cast<unsigned>(socket_cores_.size());
                num_threads_ = (threads == 0 || threads > socket_thread_count) 
                    ? socket_thread_count : threads;
            }
        } else {
            num_threads_ = threads == 0 ? get_logical_core_count() : threads;
            // Prefer P-cores first when pinning worker threads on hybrid CPUs.
            socket_cores_ = get_preferred_core_order();
        }
        
        if (high_priority_) {
            ThreadAffinityManager::set_process_priority(ProcessPriority::High);
        }
    }
    
    // Run full benchmark: warmup + ST + MT (scalar FP64 baseline + optional SIMD info)
    ComputeBenchmarkResults run(double warmup_seconds = 10.0, 
                                 double test_seconds = 5.0) {
        ComputeBenchmarkResults results;
        const bool skip_simd = env_flag_enabled("SFBENCH_NO_SIMD");
        bool skip_freq = env_flag_enabled("SFBENCH_NO_FREQ");
        bool auto_skip_simd = false;
        bool allow_worker_affinity = !env_flag_enabled("SFBENCH_NO_AFFINITY");
        bool allow_st_affinity = allow_worker_affinity && !env_flag_enabled("SFBENCH_NO_ST_AFFINITY");

#ifdef _WIN32
        // On multi-socket / multi-group systems, background frequency sampling and ST pinning
        // can be unstable on some Windows builds. Prefer safety over affinity here.
        const unsigned logical = get_logical_core_count();
        const unsigned sockets = get_socket_count();
        if (!skip_freq && (logical > 64 || sockets > 1)) {
            skip_freq = true;
        }
        if (logical > 64 || sockets > 1) {
            auto_skip_simd = true;
        }
#endif
        
        // Metadata - adjust for socket selection
        if (selected_socket_ >= 0 && !socket_cores_.empty()) {
            // When running on specific socket, show socket core counts
            results.logical_cores = static_cast<unsigned>(socket_cores_.size());
            results.physical_cores = results.logical_cores / 2;  // Assume SMT
            if (results.physical_cores == 0) results.physical_cores = results.logical_cores;
        } else {
            results.physical_cores = get_physical_core_count();
            results.logical_cores = get_logical_core_count();
        }
        results.test_duration_sec = test_seconds;
        
        // Architecture detection
#if defined(__aarch64__) || defined(_M_ARM64)
        results.arch = "arm64";
#else
        results.arch = "x86_64";
#endif
        
        // SIMD level (informational)
        SimdLevel level = CpuCapabilities::get().get_simd_level();
        results.simd_level = simd_level_to_string(level);
        results.simd_available = (level != SimdLevel::Scalar);
        
        if (compute_debug_enabled()) {
            compute_debug_log("[compute] threads=" + std::to_string(num_threads_) +
                              " logical=" + std::to_string(results.logical_cores) +
                              " physical=" + std::to_string(results.physical_cores) +
                              " socket_selected=" + std::to_string(selected_socket_) +
                              " socket_cores=" + std::to_string(socket_cores_.size()) +
                              " simd=" + results.simd_level);
            if (skip_simd) compute_debug_log("[compute] SIMD tests disabled via SFBENCH_NO_SIMD");
            if (auto_skip_simd && !skip_simd) {
                compute_debug_log("[compute] SIMD tests auto-disabled for multi-socket Windows");
            }
            if (skip_freq) compute_debug_log("[compute] frequency sampling disabled");
            if (!allow_st_affinity) compute_debug_log("[compute] ST affinity disabled");
        }

        // Initialize frequency sampler
        FrequencySampler freq_sampler;
        
        // Phase 1: Warmup (split ST vs MT so ST isn't throttled by all-core warmup)
        results.warmup_performed = warmup_seconds > 0.0;
        results.warmup_duration_sec = warmup_seconds;
        compute_debug_log("[compute] warmup start");
        double st_warmup = 0.0;
        double mt_warmup = 0.0;
        if (warmup_seconds > 0.0) {
            st_warmup = (std::min)(0.5, warmup_seconds);
            mt_warmup = warmup_seconds - st_warmup;
        }
        if (st_warmup > 0.0) {
            compute_debug_log("[compute] warmup st start");
            perform_st_warmup(st_warmup, allow_st_affinity);
            compute_debug_log("[compute] warmup st done");
        }
        compute_debug_log("[compute] warmup done");
        
        // Start background frequency sampling (every 50ms during tests)
        if (!skip_freq) {
            compute_debug_log("[compute] freq sampler start");
            freq_sampler.start_background(50);
        }
        
        // Phase 2: Single-thread scalar FP64 test (THE baseline)
        compute_debug_log("[compute] ST scalar start");
        results.single_thread = run_scalar_fp64_test(1, test_seconds, true, allow_st_affinity);
        compute_debug_log("[compute] ST scalar done");
        
        if (mt_warmup > 0.0) {
            compute_debug_log("[compute] warmup mt start");
            perform_warmup(mt_warmup, allow_worker_affinity);
            compute_debug_log("[compute] warmup mt done");
        }

        // Phase 3: Multi-thread scalar FP64 test
        compute_debug_log("[compute] MT scalar start");
        results.multi_thread = run_scalar_fp64_test(num_threads_, test_seconds, false, allow_worker_affinity);
        compute_debug_log("[compute] MT scalar done");
        
        // Phase 4: SIMD tests (informational only)
        bool run_simd = results.simd_available && !skip_simd && !auto_skip_simd;
        results.simd_available = run_simd;
        if (run_simd) {
            compute_debug_log("[compute] ST simd start");
            results.simd_single_thread = run_simd_fp32_test(1, test_seconds, allow_st_affinity);
            compute_debug_log("[compute] ST simd done");
            compute_debug_log("[compute] MT simd start");
            results.simd_multi_thread = run_simd_fp32_test(num_threads_, test_seconds, allow_worker_affinity);
            compute_debug_log("[compute] MT simd done");
        } else if (compute_debug_enabled() && level != SimdLevel::Scalar) {
            compute_debug_log("[compute] SIMD tests skipped");
        }
        
        // Stop background sampling and get results
        if (!skip_freq) {
            compute_debug_log("[compute] freq sampler stop");
            freq_sampler.stop_background();
        }
        
        // Store frequency statistics
        if (!skip_freq) {
            results.frequency = freq_sampler.get_stats();
        } else {
            results.frequency = FrequencyStats{};
        }
        
        // Calculate derived metrics (from scalar FP64 only)
        if (results.single_thread.gflops > 0) {
            results.mt_speedup = results.multi_thread.gflops / results.single_thread.gflops;
        } else {
            results.mt_speedup = 0;
        }
        
        // Calculate scores (scalar FP64 based - architecture neutral)
        calculate_scores(results);
        compute_debug_log("[compute] run done");
        
        return results;
    }

    // Format results as text table
    static std::string format_text(const ComputeBenchmarkResults& results) {
        std::ostringstream oss;
        
        constexpr int table_inner_w = 76;
        const int col_test = 15;
        const int col_threads = 7;
        const int col_time = 10;
        const int col_gflops = 10;
        const int col_score = 20;

        auto line = [&](char ch) {
            return "+" + std::string(table_inner_w, ch) + "+\n";
        };

        auto fit_text = [&](const std::string& text) {
            if (text.size() <= static_cast<size_t>(table_inner_w)) {
                return text;
            }
            return text.substr(0, static_cast<size_t>(table_inner_w));
        };

        auto print_full = [&](const std::string& text) {
            oss << "|" << std::left << std::setw(table_inner_w) << fit_text(text) << "|\n";
        };

        auto print_center = [&](const std::string& text) {
            std::string fitted = fit_text(text);
            int pad_left = (table_inner_w - static_cast<int>(fitted.size())) / 2;
            int pad_right = table_inner_w - static_cast<int>(fitted.size()) - pad_left;
            oss << "|" << std::string(pad_left, ' ') << fitted << std::string(pad_right, ' ') << "|\n";
        };

        auto make_separator = [&](int w1, int w2, int w3, int w4, int w5) {
            std::string sep = "+";
            auto add = [&](int w) {
                sep += std::string(w + 2, '-');
                sep += "+";
            };
            add(w1);
            add(w2);
            add(w3);
            add(w4);
            add(w5);
            return sep + "\n";
        };

        // Header
        oss << "\n";
        oss << line('=');
        print_center("COMPUTE BENCHMARK RESULTS");
        print_center("(Cross-Architecture Comparable)");
        oss << line('=');
        
        // System info
        std::ostringstream sys_row;
        sys_row << " Architecture: " << std::left << std::setw(12) << results.arch
                << " | Cores: " << results.physical_cores << "P/" << results.logical_cores << "L"
                << " | Test time: " << std::fixed << std::setprecision(0) << results.test_duration_sec << "s";
        print_full(sys_row.str());
        oss << line('-');
        
        // Baseline results header
        print_full(" BASELINE (Scalar FP64) - Used for scoring and cross-arch comparison");
        oss << line('-');
        oss << "| " << std::left << std::setw(col_test) << "Test"
            << " | " << std::right << std::setw(col_threads) << "Threads"
            << " | " << std::setw(col_time) << "Time (s)"
            << " | " << std::setw(col_gflops) << "GFLOPS"
            << " | " << std::setw(col_score) << "Score" << " |\n";
        oss << make_separator(col_test, col_threads, col_time, col_gflops, col_score);
        
        // Single-thread result
        oss << "| " << std::left << std::setw(col_test) << "Single-Core" << " | "
            << std::right << std::setw(col_threads) << results.single_thread.threads << " | "
            << std::setw(col_time) << std::fixed << std::setprecision(3) << results.single_thread.time_sec << " | "
            << std::setw(col_gflops) << std::fixed << std::setprecision(2) << results.single_thread.gflops << " | "
            << std::setw(col_score) << results.st_score << " |\n";
        
        // Multi-thread result
        oss << "| " << std::left << std::setw(col_test) << "All-Cores" << " | "
            << std::right << std::setw(col_threads) << results.multi_thread.threads << " | "
            << std::setw(col_time) << std::fixed << std::setprecision(3) << results.multi_thread.time_sec << " | "
            << std::setw(col_gflops) << std::fixed << std::setprecision(2) << results.multi_thread.gflops << " | "
            << std::setw(col_score) << results.mt_score << " |\n";
        
        oss << make_separator(col_test, col_threads, col_time, col_gflops, col_score);
        
        // SIMD results (informational)
        if (results.simd_available) {
            std::ostringstream simd_line;
            simd_line << " SIMD (" << results.simd_level
                      << ") - Informational only, NOT for cross-arch comparison";
            print_full(simd_line.str());
            oss << line('-');
            oss << "| " << std::left << std::setw(col_test) << "Test"
                << " | " << std::right << std::setw(col_threads) << "Threads"
                << " | " << std::setw(col_time) << "Time (s)"
                << " | " << std::setw(col_gflops) << "GFLOPS"
                << " | " << std::setw(col_score) << "(info)" << " |\n";
            oss << make_separator(col_test, col_threads, col_time, col_gflops, col_score);
            
            oss << "| " << std::left << std::setw(col_test) << "SIMD Single" << " | "
                << std::right << std::setw(col_threads) << results.simd_single_thread.threads << " | "
                << std::setw(col_time) << std::fixed << std::setprecision(3) << results.simd_single_thread.time_sec << " | "
                << std::setw(col_gflops) << std::fixed << std::setprecision(2) << results.simd_single_thread.gflops << " | "
                << std::setw(col_score) << "-" << " |\n";
            
            oss << "| " << std::left << std::setw(col_test) << "SIMD Multi" << " | "
                << std::right << std::setw(col_threads) << results.simd_multi_thread.threads << " | "
                << std::setw(col_time) << std::fixed << std::setprecision(3) << results.simd_multi_thread.time_sec << " | "
                << std::setw(col_gflops) << std::fixed << std::setprecision(2) << results.simd_multi_thread.gflops << " | "
                << std::setw(col_score) << "-" << " |\n";
            
            oss << make_separator(col_test, col_threads, col_time, col_gflops, col_score);
        }
        
        // Summary section
        oss << "\n";
        oss << line('=');
        print_center("SUMMARY");
        oss << line('=');
        
        const int summary_label_w = 22;
        auto print_summary = [&](const std::string& label, const std::string& value) {
            std::ostringstream line;
            line << " " << std::left << std::setw(summary_label_w) << label << value;
            print_full(line.str());
        };

        char speedup_buf[32];
        std::snprintf(speedup_buf, sizeof(speedup_buf), "%.2fx", results.mt_speedup);
        print_summary("Multi-Core Speedup:", speedup_buf);
        print_summary("Score Basis:", "Scalar FP64 (cross-arch)");
        
        // Show CPU frequency if available
        if (results.frequency.available) {
            char freq_buf[64];
            std::snprintf(freq_buf, sizeof(freq_buf), "%.0f - %.0f MHz (avg: %.0f, %u samples)",
                results.frequency.min_mhz, results.frequency.max_mhz, 
                results.frequency.avg_mhz, results.frequency.sample_count);
            print_summary("CPU Frequency:", freq_buf);
        }
        
        oss << line('=');
        
        // Warning about SIMD comparison
        if (results.simd_available) {
            print_full(" NOTE: SIMD results vary by architecture and should NOT be compared");
            print_full("       between x86 and ARM systems. Use Scalar FP64 scores for comparison.");
            oss << line('=');
        }
        
        return oss.str();
    }
    
    // Format results as JSON
    static std::string format_json(const ComputeBenchmarkResults& results) {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"benchmark_type\": \"compute\",\n";
        oss << "  \"score_basis\": \"scalar_fp64\",\n";
        oss << "  \"architecture\": \"" << results.arch << "\",\n";
        oss << "  \"physical_cores\": " << results.physical_cores << ",\n";
        oss << "  \"logical_cores\": " << results.logical_cores << ",\n";
        oss << "  \"test_duration_sec\": " << results.test_duration_sec << ",\n";
        oss << "  \"warmup_seconds\": " << results.warmup_duration_sec << ",\n";
        
        // Baseline results
        oss << "  \"baseline\": {\n";
        oss << "    \"type\": \"scalar_fp64\",\n";
        oss << "    \"single_core\": {\n";
        oss << "      \"threads\": " << results.single_thread.threads << ",\n";
        oss << "      \"time_sec\": " << std::fixed << std::setprecision(6) << results.single_thread.time_sec << ",\n";
        oss << "      \"gflops\": " << std::fixed << std::setprecision(2) << results.single_thread.gflops << ",\n";
        oss << "      \"score\": " << results.st_score << "\n";
        oss << "    },\n";
        oss << "    \"all_cores\": {\n";
        oss << "      \"threads\": " << results.multi_thread.threads << ",\n";
        oss << "      \"time_sec\": " << std::fixed << std::setprecision(6) << results.multi_thread.time_sec << ",\n";
        oss << "      \"gflops\": " << std::fixed << std::setprecision(2) << results.multi_thread.gflops << ",\n";
        oss << "      \"score\": " << results.mt_score << "\n";
        oss << "    }\n";
        oss << "  },\n";
        
        // SIMD results (informational)
        oss << "  \"simd_info\": {\n";
        oss << "    \"available\": " << (results.simd_available ? "true" : "false") << ",\n";
        oss << "    \"level\": \"" << results.simd_level << "\"";
        if (results.simd_available) {
            oss << ",\n";
            oss << "    \"single_thread_gflops\": " << std::fixed << std::setprecision(2) << results.simd_single_thread.gflops << ",\n";
            oss << "    \"multi_thread_gflops\": " << std::fixed << std::setprecision(2) << results.simd_multi_thread.gflops << "\n";
        } else {
            oss << "\n";
        }
        oss << "  },\n";
        
        // Summary
        oss << "  \"mt_speedup\": " << std::fixed << std::setprecision(2) << results.mt_speedup << ",\n";
        oss << "  \"overall_score\": " << results.overall_score << ",\n";
        oss << "  \"st_score\": " << results.st_score << ",\n";
        oss << "  \"mt_score\": " << results.mt_score << "\n";
        oss << "}\n";
        return oss.str();
    }

private:
    unsigned num_threads_;
    bool high_priority_;
    int selected_socket_;
    std::vector<unsigned> socket_cores_;  // Core IDs for selected socket




// Build a stable core order for pinning threads on hybrid CPUs: P-cores first, then E-cores.
// Returns an empty vector on non-hybrid systems (so existing behavior is preserved).
static std::vector<unsigned> get_preferred_core_order() {
    unsigned total = get_logical_core_count();
    if (total == 0) total = 1;

    std::vector<unsigned> perf = get_performance_cores();
    if (perf.empty() || perf.size() >= total) {
        return {};  // Non-hybrid or detection unavailable
    }

    std::vector<uint8_t> is_perf(total, 0);
    for (unsigned c : perf) {
        if (c < total) is_perf[c] = 1;
    }

    std::vector<unsigned> order;
    order.reserve(total);
    for (unsigned c : perf) {
        if (c < total) order.push_back(c);
    }
    for (unsigned i = 0; i < total; ++i) {
        if (!is_perf[i]) order.push_back(i);
    }
    return order;
}
    // Build core ID list for worker threads (uses socket/p-core ordering when available).
    std::vector<unsigned> build_thread_core_ids(unsigned threads) const {
        std::vector<unsigned> core_ids;
        if (!socket_cores_.empty()) {
            core_ids = socket_cores_;
            if (core_ids.size() > threads) {
                core_ids.resize(threads);
            }
            return core_ids;
        }

        core_ids.reserve(threads);
        for (unsigned i = 0; i < threads; ++i) {
            core_ids.push_back(i);
        }
        return core_ids;
    }

    void pin_worker_thread(std::thread& worker, unsigned core_id, unsigned thread_id, bool allow_affinity) const {
        if (!ThreadAffinityManager::is_affinity_supported()) {
            return;
        }
        if (!allow_affinity || env_flag_enabled("SFBENCH_NO_AFFINITY")) {
            return;
        }
        AffinityResult res = ThreadAffinityManager::pin_to_core(worker, core_id);
        if (res != AffinityResult::Success && compute_debug_enabled()) {
            compute_debug_log("[compute] pin thread " + std::to_string(thread_id) +
                              " core " + std::to_string(core_id) + ": " +
                              affinity_result_to_string(res));
        }
    }

    void pin_st_thread(bool allow_affinity) const {
        if (!allow_affinity || !ThreadAffinityManager::is_affinity_supported()) {
            return;
        }

        // Keep ST on a single socket in multi-socket systems for consistent scoring.
        compute_debug_log("[compute][st] pin start");
        auto perf = get_performance_cores();
        unsigned total = get_logical_core_count();
        if (total == 0) total = 1;

        std::vector<unsigned> socket_cores;
        if (selected_socket_ >= 0 && !socket_cores_.empty()) {
            socket_cores = socket_cores_;
        } else if (selected_socket_ < 0 && get_socket_count() > 1) {
            socket_cores = get_cores_for_socket(0);
        }

        std::vector<unsigned> candidates;
        std::vector<uint8_t> seen(total, 0);
        auto push_unique = [&](unsigned c) {
            if (c < total && !seen[c]) {
                seen[c] = 1;
                candidates.push_back(c);
            }
        };

        bool has_perf_split = !perf.empty() && perf.size() < total;
        if (!socket_cores.empty()) {
            std::sort(socket_cores.begin(), socket_cores.end());
            if (has_perf_split) {
                std::vector<uint8_t> is_perf(total, 0);
                for (unsigned c : perf) {
                    if (c < total) is_perf[c] = 1;
                }
                for (unsigned c : socket_cores) {
                    if (c < total && is_perf[c]) push_unique(c);
                }
                for (unsigned c : socket_cores) {
                    if (c < total && !is_perf[c]) push_unique(c);
                }
            } else {
                for (unsigned c : socket_cores) push_unique(c);
            }
        } else if (!perf.empty()) {
            std::sort(perf.begin(), perf.end());
            for (unsigned c : perf) push_unique(c);
        }

        if (candidates.empty()) {
            candidates.push_back(0u);
        }

        unsigned fallback = candidates.front();
        bool pinned = false;
        for (unsigned c : candidates) {
            if (ThreadAffinityManager::pin_current_thread(c) == AffinityResult::Success) {
                pinned = true;
                break;
            }
        }
        if (!pinned) {
            // Best-effort fallback
            ThreadAffinityManager::pin_current_thread(fallback);
        }
        compute_debug_log("[compute][st] pin done");
    }
    // Get physical core count (platform-specific)
    static unsigned get_physical_core_count() {
        // Prefer platform-provided physical core count (handles hybrid CPUs correctly).
        CpuInfo info = get_cpu_info();
        if (info.physical_cores > 0) {
            return info.physical_cores;
        }

        // Fallback heuristic
        unsigned logical = get_logical_core_count();
        if (logical == 0) logical = 1;
    #if defined(__aarch64__) || defined(_M_ARM64)
        return logical;  // ARM typically reports physical cores
    #else
        return logical / 2 > 0 ? logical / 2 : logical;  // x86 often has SMT
    #endif
    }
    // Warmup phase - run scalar FP64 compute on a single thread
    void perform_st_warmup(double seconds, bool allow_affinity) {
        if (seconds <= 0) return;

        pin_st_thread(allow_affinity);

        const size_t batch_iterations = 100000;
        double result = 0.0;
        auto start = std::chrono::steady_clock::now();
        auto end_time = start + std::chrono::duration<double>(seconds);
        while (std::chrono::steady_clock::now() < end_time) {
            kernels::compute::scalar_fp64_baseline(&result, batch_iterations);
        }
    }

    // Warmup phase - run scalar FP64 compute on all threads
    void perform_warmup(double seconds, bool allow_affinity) {
        if (seconds <= 0) return;

        const size_t batch_iterations = 100000;
        std::vector<unsigned> core_ids = build_thread_core_ids(num_threads_);
        std::atomic<bool> stop_flag{false};
        std::atomic<bool> start_flag{false};
        std::atomic<unsigned> ready{0};
        std::vector<std::thread> workers;
        workers.reserve(num_threads_);

        for (unsigned t = 0; t < num_threads_; ++t) {
            workers.emplace_back([&, t]() {
                ready.fetch_add(1, std::memory_order_release);
                while (!start_flag.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                double result = 0;
                while (!stop_flag.load(std::memory_order_relaxed)) {
                    kernels::compute::scalar_fp64_baseline(&result, batch_iterations);
                }
            });

            unsigned core_id = (!core_ids.empty() ? core_ids[t % core_ids.size()] : t);
            pin_worker_thread(workers.back(), core_id, t, allow_affinity);
        }

        while (ready.load(std::memory_order_acquire) < num_threads_) {
            std::this_thread::yield();
        }

        auto start = std::chrono::steady_clock::now();
        auto end_time = start + std::chrono::duration<double>(seconds);
        start_flag.store(true, std::memory_order_release);

        std::this_thread::sleep_until(end_time);
        stop_flag.store(true, std::memory_order_relaxed);

        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
    // Run scalar FP64 baseline test (THE cross-arch comparable test)
    // Fixed time measurement: run for test_seconds, count total FLOPs
    ComputeTestResult run_scalar_fp64_test(unsigned threads, double test_seconds, bool pin_to_core, bool allow_affinity) {
        ComputeTestResult result;
        result.test_name = (threads == 1) ? "Single-Core" : "All-Cores";
        result.test_type = "scalar_fp64";
        result.threads = threads;
        
        // For single-thread test, pin to a performance core when available.
        if (pin_to_core && threads == 1) {
            pin_st_thread(allow_affinity);
        }
        
        if (threads == 1) {
            // Single-thread: run directly, measure time and count FLOPs
            double dummy = 0;
            size_t total_flops = 0;
            const size_t batch_iterations = 1000000;  // 1M iterations per batch
            
            compute_debug_log("[compute][st] loop start");
            auto start = std::chrono::steady_clock::now();
            auto end_time = start + std::chrono::duration<double>(test_seconds);
            
            while (std::chrono::steady_clock::now() < end_time) {
                total_flops += kernels::compute::scalar_fp64_baseline(&dummy, batch_iterations);
            }
            
            auto end = std::chrono::steady_clock::now();
            compute_debug_log("[compute][st] loop done");
            result.time_sec = std::chrono::duration<double>(end - start).count();
            result.total_flops = total_flops;
            result.iterations = total_flops / 16;  // 16 FLOPs per iteration
            result.gflops = static_cast<double>(total_flops) / result.time_sec / 1e9;
        } else {
            // Multi-thread: use direct threads for stability on large Windows systems
            std::vector<size_t> thread_flops(threads, 0);
            std::vector<double> thread_results(threads, 0);
            std::atomic<bool> stop_flag{false};
            std::atomic<bool> start_flag{false};
            std::atomic<unsigned> ready{0};
            
            const size_t batch_iterations = 1000000;

            std::vector<unsigned> core_ids = build_thread_core_ids(threads);
            std::vector<std::thread> workers;
            workers.reserve(threads);

            for (unsigned t = 0; t < threads; ++t) {
                workers.emplace_back([&, t]() {
                    ready.fetch_add(1, std::memory_order_release);
                    while (!start_flag.load(std::memory_order_acquire)) {
                        std::this_thread::yield();
                    }
                    size_t local_flops = 0;
                    while (!stop_flag.load(std::memory_order_relaxed)) {
                        local_flops += kernels::compute::scalar_fp64_baseline(&thread_results[t], batch_iterations);
                    }
                    thread_flops[t] = local_flops;
                });

                unsigned core_id = (!core_ids.empty() ? core_ids[t % core_ids.size()] : t);
                pin_worker_thread(workers.back(), core_id, t, allow_affinity);
            }

            while (ready.load(std::memory_order_acquire) < threads) {
                std::this_thread::yield();
            }

            auto start = std::chrono::steady_clock::now();
            start_flag.store(true, std::memory_order_release);

            // Wait for test duration
            std::this_thread::sleep_for(std::chrono::duration<double>(test_seconds));
            stop_flag.store(true, std::memory_order_relaxed);
            for (auto& worker : workers) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            
            auto end = std::chrono::steady_clock::now();
            result.time_sec = std::chrono::duration<double>(end - start).count();
            
            // Sum up FLOPs from all threads
            size_t total_flops = 0;
            for (unsigned t = 0; t < threads; ++t) {
                total_flops += thread_flops[t];
            }
            
            result.total_flops = total_flops;
            result.iterations = total_flops / 16;
            result.gflops = static_cast<double>(total_flops) / result.time_sec / 1e9;
        }
        
        return result;
    }
    
    // Run SIMD FP32 test (informational only - NOT for cross-arch comparison)
    ComputeTestResult run_simd_fp32_test(unsigned threads, double test_seconds, bool allow_affinity) {
        ComputeTestResult result;
        result.test_name = (threads == 1) ? "SIMD Single" : "SIMD Multi";
        result.test_type = "simd_fp32";
        result.threads = threads;
        
        // Get the appropriate SIMD kernel
        ComputeKernelFloatFn kernel = RuntimeDispatcher::get_compute_kernel_float();
        
        if (threads == 1) {
            float dummy = 0;
            size_t total_flops = 0;
            const size_t batch_iterations = 1000000;
            
            auto start = std::chrono::steady_clock::now();
            auto end_time = start + std::chrono::duration<double>(test_seconds);
            
            while (std::chrono::steady_clock::now() < end_time) {
                total_flops += kernel(&dummy, batch_iterations);
            }
            
            auto end = std::chrono::steady_clock::now();
            result.time_sec = std::chrono::duration<double>(end - start).count();
            result.total_flops = total_flops;
            result.gflops = static_cast<double>(total_flops) / result.time_sec / 1e9;
        } else {
            std::vector<size_t> thread_flops(threads, 0);
            std::vector<float> thread_results(threads, 0);
            std::atomic<bool> stop_flag{false};
            std::atomic<bool> start_flag{false};
            std::atomic<unsigned> ready{0};
            
            const size_t batch_iterations = 1000000;

            std::vector<unsigned> core_ids = build_thread_core_ids(threads);
            std::vector<std::thread> workers;
            workers.reserve(threads);

            for (unsigned t = 0; t < threads; ++t) {
                workers.emplace_back([&, t]() {
                    ready.fetch_add(1, std::memory_order_release);
                    while (!start_flag.load(std::memory_order_acquire)) {
                        std::this_thread::yield();
                    }
                    size_t local_flops = 0;
                    while (!stop_flag.load(std::memory_order_relaxed)) {
                        local_flops += kernel(&thread_results[t], batch_iterations);
                    }
                    thread_flops[t] = local_flops;
                });

                unsigned core_id = (!core_ids.empty() ? core_ids[t % core_ids.size()] : t);
                pin_worker_thread(workers.back(), core_id, t, allow_affinity);
            }

            while (ready.load(std::memory_order_acquire) < threads) {
                std::this_thread::yield();
            }

            auto start = std::chrono::steady_clock::now();
            start_flag.store(true, std::memory_order_release);

            std::this_thread::sleep_for(std::chrono::duration<double>(test_seconds));
            stop_flag.store(true, std::memory_order_relaxed);
            for (auto& worker : workers) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            
            auto end = std::chrono::steady_clock::now();
            result.time_sec = std::chrono::duration<double>(end - start).count();
            
            size_t total_flops = 0;
            for (unsigned t = 0; t < threads; ++t) {
                total_flops += thread_flops[t];
            }
            
            result.total_flops = total_flops;
            result.gflops = static_cast<double>(total_flops) / result.time_sec / 1e9;
        }
        
        return result;
    }
    
    // Calculate scores based on scalar FP64 GFLOPS (architecture-neutral)
    void calculate_scores(ComputeBenchmarkResults& results) {
        // Score formula: GFLOPS * 100
        // This gives intuitive scores where 10 GFLOPS = 1000 score
        // Architecture-neutral: same formula for x86 and ARM
        const double SCORE_MULTIPLIER = 100.0;
        
        // ST score: direct GFLOPS scaling
        results.st_score = static_cast<int>(results.single_thread.gflops * SCORE_MULTIPLIER);
        
        // MT score: direct GFLOPS scaling
        results.mt_score = static_cast<int>(results.multi_thread.gflops * SCORE_MULTIPLIER);
        
        // Overall score: weighted combination
        // ST weight 30%, MT weight 70% (MT shows full system capability)
        results.overall_score = static_cast<int>(results.st_score * 0.3 + results.mt_score * 0.7);
    }
};
