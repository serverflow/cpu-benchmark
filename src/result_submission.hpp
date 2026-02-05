#pragma once
// CPU Benchmark - Result Submission Module
// Handles optional submission of benchmark results to a remote server

#include "types.hpp"
#include "platform.hpp"
#include <string>

// Result of submission attempt
struct SubmissionResult {
    bool success;
    std::string result_id;      // UUID assigned by server (if success)
    std::string error_message;  // Error description (if failure)
};

// CPU frequency data collected during benchmark
struct FrequencyData {
    double min_mhz;             // Minimum observed frequency
    double max_mhz;             // Maximum observed frequency
    double avg_mhz;             // Average frequency during test
    bool available;             // True if frequency data was collected
    
    FrequencyData() : min_mhz(0), max_mhz(0), avg_mhz(0), available(false) {}
};

// Compute benchmark data for submission
struct ComputeSubmissionData {
    double st_time_sec;
    double st_gflops;
    int st_score;
    int st_threads;
    
    double mt_time_sec;
    double mt_gflops;
    int mt_score;
    int mt_threads;
    
    int overall_score;
    std::string simd_level;
    
    // Frequency data during test
    FrequencyData frequency;
    
    // Socket info
    int socket_count;
    int selected_socket;  // -1 = all sockets
    
    // OS version
    std::string os_version;
    
    // Session ID for linking results from same benchmark run
    std::string session_id;
    
    ComputeSubmissionData() 
        : st_time_sec(0), st_gflops(0), st_score(0), st_threads(0)
        , mt_time_sec(0), mt_gflops(0), mt_score(0), mt_threads(0)
        , overall_score(0), socket_count(1), selected_socket(-1) {}
};

// Single precision result for submission
struct PrecisionResultData {
    std::string precision_name;  // "fp64", "fp32", "fp16", "int8", "fp4"
    std::string fp16_mode;       // "native" or "emulated" (only for fp16)
    double bytes_per_element;
    bool is_integer;
    bool is_emulated;
    
    double time_min_sec;
    double time_avg_sec;
    double time_stddev_sec;
    double gflops_avg;
    double gflops_max;
    uint64_t total_flops;
    uint64_t iterations;
    
    // Frequency data during test
    FrequencyData frequency;
};

// Precision=all benchmark data for submission (all precision types)
struct PrecisionAllSubmissionData {
    std::vector<PrecisionResultData> results;  // Results for each precision type
    
    // Socket info
    int socket_count;
    int selected_socket;  // -1 = all sockets
    
    // OS version
    std::string os_version;
    
    // Session ID for linking results from same benchmark run
    std::string session_id;
    
    PrecisionAllSubmissionData() : socket_count(1), selected_socket(-1) {}
};

// Full benchmark suite data for submission (all tests combined)
struct FullBenchmarkSubmissionData {
    ComputeSubmissionData compute;           // --mode=compute results
    PrecisionAllSubmissionData precision;    // --precision=all results
    BenchmarkResult mem;                     // --mode=mem results
    BenchmarkResult stencil;                 // --mode=stencil results
    // Cache results stored as BenchmarkResult (best level)
    BenchmarkResult cache;
    
    // Per-test frequency data
    FrequencyData compute_frequency;
    FrequencyData mem_frequency;
    FrequencyData stencil_frequency;
    FrequencyData cache_frequency;
    
    // Socket info
    int socket_count;
    int selected_socket;  // -1 = all sockets
    
    // OS version (detected once at start)
    std::string os_version;
    
    FullBenchmarkSubmissionData() : socket_count(1), selected_socket(-1) {}
};

// Main entry point - prompts user and submits if confirmed
// Returns true if user confirmed and submission succeeded
// Only call in text output mode after benchmark completion
// mode_override: optional mode name to override config.mode (e.g., "precision_all")
bool submit_results_interactive(
    const BenchmarkResult& result,
    const Config& config,
    const CpuInfo& cpu_info,
    const std::string& server_url = "http://5.129.211.35:8080",
    const std::string& mode_override = ""
);

// Submit compute benchmark results (with ST/MT scores) - interactive version
bool submit_compute_results_interactive(
    const ComputeSubmissionData& compute_data,
    const Config& config,
    const CpuInfo& cpu_info,
    const std::string& server_url = "http://5.129.211.35:8080"
);

// Submit compute benchmark results (with ST/MT scores) - non-interactive version
bool submit_compute_results(
    const ComputeSubmissionData& compute_data,
    const Config& config,
    const CpuInfo& cpu_info,
    const std::string& nickname,
    const std::string& server_url = "http://5.129.211.35:8080"
);

// Submit precision=all benchmark results - interactive version
bool submit_precision_all_results_interactive(
    const PrecisionAllSubmissionData& precision_data,
    const Config& config,
    const CpuInfo& cpu_info,
    const std::string& server_url = "http://5.129.211.35:8080"
);

// Submit precision=all benchmark results - non-interactive version
bool submit_precision_all_results(
    const PrecisionAllSubmissionData& precision_data,
    const Config& config,
    const CpuInfo& cpu_info,
    const std::string& nickname,
    const std::string& server_url = "http://5.129.211.35:8080"
);

// Submit full benchmark suite results (all tests combined)
bool submit_full_benchmark_results_interactive(
    const FullBenchmarkSubmissionData& full_data,
    const Config& config,
    const CpuInfo& cpu_info,
    const std::string& server_url = "http://5.129.211.35:8080"
);

// Internal functions exposed for testing
namespace result_submission {

// Build JSON payload from benchmark data (with optional nickname and mode override)
std::string build_payload(const BenchmarkResult& result, const Config& config, const CpuInfo& cpu_info, 
                          const std::string& nickname = "Anonymous", const std::string& mode_override = "");

// Generate hardware fingerprint from CPU characteristics
std::string generate_hardware_fingerprint(const CpuInfo& cpu_info);

// Generate integrity token from execution parameters (with optional mode override)
std::string generate_integrity_token(const BenchmarkResult& result, const Config& config, const std::string& hw_fingerprint, const std::string& mode_override = "");

} // namespace result_submission
