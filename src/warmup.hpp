// CPU Benchmark - Warmup Manager
// Manages CPU warmup phase before benchmark measurements

#pragma once

#include <chrono>
#include <string>
#include <atomic>

// Warmup configuration structure
struct WarmupConfig {
    std::chrono::seconds duration{2};       // Minimum warmup duration 
    bool enabled{true};                      // Whether warmup is enabled 
    bool wait_for_stable_frequency{true};    // Wait for CPU frequency stabilization 
    unsigned stabilization_iterations{3};    // Number of iterations to check for stability
    double stability_threshold{0.05};        // 5% variance threshold for stability
};

// Warmup result structure
struct WarmupResult {
    bool performed;                          // Whether warmup was actually performed
    double duration_seconds;                 // Actual warmup duration
    double final_frequency_estimate;         // Estimated CPU frequency after warmup (ops/sec)
    bool frequency_stable;                   // Whether frequency appeared stable
    std::string status_message;              // Human-readable status
};

// Warmup Manager class
class WarmupManager {
public:
    // Perform warmup with given configuration 
    // Returns result indicating what was done
    static WarmupResult perform_warmup(const WarmupConfig& config = WarmupConfig{});
    
    // Check if warmup was performed in this session
    static bool was_warmup_performed();
    
    // Get the last warmup result
    static WarmupResult get_last_result();
    
    // Reset warmup state (for testing)
    static void reset();
    
    // Get default configuration
    static WarmupConfig get_default_config();
    
    // Format warmup result as string for display
    static std::string format_result(const WarmupResult& result);
    
private:
    // Run compute-intensive load for specified duration
    static double run_compute_load(std::chrono::seconds duration);
    
    // Wait for CPU frequency to stabilize
    static bool wait_for_frequency_stabilization(
        unsigned iterations, 
        double threshold);
    
    // Measure current compute throughput (ops/sec)
    static double measure_throughput();
    
    // Static state
    static std::atomic<bool> warmup_done_;
    static WarmupResult last_result_;
};
