// CPU Benchmark - Warmup Manager Implementation
// Manages CPU warmup phase before benchmark measurements

#include "warmup.hpp"
#include <cmath>
#include <thread>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iomanip>

// Static member initialization
std::atomic<bool> WarmupManager::warmup_done_{false};
WarmupResult WarmupManager::last_result_{false, 0.0, 0.0, false, "Not performed"};

// Perform warmup with given configuration
WarmupResult WarmupManager::perform_warmup(const WarmupConfig& config) {
    WarmupResult result;
    result.performed = false;
    result.duration_seconds = 0.0;
    result.final_frequency_estimate = 0.0;
    result.frequency_stable = false;
    result.status_message = "Not performed";
    
    // Check if warmup is enabled 
    if (!config.enabled) {
        result.status_message = "Warmup disabled by configuration";
        last_result_ = result;
        return result;
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Run compute-intensive warmup loop 
    // Must run for at least the specified duration (default 2 seconds)
    double throughput = run_compute_load(config.duration);
    
    auto after_load = std::chrono::steady_clock::now();
    double load_duration = std::chrono::duration<double>(after_load - start_time).count();
    
    // Wait for frequency stabilization if requested 
    bool stable = true;
    if (config.wait_for_stable_frequency) {
        stable = wait_for_frequency_stabilization(
            config.stabilization_iterations,
            config.stability_threshold);
    }
    
    auto end_time = std::chrono::steady_clock::now();
    double total_duration = std::chrono::duration<double>(end_time - start_time).count();
    
    // Measure final throughput
    double final_throughput = measure_throughput();
    
    // Build result
    result.performed = true;
    result.duration_seconds = total_duration;
    result.final_frequency_estimate = final_throughput;
    result.frequency_stable = stable;
    
    std::ostringstream oss;
    oss << "Warmup completed in " << std::fixed << std::setprecision(2) 
        << total_duration << "s";
    if (stable) {
        oss << " (frequency stable)";
    } else {
        oss << " (frequency may not be stable)";
    }
    result.status_message = oss.str();
    
    // Update static state
    warmup_done_.store(true);
    last_result_ = result;
    
    return result;
}

// Check if warmup was performed 
bool WarmupManager::was_warmup_performed() {
    return warmup_done_.load();
}

// Get the last warmup result
WarmupResult WarmupManager::get_last_result() {
    return last_result_;
}

// Reset warmup state (for testing)
void WarmupManager::reset() {
    warmup_done_.store(false);
    last_result_ = WarmupResult{false, 0.0, 0.0, false, "Not performed"};
}

// Get default configuration
WarmupConfig WarmupManager::get_default_config() {
    return WarmupConfig{};
}

// Format warmup result as string
std::string WarmupManager::format_result(const WarmupResult& result) {
    std::ostringstream oss;
    oss << "Warmup: ";
    if (result.performed) {
        oss << "Yes (" << std::fixed << std::setprecision(2) 
            << result.duration_seconds << "s)";
        if (result.frequency_stable) {
            oss << " [stable]";
        }
    } else {
        oss << "No";
    }
    return oss.str();
}

// Run compute-intensive load for specified duration 
// Uses FMA-like operations to stress the CPU and trigger turbo boost
double WarmupManager::run_compute_load(std::chrono::seconds duration) {
    auto start = std::chrono::steady_clock::now();
    auto end_time = start + duration;
    
    // Use volatile to prevent compiler optimization
    volatile double result = 1.0;
    volatile double a = 1.0000001;
    volatile double b = 0.9999999;
    
    size_t iterations = 0;
    
    // Run compute-intensive loop until duration is reached
    while (std::chrono::steady_clock::now() < end_time) {
        // Perform many FMA-like operations per iteration
        // This stresses the FPU and should trigger turbo boost
        for (int i = 0; i < 10000; ++i) {
            result = result * a + b;
            result = result * b + a;
            result = result * a + b;
            result = result * b + a;
        }
        iterations++;
    }
    
    auto actual_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(actual_end - start).count();
    
    // Return throughput in operations per second
    // Each iteration does 40000 FMA-like ops
    return (iterations * 40000.0) / elapsed;
}

// Wait for CPU frequency to stabilize
bool WarmupManager::wait_for_frequency_stabilization(
    unsigned iterations, 
    double threshold) {
    
    if (iterations < 2) {
        return true;  // Can't check stability with less than 2 samples
    }
    
    std::vector<double> throughputs;
    throughputs.reserve(iterations);
    
    // Measure throughput multiple times
    for (unsigned i = 0; i < iterations; ++i) {
        double t = measure_throughput();
        throughputs.push_back(t);
        
        // Small delay between measurements
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Calculate mean and variance
    double sum = 0.0;
    for (double t : throughputs) {
        sum += t;
    }
    double mean = sum / throughputs.size();
    
    if (mean <= 0.0) {
        return false;
    }
    
    // Check if all measurements are within threshold of mean
    for (double t : throughputs) {
        double deviation = std::abs(t - mean) / mean;
        if (deviation > threshold) {
            return false;
        }
    }
    
    return true;
}

// Measure current compute throughput (ops/sec)
double WarmupManager::measure_throughput() {
    const int measurement_iterations = 1000;
    
    volatile double result = 1.0;
    volatile double a = 1.0000001;
    volatile double b = 0.9999999;
    
    auto start = std::chrono::steady_clock::now();
    
    for (int iter = 0; iter < measurement_iterations; ++iter) {
        for (int i = 0; i < 1000; ++i) {
            result = result * a + b;
            result = result * b + a;
        }
    }
    
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    if (elapsed <= 0.0) {
        return 0.0;
    }
    
    // Return ops per second (2000 ops per inner iteration * measurement_iterations)
    return (measurement_iterations * 2000.0) / elapsed;
}
