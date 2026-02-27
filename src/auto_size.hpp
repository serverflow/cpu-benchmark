// CPU Benchmark - Auto-Size Detector
// Automatically determines optimal test sizes based on system resources

#pragma once

#include "types.hpp"
#include "platform.hpp"
#include <string>
#include <cstddef>

// Test size description structure
struct TestSize {
    Size3D size;                // 3D dimensions
    size_t memory_bytes;        // Total memory required in bytes
    std::string description;    // Human-readable description ("Small", "Medium", "Large")
    std::string cache_target;   // Target cache level ("L1", "L2", "L3", "RAM")
};

// Auto-size configuration
struct AutoSizeConfig {
    double safety_margin;       // Fraction of RAM to leave free (default: 0.5 = 50%)
    double cache_utilization;   // Fraction of cache to use (default: 0.75 = 75%)
    size_t min_elements;        // Minimum number of elements (default: 8*8*8 = 512)
    size_t max_elements;        // Maximum number of elements (default: 2048^3)
    
    AutoSizeConfig()
        : safety_margin(0.5)
        , cache_utilization(0.75)
        , min_elements(512)
        , max_elements(500ULL * 500ULL * 500ULL) {}  // Max 500x500x500
};

// Auto-Size Detector class
// Determines optimal test sizes based on available RAM and cache
class AutoSizeDetector {
public:
    // Detect optimal size for general benchmarking
    // Uses available RAM with safety margin
    static TestSize detect_optimal_size(
        Precision precision = Precision::Float,
        const AutoSizeConfig& config = AutoSizeConfig());
    
    // Get size that fits in specific cache level
    static TestSize get_size_for_cache_level(
        int cache_level,  // 1=L1, 2=L2, 3=L3
        Precision precision = Precision::Float,
        const AutoSizeConfig& config = AutoSizeConfig());
    
    // Get size for RAM-bound test (larger than L3)
    static TestSize get_size_for_ram_bound(
        Precision precision = Precision::Float,
        const AutoSizeConfig& config = AutoSizeConfig());
    
    // Check if explicit size is safe to use
    // Returns true if size fits in available memory with safety margin
    static bool is_size_safe(
        const Size3D& size,
        Precision precision,
        const AutoSizeConfig& config = AutoSizeConfig());
    
    // Get available memory in bytes (Requirement 7.1)
    static size_t get_available_memory();
    
    // Get cache info
    static CacheInfo get_cache_info();
    
    // Calculate memory required for a given size and precision
    static size_t calculate_memory_required(
        const Size3D& size,
        Precision precision);
    
private:
    // Calculate cubic dimension from total elements
    static size_t calculate_cubic_dimension(size_t total_elements);
    
    // Get bytes per element for precision
    static double get_bytes_per_element(Precision precision);
    
    // Create TestSize from total elements
    static TestSize create_test_size(
        size_t total_elements,
        Precision precision,
        const std::string& description,
        const std::string& cache_target);
};

