// CPU Benchmark - Auto-Size Detector Implementation


#include "auto_size.hpp"
#include <cmath>
#include <algorithm>

// Get bytes per element for precision
double AutoSizeDetector::get_bytes_per_element(Precision precision) {
    switch (precision) {
        case Precision::Double:
        case Precision::FP64:
            return 8.0;
        case Precision::Float:
            return 4.0;
        case Precision::FP16:
            return 2.0;
        case Precision::INT8:
            return 1.0;
        case Precision::FP4:
            return 0.5;
    }
    return 4.0;  // Default to float
}

// Calculate cubic dimension from total elements
size_t AutoSizeDetector::calculate_cubic_dimension(size_t total_elements) {
    // Calculate cube root
    double cube_root = std::cbrt(static_cast<double>(total_elements));
    size_t dim = static_cast<size_t>(cube_root);

    // Correct for rounding error so dim^3 stays within total_elements
    while ((dim + 1) <= 500) {
        size_t next = dim + 1;
        size_t next_elements = next * next * next;
        if (next_elements > total_elements) {
            break;
        }
        dim = next;
    }
    
    // Ensure minimum dimension of 4 for stencil operations
    dim = (std::max)(dim, static_cast<size_t>(4));
    
    // Limit maximum dimension to 500 to prevent excessive computation time
    dim = (std::min)(dim, static_cast<size_t>(500));
    
    return dim;
}

// Calculate memory required for a given size and precision
// Benchmark uses 3 arrays (A, B, C) of the same size
size_t AutoSizeDetector::calculate_memory_required(
    const Size3D& size,
    Precision precision) {
    
    double bytes_per_elem = get_bytes_per_element(precision);
    size_t total_elements = size.total();
    
    // 3 arrays for benchmark (A, B, C)
    const size_t num_arrays = 3;
    
    return static_cast<size_t>(total_elements * bytes_per_elem * num_arrays);
}


// TestSize from total elements
TestSize AutoSizeDetector::create_test_size(
    size_t total_elements,
    Precision precision,
    const std::string& description,
    const std::string& cache_target) {
    
    TestSize result;
    size_t dim = calculate_cubic_dimension(total_elements);
    
    result.size = {dim, dim, dim};
    result.memory_bytes = calculate_memory_required(result.size, precision);
    result.description = description;
    result.cache_target = cache_target;
    
    return result;
}

// Get available memory in bytes
size_t AutoSizeDetector::get_available_memory() {
    size_t available = get_available_ram();
    
    // If detection fails, try total RAM
    if (available == 0) {
        available = get_total_ram();
    }
    
    // If still zero, use conservative default (4GB)
    if (available == 0) {
        available = 4ULL * 1024 * 1024 * 1024;
    }
    
    return available;
}

// Get cache info
CacheInfo AutoSizeDetector::get_cache_info() {
    return ::get_cache_info();
}

// Check if explicit size is safe to use
bool AutoSizeDetector::is_size_safe(
    const Size3D& size,
    Precision precision,
    const AutoSizeConfig& config) {
    
    size_t required = calculate_memory_required(size, precision);
    size_t available = get_available_memory();
    size_t safe_limit = static_cast<size_t>(available * (1.0 - config.safety_margin));
    
    return required <= safe_limit;
}


// Detect optimal size for general benchmarking 
TestSize AutoSizeDetector::detect_optimal_size(
    Precision precision,
    const AutoSizeConfig& config) {
    
    size_t available = get_available_memory();
    
    // Apply safety margin - use at most (1 - safety_margin) of available RAM

    size_t safe_memory = static_cast<size_t>(available * (1.0 - config.safety_margin));
    
    // Calculate max elements that fit in safe memory
    double bytes_per_elem = get_bytes_per_element(precision);
    const size_t num_arrays = 3;  // A, B, C arrays
    
    size_t max_elements = static_cast<size_t>(
        safe_memory / (bytes_per_elem * num_arrays));
    
    // Clamp to configured limits
    max_elements = (std::min)(max_elements, config.max_elements);
    max_elements = (std::max)(max_elements, config.min_elements);
    
    // Determine description based on size
    std::string description;
    size_t dim = calculate_cubic_dimension(max_elements);
    
    if (dim <= 64) {
        description = "Small";
    } else if (dim <= 256) {
        description = "Medium";
    } else if (dim <= 512) {
        description = "Large";
    } else {
        description = "Extra Large";
    }
    
    return create_test_size(max_elements, precision, description, "RAM");
}

// Get size that fits in specific cache level 
TestSize AutoSizeDetector::get_size_for_cache_level(
    int cache_level,
    Precision precision,
    const AutoSizeConfig& config) {
    
    CacheInfo cache = get_cache_info();
    size_t cache_size = 0;
    std::string cache_target;
    
    switch (cache_level) {
        case 1:
            cache_size = cache.l1_data_size;
            cache_target = "L1";
            break;
        case 2:
            cache_size = cache.l2_size;
            cache_target = "L2";
            break;
        case 3:
            cache_size = cache.l3_size;
            cache_target = "L3";
            break;
        default:
            cache_size = cache.l2_size;  // Default to L2
            cache_target = "L2";
            break;
    }

    
    // Use fallback values if cache detection failed
    if (cache_size == 0) {
        switch (cache_level) {
            case 1: cache_size = 32 * 1024; break;       // 32KB default L1
            case 2: cache_size = 256 * 1024; break;      // 256KB default L2
            case 3: cache_size = 8 * 1024 * 1024; break; // 8MB default L3
            default: cache_size = 256 * 1024; break;
        }
    }
    
    // Apply cache utilization factor (default 75%)
    size_t usable_cache = static_cast<size_t>(cache_size * config.cache_utilization);
    
    // Calculate max elements that fit
    double bytes_per_elem = get_bytes_per_element(precision);
    const size_t num_arrays = 3;
    
    size_t max_elements = static_cast<size_t>(
        usable_cache / (bytes_per_elem * num_arrays));
    
    // Clamp to minimum
    max_elements = (std::max)(max_elements, config.min_elements);
    
    std::string description = cache_target + "-fit";
    
    return create_test_size(max_elements, precision, description, cache_target);
}

// Get size for RAM-bound test (larger than L3)
TestSize AutoSizeDetector::get_size_for_ram_bound(
    Precision precision,
    const AutoSizeConfig& config) {
    
    CacheInfo cache = get_cache_info();
    size_t l3_size = cache.l3_size;
    
    // Use fallback if L3 detection failed
    if (l3_size == 0) {
        l3_size = 8 * 1024 * 1024;  // 8MB default
    }
    
    // Target 4x L3 size for RAM-bound test
    size_t target_memory = l3_size * 4;
    
    // But don't exceed safe RAM limit
    size_t available = get_available_memory();
    size_t safe_memory = static_cast<size_t>(available * (1.0 - config.safety_margin));
    target_memory = (std::min)(target_memory, safe_memory);
    
    // Calculate elements
    double bytes_per_elem = get_bytes_per_element(precision);
    const size_t num_arrays = 3;
    
    size_t max_elements = static_cast<size_t>(
        target_memory / (bytes_per_elem * num_arrays));
    
    max_elements = (std::max)(max_elements, config.min_elements);
    
    return create_test_size(max_elements, precision, "RAM-bound", "RAM");
}
