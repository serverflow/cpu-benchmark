#pragma once
// CPU Benchmark - Memory Validator


#include "types.hpp"
#include <cstddef>
#include <string>

// Memory requirement result structure 
struct MemoryRequirement {
    size_t required_bytes;      // Total memory required for benchmark
    size_t available_bytes;     // Available system memory
    bool sufficient;            // True if available >= required
    
    // Human-readable description
    std::string to_string() const;
};

// Memory Validator class
// Validates that sufficient memory is available before benchmark allocation
class MemoryValidator {
public:
    // Estimate memory requirement for benchmark 
    // Calculates: num_arrays * size.total() * bytes_per_element
    // 
    // Parameters:
    //   size - 3D dimensions of arrays
    //   precision - Data type precision (determines bytes_per_element)
    //   num_arrays - Number of arrays to allocate (default 3: A, B, C)
    //
    // Returns: Total bytes required
    // Throws: std::overflow_error if calculation would overflow
    static size_t estimate_required_memory(
        const Size3D& size,
        Precision precision,
        int num_arrays = 3
    );
    
    // Query available system memory
    // Platform-specific implementation:
    //   Windows: GlobalMemoryStatusEx
    //   Linux: sysinfo or /proc/meminfo
    //   macOS: sysctl hw.memsize
    //
    // Returns: Available memory in bytes, or 0 if detection fails
    static size_t get_available_memory();
    
    // Validate memory before allocation 
    // Compares required memory against available memory
    //
    // Parameters:
    //   size - 3D dimensions of arrays
    //   precision - Data type precision
    //   num_arrays - Number of arrays (default 3)
    //
    // Returns: MemoryRequirement with validation result
    static MemoryRequirement validate(
        const Size3D& size,
        Precision precision,
        int num_arrays = 3
    );
    
    // Get bytes per element for a given precision
    static double get_bytes_per_element(Precision precision);
    
    // Format bytes as human-readable string (e.g., "1.5 GB")
    static std::string format_bytes(size_t bytes);
};

