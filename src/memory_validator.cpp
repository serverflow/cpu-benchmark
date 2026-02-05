// CPU Benchmark - Memory Validator Implementation


#include "memory_validator.hpp"
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdint>


#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#elif defined(__linux__)
    #include <sys/sysinfo.h>
    #include <fstream>
#elif defined(__APPLE__)
    #include <sys/sysctl.h>
    #include <mach/mach.h>
#endif

// ============================================================================
// MemoryRequirement Implementation
// ============================================================================

std::string MemoryRequirement::to_string() const {
    std::ostringstream oss;
    oss << "Required: " << MemoryValidator::format_bytes(required_bytes)
        << ", Available: " << MemoryValidator::format_bytes(available_bytes)
        << ", Status: " << (sufficient ? "OK" : "INSUFFICIENT");
    return oss.str();
}

// ============================================================================
// MemoryValidator Implementation
// ============================================================================

// Get bytes per element for a given precision
double MemoryValidator::get_bytes_per_element(Precision precision) {
    PrecisionConfig config = get_precision_config(precision);
    return config.bytes_per_element;
}

// Estimate memory requirement 
// Formula: num_arrays * size.total() * bytes_per_element
size_t MemoryValidator::estimate_required_memory(
    const Size3D& size,
    Precision precision,
    int num_arrays
) {
    // Get bytes per element from precision config
    double bytes_per_element = get_bytes_per_element(precision);
    
    // Calculate total elements (may throw overflow_error)
    size_t total_elements = size.total();
    
    // Calculate bytes for one array
    size_t bytes_per_array = size.total_bytes(bytes_per_element);
    
    // Check overflow for num_arrays multiplication
    if (num_arrays > 0 && bytes_per_array > SIZE_MAX / static_cast<size_t>(num_arrays)) {
        throw std::overflow_error(
            "Memory requirement overflow: " + std::to_string(num_arrays) + 
            " arrays * " + std::to_string(bytes_per_array) + " bytes"
        );
    }
    
    return static_cast<size_t>(num_arrays) * bytes_per_array;
}

// ============================================================================
// Platform-specific get_available_memory implementations
// ============================================================================

#ifdef _WIN32
// Windows: Use GlobalMemoryStatusEx
size_t MemoryValidator::get_available_memory() {
    MEMORYSTATUSEX mem_info;
    mem_info.dwLength = sizeof(MEMORYSTATUSEX);
    
    if (GlobalMemoryStatusEx(&mem_info)) {
        // Return available physical memory
        return static_cast<size_t>(mem_info.ullAvailPhys);
    }
    
    // Fallback: return 0 if detection fails
    return 0;
}

#elif defined(__linux__)
// Linux: Use sysinfo or /proc/meminfo
size_t MemoryValidator::get_available_memory() {
    // Try sysinfo first
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        // Calculate available memory: free + buffers + cached
        // sysinfo provides freeram, but for better accuracy we use /proc/meminfo
        size_t available = static_cast<size_t>(info.freeram) * info.mem_unit;
        
        // Try to get more accurate "MemAvailable" from /proc/meminfo
        std::ifstream meminfo("/proc/meminfo");
        if (meminfo.is_open()) {
            std::string line;
            while (std::getline(meminfo, line)) {
                if (line.find("MemAvailable:") == 0) {
                    // Parse "MemAvailable:    12345678 kB"
                    size_t kb = 0;
                    if (sscanf(line.c_str(), "MemAvailable: %zu kB", &kb) == 1) {
                        return kb * 1024;  // Convert to bytes
                    }
                }
            }
        }
        
        // Fallback to sysinfo freeram
        return available;
    }
    
    return 0;
}

#elif defined(__APPLE__)
// macOS: Use sysctl and mach APIs
size_t MemoryValidator::get_available_memory() {
    // Get total physical memory
    int64_t total_mem = 0;
    size_t len = sizeof(total_mem);
    
    if (sysctlbyname("hw.memsize", &total_mem, &len, nullptr, 0) != 0) {
        return 0;
    }
    
    // Get VM statistics for free pages
    vm_size_t page_size;
    mach_port_t mach_port = mach_host_self();
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = sizeof(vm_stats) / sizeof(natural_t);
    
    if (host_page_size(mach_port, &page_size) != KERN_SUCCESS) {
        // Fallback: assume 50% of total memory is available
        return static_cast<size_t>(total_mem / 2);
    }
    
    if (host_statistics64(mach_port, HOST_VM_INFO64, 
                          reinterpret_cast<host_info64_t>(&vm_stats), 
                          &count) != KERN_SUCCESS) {
        return static_cast<size_t>(total_mem / 2);
    }
    
    // Calculate available memory: free + inactive pages
    size_t available = (vm_stats.free_count + vm_stats.inactive_count) * page_size;
    return available;
}

#else
// Unknown platform: return 0 (detection not supported)
size_t MemoryValidator::get_available_memory() {
    return 0;
}
#endif

// Validate memory before allocation
MemoryRequirement MemoryValidator::validate(
    const Size3D& size,
    Precision precision,
    int num_arrays
) {
    MemoryRequirement result;
    
    // Estimate required memory (may throw on overflow)
    result.required_bytes = estimate_required_memory(size, precision, num_arrays);
    
    // Get available memory
    result.available_bytes = get_available_memory();
    
    // Check if sufficient
    // If available_bytes is 0, we couldn't detect memory - assume sufficient
    // to avoid blocking on platforms where detection isn't supported
    if (result.available_bytes == 0) {
        result.sufficient = true;  // Can't detect, assume OK
    } else {
        result.sufficient = (result.available_bytes >= result.required_bytes);
    }
    
    return result;
}

// Format bytes as human-readable string
std::string MemoryValidator::format_bytes(size_t bytes) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    
    const double KB = 1024.0;
    const double MB = KB * 1024.0;
    const double GB = MB * 1024.0;
    const double TB = GB * 1024.0;
    
    double value = static_cast<double>(bytes);
    
    if (value >= TB) {
        oss << (value / TB) << " TB";
    } else if (value >= GB) {
        oss << (value / GB) << " GB";
    } else if (value >= MB) {
        oss << (value / MB) << " MB";
    } else if (value >= KB) {
        oss << (value / KB) << " KB";
    } else {
        oss << bytes << " B";
    }
    
    return oss.str();
}
