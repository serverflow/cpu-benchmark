#pragma once
// CPU Benchmark - Platform detection header


#include <string>
#include <cstddef>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

// Cache information structure
struct CacheInfo {
    size_t l1_data_size;        // L1 data cache size in bytes
    size_t l1_inst_size;        // L1 instruction cache size in bytes
    size_t l2_size;             // L2 cache size in bytes
    size_t l3_size;             // L3 cache size in bytes
    size_t cache_line_size;     // Cache line size in bytes
    
    // Availability flags for each cache level
    bool l1_available;
    bool l2_available;
    bool l3_available;
    
    // Default constructor - initialize all to zero/false
    CacheInfo()
        : l1_data_size(0)
        , l1_inst_size(0)
        , l2_size(0)
        , l3_size(0)
        , cache_line_size(0)
        , l1_available(false)
        , l2_available(false)
        , l3_available(false) {}
};

// CPU frequency information structure
struct CpuFrequencyInfo {
    double min_mhz;             // Minimum frequency in MHz
    double max_mhz;             // Maximum frequency in MHz  
    double current_mhz;         // Current frequency in MHz (at time of query)
    double base_mhz;            // Base/nominal frequency in MHz
    bool available;             // True if frequency info was successfully retrieved
    
    CpuFrequencyInfo()
        : min_mhz(0.0)
        , max_mhz(0.0)
        , current_mhz(0.0)
        , base_mhz(0.0)
        , available(false) {}
};

// Per-socket CPU information (for multi-socket systems)
struct SocketInfo {
    unsigned socket_id;         // Socket/package ID (0-based)
    std::string model;          // CPU model name for this socket
    unsigned logical_cores;     // Logical cores on this socket
    unsigned physical_cores;    // Physical cores on this socket
    std::vector<unsigned> core_ids;  // List of core IDs belonging to this socket
};

// CPU information structure 
struct CpuInfo {
    std::string arch;           // "x86_64", "arm64", "unknown"
    unsigned logical_cores;     // Number of logical cores (total)
    unsigned physical_cores;    // Number of physical cores (total)
    std::string vendor;         // CPU vendor (Intel, AMD, ARM, etc.)
    std::string model;          // CPU model name (or "model1; model2" for multi-socket)
    CacheInfo cache;            // Cache information
    
    // Multi-socket support
    unsigned socket_count;      // Number of CPU sockets (1 for most systems)
    std::vector<SocketInfo> sockets;  // Per-socket information
    
    CpuInfo()
        : logical_cores(0)
        , physical_cores(0)
        , socket_count(1) {}
};

// Get CPU information 
// Detects architecture, core count, vendor, and model
CpuInfo get_cpu_info();

// Get total logical core count
// On Windows, includes all processor groups (>64 logical CPUs).
unsigned get_logical_core_count();

// Get operating system name 
// Returns "Linux", "Windows", or "Unknown"
std::string get_os_name();

// Get detailed OS version string
// Returns e.g. "Windows 10 Build 19045", "Ubuntu 22.04.3 LTS", "macOS 14.2"
std::string get_os_version();

// Get compiler information
// Returns compiler name and version
std::string get_compiler_info();

// Get architecture string from preprocessor macros 
// Uses _WIN32, __linux__, __x86_64__, __aarch64__
std::string get_arch_string();

// Get cache information 
// Detects L1, L2, L3 cache sizes and cache line size
CacheInfo get_cache_info();

// Get CPU frequency information
// Returns min/max/current/base frequencies
CpuFrequencyInfo get_cpu_frequency();

// Sample CPU frequency during benchmark execution
// Call this periodically during tests to track frequency changes
// Returns current frequency in MHz, or 0 if not available
double sample_current_frequency();

// Get number of CPU sockets in the system
unsigned get_socket_count();

// Get list of core IDs belonging to a specific socket
// socket_id: 0-based socket index
// Returns empty vector if socket doesn't exist or info unavailable
std::vector<unsigned> get_cores_for_socket(unsigned socket_id);

// Get list of performance core IDs (P-cores) for hybrid architectures
// On Intel 12th Gen+ (Alder Lake, Raptor Lake): returns P-cores only
// On non-hybrid CPUs: returns all cores
// Returns empty vector if detection fails
std::vector<unsigned> get_performance_cores();

// Get the best performance core ID for single-threaded workloads
// Returns the first P-core on hybrid architectures, or core 0 as fallback
unsigned get_best_performance_core();

// Get fallback CPU name when detection fails 
// Returns string in format "{arch} {cores}C/{threads}T"
std::string get_fallback_cpu_name(const std::string& arch, unsigned physical_cores, unsigned logical_cores);

// Get total system RAM in bytes 
// Returns 0 if detection fails
size_t get_total_ram();

// Get available system RAM in bytes 
// Returns 0 if detection fails
size_t get_available_ram();

// ============================================================================
// Frequency Sampling During Benchmark
// ============================================================================

// Collected frequency statistics during a benchmark run
struct FrequencyStats {
    double min_mhz;
    double max_mhz;
    double avg_mhz;
    unsigned sample_count;
    bool available;
    
    FrequencyStats()
        : min_mhz(0.0)
        , max_mhz(0.0)
        , avg_mhz(0.0)
        , sample_count(0)
        , available(false) {}
};

// Class to sample CPU frequency during benchmark execution
// Supports both manual sampling and background thread sampling
class FrequencySampler {
public:
    FrequencySampler() : sum_mhz_(0.0), min_mhz_(1e9), max_mhz_(0.0), sample_count_(0),
                         running_(false) {}
    
    ~FrequencySampler() {
        stop_background();
    }
    
    // Take a frequency sample (call periodically during test)
    void sample() {
        double freq = sample_current_frequency();
        if (freq > 0.0) {
            std::lock_guard<std::mutex> lock(mutex_);
            sum_mhz_ += freq;
            if (freq < min_mhz_) min_mhz_ = freq;
            if (freq > max_mhz_) max_mhz_ = freq;
            ++sample_count_;
        }
    }
    
    // Start background sampling thread (samples every interval_ms milliseconds)
    void start_background(unsigned interval_ms = 100) {
        if (running_.load()) return;
        running_.store(true);
        sample_thread_ = std::thread([this, interval_ms]() {
            while (running_.load()) {
                sample();
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
            }
        });
    }
    
    // Stop background sampling thread
    void stop_background() {
        if (running_.load()) {
            running_.store(false);
            if (sample_thread_.joinable()) {
                sample_thread_.join();
            }
        }
    }
    
    // Get collected statistics
    FrequencyStats get_stats() const {
        FrequencyStats stats;
        std::lock_guard<std::mutex> lock(mutex_);
        if (sample_count_ > 0) {
            stats.min_mhz = min_mhz_;
            stats.max_mhz = max_mhz_;
            stats.avg_mhz = sum_mhz_ / sample_count_;
            stats.sample_count = sample_count_;
            stats.available = true;
        }
        return stats;
    }
    
    // Reset for new measurement
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        sum_mhz_ = 0.0;
        min_mhz_ = 1e9;
        max_mhz_ = 0.0;
        sample_count_ = 0;
    }
    
private:
    double sum_mhz_;
    double min_mhz_;
    double max_mhz_;
    unsigned sample_count_;
    
    std::atomic<bool> running_;
    std::thread sample_thread_;
    mutable std::mutex mutex_;
};

// Check if OpenMP is enabled
inline bool is_openmp_enabled() {
#ifdef USE_OPENMP
    return true;
#else
    return false;
#endif
}
