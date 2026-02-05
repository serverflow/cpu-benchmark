#pragma once
// CPU Benchmark - Thread Affinity Manager


#include <thread>

// Thread affinity result codes
enum class AffinityResult {
    Success,            // Affinity was set successfully
    NotSupported,       // Platform doesn't support thread affinity
    InvalidCore,        // Core ID is out of range
    PermissionDenied,   // Insufficient permissions
    Failed              // Other failure
};

// Process priority levels
enum class ProcessPriority {
    Normal,
    AboveNormal,
    High,
    Realtime            // Requires admin/root privileges
};

// Priority result codes
enum class PriorityResult {
    Success,            // Priority was set successfully
    NotSupported,       // Platform doesn't support priority setting
    PermissionDenied,   // Insufficient permissions (common for High/Realtime)
    Failed              // Other failure
};

// Thread Affinity Manager 
// Provides cross-platform thread pinning and process priority control
class ThreadAffinityManager {
public:
    // Pin a thread to a specific CPU core 
    // core_id: 0-based index of the CPU core
    // Returns: AffinityResult indicating success or failure reason
    static AffinityResult pin_to_core(std::thread& thread, unsigned core_id);
    
    // Pin the current thread to a specific CPU core 
    // core_id: 0-based index of the CPU core
    // Returns: AffinityResult indicating success or failure reason
    static AffinityResult pin_current_thread(unsigned core_id);
    
    // Pin the current thread to cores belonging to a specific socket
    // socket_id: 0-based socket index
    // Returns: AffinityResult indicating success or failure reason
    static AffinityResult pin_to_socket(unsigned socket_id);
    
    // Set thread affinity mask to use only cores from a specific socket
    // This affects all threads in the current process
    // socket_id: 0-based socket index
    // Returns: AffinityResult indicating success or failure reason
    static AffinityResult set_process_socket_affinity(unsigned socket_id);
    
    // Set process priority 
    // priority: Desired priority level
    // Returns: PriorityResult indicating success or failure reason
    static PriorityResult set_process_priority(ProcessPriority priority);
    
    // Get the number of available CPU cores
    // Returns: Number of logical CPU cores
    static unsigned get_core_count();
    
    // Check if thread affinity is supported on this platform
    static bool is_affinity_supported();
    
    // Check if priority setting is supported on this platform
    static bool is_priority_supported();
    
    // Get the current thread's affinity mask (if supported)
    // Returns: Bitmask of cores the thread can run on, or 0 if not supported
    static unsigned long long get_current_affinity();
};

// Convert AffinityResult to string for logging/debugging
inline const char* affinity_result_to_string(AffinityResult result) {
    switch (result) {
        case AffinityResult::Success: return "Success";
        case AffinityResult::NotSupported: return "Not Supported";
        case AffinityResult::InvalidCore: return "Invalid Core ID";
        case AffinityResult::PermissionDenied: return "Permission Denied";
        case AffinityResult::Failed: return "Failed";
        default: return "Unknown";
    }
}

// Convert PriorityResult to string for logging/debugging
inline const char* priority_result_to_string(PriorityResult result) {
    switch (result) {
        case PriorityResult::Success: return "Success";
        case PriorityResult::NotSupported: return "Not Supported";
        case PriorityResult::PermissionDenied: return "Permission Denied";
        case PriorityResult::Failed: return "Failed";
        default: return "Unknown";
    }
}

// Convert ProcessPriority to string for logging/debugging
inline const char* priority_to_string(ProcessPriority priority) {
    switch (priority) {
        case ProcessPriority::Normal: return "Normal";
        case ProcessPriority::AboveNormal: return "Above Normal";
        case ProcessPriority::High: return "High";
        case ProcessPriority::Realtime: return "Realtime";
        default: return "Unknown";
    }
}
