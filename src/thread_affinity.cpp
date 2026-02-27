// CPU Benchmark - Thread Affinity Manager Implementation


#include "thread_affinity.hpp"
#include "platform.hpp"
#include <thread>
#include <vector>

// Platform-specific includes
#ifdef _WIN32


    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>

static AffinityResult set_thread_affinity_handle(HANDLE handle, unsigned core_id) {
    // Support CPU groups (>= 64 logical processors). We treat core_id as a global
    // index: group = core_id / 64, index = core_id % 64.
    WORD group = static_cast<WORD>(core_id / 64);
    unsigned index = core_id % 64;

    KAFFINITY mask = (static_cast<KAFFINITY>(1) << index);

    GROUP_AFFINITY ga{};
    ga.Group = group;
    ga.Mask = mask;

    // Works on modern Windows, including for group 0.
    if (SetThreadGroupAffinity(handle, &ga, nullptr)) {
        return AffinityResult::Success;
    }

    // Fallback for older systems/toolchains: group 0 only.
    if (group == 0) {
        DWORD_PTR result = SetThreadAffinityMask(handle, static_cast<DWORD_PTR>(mask));
        if (result != 0) {
            return AffinityResult::Success;
        }
    }

    DWORD error = GetLastError();
    if (error == ERROR_ACCESS_DENIED) {
        return AffinityResult::PermissionDenied;
    }
    return AffinityResult::Failed;
}
#elif defined(__linux__)
    #ifndef _GNU_SOURCE
        #define _GNU_SOURCE
    #endif
    #include <pthread.h>
    #include <sched.h>
    #include <sys/resource.h>
    #include <unistd.h>
    #include <errno.h>
#elif defined(__APPLE__)
    #include <pthread.h>
    #include <mach/mach.h>
    #include <mach/thread_policy.h>
    #include <mach/thread_act.h>
    #include <sys/resource.h>
    #include <unistd.h>
    #include <errno.h>
#endif

// Get the number of available CPU cores
unsigned ThreadAffinityManager::get_core_count() {
    return get_logical_core_count();
}

// ============================================================================
// Windows Implementation
// ============================================================================
#ifdef _WIN32


AffinityResult ThreadAffinityManager::pin_to_core(std::thread& thread, unsigned core_id) {
    if (core_id >= get_core_count()) {
        return AffinityResult::InvalidCore;
    }

    HANDLE handle = thread.native_handle();
    return set_thread_affinity_handle(handle, core_id);
}



AffinityResult ThreadAffinityManager::pin_current_thread(unsigned core_id) {
    if (core_id >= get_core_count()) {
        return AffinityResult::InvalidCore;
    }

    HANDLE handle = GetCurrentThread();
    return set_thread_affinity_handle(handle, core_id);
}


AffinityResult ThreadAffinityManager::pin_to_socket(unsigned socket_id) {
    // Get cores for the specified socket
    std::vector<unsigned> cores = get_cores_for_socket(socket_id);
    if (cores.empty()) {
        return AffinityResult::InvalidCore;
    }
    
    // Build affinity mask from socket cores
    DWORD_PTR mask = 0;
    for (unsigned core : cores) {
        if (core < 64) {  // DWORD_PTR is 64-bit on x64
            mask |= (static_cast<DWORD_PTR>(1) << core);
        }
    }
    
    if (mask == 0) {
        return AffinityResult::Failed;
    }
    
    HANDLE handle = GetCurrentThread();
    DWORD_PTR result = SetThreadAffinityMask(handle, mask);
    if (result == 0) {
        DWORD error = GetLastError();
        if (error == ERROR_ACCESS_DENIED) {
            return AffinityResult::PermissionDenied;
        }
        return AffinityResult::Failed;
    }
    
    return AffinityResult::Success;
}

AffinityResult ThreadAffinityManager::set_process_socket_affinity(unsigned socket_id) {
    // Get cores for the specified socket
    std::vector<unsigned> cores = get_cores_for_socket(socket_id);
    if (cores.empty()) {
        return AffinityResult::InvalidCore;
    }
    
    // Build affinity mask from socket cores
    DWORD_PTR mask = 0;
    for (unsigned core : cores) {
        if (core < 64) {
            mask |= (static_cast<DWORD_PTR>(1) << core);
        }
    }
    
    if (mask == 0) {
        return AffinityResult::Failed;
    }
    
    HANDLE process = GetCurrentProcess();
    if (!SetProcessAffinityMask(process, mask)) {
        DWORD error = GetLastError();
        if (error == ERROR_ACCESS_DENIED) {
            return AffinityResult::PermissionDenied;
        }
        return AffinityResult::Failed;
    }
    
    return AffinityResult::Success;
}

PriorityResult ThreadAffinityManager::set_process_priority(ProcessPriority priority) {
    HANDLE process = GetCurrentProcess();
    DWORD priority_class;
    
    switch (priority) {
        case ProcessPriority::Normal:
            priority_class = NORMAL_PRIORITY_CLASS;
            break;
        case ProcessPriority::AboveNormal:
            priority_class = ABOVE_NORMAL_PRIORITY_CLASS;
            break;
        case ProcessPriority::High:
            priority_class = HIGH_PRIORITY_CLASS;
            break;
        case ProcessPriority::Realtime:
            priority_class = REALTIME_PRIORITY_CLASS;
            break;
        default:
            return PriorityResult::Failed;
    }
    
    if (!SetPriorityClass(process, priority_class)) {
        DWORD error = GetLastError();
        if (error == ERROR_ACCESS_DENIED) {
            return PriorityResult::PermissionDenied;
        }
        return PriorityResult::Failed;
    }
    
    return PriorityResult::Success;
}

bool ThreadAffinityManager::is_affinity_supported() {
    return true;
}

bool ThreadAffinityManager::is_priority_supported() {
    return true;
}

unsigned long long ThreadAffinityManager::get_current_affinity() {
    HANDLE thread = GetCurrentThread();
    DWORD_PTR process_mask, system_mask;
    
    if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
        return static_cast<unsigned long long>(process_mask);
    }
    return 0;
}

// ============================================================================
// Linux Implementation
// ============================================================================
#elif defined(__linux__)

AffinityResult ThreadAffinityManager::pin_to_core(std::thread& thread, unsigned core_id) {
    if (core_id >= get_core_count()) {
        return AffinityResult::InvalidCore;
    }
    
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    
    pthread_t native_handle = thread.native_handle();
    int result = pthread_setaffinity_np(native_handle, sizeof(cpu_set_t), &cpuset);
    
    if (result != 0) {
        if (result == EPERM) {
            return AffinityResult::PermissionDenied;
        }
        if (result == EINVAL) {
            return AffinityResult::InvalidCore;
        }
        return AffinityResult::Failed;
    }
    
    return AffinityResult::Success;
}

AffinityResult ThreadAffinityManager::pin_current_thread(unsigned core_id) {
    if (core_id >= get_core_count()) {
        return AffinityResult::InvalidCore;
    }
    
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    
    int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    
    if (result != 0) {
        if (result == EPERM) {
            return AffinityResult::PermissionDenied;
        }
        if (result == EINVAL) {
            return AffinityResult::InvalidCore;
        }
        return AffinityResult::Failed;
    }
    
    return AffinityResult::Success;
}

AffinityResult ThreadAffinityManager::pin_to_socket(unsigned socket_id) {
    // Get cores for the specified socket
    std::vector<unsigned> cores = get_cores_for_socket(socket_id);
    if (cores.empty()) {
        return AffinityResult::InvalidCore;
    }
    
    // Build CPU set from socket cores
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (unsigned core : cores) {
        CPU_SET(core, &cpuset);
    }
    
    int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    
    if (result != 0) {
        if (result == EPERM) {
            return AffinityResult::PermissionDenied;
        }
        if (result == EINVAL) {
            return AffinityResult::InvalidCore;
        }
        return AffinityResult::Failed;
    }
    
    return AffinityResult::Success;
}

AffinityResult ThreadAffinityManager::set_process_socket_affinity(unsigned socket_id) {
    // Get cores for the specified socket
    std::vector<unsigned> cores = get_cores_for_socket(socket_id);
    if (cores.empty()) {
        return AffinityResult::InvalidCore;
    }
    
    // Build CPU set from socket cores
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (unsigned core : cores) {
        CPU_SET(core, &cpuset);
    }
    
    // Set affinity for the current process (affects all threads)
    int result = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    
    if (result != 0) {
        if (errno == EPERM) {
            return AffinityResult::PermissionDenied;
        }
        if (errno == EINVAL) {
            return AffinityResult::InvalidCore;
        }
        return AffinityResult::Failed;
    }
    
    return AffinityResult::Success;
}

PriorityResult ThreadAffinityManager::set_process_priority(ProcessPriority priority) {
    int nice_value;
    
    switch (priority) {
        case ProcessPriority::Normal:
            nice_value = 0;
            break;
        case ProcessPriority::AboveNormal:
            nice_value = -5;
            break;
        case ProcessPriority::High:
            nice_value = -10;
            break;
        case ProcessPriority::Realtime:
            nice_value = -20;  // Lowest nice value (highest priority)
            break;
        default:
            return PriorityResult::Failed;
    }
    
    // setpriority returns -1 on error, but -1 is also a valid nice value
    // So we need to clear errno first and check it after
    errno = 0;
    int result = setpriority(PRIO_PROCESS, 0, nice_value);
    
    if (result == -1 && errno != 0) {
        if (errno == EPERM || errno == EACCES) {
            return PriorityResult::PermissionDenied;
        }
        return PriorityResult::Failed;
    }
    
    return PriorityResult::Success;
}

bool ThreadAffinityManager::is_affinity_supported() {
    return true;
}

bool ThreadAffinityManager::is_priority_supported() {
    return true;
}

unsigned long long ThreadAffinityManager::get_current_affinity() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0) {
        unsigned long long mask = 0;
        unsigned core_count = get_core_count();
        for (unsigned i = 0; i < core_count && i < 64; ++i) {
            if (CPU_ISSET(i, &cpuset)) {
                mask |= (1ULL << i);
            }
        }
        return mask;
    }
    return 0;
}

// ============================================================================
// macOS Implementation
// ============================================================================
#elif defined(__APPLE__)

AffinityResult ThreadAffinityManager::pin_to_core(std::thread& thread, unsigned core_id) {
    if (core_id >= get_core_count()) {
        return AffinityResult::InvalidCore;
    }
    
    // macOS uses thread_policy_set with THREAD_AFFINITY_POLICY
    // Note: macOS doesn't support true CPU pinning like Linux/Windows
    // It uses affinity tags to hint the scheduler to keep threads together
    pthread_t native_handle = thread.native_handle();
    mach_port_t mach_thread = pthread_mach_thread_np(native_handle);
    
    thread_affinity_policy_data_t policy;
    policy.affinity_tag = core_id + 1;  // Tags should be non-zero
    
    kern_return_t result = thread_policy_set(
        mach_thread,
        THREAD_AFFINITY_POLICY,
        reinterpret_cast<thread_policy_t>(&policy),
        THREAD_AFFINITY_POLICY_COUNT
    );
    
    if (result != KERN_SUCCESS) {
        if (result == KERN_INVALID_ARGUMENT) {
            return AffinityResult::InvalidCore;
        }
        return AffinityResult::Failed;
    }
    
    return AffinityResult::Success;
}

AffinityResult ThreadAffinityManager::pin_current_thread(unsigned core_id) {
    if (core_id >= get_core_count()) {
        return AffinityResult::InvalidCore;
    }
    
    mach_port_t mach_thread = mach_thread_self();
    
    thread_affinity_policy_data_t policy;
    policy.affinity_tag = core_id + 1;  // Tags should be non-zero
    
    kern_return_t result = thread_policy_set(
        mach_thread,
        THREAD_AFFINITY_POLICY,
        reinterpret_cast<thread_policy_t>(&policy),
        THREAD_AFFINITY_POLICY_COUNT
    );
    
    mach_port_deallocate(mach_task_self(), mach_thread);
    
    if (result != KERN_SUCCESS) {
        if (result == KERN_INVALID_ARGUMENT) {
            return AffinityResult::InvalidCore;
        }
        return AffinityResult::Failed;
    }
    
    return AffinityResult::Success;
}

AffinityResult ThreadAffinityManager::pin_to_socket(unsigned socket_id) {
    // macOS typically has single socket, so socket 0 is always valid
    if (socket_id != 0) {
        return AffinityResult::InvalidCore;
    }
    // macOS doesn't support true socket affinity, return success for socket 0
    return AffinityResult::Success;
}

AffinityResult ThreadAffinityManager::set_process_socket_affinity(unsigned socket_id) {
    // macOS typically has single socket, so socket 0 is always valid
    if (socket_id != 0) {
        return AffinityResult::InvalidCore;
    }
    // macOS doesn't support true socket affinity, return success for socket 0
    return AffinityResult::Success;
}

PriorityResult ThreadAffinityManager::set_process_priority(ProcessPriority priority) {
    int nice_value;
    
    switch (priority) {
        case ProcessPriority::Normal:
            nice_value = 0;
            break;
        case ProcessPriority::AboveNormal:
            nice_value = -5;
            break;
        case ProcessPriority::High:
            nice_value = -10;
            break;
        case ProcessPriority::Realtime:
            nice_value = -20;  // Lowest nice value (highest priority)
            break;
        default:
            return PriorityResult::Failed;
    }
    
    // setpriority returns -1 on error
    errno = 0;
    int result = setpriority(PRIO_PROCESS, 0, nice_value);
    
    if (result == -1 && errno != 0) {
        if (errno == EPERM || errno == EACCES) {
            return PriorityResult::PermissionDenied;
        }
        return PriorityResult::Failed;
    }
    
    return PriorityResult::Success;
}

bool ThreadAffinityManager::is_affinity_supported() {
    // macOS supports affinity hints but not true CPU pinning
    return true;
}

bool ThreadAffinityManager::is_priority_supported() {
    return true;
}

unsigned long long ThreadAffinityManager::get_current_affinity() {
    // macOS doesn't provide a way to query the current affinity mask
    // Return a mask with all cores set
    unsigned core_count = get_core_count();
    if (core_count >= 64) {
        return ~0ULL;
    }
    return (1ULL << core_count) - 1;
}

// ============================================================================
// Fallback Implementation for unsupported platforms
// ============================================================================
#else

AffinityResult ThreadAffinityManager::pin_to_core(std::thread& /*thread*/, unsigned /*core_id*/) {
    return AffinityResult::NotSupported;
}

AffinityResult ThreadAffinityManager::pin_current_thread(unsigned /*core_id*/) {
    return AffinityResult::NotSupported;
}

AffinityResult ThreadAffinityManager::pin_to_socket(unsigned /*socket_id*/) {
    return AffinityResult::NotSupported;
}

AffinityResult ThreadAffinityManager::set_process_socket_affinity(unsigned /*socket_id*/) {
    return AffinityResult::NotSupported;
}

PriorityResult ThreadAffinityManager::set_process_priority(ProcessPriority /*priority*/) {
    return PriorityResult::NotSupported;
}

bool ThreadAffinityManager::is_affinity_supported() {
    return false;
}

bool ThreadAffinityManager::is_priority_supported() {
    return false;
}

unsigned long long ThreadAffinityManager::get_current_affinity() {
    return 0;
}

#endif
