// CPU Benchmark - Threading module

#include "threading.hpp"
#include "thread_affinity.hpp"
#include <algorithm>

// Compute work ranges for distributing work among threads 
// Partitions [0, total) into num_parts ranges without overlap
std::vector<WorkRange> compute_ranges(size_t total, unsigned num_parts) {
    std::vector<WorkRange> ranges;
    
    if (num_parts == 0 || total == 0) {
        return ranges;
    }
    
    size_t base_size = total / num_parts;
    size_t remainder = total % num_parts;
    
    size_t current = 0;
    for (unsigned i = 0; i < num_parts; ++i) {
        size_t size = base_size + (i < remainder ? 1 : 0);
        if (size > 0) {
            ranges.push_back({current, current + size});
            current += size;
        }
    }
    
    return ranges;
}

// ThreadPool constructor
// If num_threads is 0, uses logical core count (auto-detect)
ThreadPool::ThreadPool(unsigned num_threads) : core_ids_() {
    if (num_threads == 0) {
        // Use logical core count when threads=0
        num_threads_ = ThreadAffinityManager::get_core_count();
        if (num_threads_ == 0) {
            num_threads_ = 1; // Fallback if logical core count returns 0
        }
    } else {
        //  Create exactly N threads when N > 0
        num_threads_ = num_threads;
    }
}

// ThreadPool constructor with core affinity for NUMA systems
ThreadPool::ThreadPool(unsigned num_threads, const std::vector<unsigned>& core_ids)
    : core_ids_(core_ids)
{
    if (num_threads == 0) {
        // Use core_ids size if available, otherwise logical core count
        if (!core_ids_.empty()) {
            num_threads_ = static_cast<unsigned>(core_ids_.size());
        } else {
            num_threads_ = ThreadAffinityManager::get_core_count();
            if (num_threads_ == 0) {
                num_threads_ = 1;
            }
        }
    } else {
        num_threads_ = num_threads;
    }
    
    // Limit to available core_ids if specified
    if (!core_ids_.empty() && num_threads_ > core_ids_.size()) {
        num_threads_ = static_cast<unsigned>(core_ids_.size());
    }
}

// ThreadPool destructor - ensure all threads are joined 
ThreadPool::~ThreadPool() {
    for (auto& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

// Partition z-axis for parallel execution 
std::vector<WorkRange> ThreadPool::partition_z(size_t Nz) const {
    return compute_ranges(Nz, num_threads_);
}
