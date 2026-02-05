#pragma once
// CPU Benchmark - Threading module header

#include "thread_affinity.hpp"

#include <vector>
#include <thread>
#include <functional>
#include <cstdlib>

// Work range for distributing work among threads
struct WorkRange {
    size_t begin;
    size_t end;
};

// Compute work ranges for distributing work among threads
// Partitions [0, total) into num_parts ranges without overlap
std::vector<WorkRange> compute_ranges(size_t total, unsigned num_parts);

// Thread pool for parallel execution 
class ThreadPool {
public:
    explicit ThreadPool(unsigned num_threads);
    
    // Constructor with specific core affinity for NUMA systems
    ThreadPool(unsigned num_threads, const std::vector<unsigned>& core_ids);
    
    ~ThreadPool();
    
    // Partition z-axis for parallel execution 
    std::vector<WorkRange> partition_z(size_t Nz) const;
    
    // Parallel execution over z-axis ranges 
    // func signature: void(size_t z_begin, size_t z_end)
    template<typename Func>
    void parallel_for_z(size_t Nz, Func&& func);
    
    unsigned thread_count() const { return num_threads_; }
    
private:
    std::vector<std::thread> threads_;
    unsigned num_threads_;
    std::vector<unsigned> core_ids_;  // Core IDs for thread affinity (empty = no affinity)
};

namespace {
inline bool threadpool_affinity_enabled() {
    const char* env = std::getenv("SFBENCH_NO_AFFINITY");
    return !(env && env[0] != '\0' && env[0] != '0');
}
} // namespace

// Template implementation must be in header
template<typename Func>
void ThreadPool::parallel_for_z(size_t Nz, Func&& func) {
    auto ranges = partition_z(Nz);
    
    if (ranges.empty()) {
        return;
    }
    
    // Single-threaded case: execute directly
    if (ranges.size() == 1) {
        func(ranges[0].begin, ranges[0].end);
        return;
    }
    
    const bool use_affinity = !core_ids_.empty() &&
        ThreadAffinityManager::is_affinity_supported() &&
        threadpool_affinity_enabled();

    threads_.clear();
    if (use_affinity) {
        threads_.reserve(ranges.size());
        for (size_t i = 0; i < ranges.size(); ++i) {
            unsigned core_id = core_ids_[i % core_ids_.size()];
            threads_.emplace_back([&func, begin = ranges[i].begin, end = ranges[i].end, core_id]() {
                ThreadAffinityManager::pin_current_thread(core_id);
                func(begin, end);
            });
        }

        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        threads_.clear();
        return;
    }

    // Multi-threaded case: spawn threads for all but last range
    threads_.reserve(ranges.size() - 1);

    for (size_t i = 0; i < ranges.size() - 1; ++i) {
        threads_.emplace_back([&func, begin = ranges[i].begin, end = ranges[i].end]() {
            func(begin, end);
        });
    }

    // Execute last range on current thread
    func(ranges.back().begin, ranges.back().end);

    // Synchronization: wait for all threads to complete 
    for (auto& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    threads_.clear();
}
