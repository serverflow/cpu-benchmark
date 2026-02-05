#pragma once
// CPU Benchmark - Persistent Thread Pool with Work Stealing

#include <vector>
#include <deque>
#include <thread>
#include <atomic>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <memory>

// Work item for the thread pool
struct WorkItem {
    std::function<void()> task;
    std::atomic<bool> taken{false};
    
    WorkItem() = default;
    WorkItem(std::function<void()> t) : task(std::move(t)), taken(false) {}
    
    // Non-copyable, movable
    WorkItem(const WorkItem&) = delete;
    WorkItem& operator=(const WorkItem&) = delete;
    WorkItem(WorkItem&& other) noexcept 
        : task(std::move(other.task))
        , taken(other.taken.load()) {}
    WorkItem& operator=(WorkItem&& other) noexcept {
        task = std::move(other.task);
        taken.store(other.taken.load());
        return *this;
    }
};

// Per-thread work queue with lock-free stealing support

class WorkQueue {
public:
    WorkQueue() = default;
    
    // Push work to the back (owner thread)
    void push(std::function<void()> task);
    
    // Pop work from the back (owner thread) - LIFO for cache locality
    bool pop(std::function<void()>& task);
    
    // Steal work from the front (other threads) - FIFO for fairness
    
    bool steal(std::function<void()>& task);
    
    // Check if queue is empty
    bool empty() const;
    
    // Clear all pending work
    void clear();
    
private:
    mutable std::mutex mutex_;
    std::deque<std::function<void()>> tasks_;
};


// Persistent Thread Pool with Work Stealing

class PersistentThreadPool {
public:
    // Create thread pool with specified number of threads
    // If num_threads is 0, uses logical core count
    explicit PersistentThreadPool(unsigned num_threads = 0);
    
    // Create thread pool with specific core affinity
    // Threads will be pinned to the specified core IDs
    PersistentThreadPool(unsigned num_threads, const std::vector<unsigned>& core_ids);
    
    // Destructor - stops all threads
    ~PersistentThreadPool();
    
    // Non-copyable, non-movable
    PersistentThreadPool(const PersistentThreadPool&) = delete;
    PersistentThreadPool& operator=(const PersistentThreadPool&) = delete;
    PersistentThreadPool(PersistentThreadPool&&) = delete;
    PersistentThreadPool& operator=(PersistentThreadPool&&) = delete;
    
    // Submit a single task
    void submit(std::function<void()> task);
    
    // Submit multiple tasks at once (more efficient)
    void submit_batch(std::vector<std::function<void()>>& tasks);
    
    // Wait for all submitted work to complete
    void wait_all();
    
    // Enable/disable spin-wait mode
   
    void set_spin_wait(bool enabled);
    bool is_spin_wait() const { return spin_wait_.load(std::memory_order_relaxed); }
    
    // Get number of threads
    unsigned thread_count() const { return num_threads_; }
    
    // Get statistics for work stealing effectiveness
    struct Stats {
        std::atomic<size_t> tasks_executed{0};
        std::atomic<size_t> tasks_stolen{0};
        std::atomic<size_t> steal_attempts{0};
        
        void reset() {
            tasks_executed.store(0, std::memory_order_relaxed);
            tasks_stolen.store(0, std::memory_order_relaxed);
            steal_attempts.store(0, std::memory_order_relaxed);
        }
    };
    
    const Stats& stats() const { return stats_; }
    void reset_stats() { stats_.reset(); }
    
    // Parallel for over z-axis ranges (compatible with existing ThreadPool interface)
    template<typename Func>
    void parallel_for_z(size_t Nz, Func&& func);
    
    // Partition z-axis for parallel execution
    std::vector<std::pair<size_t, size_t>> partition_z(size_t Nz) const;
    
private:
    // Worker thread main loop
    void worker_loop(unsigned thread_id);
    
    // Try to steal work from another thread's queue
   
    bool try_steal_work(unsigned thread_id, std::function<void()>& task);
    
    // Worker threads
    std::vector<std::thread> workers_;
    unsigned num_threads_;
    
    // Core IDs for thread affinity (empty = use thread_id as core_id)
    std::vector<unsigned> core_ids_;
    
    // Per-thread work queues
    
    std::vector<std::unique_ptr<WorkQueue>> work_queues_;
    
    // Control flags
    std::atomic<bool> running_{true};
    std::atomic<bool> spin_wait_{true};  
    
    // Synchronization for wait_all
    std::atomic<size_t> pending_tasks_{0};
    std::mutex wait_mutex_;
    std::condition_variable wait_cv_;
    
    // Wake-up mechanism for non-spin-wait mode
    std::mutex wake_mutex_;
    std::condition_variable wake_cv_;
    std::atomic<bool> has_work_{false};
    
    // Statistics
    mutable Stats stats_;
    
    // Round-robin counter for task distribution
    std::atomic<unsigned> next_queue_{0};
};

// Template implementation for parallel_for_z
template<typename Func>
void PersistentThreadPool::parallel_for_z(size_t Nz, Func&& func) {
    if (Nz == 0) return;
    
    auto ranges = partition_z(Nz);
    
    if (ranges.empty()) return;
    
    // Single-threaded case: execute directly
    if (ranges.size() == 1) {
        func(ranges[0].first, ranges[0].second);
        return;
    }
    
    // Submit all but last range to thread pool
    for (size_t i = 0; i < ranges.size() - 1; ++i) {
        submit([&func, begin = ranges[i].first, end = ranges[i].second]() {
            func(begin, end);
        });
    }
    
    // Execute last range on current thread
    func(ranges.back().first, ranges.back().second);
    
    // Wait for all submitted tasks to complete
    wait_all();
}
