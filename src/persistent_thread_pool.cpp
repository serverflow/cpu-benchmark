// CPU Benchmark - Persistent Thread Pool with Work Stealing

#include "persistent_thread_pool.hpp"
#include "thread_affinity.hpp"
#include <algorithm>
#include <random>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <system_error>
#include <stdexcept>
#include <string>

namespace {
bool debug_enabled() {
    const char* env = std::getenv("SFBENCH_DEBUG");
    return env && env[0] != '\0' && env[0] != '0';
}

void debug_log(const std::string& msg) {
    if (debug_enabled()) {
        std::cerr << msg << "\n";
    }
}

bool affinity_enabled() {
    const char* env = std::getenv("SFBENCH_NO_AFFINITY");
    return !(env && env[0] != '\0' && env[0] != '0');
}
} // namespace

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

// WorkQueue implementation

void WorkQueue::push(std::function<void()> task) {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.push_back(std::move(task));
}

bool WorkQueue::pop(std::function<void()>& task) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tasks_.empty()) {
        return false;
    }
    // Pop from back (LIFO) for cache locality
    task = std::move(tasks_.back());
    tasks_.pop_back();
    return true;
}

bool WorkQueue::steal(std::function<void()>& task) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tasks_.empty()) {
        return false;
    }
    // Steal from front (FIFO) for fairness
    task = std::move(tasks_.front());
    tasks_.pop_front();
    return true;
}

bool WorkQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tasks_.empty();
}

void WorkQueue::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.clear();
}

// PersistentThreadPool implementation


// persistent thread pool
PersistentThreadPool::PersistentThreadPool(unsigned num_threads) {
    if (num_threads == 0) {
        num_threads_ = ThreadAffinityManager::get_core_count();
        if (num_threads_ == 0) {
            num_threads_ = 1;
        }
    } else {
        num_threads_ = num_threads;
    }
    
    // Create per-thread work queues 
    work_queues_.reserve(num_threads_);
    for (unsigned i = 0; i < num_threads_; ++i) {
        work_queues_.push_back(std::make_unique<WorkQueue>());
    }
    
    // Create worker threads
    workers_.reserve(num_threads_);
    for (unsigned i = 0; i < num_threads_; ++i) {
        try {
            workers_.emplace_back(&PersistentThreadPool::worker_loop, this, i);
        } catch (const std::system_error& e) {
            debug_log(std::string("[thread_pool] Failed to create worker ") +
                      std::to_string(i) + "/" + std::to_string(num_threads_) +
                      ": " + e.what());
            num_threads_ = i;
            break;
        }
    }
    if (workers_.empty()) {
        throw std::runtime_error("Failed to create worker threads");
    }
}

// Constructor with specific core affinity
PersistentThreadPool::PersistentThreadPool(unsigned num_threads, const std::vector<unsigned>& core_ids)
    : core_ids_(core_ids)
{
    if (num_threads == 0) {
        num_threads_ = core_ids.empty() ? ThreadAffinityManager::get_core_count() 
                                        : static_cast<unsigned>(core_ids.size());
        if (num_threads_ == 0) {
            num_threads_ = 1;
        }
    } else {
        num_threads_ = num_threads;
    }
    
    // Create per-thread work queues 
    work_queues_.reserve(num_threads_);
    for (unsigned i = 0; i < num_threads_; ++i) {
        work_queues_.push_back(std::make_unique<WorkQueue>());
    }
    
    // Create worker threads
    workers_.reserve(num_threads_);
    for (unsigned i = 0; i < num_threads_; ++i) {
        try {
            workers_.emplace_back(&PersistentThreadPool::worker_loop, this, i);
        } catch (const std::system_error& e) {
            debug_log(std::string("[thread_pool] Failed to create worker ") +
                      std::to_string(i) + "/" + std::to_string(num_threads_) +
                      ": " + e.what());
            num_threads_ = i;
            break;
        }
    }
    if (workers_.empty()) {
        throw std::runtime_error("Failed to create worker threads");
    }
}

PersistentThreadPool::~PersistentThreadPool() {
    // Signal threads to stop
    running_.store(false, std::memory_order_release);
    
    // Wake up any sleeping threads
    {
        std::lock_guard<std::mutex> lock(wake_mutex_);
        has_work_.store(true, std::memory_order_release);
    }
    wake_cv_.notify_all();
    
    // Join all threads
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void PersistentThreadPool::submit(std::function<void()> task) {
    // Increment pending count
    pending_tasks_.fetch_add(1, std::memory_order_release);
    
    // Round-robin distribution to queues
    unsigned queue_idx = next_queue_.fetch_add(1, std::memory_order_relaxed) % num_threads_;
    work_queues_[queue_idx]->push(std::move(task));
    
    // Signal that work is available
    if (!spin_wait_.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lock(wake_mutex_);
        has_work_.store(true, std::memory_order_release);
        wake_cv_.notify_one();
    }
}

void PersistentThreadPool::submit_batch(std::vector<std::function<void()>>& tasks) {
    if (tasks.empty()) return;
    
    // Increment pending count
    pending_tasks_.fetch_add(tasks.size(), std::memory_order_release);
    
    // Distribute tasks across queues
    for (size_t i = 0; i < tasks.size(); ++i) {
        unsigned queue_idx = (next_queue_.fetch_add(1, std::memory_order_relaxed)) % num_threads_;
        work_queues_[queue_idx]->push(std::move(tasks[i]));
    }
    
    // Signal that work is available
    if (!spin_wait_.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lock(wake_mutex_);
        has_work_.store(true, std::memory_order_release);
        wake_cv_.notify_all();
    }
}

void PersistentThreadPool::wait_all() {
    // Wait until all pending tasks are complete
    std::unique_lock<std::mutex> lock(wait_mutex_);
    wait_cv_.wait(lock, [this]() {
        return pending_tasks_.load(std::memory_order_acquire) == 0;
    });
}

void PersistentThreadPool::set_spin_wait(bool enabled) {
    spin_wait_.store(enabled, std::memory_order_release);
    
    // Wake up threads if switching from spin to sleep mode
    if (!enabled) {
        std::lock_guard<std::mutex> lock(wake_mutex_);
        has_work_.store(true, std::memory_order_release);
        wake_cv_.notify_all();
    }
}


// Worker thread main loop

void PersistentThreadPool::worker_loop(unsigned thread_id) {
    // Pin thread to specific core for NUMA locality
    // This is critical for multi-socket systems to avoid cross-socket memory access
    bool do_affinity = ThreadAffinityManager::is_affinity_supported() && affinity_enabled();
    if (do_affinity) {
        unsigned core_id;
        if (!core_ids_.empty() && thread_id < core_ids_.size()) {
            // Use specified core ID from the list
            core_id = core_ids_[thread_id];
        } else {
            // Fallback to thread_id as core_id
            core_id = thread_id;
        }
        
        unsigned core_count = ThreadAffinityManager::get_core_count();
        if (core_id < core_count) {
            AffinityResult result = ThreadAffinityManager::pin_current_thread(core_id);
            if (result != AffinityResult::Success) {
                debug_log(std::string("[thread_pool] Pin failed for thread ") +
                          std::to_string(thread_id) + " core " +
                          std::to_string(core_id) + ": " +
                          affinity_result_to_string(result));
            }
        }
    } else if (debug_enabled() && thread_id == 0 && !affinity_enabled()) {
        debug_log("[thread_pool] Affinity disabled via SFBENCH_NO_AFFINITY");
    }
    
    std::function<void()> task;
    
    while (running_.load(std::memory_order_acquire)) {
        bool got_work = false;
        
        // First, try to get work from own queue
        if (work_queues_[thread_id]->pop(task)) {
            got_work = true;
        }
        // If own queue is empty, try to steal from others 
        else if (try_steal_work(thread_id, task)) {
            got_work = true;
            stats_.tasks_stolen.fetch_add(1, std::memory_order_relaxed);
        }
        
        if (got_work) {
            // Execute the task
            try {
                task();
            } catch (const std::exception& e) {
                debug_log(std::string("[thread_pool] Task threw exception: ") + e.what());
            } catch (...) {
                debug_log("[thread_pool] Task threw unknown exception");
            }
            stats_.tasks_executed.fetch_add(1, std::memory_order_relaxed);
            
            // Decrement pending count and notify waiters
            size_t prev = pending_tasks_.fetch_sub(1, std::memory_order_release);
            if (prev == 1) {
                // Last task completed, notify wait_all
                std::lock_guard<std::mutex> lock(wait_mutex_);
                wait_cv_.notify_all();
            }
        } else {
            // No work available
            if (spin_wait_.load(std::memory_order_relaxed)) {
                // Use a brief pause instruction for better power efficiency
                #if defined(_MSC_VER)
                    _mm_pause();
                #elif defined(__x86_64__) || defined(__i386__)
                    __builtin_ia32_pause();
                #elif defined(__aarch64__)
                    __asm__ volatile("yield");
                #else
                    std::this_thread::yield();
                #endif
            } else {
                // Sleep mode - wait for notification
                std::unique_lock<std::mutex> lock(wake_mutex_);
                wake_cv_.wait_for(lock, std::chrono::microseconds(100), [this]() {
                    return has_work_.load(std::memory_order_acquire) || 
                           !running_.load(std::memory_order_acquire);
                });
                has_work_.store(false, std::memory_order_release);
            }
        }
    }
}

// Try to steal work from another thread's queue

bool PersistentThreadPool::try_steal_work(unsigned thread_id, std::function<void()>& task) {
    stats_.steal_attempts.fetch_add(1, std::memory_order_relaxed);
    
    // Try to steal from other queues in round-robin fashion
    for (unsigned i = 1; i < num_threads_; ++i) {
        unsigned victim = (thread_id + i) % num_threads_;
        if (work_queues_[victim]->steal(task)) {
            return true;
        }
    }
    return false;
}

// Partition z-axis for parallel execution
std::vector<std::pair<size_t, size_t>> PersistentThreadPool::partition_z(size_t Nz) const {
    std::vector<std::pair<size_t, size_t>> ranges;
    
    if (num_threads_ == 0 || Nz == 0) {
        return ranges;
    }
    
    size_t base_size = Nz / num_threads_;
    size_t remainder = Nz % num_threads_;
    
    size_t current = 0;
    for (unsigned i = 0; i < num_threads_; ++i) {
        size_t size = base_size + (i < remainder ? 1 : 0);
        if (size > 0) {
            ranges.emplace_back(current, current + size);
            current += size;
        }
    }
    
    return ranges;
}
