// CPU Benchmark - Cache-Level Sizes Calculator
// Calculates optimal data sizes for different cache levels

#pragma once

#include "platform.hpp"
#include "types.hpp"
#include <cstddef>
#include <cstdlib>
#include <string>

// Cache line size constant 
constexpr size_t CACHE_LINE_SIZE = 64;

// Cache level enumeration for test targeting 
enum class CacheLevel {
    L1,         // L1 data cache
    L2,         // L2 cache
    L3,         // L3 cache (LLC)
    RAM         // Main memory (RAM-bound)
};

// Convert CacheLevel to string for output 
inline std::string cache_level_to_string(CacheLevel level) {
    switch (level) {
        case CacheLevel::L1: return "L1";
        case CacheLevel::L2: return "L2";
        case CacheLevel::L3: return "L3";
        case CacheLevel::RAM: return "RAM";
    }
    return "Unknown";
}

// Cache-Level Sizes structure 
// Stores calculated element counts for each cache level
struct CacheLevelSizes {
    size_t l1_fit_elements;     // Elements that fit in L1 cache
    size_t l2_fit_elements;     // Elements that fit in L2 cache
    size_t l3_fit_elements;     // Elements that fit in L3 cache
    size_t ram_bound_elements;  // Elements for RAM-bound test (> L3)
    
    // Cache sizes used for calculation (for reporting)
    size_t l1_cache_bytes;
    size_t l2_cache_bytes;
    size_t l3_cache_bytes;
    
    // Default constructor
    CacheLevelSizes()
        : l1_fit_elements(0)
        , l2_fit_elements(0)
        , l3_fit_elements(0)
        , ram_bound_elements(0)
        , l1_cache_bytes(0)
        , l2_cache_bytes(0)
        , l3_cache_bytes(0) {}
    
    // Calculate sizes based on cache info and element size 
    // Uses 75% of cache to leave room for other data
    static CacheLevelSizes calculate(const CacheInfo& cache, size_t element_size) {
        CacheLevelSizes sizes;
        
        // Store cache sizes for reporting
        sizes.l1_cache_bytes = cache.l1_data_size;
        sizes.l2_cache_bytes = cache.l2_size;
        sizes.l3_cache_bytes = cache.l3_size;
        
        // Use fallback values if cache detection failed
        if (sizes.l1_cache_bytes == 0) {
            sizes.l1_cache_bytes = 32 * 1024;  // 32KB default L1
        }
        if (sizes.l2_cache_bytes == 0) {
            sizes.l2_cache_bytes = 256 * 1024;  // 256KB default L2
        }
        if (sizes.l3_cache_bytes == 0) {
            sizes.l3_cache_bytes = 8 * 1024 * 1024;  // 8MB default L3
        }
        
        // Ensure element_size is at least 1
        if (element_size == 0) {
            element_size = 1;
        }
        
        // Calculate elements using 75% of cache
        // This leaves room for other data and reduces cache conflicts
        // Note: Benchmark uses 3 arrays (A, B, C), so divide by 3
        constexpr double CACHE_UTILIZATION = 0.75;
        constexpr size_t NUM_ARRAYS = 3;
        
        sizes.l1_fit_elements = static_cast<size_t>(
            (sizes.l1_cache_bytes * CACHE_UTILIZATION) / (element_size * NUM_ARRAYS));
        
        sizes.l2_fit_elements = static_cast<size_t>(
            (sizes.l2_cache_bytes * CACHE_UTILIZATION) / (element_size * NUM_ARRAYS));
        
        sizes.l3_fit_elements = static_cast<size_t>(
            (sizes.l3_cache_bytes * CACHE_UTILIZATION) / (element_size * NUM_ARRAYS));
        
        // RAM-bound: 4x L3 size to ensure we're memory-bound
        sizes.ram_bound_elements = (sizes.l3_cache_bytes * 4) / (element_size * NUM_ARRAYS);
        
        // Ensure minimum sizes for valid tests
        constexpr size_t MIN_ELEMENTS = 64;
        if (sizes.l1_fit_elements < MIN_ELEMENTS) {
            sizes.l1_fit_elements = MIN_ELEMENTS;
        }
        if (sizes.l2_fit_elements < MIN_ELEMENTS) {
            sizes.l2_fit_elements = MIN_ELEMENTS;
        }
        if (sizes.l3_fit_elements < MIN_ELEMENTS) {
            sizes.l3_fit_elements = MIN_ELEMENTS;
        }
        if (sizes.ram_bound_elements < MIN_ELEMENTS) {
            sizes.ram_bound_elements = MIN_ELEMENTS;
        }
        
        return sizes;
    }
    
    // Get element count for a specific cache level
    size_t get_elements_for_level(CacheLevel level) const {
        switch (level) {
            case CacheLevel::L1: return l1_fit_elements;
            case CacheLevel::L2: return l2_fit_elements;
            case CacheLevel::L3: return l3_fit_elements;
            case CacheLevel::RAM: return ram_bound_elements;
        }
        return l2_fit_elements;  // Default to L2
    }
    
    // Get cache size in bytes for a specific level
    size_t get_cache_size_for_level(CacheLevel level) const {
        switch (level) {
            case CacheLevel::L1: return l1_cache_bytes;
            case CacheLevel::L2: return l2_cache_bytes;
            case CacheLevel::L3: return l3_cache_bytes;
            case CacheLevel::RAM: return l3_cache_bytes * 4;  // Target 4x L3
        }
        return l2_cache_bytes;
    }
};



// ============================================================================
// Cache-Aligned Memory Allocation 
// ============================================================================

// Allocate cache-aligned memory 
// Aligns to 64-byte cache line boundaries
// Returns nullptr on failure
inline void* aligned_alloc_cache(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    
#ifdef _WIN32
    // Windows: use _aligned_malloc
    return _aligned_malloc(size, CACHE_LINE_SIZE);
#else
    // POSIX: use aligned_alloc (C11) or posix_memalign
    // aligned_alloc requires size to be a multiple of alignment
    size_t aligned_size = ((size + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;
    
    #if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
        // C11 aligned_alloc
        return aligned_alloc(CACHE_LINE_SIZE, aligned_size);
    #else
        // POSIX posix_memalign
        void* ptr = nullptr;
        if (posix_memalign(&ptr, CACHE_LINE_SIZE, aligned_size) != 0) {
            return nullptr;
        }
        return ptr;
    #endif
#endif
}

// Free cache-aligned memory 
inline void aligned_free_cache(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Check if a pointer is cache-aligned 
inline bool is_cache_aligned(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % CACHE_LINE_SIZE) == 0;
}

// RAII wrapper for cache-aligned memory 
template<typename T>
class AlignedBuffer {
public:
    // Allocate aligned buffer for 'count' elements
    explicit AlignedBuffer(size_t count)
        : data_(nullptr)
        , count_(count)
        , size_bytes_(count * sizeof(T))
    {
        if (count > 0) {
            data_ = static_cast<T*>(aligned_alloc_cache(size_bytes_));
        }
    }
    
    // Destructor - free aligned memory
    ~AlignedBuffer() {
        if (data_) {
            aligned_free_cache(data_);
        }
    }
    
    // Non-copyable
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    
    // Movable
    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data_(other.data_)
        , count_(other.count_)
        , size_bytes_(other.size_bytes_)
    {
        other.data_ = nullptr;
        other.count_ = 0;
        other.size_bytes_ = 0;
    }
    
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) {
                aligned_free_cache(data_);
            }
            data_ = other.data_;
            count_ = other.count_;
            size_bytes_ = other.size_bytes_;
            other.data_ = nullptr;
            other.count_ = 0;
            other.size_bytes_ = 0;
        }
        return *this;
    }
    
    // Access
    T* data() { return data_; }
    const T* data() const { return data_; }
    
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
    
    // Info
    size_t count() const { return count_; }
    size_t size_bytes() const { return size_bytes_; }
    bool valid() const { return data_ != nullptr; }
    bool is_aligned() const { return is_cache_aligned(data_); }
    
    // Fill with value
    void fill(T value) {
        for (size_t i = 0; i < count_; ++i) {
            data_[i] = value;
        }
    }
    
    // Fill with zeros
    void fill_zero() {
        fill(T(0));
    }
    
private:
    T* data_;
    size_t count_;
    size_t size_bytes_;
};

