#pragma once
// CPU Benchmark - Runtime SIMD Dispatcher

// Provides runtime selection of optimal SIMD kernels based on CPU capabilities

#include <cstddef>
#include <cstdint>
#include <string>
#include "cpu_capabilities.hpp"
#include "kernels/kernel_common.hpp"
#include "kernels/kernel_compute.hpp"

// RuntimeDispatcher class
// Selects optimal SIMD kernel implementations at runtime based on detected CPU capabilities
class RuntimeDispatcher {
public:
    // Initialize the dispatcher (detects SIMD capabilities)
    // Should be called once at program startup
    static void initialize();
    
    // Check if dispatcher has been initialized
    static bool is_initialized();
    
    // Get the active SIMD level
    static SimdLevel get_active_level();
    
    // Get human-readable name of active SIMD level
    static const char* get_active_level_name();
    
    // Get memory kernel for float
    static MemKernelFn<float> get_mem_kernel_float();
    
    // Get memory kernel for double
    static MemKernelFn<double> get_mem_kernel_double();
    
    // Get memory kernel for int8_t
    static MemKernelFn<int8_t> get_mem_kernel_int8();
    
    // Get stencil kernel for float
    static StencilKernelFn<float> get_stencil_kernel_float();
    
    // Get stencil kernel for double
    static StencilKernelFn<double> get_stencil_kernel_double();
    
    // Get stencil kernel for int8_t
    static StencilKernelFn<int8_t> get_stencil_kernel_int8();
    
    // Get compute kernel for float 
    static ComputeKernelFloatFn get_compute_kernel_float();
    
    // Get compute kernel for double 
    static ComputeKernelDoubleFn get_compute_kernel_double();
    
    // Template versions for generic code
    template<typename T>
    static MemKernelFn<T> get_mem_kernel();
    
    template<typename T>
    static StencilKernelFn<T> get_stencil_kernel();
    
    // Force a specific SIMD level (for testing)
    static void force_level(SimdLevel level);
    
    // Reset to auto-detected level
    static void reset_to_auto();
    
    // Get diagnostic information
    static std::string get_diagnostics();
    
private:
    static bool initialized_;
    static SimdLevel active_level_;
    static SimdLevel detected_level_;
    static bool forced_;
    
    // Cached kernel pointers
    static MemKernelFn<float> mem_kernel_float_;
    static MemKernelFn<double> mem_kernel_double_;
    static MemKernelFn<int8_t> mem_kernel_int8_;
    static StencilKernelFn<float> stencil_kernel_float_;
    static StencilKernelFn<double> stencil_kernel_double_;
    static StencilKernelFn<int8_t> stencil_kernel_int8_;
    static ComputeKernelFloatFn compute_kernel_float_;
    static ComputeKernelDoubleFn compute_kernel_double_;
    
    // Select kernels based on SIMD level
    static void select_kernels(SimdLevel level);
};

// Template specializations
template<>
inline MemKernelFn<float> RuntimeDispatcher::get_mem_kernel<float>() {
    return get_mem_kernel_float();
}

template<>
inline MemKernelFn<double> RuntimeDispatcher::get_mem_kernel<double>() {
    return get_mem_kernel_double();
}

template<>
inline MemKernelFn<int8_t> RuntimeDispatcher::get_mem_kernel<int8_t>() {
    return get_mem_kernel_int8();
}

template<>
inline StencilKernelFn<float> RuntimeDispatcher::get_stencil_kernel<float>() {
    return get_stencil_kernel_float();
}

template<>
inline StencilKernelFn<double> RuntimeDispatcher::get_stencil_kernel<double>() {
    return get_stencil_kernel_double();
}

template<>
inline StencilKernelFn<int8_t> RuntimeDispatcher::get_stencil_kernel<int8_t>() {
    return get_stencil_kernel_int8();
}


