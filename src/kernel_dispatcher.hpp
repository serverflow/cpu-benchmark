#pragma once
// CPU Benchmark - Kernel Dispatcher
// Requirements: 11.1, 11.2, 11.3, 11.4
// Provides runtime selection of optimal kernel implementations based on CPU capabilities

#include <string>
#include <sstream>
#include "cpu_capabilities.hpp"
#include "simd_kernels.hpp"
#include "math_kernels.hpp"

// ============================================================================
// Function pointer types for kernels
// ============================================================================

template<typename T>
using MemKernelFnPtr = void(*)(T*, const T*, const T*, T, T, size_t, size_t, size_t, size_t, size_t);

template<typename T>
using StencilKernelFnPtr = void(*)(T*, const T*, T, T, size_t, size_t, size_t, size_t, size_t);

template<typename T>
using MatmulKernelFnPtr = void(*)(T*, const T*, const T*, size_t, size_t, size_t);

// ============================================================================
// KernelDispatcher class
// ============================================================================

class KernelDispatcher {
public:
    // Get singleton instance
    static KernelDispatcher& instance() {
        static KernelDispatcher inst;
        return inst;
    }
    
    // Get the selected SIMD level
    SimdLevel get_selected_level() const { return selected_level_; }
    
    // Check if force_scalar mode is enabled
    bool is_force_scalar() const { return force_scalar_; }
    
    // Set force_scalar mode 
    void set_force_scalar(bool force) {
        force_scalar_ = force;
        update_selected_level();
    }
    
    // Get description of selected kernel
    std::string get_selected_kernel_info() const {
        std::ostringstream oss;
        oss << "Kernel Implementation: ";
        if (force_scalar_) {
            oss << "Scalar (forced)";
        } else {
            oss << simd_level_to_string(selected_level_);
        }
        return oss.str();
    }

    // ========================================================================
    // Memory kernel dispatchers
    // ========================================================================
    
    // Get optimal memory kernel for float
    MemKernelFnPtr<float> get_mem_kernel_float() const {
        if (force_scalar_) {
            return kernel_mem_scalar_float;
        }
        return ::get_mem_kernel_float(false);
    }
    
    // Get optimal memory kernel for double
    MemKernelFnPtr<double> get_mem_kernel_double() const {
        if (force_scalar_) {
            return kernel_mem_scalar_double;
        }
        return ::get_mem_kernel_double(false);
    }
    
    // Template version for generic code
    template<typename T>
    MemKernelFnPtr<T> get_mem_kernel() const;
    
    // ========================================================================
    // Stencil kernel dispatchers 
    // ========================================================================
    
    // Get optimal stencil kernel for float
    StencilKernelFnPtr<float> get_stencil_kernel_float() const {
        if (force_scalar_) {
            return kernel_stencil_scalar_float;
        }
        return ::get_stencil_kernel_float(false);
    }
    
    // Get optimal stencil kernel for double
    StencilKernelFnPtr<double> get_stencil_kernel_double() const {
        if (force_scalar_) {
            return kernel_stencil_scalar_double;
        }
        return ::get_stencil_kernel_double(false);
    }
    
    // Template version for generic code
    template<typename T>
    StencilKernelFnPtr<T> get_stencil_kernel() const;
    
    // ========================================================================
    // Matmul kernel dispatchers 
    // ========================================================================
    
    // Get optimal matmul kernel for float
    // Note: Currently only scalar implementation exists
    MatmulKernelFnPtr<float> get_matmul_kernel_float() const {
        // Matmul uses the generic kernel_matmul3d template
        // Return a wrapper that calls the template
        return &matmul_wrapper_float;
    }
    
    // Get optimal matmul kernel for double
    MatmulKernelFnPtr<double> get_matmul_kernel_double() const {
        return &matmul_wrapper_double;
    }
    
    // Template version for generic code
    template<typename T>
    MatmulKernelFnPtr<T> get_matmul_kernel() const;
    
    // ========================================================================
    // Kernel name getters
    // ========================================================================
    
    const char* get_kernel_name_float() const {
        if (force_scalar_) return "Scalar";
        return ::get_selected_kernel_name_float(false);
    }
    
    const char* get_kernel_name_double() const {
        if (force_scalar_) return "Scalar";
        return ::get_selected_kernel_name_double(false);
    }
    
    template<typename T>
    const char* get_kernel_name() const;

private:
    KernelDispatcher() : force_scalar_(false) {
        update_selected_level();
    }
    
    void update_selected_level() {
        if (force_scalar_) {
            selected_level_ = SimdLevel::Scalar;
        } else {
            selected_level_ = CpuCapabilities::get().get_simd_level();
        }
    }
    
    // Static wrapper functions for matmul (to match function pointer signature)
    static void matmul_wrapper_float(float* C, const float* A, const float* B,
                                     size_t z_begin, size_t z_end, size_t N) {
        kernel_matmul3d<float>(C, A, B, z_begin, z_end, N);
    }
    
    static void matmul_wrapper_double(double* C, const double* A, const double* B,
                                      size_t z_begin, size_t z_end, size_t N) {
        kernel_matmul3d<double>(C, A, B, z_begin, z_end, N);
    }
    
    SimdLevel selected_level_;
    bool force_scalar_;
};

// ============================================================================
// Template specializations
// ============================================================================

template<>
inline MemKernelFnPtr<float> KernelDispatcher::get_mem_kernel<float>() const {
    return get_mem_kernel_float();
}

template<>
inline MemKernelFnPtr<double> KernelDispatcher::get_mem_kernel<double>() const {
    return get_mem_kernel_double();
}

template<>
inline StencilKernelFnPtr<float> KernelDispatcher::get_stencil_kernel<float>() const {
    return get_stencil_kernel_float();
}

template<>
inline StencilKernelFnPtr<double> KernelDispatcher::get_stencil_kernel<double>() const {
    return get_stencil_kernel_double();
}

template<>
inline MatmulKernelFnPtr<float> KernelDispatcher::get_matmul_kernel<float>() const {
    return get_matmul_kernel_float();
}

template<>
inline MatmulKernelFnPtr<double> KernelDispatcher::get_matmul_kernel<double>() const {
    return get_matmul_kernel_double();
}

template<>
inline const char* KernelDispatcher::get_kernel_name<float>() const {
    return get_kernel_name_float();
}

template<>
inline const char* KernelDispatcher::get_kernel_name<double>() const {
    return get_kernel_name_double();
}

// ============================================================================
// Convenience free functions
// ============================================================================

// Get the global dispatcher instance
inline KernelDispatcher& get_kernel_dispatcher() {
    return KernelDispatcher::instance();
}

// Log kernel selection info 
inline std::string get_kernel_selection_log(bool verbose = false) {
    const auto& dispatcher = KernelDispatcher::instance();
    const auto& caps = CpuCapabilities::get();
    
    std::ostringstream oss;
    
    if (verbose) {
        oss << "=== Kernel Selection ===\n";
        oss << caps.to_string() << "\n";
        oss << dispatcher.get_selected_kernel_info() << "\n";
        oss << "Float kernel: " << dispatcher.get_kernel_name_float() << "\n";
        oss << "Double kernel: " << dispatcher.get_kernel_name_double() << "\n";
    } else {
        oss << "SIMD: " << simd_level_to_string(dispatcher.get_selected_level());
        if (dispatcher.is_force_scalar()) {
            oss << " (forced scalar)";
        }
    }
    
    return oss.str();
}
