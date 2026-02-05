// CPU Benchmark - Runtime SIMD Dispatcher Implementation

#include "runtime_dispatcher.hpp"
#include <iostream>
#include <sstream>

// Static member initialization
bool RuntimeDispatcher::initialized_ = false;
SimdLevel RuntimeDispatcher::active_level_ = SimdLevel::Scalar;
SimdLevel RuntimeDispatcher::detected_level_ = SimdLevel::Scalar;
bool RuntimeDispatcher::forced_ = false;

MemKernelFn<float> RuntimeDispatcher::mem_kernel_float_ = nullptr;
MemKernelFn<double> RuntimeDispatcher::mem_kernel_double_ = nullptr;
MemKernelFn<int8_t> RuntimeDispatcher::mem_kernel_int8_ = nullptr;
StencilKernelFn<float> RuntimeDispatcher::stencil_kernel_float_ = nullptr;
StencilKernelFn<double> RuntimeDispatcher::stencil_kernel_double_ = nullptr;
StencilKernelFn<int8_t> RuntimeDispatcher::stencil_kernel_int8_ = nullptr;
ComputeKernelFloatFn RuntimeDispatcher::compute_kernel_float_ = nullptr;
ComputeKernelDoubleFn RuntimeDispatcher::compute_kernel_double_ = nullptr;

void RuntimeDispatcher::initialize() {
    if (initialized_) {
        return;
    }
    
    // Detect CPU capabilities
    const auto& caps = CpuCapabilities::get();
    detected_level_ = caps.get_simd_level();
    active_level_ = detected_level_;
    
    // Select kernels based on detected level
    select_kernels(active_level_);
    
    initialized_ = true;
    
    // Log the selected level
    std::cerr << "[RuntimeDispatcher] Initialized with SIMD level: " 
              << simd_level_to_string(active_level_) << std::endl;
}

bool RuntimeDispatcher::is_initialized() {
    return initialized_;
}

SimdLevel RuntimeDispatcher::get_active_level() {
    if (!initialized_) {
        initialize();
    }
    return active_level_;
}

const char* RuntimeDispatcher::get_active_level_name() {
    if (!initialized_) {
        initialize();
    }
    
    switch (active_level_) {
        case SimdLevel::Scalar: return "Scalar";
        case SimdLevel::SSE2: return "SSE2";
        case SimdLevel::SSE4_2: return "SSE4.2";
        case SimdLevel::AVX: return "AVX";
        case SimdLevel::AVX2: return "AVX2";
        case SimdLevel::AVX512: return "AVX-512";
        case SimdLevel::NEON: return "ARM NEON";
        case SimdLevel::NEON_FP16: return "ARM NEON FP16";
    }
    return "Unknown";
}

void RuntimeDispatcher::select_kernels(SimdLevel level) {
    // Select kernels based on SIMD level
    // Priority: AVX-512 > AVX2 > AVX > SSE2 > Scalar (x86)
    //           NEON > Scalar (ARM)
    // 
    // Note: All cases are always compiled. Kernel functions have stub implementations
    // for non-native platforms that fall back to scalar. This ensures the switch
    // statement is complete regardless of compile-time platform detection.
    
    switch (level) {
        case SimdLevel::AVX512:
            mem_kernel_float_ = kernels::avx512::mem_float;
            mem_kernel_double_ = kernels::avx512::mem_double;
            mem_kernel_int8_ = kernels::avx2::mem_int8;  // AVX-512 INT8 uses AVX2 for now
            stencil_kernel_float_ = kernels::avx512::stencil_float;
            stencil_kernel_double_ = kernels::avx512::stencil_double;
            stencil_kernel_int8_ = kernels::scalar::stencil_int8;
            compute_kernel_float_ = kernels::compute::avx512_float;
            compute_kernel_double_ = kernels::compute::avx512_double;
            break;
            
        case SimdLevel::AVX2:
            mem_kernel_float_ = kernels::avx2::mem_float;
            mem_kernel_double_ = kernels::avx2::mem_double;
            mem_kernel_int8_ = kernels::avx2::mem_int8;
            stencil_kernel_float_ = kernels::avx2::stencil_float;
            stencil_kernel_double_ = kernels::avx2::stencil_double;
            stencil_kernel_int8_ = kernels::scalar::stencil_int8;
            compute_kernel_float_ = kernels::compute::avx2_float;
            compute_kernel_double_ = kernels::compute::avx2_double;
            break;
            
        case SimdLevel::AVX:
            mem_kernel_float_ = kernels::avx::mem_float;
            mem_kernel_double_ = kernels::avx::mem_double;
            mem_kernel_int8_ = kernels::sse2::mem_int8;
            stencil_kernel_float_ = kernels::avx::stencil_float;
            stencil_kernel_double_ = kernels::avx::stencil_double;
            stencil_kernel_int8_ = kernels::scalar::stencil_int8;
            compute_kernel_float_ = kernels::compute::avx_float;
            compute_kernel_double_ = kernels::compute::avx_double;
            break;
            
        case SimdLevel::SSE4_2:
        case SimdLevel::SSE2:
            mem_kernel_float_ = kernels::sse2::mem_float;
            mem_kernel_double_ = kernels::sse2::mem_double;
            mem_kernel_int8_ = kernels::sse2::mem_int8;
            stencil_kernel_float_ = kernels::sse2::stencil_float;
            stencil_kernel_double_ = kernels::sse2::stencil_double;
            stencil_kernel_int8_ = kernels::scalar::stencil_int8;
            compute_kernel_float_ = kernels::compute::sse2_float;
            compute_kernel_double_ = kernels::compute::sse2_double;
            break;

        case SimdLevel::NEON_FP16:
        case SimdLevel::NEON:
            mem_kernel_float_ = kernels::neon::mem_float;
            mem_kernel_double_ = kernels::scalar::mem_double;  // NEON doesn't have double
            mem_kernel_int8_ = kernels::scalar::mem_int8;
            stencil_kernel_float_ = kernels::neon::stencil_float;
            stencil_kernel_double_ = kernels::scalar::stencil_double;
            stencil_kernel_int8_ = kernels::scalar::stencil_int8;
            compute_kernel_float_ = kernels::compute::neon_float;
            compute_kernel_double_ = kernels::compute::neon_double;
            break;

        case SimdLevel::Scalar:
        default:
            mem_kernel_float_ = kernels::scalar::mem_float;
            mem_kernel_double_ = kernels::scalar::mem_double;
            mem_kernel_int8_ = kernels::scalar::mem_int8;
            stencil_kernel_float_ = kernels::scalar::stencil_float;
            stencil_kernel_double_ = kernels::scalar::stencil_double;
            stencil_kernel_int8_ = kernels::scalar::stencil_int8;
            compute_kernel_float_ = kernels::compute::scalar_float;
            compute_kernel_double_ = kernels::compute::scalar_double;
            break;
    }
}

MemKernelFn<float> RuntimeDispatcher::get_mem_kernel_float() {
    if (!initialized_) {
        initialize();
    }
    return mem_kernel_float_;
}

MemKernelFn<double> RuntimeDispatcher::get_mem_kernel_double() {
    if (!initialized_) {
        initialize();
    }
    return mem_kernel_double_;
}

MemKernelFn<int8_t> RuntimeDispatcher::get_mem_kernel_int8() {
    if (!initialized_) {
        initialize();
    }
    return mem_kernel_int8_;
}

StencilKernelFn<float> RuntimeDispatcher::get_stencil_kernel_float() {
    if (!initialized_) {
        initialize();
    }
    return stencil_kernel_float_;
}

StencilKernelFn<double> RuntimeDispatcher::get_stencil_kernel_double() {
    if (!initialized_) {
        initialize();
    }
    return stencil_kernel_double_;
}

StencilKernelFn<int8_t> RuntimeDispatcher::get_stencil_kernel_int8() {
    if (!initialized_) {
        initialize();
    }
    return stencil_kernel_int8_;
}

ComputeKernelFloatFn RuntimeDispatcher::get_compute_kernel_float() {
    if (!initialized_) {
        initialize();
    }
    return compute_kernel_float_;
}

ComputeKernelDoubleFn RuntimeDispatcher::get_compute_kernel_double() {
    if (!initialized_) {
        initialize();
    }
    return compute_kernel_double_;
}

void RuntimeDispatcher::force_level(SimdLevel level) {
    if (!initialized_) {
        initialize();
    }
    
    active_level_ = level;
    forced_ = true;
    select_kernels(level);
    
    std::cerr << "[RuntimeDispatcher] Forced SIMD level: " 
              << simd_level_to_string(level) << std::endl;
}

void RuntimeDispatcher::reset_to_auto() {
    if (!initialized_) {
        initialize();
        return;
    }
    
    active_level_ = detected_level_;
    forced_ = false;
    select_kernels(active_level_);
    
    std::cerr << "[RuntimeDispatcher] Reset to auto-detected level: " 
              << simd_level_to_string(active_level_) << std::endl;
}

std::string RuntimeDispatcher::get_diagnostics() {
    if (!initialized_) {
        initialize();
    }
    
    std::ostringstream oss;
    oss << "RuntimeDispatcher Diagnostics:\n";
    oss << "  Initialized: " << (initialized_ ? "Yes" : "No") << "\n";
    oss << "  Detected Level: " << simd_level_to_string(detected_level_) << "\n";
    oss << "  Active Level: " << simd_level_to_string(active_level_) << "\n";
    oss << "  Forced: " << (forced_ ? "Yes" : "No") << "\n";
    oss << "  Kernel Pointers:\n";
    oss << "    mem_float: " << (mem_kernel_float_ ? "Set" : "NULL") << "\n";
    oss << "    mem_double: " << (mem_kernel_double_ ? "Set" : "NULL") << "\n";
    oss << "    mem_int8: " << (mem_kernel_int8_ ? "Set" : "NULL") << "\n";
    oss << "    stencil_float: " << (stencil_kernel_float_ ? "Set" : "NULL") << "\n";
    oss << "    stencil_double: " << (stencil_kernel_double_ ? "Set" : "NULL") << "\n";
    oss << "    stencil_int8: " << (stencil_kernel_int8_ ? "Set" : "NULL") << "\n";
    oss << "    compute_float: " << (compute_kernel_float_ ? "Set" : "NULL") << "\n";
    oss << "    compute_double: " << (compute_kernel_double_ ? "Set" : "NULL") << "\n";
    
    return oss.str();
}
