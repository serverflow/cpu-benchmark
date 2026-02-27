#pragma once
// CPU Benchmark - Extended Benchmark Output
// SIMD Optimization 
#include "cpu_capabilities.hpp"
#include "platform.hpp"
#include "types.hpp"
#include <string>
#include <sstream>
#include <iomanip>

// Forward declaration for FP16 diagnostic functions from precision_dispatcher.hpp
// These are defined there to avoid circular dependencies
inline bool is_native_fp16_compiled();
inline FP16Mode get_actual_fp16_mode();

// Format SIMD capabilities for display 
// Returns a formatted string showing all SIMD flags with their support status
// and the active FP16 mode (considering both compile-time and runtime)
inline std::string format_simd_capabilities(const CpuCapabilities& caps) {
    std::ostringstream oss;
    
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    oss << "  SSE2:          " << (caps.has_sse2 ? "Yes" : "No") << "\n";
    oss << "  SSE4.2:        " << (caps.has_sse4_2 ? "Yes" : "No") << "\n";
    oss << "  AVX:           " << (caps.has_avx ? "Yes" : "No") << "\n";
    oss << "  AVX2:          " << (caps.has_avx2 ? "Yes" : "No") << "\n";
    oss << "  AVX-512F:      " << (caps.has_avx512f ? "Yes" : "No") << "\n";
    oss << "  AVX-512 FP16:  " << (caps.has_avx512_fp16 ? "Yes" : "No") << "\n";
    oss << "  AVX-512 VNNI:  " << (caps.has_avx512_vnni ? "Yes" : "No") << "\n";
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    oss << "  ARM NEON:      " << (caps.has_arm_neon ? "Yes" : "No") << "\n";
    oss << "  NEON FP16:     " << (caps.has_arm_neon_fp16 ? "Yes" : "No") << "\n";
#endif

    // Active FP16 mode - show ACTUAL execution path, not just CPU capability
    // This considers both compile-time support and runtime availability
    oss << "  FP16 Mode:     ";
    
    // Check if native FP16 code was compiled
    bool native_compiled = false;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
    native_compiled = true;
#endif
    
    bool cpu_supports_native = caps.has_avx512_fp16 || caps.has_arm_neon_fp16;
    
    if (native_compiled && cpu_supports_native) {
        // Both compiled and runtime support - will use native path
        if (caps.has_avx512_fp16) {
            oss << "Native (AVX-512 FP16)";
        } else if (caps.has_arm_neon_fp16) {
            oss << "Native (ARM NEON FP16)";
        }
    } else if (cpu_supports_native && !native_compiled) {
        // CPU supports but binary wasn't compiled with support
        oss << "Emulated (CPU supports native, but binary not compiled with FP16)";
    } else {
        // No native support
        oss << "Emulated";
    }
    
    return oss.str();
}

// Format cache information for display
// Returns a formatted string showing L1/L2/L3 cache sizes and cache line size
// Handles unavailable cache levels gracefully
inline std::string format_cache_info(const CacheInfo& cache, unsigned physical_cores = 0) {
    std::ostringstream oss;
    
    auto format_size_kb = [](size_t bytes) -> std::string {
        if (bytes == 0) return "N/A";
        std::ostringstream s;
        s << (bytes / 1024) << " KB";
        return s.str();
    };

    auto format_size_pretty = [](size_t bytes) -> std::string {
        if (bytes == 0) return "N/A";
        const size_t mb = 1024 * 1024;
        if (bytes >= mb) {
            if (bytes % mb == 0) {
                return std::to_string(bytes / mb) + " MB";
            }
            std::ostringstream s;
            s << std::fixed << std::setprecision(1) << (bytes / (1024.0 * 1024.0)) << " MB";
            return s.str();
        }
        if (bytes >= 1024) {
            return std::to_string(bytes / 1024) + " KB";
        }
        return std::to_string(bytes) + " B";
    };

    unsigned core_count = physical_cores;
    if (core_count == 0) {
        core_count = get_logical_core_count();
    }
    if (core_count == 0) {
        core_count = 1;
    }

    auto format_per_core = [&](size_t bytes) -> std::string {
        if (bytes == 0) return "N/A";
        std::ostringstream s;
        size_t total_bytes = bytes * static_cast<size_t>(core_count);
        s << core_count << " x " << format_size_kb(bytes)
          << " (" << format_size_pretty(total_bytes) << ")";
        return s.str();
    };
    
    auto cache_row = [&](const std::string& label, const std::string& value) {
        const int label_w = 22;
        oss << "  " << std::left << std::setw(label_w) << label << value << "\n";
    };

    // L1 Cache
    if (cache.l1_available) {
        if (cache.l1_inst_size > 0) {
            cache_row("L1 Instruction Cache:", format_per_core(cache.l1_inst_size));
        }
        if (cache.l1_data_size > 0) {
            cache_row("L1 Data Cache:", format_per_core(cache.l1_data_size));
        }
    } else {
        cache_row("L1 Cache:", "N/A");
    }
    
    // L2 Cache
    if (cache.l2_available && cache.l2_size > 0) {
        cache_row("L2 Cache:", format_per_core(cache.l2_size));
    } else {
        cache_row("L2 Cache:", "N/A");
    }
    
    // L3 Cache
    if (cache.l3_available && cache.l3_size > 0) {
#if defined(__APPLE__) && defined(__aarch64__)
        // Apple Silicon uses distributed System Level Cache (SLC)
        cache_row("L3 Cache:", format_size_pretty(cache.l3_size) + " (SLC, distributed)");
#else
        cache_row("L3 Cache:", format_size_pretty(cache.l3_size));
#endif
    } else {
#if defined(__APPLE__) && defined(__aarch64__)
        // Apple Silicon uses System Level Cache (SLC) instead of traditional L3
        cache_row("L3 Cache:", "Distributed (SLC)");
#else
        cache_row("L3 Cache:", "N/A");
#endif
    }
    
    // Cache line size
    if (cache.cache_line_size > 0) {
        cache_row("Line Size:", std::to_string(cache.cache_line_size) + " bytes");
    } else {
        cache_row("Line Size:", "N/A");
    }
    
    return oss.str();
}

// Get FP16 mode string for output
// This considers both compile-time and runtime availability
inline std::string get_fp16_mode_string(const CpuCapabilities& caps) {
    // Check if native FP16 code was compiled
    bool native_compiled = false;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
    native_compiled = true;
#endif
    
    bool cpu_supports_native = caps.has_avx512_fp16 || caps.has_arm_neon_fp16;
    
    if (native_compiled && cpu_supports_native) {
        if (caps.has_avx512_fp16) {
            return "native (AVX-512 FP16)";
        } else if (caps.has_arm_neon_fp16) {
            return "native (ARM NEON FP16)";
        }
    } else if (cpu_supports_native && !native_compiled) {
        return "emulated (CPU supports native, binary not compiled with FP16)";
    }
    return "emulated";
}

// Format SIMD capabilities as JSON object
inline std::string format_simd_capabilities_json(const CpuCapabilities& caps) {
    std::ostringstream oss;
    oss << "{\n";
    oss << "      \"sse2\": " << (caps.has_sse2 ? "true" : "false") << ",\n";
    oss << "      \"sse4_2\": " << (caps.has_sse4_2 ? "true" : "false") << ",\n";
    oss << "      \"avx\": " << (caps.has_avx ? "true" : "false") << ",\n";
    oss << "      \"avx2\": " << (caps.has_avx2 ? "true" : "false") << ",\n";
    oss << "      \"avx512f\": " << (caps.has_avx512f ? "true" : "false") << ",\n";
    oss << "      \"avx512_fp16\": " << (caps.has_avx512_fp16 ? "true" : "false") << ",\n";
    oss << "      \"avx512_vnni\": " << (caps.has_avx512_vnni ? "true" : "false") << ",\n";
    oss << "      \"arm_neon\": " << (caps.has_arm_neon ? "true" : "false") << ",\n";
    oss << "      \"arm_neon_fp16\": " << (caps.has_arm_neon_fp16 ? "true" : "false") << "\n";
    oss << "    }";
    return oss.str();
}

// Format cache information as JSON object
inline std::string format_cache_info_json(const CacheInfo& cache) {
    std::ostringstream oss;
    oss << "{\n";
    
    // Convert to KB for JSON output
    auto to_kb = [](size_t bytes) -> size_t {
        return bytes / 1024;
    };
    
    oss << "      \"l1_data_kb\": " << (cache.l1_available ? std::to_string(to_kb(cache.l1_data_size)) : "null") << ",\n";
    oss << "      \"l1_inst_kb\": " << (cache.l1_available ? std::to_string(to_kb(cache.l1_inst_size)) : "null") << ",\n";
    oss << "      \"l2_kb\": " << (cache.l2_available ? std::to_string(to_kb(cache.l2_size)) : "null") << ",\n";
    oss << "      \"l3_kb\": " << (cache.l3_available ? std::to_string(to_kb(cache.l3_size)) : "null") << ",\n";
    oss << "      \"line_size\": " << cache.cache_line_size << "\n";
    oss << "    }";
    return oss.str();
}

