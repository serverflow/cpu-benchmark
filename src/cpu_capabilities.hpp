#pragma once
// CPU Benchmark - CPU Capabilities Detection



#include "types.hpp"
#include <atomic>
#include <string>

// Platform-specific includes for CPUID
#ifdef _WIN32
    #include <intrin.h>
#elif defined(__linux__) || defined(__APPLE__)
    #if defined(__x86_64__) || defined(__i386__)
        #include <cpuid.h>
    #endif
    #if defined(__aarch64__)
        #ifdef __linux__
            #include <sys/auxv.h>
            #include <asm/hwcap.h>
        #endif
    #endif
#endif

// SIMD Level enum 
enum class SimdLevel {
    Scalar = 0,
    SSE2 = 1,
    SSE4_2 = 2,
    AVX = 3,
    AVX2 = 4,
    AVX512 = 5,
    NEON = 10,
    NEON_FP16 = 11
};

// Convert SimdLevel to string
inline std::string simd_level_to_string(SimdLevel level) {
    switch (level) {
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

// Priority comparison for SIMD levels
inline bool is_higher_priority(SimdLevel a, SimdLevel b) {
    return static_cast<int>(a) > static_cast<int>(b);
}

// CPU capabilities structure 
struct CpuCapabilities {
    // x86-64 SIMD
    bool has_sse2;
    bool has_sse4_2;
    bool has_avx;
    bool has_avx2;
    bool has_avx512f;
    bool has_avx512_fp16;       // AVX-512 FP16 support (x86-64)
    bool has_avx512_vnni;       // AVX-512 VNNI support
    
    // ARM64 SIMD 
    bool has_arm_neon;
    bool has_arm_neon_fp16;     // ARM NEON FP16 support (ARM64)
    
    // Derived flags
    bool fp16_native_available; // Any native FP16 support available
    
    // Singleton accessor with caching
    static const CpuCapabilities& get();
    
    // Get highest available SIMD level 
    SimdLevel get_simd_level() const;
    
    // Get human-readable description
    std::string to_string() const;
    
private:
    CpuCapabilities();
    static CpuCapabilities detect();
};


// Helper: Check OS support for AVX via XGETBV
inline bool check_os_avx_support() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef _WIN32
        // Check OSXSAVE bit first (CPUID.01H:ECX[bit 27])
        int info[4] = {0};
        __cpuid(info, 1);
        bool has_osxsave = (info[2] & (1 << 27)) != 0;
        if (!has_osxsave) return false;
        
        // Check XCR0 for AVX state support (bits 1, 2 for XMM, YMM)
        unsigned long long xcr0 = _xgetbv(0);
        return ((xcr0 & 0x6) == 0x6);
    #else
        // Linux/macOS
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0) return false;
        bool has_osxsave = (ecx & (1 << 27)) != 0;
        if (!has_osxsave) return false;
        
        unsigned int xcr0_lo, xcr0_hi;
        __asm__ __volatile__ (
            "xgetbv"
            : "=a"(xcr0_lo), "=d"(xcr0_hi)
            : "c"(0)
        );
        return ((xcr0_lo & 0x6) == 0x6);
    #endif
#else
    return false;
#endif
}

// Helper: Check OS support for AVX-512 via XGETBV
inline bool check_os_avx512_support() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef _WIN32
        int info[4] = {0};
        __cpuid(info, 1);
        bool has_osxsave = (info[2] & (1 << 27)) != 0;
        if (!has_osxsave) return false;
        
        // Check XCR0 for AVX-512 state support (bits 5, 6, 7 for opmask, ZMM_Hi256, Hi16_ZMM)
        unsigned long long xcr0 = _xgetbv(0);
        return ((xcr0 & 0xE6) == 0xE6);
    #else
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0) return false;
        bool has_osxsave = (ecx & (1 << 27)) != 0;
        if (!has_osxsave) return false;
        
        unsigned int xcr0_lo, xcr0_hi;
        __asm__ __volatile__ (
            "xgetbv"
            : "=a"(xcr0_lo), "=d"(xcr0_hi)
            : "c"(0)
        );
        return ((xcr0_lo & 0xE6) == 0xE6);
    #endif
#else
    return false;
#endif
}

// Detect SSE2 support
inline bool detect_sse2() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef _WIN32
        int info[4] = {0};
        __cpuid(info, 1);
        // SSE2: CPUID.01H:EDX[bit 26]
        return (info[3] & (1 << 26)) != 0;
    #else
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0) return false;
        return (edx & (1 << 26)) != 0;
    #endif
#else
    return false;
#endif
}

// Detect SSE4.2 support 
inline bool detect_sse4_2() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef _WIN32
        int info[4] = {0};
        __cpuid(info, 1);
        // SSE4.2: CPUID.01H:ECX[bit 20]
        return (info[2] & (1 << 20)) != 0;
    #else
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0) return false;
        return (ecx & (1 << 20)) != 0;
    #endif
#else
    return false;
#endif
}

// Detect AVX support
inline bool detect_avx() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef _WIN32
        int info[4] = {0};
        __cpuid(info, 1);
        // AVX: CPUID.01H:ECX[bit 28]
        bool has_avx = (info[2] & (1 << 28)) != 0;
        return has_avx && check_os_avx_support();
    #else
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0) return false;
        bool has_avx = (ecx & (1 << 28)) != 0;
        return has_avx && check_os_avx_support();
    #endif
#else
    return false;
#endif
}

// Detect AVX2 support 
inline bool detect_avx2() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef _WIN32
        int info[4] = {0};
        // Check max CPUID level
        __cpuid(info, 0);
        if (info[0] < 7) return false;
        
        // AVX2: CPUID.07H.0H:EBX[bit 5]
        __cpuidex(info, 7, 0);
        bool has_avx2 = (info[1] & (1 << 5)) != 0;
        return has_avx2 && check_os_avx_support();
    #else
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid(0, &eax, &ebx, &ecx, &edx) == 0 || eax < 7) return false;
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) return false;
        bool has_avx2 = (ebx & (1 << 5)) != 0;
        return has_avx2 && check_os_avx_support();
    #endif
#else
    return false;
#endif
}

// Detect AVX-512F support 
inline bool detect_avx512f() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef _WIN32
        int info[4] = {0};
        __cpuid(info, 0);
        if (info[0] < 7) return false;
        
        // AVX-512F: CPUID.07H.0H:EBX[bit 16]
        __cpuidex(info, 7, 0);
        bool has_avx512f = (info[1] & (1 << 16)) != 0;
        return has_avx512f && check_os_avx512_support();
    #else
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid(0, &eax, &ebx, &ecx, &edx) == 0 || eax < 7) return false;
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) return false;
        bool has_avx512f = (ebx & (1 << 16)) != 0;
        return has_avx512f && check_os_avx512_support();
    #endif
#else
    return false;
#endif
}


// Detect AVX-512 FP16 support using CPUID on x86-64
inline bool detect_avx512_fp16() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef _WIN32
        int info[4] = {0};
        __cpuid(info, 0);
        if (info[0] < 7) return false;
        
        // Check AVX-512 FP16 (CPUID.07H.0H:EDX[bit 23])
        __cpuidex(info, 7, 0);
        bool has_avx512_fp16 = (info[3] & (1 << 23)) != 0;
        return has_avx512_fp16 && check_os_avx512_support();
    #else
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid(0, &eax, &ebx, &ecx, &edx) == 0 || eax < 7) return false;
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) return false;
        bool has_avx512_fp16 = (edx & (1 << 23)) != 0;
        return has_avx512_fp16 && check_os_avx512_support();
    #endif
#else
    return false;
#endif
}

// Detect AVX-512 VNNI support 
inline bool detect_avx512_vnni() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef _WIN32
        int info[4] = {0};
        __cpuid(info, 0);
        if (info[0] < 7) return false;
        
        // AVX-512 VNNI: CPUID.07H.0H:ECX[bit 11]
        __cpuidex(info, 7, 0);
        bool has_avx512_vnni = (info[2] & (1 << 11)) != 0;
        return has_avx512_vnni && check_os_avx512_support();
    #else
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid(0, &eax, &ebx, &ecx, &edx) == 0 || eax < 7) return false;
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) return false;
        bool has_avx512_vnni = (ecx & (1 << 11)) != 0;
        return has_avx512_vnni && check_os_avx512_support();
    #endif
#else
    return false;
#endif
}

// Detect ARM NEON support
inline bool detect_arm_neon() {
#if defined(__aarch64__) || defined(_M_ARM64)
    // ARM64 always has NEON
    return true;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    return true;
#else
    return false;
#endif
}

// Detect ARM NEON FP16 support 
inline bool detect_arm_neon_fp16() {
#if defined(__aarch64__) || defined(_M_ARM64)
    #if defined(__linux__)
        // Linux ARM64: check HWCAP for FP16 support
        unsigned long hwcap = getauxval(AT_HWCAP);
        // HWCAP_FPHP (bit 9) indicates FP16 support
        // HWCAP_ASIMDHP (bit 10) indicates Advanced SIMD FP16 support
        return (hwcap & HWCAP_FPHP) != 0 && (hwcap & HWCAP_ASIMDHP) != 0;
    #elif defined(__APPLE__)
        // Apple Silicon (M1/M2/etc.) always supports FP16
        return true;
    #elif defined(_WIN32)
        // Windows ARM64: assume FP16 support on modern ARM64 chips
        return true;
    #else
        return false;
    #endif
#else
    return false;
#endif
}

// CpuCapabilities implementation
inline CpuCapabilities CpuCapabilities::detect() {
    CpuCapabilities caps;
    
    // x86-64 SIMD detection 
    caps.has_sse2 = detect_sse2();
    caps.has_sse4_2 = detect_sse4_2();
    caps.has_avx = detect_avx();
    caps.has_avx2 = detect_avx2();
    caps.has_avx512f = detect_avx512f();
    caps.has_avx512_fp16 = detect_avx512_fp16();
    caps.has_avx512_vnni = detect_avx512_vnni();
    
    // ARM64 SIMD detection 
    caps.has_arm_neon = detect_arm_neon();
    caps.has_arm_neon_fp16 = detect_arm_neon_fp16();
    
    // Derived flags
    caps.fp16_native_available = caps.has_avx512_fp16 || caps.has_arm_neon_fp16;
    
    return caps;
}

inline CpuCapabilities::CpuCapabilities()
    : has_sse2(false)
    , has_sse4_2(false)
    , has_avx(false)
    , has_avx2(false)
    , has_avx512f(false)
    , has_avx512_fp16(false)
    , has_avx512_vnni(false)
    , has_arm_neon(false)
    , has_arm_neon_fp16(false)
    , fp16_native_available(false) {
}

// Singleton with thread-safe lazy initialization 
inline const CpuCapabilities& CpuCapabilities::get() {
    static const CpuCapabilities instance = detect();
    return instance;
}

// Get highest available SIMD level 
inline SimdLevel CpuCapabilities::get_simd_level() const {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    // x86-64: prefer higher capability level (AVX-512 > AVX2 > AVX > SSE4.2 > SSE2 > Scalar)
    if (has_avx512f) return SimdLevel::AVX512;
    if (has_avx2) return SimdLevel::AVX2;
    if (has_avx) return SimdLevel::AVX;
    if (has_sse4_2) return SimdLevel::SSE4_2;
    if (has_sse2) return SimdLevel::SSE2;
    return SimdLevel::Scalar;
#elif defined(__aarch64__) || defined(_M_ARM64)
    // ARM64: prefer NEON FP16 > NEON > Scalar
    if (has_arm_neon_fp16) return SimdLevel::NEON_FP16;
    if (has_arm_neon) return SimdLevel::NEON;
    return SimdLevel::Scalar;
#else
    return SimdLevel::Scalar;
#endif
}

// Get human-readable description of capabilities
inline std::string CpuCapabilities::to_string() const {
    std::string result = "SIMD Capabilities:\n";
    
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    result += "  SSE2:        " + std::string(has_sse2 ? "Yes" : "No") + "\n";
    result += "  SSE4.2:      " + std::string(has_sse4_2 ? "Yes" : "No") + "\n";
    result += "  AVX:         " + std::string(has_avx ? "Yes" : "No") + "\n";
    result += "  AVX2:        " + std::string(has_avx2 ? "Yes" : "No") + "\n";
    result += "  AVX-512F:    " + std::string(has_avx512f ? "Yes" : "No") + "\n";
    result += "  AVX-512 FP16:" + std::string(has_avx512_fp16 ? "Yes" : "No") + "\n";
    result += "  AVX-512 VNNI:" + std::string(has_avx512_vnni ? "Yes" : "No") + "\n";
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    result += "  ARM NEON:    " + std::string(has_arm_neon ? "Yes" : "No") + "\n";
    result += "  NEON FP16:   " + std::string(has_arm_neon_fp16 ? "Yes" : "No") + "\n";
#endif

    result += "  Active Level: " + simd_level_to_string(get_simd_level()) + "\n";
    result += "  FP16 Native:  " + std::string(fp16_native_available ? "Yes" : "No");
    
    return result;
}

// Get FP16 execution mode based on hardware capabilities 
inline FP16Mode get_fp16_mode() {
    const auto& caps = CpuCapabilities::get();
    return caps.fp16_native_available ? FP16Mode::Native : FP16Mode::Emulated;
}

// Get a human-readable string describing FP16 support
inline const char* get_fp16_support_description() {
    const auto& caps = CpuCapabilities::get();
    if (caps.has_avx512_fp16) {
        return "AVX-512 FP16 (native)";
    } else if (caps.has_arm_neon_fp16) {
        return "ARM NEON FP16 (native)";
    } else {
        return "Software emulation";
    }
}

// Get comma-separated string of available CPU instructions
inline std::string get_cpu_instructions_string() {
    const auto& caps = CpuCapabilities::get();
    std::string result;
    
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if (caps.has_sse2) {
        if (!result.empty()) result += ",";
        result += "SSE2";
    }
    if (caps.has_sse4_2) {
        if (!result.empty()) result += ",";
        result += "SSE4.2";
    }
    if (caps.has_avx) {
        if (!result.empty()) result += ",";
        result += "AVX";
    }
    if (caps.has_avx2) {
        if (!result.empty()) result += ",";
        result += "AVX2";
    }
    if (caps.has_avx512f) {
        if (!result.empty()) result += ",";
        result += "AVX-512F";
    }
    if (caps.has_avx512_fp16) {
        if (!result.empty()) result += ",";
        result += "AVX-512_FP16";
    }
    if (caps.has_avx512_vnni) {
        if (!result.empty()) result += ",";
        result += "AVX-512_VNNI";
    }
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    if (caps.has_arm_neon) {
        if (!result.empty()) result += ",";
        result += "NEON";
    }
    if (caps.has_arm_neon_fp16) {
        if (!result.empty()) result += ",";
        result += "NEON_FP16";
    }
#endif

    return result.empty() ? "Scalar" : result;
}
