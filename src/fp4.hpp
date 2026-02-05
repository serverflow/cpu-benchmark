#pragma once
// CPU Benchmark - FP4 (4-bit Float) Emulation

//
// FP4 Format Specification:
// - 4-bit floating point (emulated on CPU)
// - Format: 1 sign bit, 2 exponent bits, 1 mantissa bit
// - Range: approximately [-6, 6] with 16 discrete values
// - Two FP4 values packed per byte (low nibble = even index, high nibble = odd index)
//
// Value Mapping (16 discrete values):
// | 4-bit | Float Value |
// |-------|-------------|
// | 0000  |  0.0        |
// | 0001  |  0.5        |
// | 0010  |  1.0        |
// | 0011  |  1.5        |
// | 0100  |  2.0        |
// | 0101  |  3.0        |
// | 0110  |  4.0        |
// | 0111  |  6.0        |
// | 1000  | -0.0        |
// | 1001  | -0.5        |
// | 1010  | -1.0        |
// | 1011  | -1.5        |
// | 1100  | -2.0        |
// | 1101  | -3.0        |
// | 1110  | -4.0        |
// | 1111  | -6.0        |

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>

// ============================================================================
// SIMD Detection for FP4 Batch Operations
// Must be defined before FP4Array class
// ============================================================================

// Detect x86-64 platform for SIMD
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define FP4_SIMD_X86 1
#else
    #define FP4_SIMD_X86 0
#endif

// AVX2 detection for FP4 batch operations
#if FP4_SIMD_X86 && defined(__AVX2__)
    #define FP4_SIMD_AVX2 1
    #include <immintrin.h>
#else
    #define FP4_SIMD_AVX2 0
#endif

// SSE2 detection (baseline for x86-64)
#if FP4_SIMD_X86 && (defined(__SSE2__) || defined(_M_X64))
    #define FP4_SIMD_SSE2 1
    #if !FP4_SIMD_AVX2
        #include <emmintrin.h>
    #endif
#else
    #define FP4_SIMD_SSE2 0
#endif

// ARM NEON detection
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define FP4_SIMD_NEON 1
    #include <arm_neon.h>
#else
    #define FP4_SIMD_NEON 0
#endif

// SIMD chunk sizes based on register width 
// AVX2: 256-bit = 8 floats
// SSE2: 128-bit = 4 floats
// NEON: 128-bit = 4 floats
#if FP4_SIMD_AVX2
    constexpr size_t FP4_SIMD_CHUNK_SIZE = 8;
#elif FP4_SIMD_SSE2 || FP4_SIMD_NEON
    constexpr size_t FP4_SIMD_CHUNK_SIZE = 4;
#else
    constexpr size_t FP4_SIMD_CHUNK_SIZE = 1;
#endif

// ============================================================================
// FP4 Lookup Table and Basic Conversion Functions
// ============================================================================

// FP4 value lookup table (4-bit index -> float value)
// Index 0-7: positive values, Index 8-15: negative values
constexpr float FP4_TO_FLOAT_TABLE[16] = {
    0.0f,   // 0000
    0.5f,   // 0001
    1.0f,   // 0010
    1.5f,   // 0011
    2.0f,   // 0100
    3.0f,   // 0101
    4.0f,   // 0110
    6.0f,   // 0111
    -0.0f,  // 1000 (negative zero, treated as 0.0f)
    -0.5f,  // 1001
    -1.0f,  // 1010
    -1.5f,  // 1011
    -2.0f,  // 1100
    -3.0f,  // 1101
    -4.0f,  // 1110
    -6.0f   // 1111
};

// Convert 4-bit FP4 nibble to float 
inline float fp4_to_float(uint8_t nibble) {
    return FP4_TO_FLOAT_TABLE[nibble & 0x0F];
}

// Convert float to 4-bit FP4 value 
inline uint8_t float_to_fp4(float x) {
    uint8_t sign = (x < 0.0f) ? 0x08 : 0x00;
    float abs_x = std::fabs(x);
    uint8_t magnitude;
    
    if (abs_x < 0.25f) magnitude = 0;
    else if (abs_x < 0.75f) magnitude = 1;
    else if (abs_x < 1.25f) magnitude = 2;
    else if (abs_x < 1.75f) magnitude = 3;
    else if (abs_x < 2.5f) magnitude = 4;
    else if (abs_x < 3.5f) magnitude = 5;
    else if (abs_x < 5.0f) magnitude = 6;
    else magnitude = 7;
    
    return sign | magnitude;
}

// Extract FP4 value from packed byte 
inline float fp4_extract(uint8_t packed, bool high_nibble) {
    uint8_t nibble = high_nibble ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
    return fp4_to_float(nibble);
}

// Pack two FP4 values into a single byte 
inline uint8_t fp4_pack(float low, float high) {
    uint8_t low_nibble = float_to_fp4(low);
    uint8_t high_nibble = float_to_fp4(high);
    return (high_nibble << 4) | low_nibble;
}


// ============================================================================
// FP4 Batch Operations (Requirements 7.1, 7.2, 7.3, 7.4)
// Must be defined before FP4Array class
// ============================================================================

// Batch unpack: Convert multiple FP4 values to float array 
inline void fp4_batch_unpack(float* out, const uint8_t* packed_data, 
                              size_t start_idx, size_t count, size_t total_count) {
    for (size_t i = 0; i < count; ++i) {
        size_t idx = start_idx + i;
        if (idx >= total_count) {
            out[i] = 0.0f;
            continue;
        }
        size_t byte_idx = idx / 2;
        bool high_nibble = (idx % 2) != 0;
        out[i] = fp4_extract(packed_data[byte_idx], high_nibble);
    }
}

// Batch pack: Convert multiple float values to FP4 array 
inline void fp4_batch_pack(uint8_t* packed_data, const float* in,
                           size_t start_idx, size_t count, size_t total_count) {
    for (size_t i = 0; i < count; ++i) {
        size_t idx = start_idx + i;
        if (idx >= total_count) continue;
        size_t byte_idx = idx / 2;
        bool high_nibble = (idx % 2) != 0;
        uint8_t nibble = float_to_fp4(in[i]);
        
        if (high_nibble) {
            packed_data[byte_idx] = (packed_data[byte_idx] & 0x0F) | (nibble << 4);
        } else {
            packed_data[byte_idx] = (packed_data[byte_idx] & 0xF0) | nibble;
        }
    }
}

#if FP4_SIMD_AVX2
// AVX2-optimized batch unpack 
inline void fp4_batch_unpack_avx2(float* out, const uint8_t* packed_data,
                                   size_t start_idx, size_t count, size_t total_count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        float temp[8];
        for (size_t j = 0; j < 8; ++j) {
            size_t idx = start_idx + i + j;
            if (idx < total_count) {
                size_t byte_idx = idx / 2;
                bool high_nibble = (idx % 2) != 0;
                temp[j] = fp4_extract(packed_data[byte_idx], high_nibble);
            } else {
                temp[j] = 0.0f;
            }
        }
        _mm256_storeu_ps(&out[i], _mm256_loadu_ps(temp));
    }
    for (; i < count; ++i) {
        size_t idx = start_idx + i;
        if (idx < total_count) {
            size_t byte_idx = idx / 2;
            bool high_nibble = (idx % 2) != 0;
            out[i] = fp4_extract(packed_data[byte_idx], high_nibble);
        } else {
            out[i] = 0.0f;
        }
    }
}

inline void fp4_batch_pack_avx2(uint8_t* packed_data, const float* in,
                                 size_t start_idx, size_t count, size_t total_count) {
    fp4_batch_pack(packed_data, in, start_idx, count, total_count);
}
#endif // FP4_SIMD_AVX2

#if FP4_SIMD_SSE2 && !FP4_SIMD_AVX2
// SSE2-optimized batch unpack
inline void fp4_batch_unpack_sse2(float* out, const uint8_t* packed_data,
                                   size_t start_idx, size_t count, size_t total_count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float temp[4];
        for (size_t j = 0; j < 4; ++j) {
            size_t idx = start_idx + i + j;
            if (idx < total_count) {
                size_t byte_idx = idx / 2;
                bool high_nibble = (idx % 2) != 0;
                temp[j] = fp4_extract(packed_data[byte_idx], high_nibble);
            } else {
                temp[j] = 0.0f;
            }
        }
        _mm_storeu_ps(&out[i], _mm_loadu_ps(temp));
    }
    for (; i < count; ++i) {
        size_t idx = start_idx + i;
        if (idx < total_count) {
            size_t byte_idx = idx / 2;
            bool high_nibble = (idx % 2) != 0;
            out[i] = fp4_extract(packed_data[byte_idx], high_nibble);
        } else {
            out[i] = 0.0f;
        }
    }
}

inline void fp4_batch_pack_sse2(uint8_t* packed_data, const float* in,
                                 size_t start_idx, size_t count, size_t total_count) {
    fp4_batch_pack(packed_data, in, start_idx, count, total_count);
}
#endif // FP4_SIMD_SSE2

#if FP4_SIMD_NEON
// NEON-optimized batch unpack
inline void fp4_batch_unpack_neon(float* out, const uint8_t* packed_data,
                                   size_t start_idx, size_t count, size_t total_count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float temp[4];
        for (size_t j = 0; j < 4; ++j) {
            size_t idx = start_idx + i + j;
            if (idx < total_count) {
                size_t byte_idx = idx / 2;
                bool high_nibble = (idx % 2) != 0;
                temp[j] = fp4_extract(packed_data[byte_idx], high_nibble);
            } else {
                temp[j] = 0.0f;
            }
        }
        vst1q_f32(&out[i], vld1q_f32(temp));
    }
    for (; i < count; ++i) {
        size_t idx = start_idx + i;
        if (idx < total_count) {
            size_t byte_idx = idx / 2;
            bool high_nibble = (idx % 2) != 0;
            out[i] = fp4_extract(packed_data[byte_idx], high_nibble);
        } else {
            out[i] = 0.0f;
        }
    }
}

inline void fp4_batch_pack_neon(uint8_t* packed_data, const float* in,
                                 size_t start_idx, size_t count, size_t total_count) {
    fp4_batch_pack(packed_data, in, start_idx, count, total_count);
}
#endif // FP4_SIMD_NEON

// Dispatcher functions that select the best SIMD implementation 
inline void fp4_batch_unpack_simd(float* out, const uint8_t* packed_data,
                                   size_t start_idx, size_t count, size_t total_count) {
#if FP4_SIMD_AVX2
    fp4_batch_unpack_avx2(out, packed_data, start_idx, count, total_count);
#elif FP4_SIMD_SSE2
    fp4_batch_unpack_sse2(out, packed_data, start_idx, count, total_count);
#elif FP4_SIMD_NEON
    fp4_batch_unpack_neon(out, packed_data, start_idx, count, total_count);
#else
    fp4_batch_unpack(out, packed_data, start_idx, count, total_count);
#endif
}

inline void fp4_batch_pack_simd(uint8_t* packed_data, const float* in,
                                 size_t start_idx, size_t count, size_t total_count) {
#if FP4_SIMD_AVX2
    fp4_batch_pack_avx2(packed_data, in, start_idx, count, total_count);
#elif FP4_SIMD_SSE2
    fp4_batch_pack_sse2(packed_data, in, start_idx, count, total_count);
#elif FP4_SIMD_NEON
    fp4_batch_pack_neon(packed_data, in, start_idx, count, total_count);
#else
    fp4_batch_pack(packed_data, in, start_idx, count, total_count);
#endif
}


// ============================================================================
// Packed FP4 Storage Structure
// ============================================================================

struct fp4_packed {
    uint8_t data;
    
    fp4_packed() : data(0) {}
    explicit fp4_packed(uint8_t raw) : data(raw) {}
    fp4_packed(float low, float high) : data(fp4_pack(low, high)) {}
    
    float get(int index) const { return fp4_extract(data, index != 0); }
    
    void set(int index, float value) {
        uint8_t nibble = float_to_fp4(value);
        if (index == 0) {
            data = (data & 0xF0) | nibble;
        } else {
            data = (data & 0x0F) | (nibble << 4);
        }
    }
};

// ============================================================================
// FP4 Array Class
// ============================================================================

class FP4Array {
public:
    explicit FP4Array(size_t count)
        : buffer_((count + 1) / 2, 0), count_(count) {}
    
    float get(size_t index) const {
        if (index >= count_) return 0.0f;
        size_t byte_idx = index / 2;
        bool high_nibble = (index % 2) != 0;
        return fp4_extract(buffer_[byte_idx], high_nibble);
    }
    
    void set(size_t index, float value) {
        if (index >= count_) return;
        size_t byte_idx = index / 2;
        bool high_nibble = (index % 2) != 0;
        uint8_t nibble = float_to_fp4(value);
        
        if (high_nibble) {
            buffer_[byte_idx] = (buffer_[byte_idx] & 0x0F) | (nibble << 4);
        } else {
            buffer_[byte_idx] = (buffer_[byte_idx] & 0xF0) | nibble;
        }
    }
    
    size_t size() const { return count_; }
    size_t byte_size() const { return buffer_.size(); }
    uint8_t* data() { return buffer_.data(); }
    const uint8_t* data() const { return buffer_.data(); }
    
    void fill(float value) {
        uint8_t nibble = float_to_fp4(value);
        uint8_t packed = (nibble << 4) | nibble;
        for (auto& byte : buffer_) byte = packed;
    }
    
    void fill_zero() {
        for (auto& byte : buffer_) byte = 0;
    }
    
    void fill_random() {
        uint32_t seed = 12345;
        for (size_t i = 0; i < count_; ++i) {
            seed = seed * 1103515245 + 12345;
            uint8_t nibble = (seed >> 16) & 0x0F;
            set(i, fp4_to_float(nibble));
        }
    }
    
    // Batch Operations (Requirements 7.1, 7.2, 7.3, 7.4)
    void batch_unpack(float* out, size_t start_idx, size_t count) const {
        fp4_batch_unpack(out, buffer_.data(), start_idx, count, count_);
    }
    
    void batch_pack(const float* in, size_t start_idx, size_t count) {
        fp4_batch_pack(buffer_.data(), in, start_idx, count, count_);
    }
    
    void batch_unpack_simd(float* out, size_t start_idx, size_t count) const {
        fp4_batch_unpack_simd(out, buffer_.data(), start_idx, count, count_);
    }
    
    void batch_pack_simd(const float* in, size_t start_idx, size_t count) {
        fp4_batch_pack_simd(buffer_.data(), in, start_idx, count, count_);
    }
    
    static constexpr size_t simd_chunk_size() { return FP4_SIMD_CHUNK_SIZE; }
    
private:
    std::vector<uint8_t> buffer_;
    size_t count_;
};

// ============================================================================
// Helper Functions
// ============================================================================

inline const float* get_fp4_value_table() { return FP4_TO_FLOAT_TABLE; }
constexpr size_t FP4_VALUE_COUNT = 16;

inline size_t fp4_storage_bytes(size_t element_count) {
    return (element_count + 1) / 2;
}
