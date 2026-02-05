#pragma once
// CPU Benchmark - IEEE 754-2008 Half-Precision Floating-Point Type
// Requirements: 3.1, 3.5

#include <cstdint>
#include <cmath>
#include <limits>

// IEEE 754-2008 half-precision floating-point
// Format: 1 sign bit, 5 exponent bits, 10 mantissa bits
// Range: ±65504 (max normal), ±6.10e-5 (min positive normal)
// Precision: ~3.3 decimal digits
struct half {
    uint16_t bits;

    // Default constructor - zero
    constexpr half() : bits(0) {}

    // Private constructor from raw bits (for constexpr from_bits)
    constexpr explicit half(uint16_t raw, bool) : bits(raw) {}

    // Construct from raw bits
    static constexpr half from_bits(uint16_t raw) {
        return half(raw, true);
    }

    // Explicit constructor from float
    explicit half(float f) : bits(float_to_half_bits(f)) {}

    // Explicit constructor from double
    explicit half(double d) : bits(float_to_half_bits(static_cast<float>(d))) {}

    // Conversion to float (for computations)
    operator float() const {
        return half_bits_to_float(bits);
    }

    // Static conversion functions
    static half from_float(float f) {
        return half(f);
    }

    static float to_float(half h) {
        return static_cast<float>(h);
    }

    // Arithmetic operators (compute in float, store as half)
    half operator+(half other) const {
        return half(static_cast<float>(*this) + static_cast<float>(other));
    }

    half operator-(half other) const {
        return half(static_cast<float>(*this) - static_cast<float>(other));
    }

    half operator*(half other) const {
        return half(static_cast<float>(*this) * static_cast<float>(other));
    }

    half operator/(half other) const {
        return half(static_cast<float>(*this) / static_cast<float>(other));
    }

    // Compound assignment operators
    half& operator+=(half other) {
        *this = *this + other;
        return *this;
    }

    half& operator-=(half other) {
        *this = *this - other;
        return *this;
    }

    half& operator*=(half other) {
        *this = *this * other;
        return *this;
    }

    half& operator/=(half other) {
        *this = *this / other;
        return *this;
    }

    // Unary minus
    half operator-() const {
        half result;
        result.bits = bits ^ 0x8000; // Flip sign bit
        return result;
    }

    // Comparison operators
    bool operator==(half other) const {
        // Handle +0 == -0
        if ((bits & 0x7FFF) == 0 && (other.bits & 0x7FFF) == 0) return true;
        return bits == other.bits;
    }

    bool operator!=(half other) const {
        return !(*this == other);
    }

    bool operator<(half other) const {
        return static_cast<float>(*this) < static_cast<float>(other);
    }

    bool operator<=(half other) const {
        return static_cast<float>(*this) <= static_cast<float>(other);
    }

    bool operator>(half other) const {
        return static_cast<float>(*this) > static_cast<float>(other);
    }

    bool operator>=(half other) const {
        return static_cast<float>(*this) >= static_cast<float>(other);
    }

private:
    // Convert float to half bits
    // IEEE 754 single: 1 sign, 8 exp (bias 127), 23 mantissa
    // IEEE 754 half:   1 sign, 5 exp (bias 15),  10 mantissa
    static uint16_t float_to_half_bits(float f) {
        union { float f; uint32_t u; } fu;
        fu.f = f;
        uint32_t fbits = fu.u;

        uint32_t sign = (fbits >> 16) & 0x8000;
        int32_t exp = ((fbits >> 23) & 0xFF) - 127 + 15; // Rebias exponent
        uint32_t mantissa = (fbits >> 13) & 0x3FF;       // Top 10 bits of mantissa

        // Handle special cases
        if (((fbits >> 23) & 0xFF) == 0xFF) {
            // Infinity or NaN
            if ((fbits & 0x7FFFFF) == 0) {
                // Infinity
                return static_cast<uint16_t>(sign | 0x7C00);
            } else {
                // NaN - preserve some mantissa bits
                return static_cast<uint16_t>(sign | 0x7C00 | (mantissa ? mantissa : 1));
            }
        }

        if (exp <= 0) {
            // Underflow to zero or denormal
            if (exp < -10) {
                // Too small, flush to zero
                return static_cast<uint16_t>(sign);
            }
            // Denormalized number
            mantissa = (mantissa | 0x400) >> (1 - exp);
            // Round to nearest even
            if ((mantissa & 1) && (mantissa & 2)) {
                mantissa++;
            }
            return static_cast<uint16_t>(sign | (mantissa >> 1));
        }

        if (exp >= 31) {
            // Overflow to infinity
            return static_cast<uint16_t>(sign | 0x7C00);
        }

        // Round to nearest even
        uint32_t round_bit = (fbits >> 12) & 1;
        uint32_t sticky_bits = fbits & 0xFFF;
        if (round_bit && (sticky_bits || (mantissa & 1))) {
            mantissa++;
            if (mantissa > 0x3FF) {
                mantissa = 0;
                exp++;
                if (exp >= 31) {
                    return static_cast<uint16_t>(sign | 0x7C00);
                }
            }
        }

        return static_cast<uint16_t>(sign | (exp << 10) | mantissa);
    }

    // Convert half bits to float
    static float half_bits_to_float(uint16_t h) {
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mantissa = h & 0x3FF;

        uint32_t fbits;

        if (exp == 0) {
            if (mantissa == 0) {
                // Zero (positive or negative)
                fbits = sign;
            } else {
                // Denormalized number - normalize it
                exp = 1;
                while ((mantissa & 0x400) == 0) {
                    mantissa <<= 1;
                    exp--;
                }
                mantissa &= 0x3FF;
                fbits = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
            }
        } else if (exp == 31) {
            // Infinity or NaN
            fbits = sign | 0x7F800000 | (mantissa << 13);
        } else {
            // Normalized number
            fbits = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
        }

        union { uint32_t u; float f; } uf;
        uf.u = fbits;
        return uf.f;
    }
};

// Size verification 
static_assert(sizeof(half) == 2, "half must be 2 bytes");

// Numeric limits specialization for half
namespace std {
    template<>
    struct numeric_limits<half> {
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = false;
        static constexpr bool has_infinity = true;
        static constexpr bool has_quiet_NaN = true;
        static constexpr bool has_signaling_NaN = true;
        static constexpr int digits = 11;        // Including implicit bit
        static constexpr int digits10 = 3;       // ~3.3 decimal digits
        static constexpr int max_digits10 = 5;
        static constexpr int radix = 2;
        static constexpr int min_exponent = -13;
        static constexpr int min_exponent10 = -4;
        static constexpr int max_exponent = 16;
        static constexpr int max_exponent10 = 4;

        static constexpr half min() noexcept { return half::from_bits(0x0400); }          // Smallest positive normal
        static constexpr half max() noexcept { return half::from_bits(0x7BFF); }          // 65504
        static constexpr half lowest() noexcept { return half::from_bits(0xFBFF); }       // -65504
        static constexpr half epsilon() noexcept { return half::from_bits(0x1400); }      // 2^-10
        static constexpr half round_error() noexcept { return half::from_bits(0x3800); }  // 0.5
        static constexpr half infinity() noexcept { return half::from_bits(0x7C00); }
        static constexpr half quiet_NaN() noexcept { return half::from_bits(0x7E00); }
        static constexpr half signaling_NaN() noexcept { return half::from_bits(0x7D00); }
        static constexpr half denorm_min() noexcept { return half::from_bits(0x0001); }   // Smallest positive denormal
    };
}
