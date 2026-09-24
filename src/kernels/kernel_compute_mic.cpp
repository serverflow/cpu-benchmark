// CPU Benchmark - Intel Xeon Phi Knights Corner / MIC compute kernels
//
// Knights Corner uses IMCI rather than regular AVX-512. K1OM builds call small
// assembly hot loops so the old MPSS GCC toolchain cannot silently fall back
// to x87/scalar code. Portable C++ fallbacks keep host builds linkable.

#include "kernel_compute.hpp"

#include <cstddef>

#if defined(SFBENCH_K1OM)
extern "C" void sfbench_k1om_fma_loop(unsigned long iterations);
extern "C" void sfbench_k1om_fma_loop_masked_fp64(unsigned long iterations);
extern "C" void sfbench_k1om_fma_loop_fp32(unsigned long iterations);
#endif

namespace kernels {
namespace compute {

size_t mic_imci_double(double* result, size_t iterations) {
#if defined(SFBENCH_K1OM)
    sfbench_k1om_fma_loop(static_cast<unsigned long>(iterations));
    if (result) {
        *result = static_cast<double>(iterations);
    }
    // 14 zmm accumulators * 8 FP64 lanes * 2 FLOPs per FMA.
    return iterations * 224;
#else
    alignas(64) double acc[64];
    alignas(64) double mul[64];
    alignas(64) double add[64];

    for (size_t lane = 0; lane < 64; ++lane) {
        acc[lane] = 1.0 + static_cast<double>(lane) * 0.01;
        mul[lane] = 1.0000001 + static_cast<double>(lane % 7) * 1.0e-10;
        add[lane] = 0.0000001 + static_cast<double>(lane % 5) * 1.0e-10;
    }

    for (size_t i = 0; i < iterations; ++i) {
#if defined(__INTEL_COMPILER)
        #pragma vector aligned
        #pragma vector always
#endif
        for (size_t lane = 0; lane < 64; ++lane) {
            acc[lane] = acc[lane] * mul[lane] + add[lane];
        }
    }

    double sum = 0.0;
    for (size_t lane = 0; lane < 64; ++lane) {
        sum += acc[lane];
    }
    *result = sum;

    return iterations * 64 * 2;
#endif
}

size_t mic_imci_score_double(double* result, size_t iterations) {
#if defined(SFBENCH_K1OM)
    sfbench_k1om_fma_loop_masked_fp64(static_cast<unsigned long>(iterations));
    if (result) {
        *result = static_cast<double>(iterations);
    }
    // Match the regular scalar baseline: 8 FMA chains, one active FP64 lane.
    return iterations * 16;
#else
    return scalar_fp64_baseline(result, iterations);
#endif
}

size_t mic_imci_float(float* result, size_t iterations) {
#if defined(SFBENCH_K1OM)
    sfbench_k1om_fma_loop_fp32(static_cast<unsigned long>(iterations));
    if (result) {
        *result = static_cast<float>(iterations);
    }
    // 14 zmm accumulators * 16 FP32 lanes * 2 FLOPs per FMA.
    return iterations * 448;
#else
    alignas(64) float acc[128];
    alignas(64) float mul[128];
    alignas(64) float add[128];

    for (size_t lane = 0; lane < 128; ++lane) {
        acc[lane] = 1.0f + static_cast<float>(lane) * 0.01f;
        mul[lane] = 1.0000001f + static_cast<float>(lane % 7) * 1.0e-7f;
        add[lane] = 0.0000001f + static_cast<float>(lane % 5) * 1.0e-7f;
    }

    for (size_t i = 0; i < iterations; ++i) {
#if defined(__INTEL_COMPILER)
        #pragma vector aligned
        #pragma vector always
#endif
        for (size_t lane = 0; lane < 128; ++lane) {
            acc[lane] = acc[lane] * mul[lane] + add[lane];
        }
    }

    float sum = 0.0f;
    for (size_t lane = 0; lane < 128; ++lane) {
        sum += acc[lane];
    }
    *result = sum;

    return iterations * 128 * 2;
#endif
}

} // namespace compute
} // namespace kernels
