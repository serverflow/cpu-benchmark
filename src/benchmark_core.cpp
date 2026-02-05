// CPU Benchmark - Benchmark core implementation


#include "benchmark_core.hpp"
#include "half.hpp"
#include "math_kernels_extended.hpp"

// Template instantiations for float and double
template class Benchmark<float>;
template class Benchmark<double>;

// Template instantiations for extended precision types
template class Benchmark<half>;
template class Benchmark<int8_t>;
