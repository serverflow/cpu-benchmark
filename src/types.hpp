#pragma once
// CPU Benchmark - Core types and enums

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <sstream>

// Benchmark mode enum 
enum class BenchmarkMode {
    Mem,        // Memory bandwidth test
    Stencil,    // 7-point stencil test
    Matmul3D,   // Batch matrix multiplication
    Compute,    // Pure compute test (FMA-heavy, no memory bottleneck)
    CacheLevel  // Cache-level tests (L1, L2, L3, RAM-bound) 
};

// Precision enum 
enum class Precision {
    Float,      // 32-bit floating-point (backward compatibility)
    Double,     // 64-bit floating-point (backward compatibility)
    FP64,       // Explicit 64-bit double precision
    FP16,       // 16-bit half precision
    INT8,       // 8-bit integer
    FP4         // 4-bit emulated float
};

// FP16 execution mode 
enum class FP16Mode {
    Native,     // Hardware AVX-512 FP16 or ARM NEON FP16
    Emulated    // Software emulation via float32
};

// Precision configuration structure 
struct PrecisionConfig {
    Precision precision;
    const char* name;           // Display name: "FP64", "FP16", etc.
    const char* description;    // "64-bit double", "16-bit half (native)", etc.
    double bytes_per_element;   // 8, 2, 1, 0.5
    bool is_integer;            // true for INT8
    bool is_emulated;           // true for FP4, possibly FP16
};

// Output format enum 
enum class OutputFormat {
    Text,       // Human-readable
    Json,       // JSON format
    Csv         // CSV format
};

// 3D size structure with overflow-safe operations 
struct Size3D {
    size_t Nx;
    size_t Ny;
    size_t Nz;
    
    // Safe total calculation with overflow check
    // Throws std::overflow_error if multiplication would overflow
    size_t total() const {
        // Handle zero dimensions - no overflow possible
        if (Nx == 0 || Ny == 0 || Nz == 0) {
            return 0;
        }
        
        // Check Nx * Ny overflow
        if (Ny > SIZE_MAX / Nx) {
            throw std::overflow_error(
                "Size overflow: Nx * Ny exceeds SIZE_MAX (" +
                std::to_string(Nx) + " * " + std::to_string(Ny) + ")"
            );
        }
        size_t xy = Nx * Ny;
        
        // Check xy * Nz overflow
        if (Nz > SIZE_MAX / xy) {
            throw std::overflow_error(
                "Size overflow: Nx * Ny * Nz exceeds SIZE_MAX (" +
                std::to_string(Nx) + " * " + std::to_string(Ny) + " * " + std::to_string(Nz) + ")"
            );
        }
        return xy * Nz;
    }
    
    // Unsafe total for cases where overflow check is not needed
    // Use only when dimensions are known to be small
    size_t total_unchecked() const noexcept {
        return Nx * Ny * Nz;
    }
    
    // Safe total_bytes calculation with overflow check 
    // Handles fractional bytes_per_element for FP4 (0.5 bytes)
    size_t total_bytes(double bytes_per_element) const {
        size_t t = total();  // May throw overflow_error
        
        if (bytes_per_element <= 0) {
            throw std::invalid_argument("bytes_per_element must be positive");
        }
        
        // For fractional bytes (FP4 = 0.5), we need ceiling division
        // total_bytes = ceil(total * bytes_per_element)
        if (bytes_per_element < 1.0) {
            // For FP4: 2 values per byte, so (total + 1) / 2
            // General formula: ceil(total * bytes_per_element)
            double exact_bytes = static_cast<double>(t) * bytes_per_element;
            size_t result = static_cast<size_t>(exact_bytes);
            // Round up if there's a fractional part
            if (exact_bytes > static_cast<double>(result)) {
                result++;
            }
            return result;
        }
        
        // For whole bytes, check overflow
        size_t whole_bytes = static_cast<size_t>(bytes_per_element);
        if (t > SIZE_MAX / whole_bytes) {
            throw std::overflow_error(
                "Size overflow: total_bytes exceeds SIZE_MAX (" +
                std::to_string(t) + " * " + std::to_string(whole_bytes) + " bytes)"
            );
        }
        return t * whole_bytes;
    }
    
    // Parse from string "NxNxN" format
    static Size3D parse(const std::string& s) {
        Size3D result{0, 0, 0};
        char sep1, sep2;
        std::istringstream iss(s);
        if (!(iss >> result.Nx >> sep1 >> result.Ny >> sep2 >> result.Nz) ||
            sep1 != 'x' || sep2 != 'x') {
            throw std::invalid_argument("Invalid size format. Expected NxNxN");
        }
        return result;
    }
    
    std::string to_string() const {
        return std::to_string(Nx) + "x" + std::to_string(Ny) + "x" + std::to_string(Nz);
    }
};


// Configuration structure 
struct Config {
    BenchmarkMode mode = BenchmarkMode::Mem;
    Size3D size = {128, 128, 128};
    unsigned threads = 0;           // 0 = auto-detect
    double min_time = 3.0;          // seconds (for mem/stencil/precision modes)
    Precision precision = Precision::Float;
    unsigned repeats = 5;
    OutputFormat output = OutputFormat::Text;
    int selected_socket = -1;       // -1 = all sockets, 0+ = specific socket (for NUMA systems)
};

// Benchmark result structure
struct BenchmarkResult {
    std::vector<double> times_sec;
    double time_avg_sec = 0.0;
    double time_min_sec = 0.0;
    double time_stddev_sec = 0.0;
    double gflops_avg = 0.0;
    double gflops_max = 0.0;
    double mlups_avg = 0.0;         // Only for stencil mode
    double bandwidth_gbs = 0.0;     // Only for mem mode
    size_t total_flops = 0;
    size_t iterations = 0;
};

// Error codes
enum class ErrorCode {
    Success = 0,
    InvalidArguments = 1,
    OutOfMemory = 2,
    ThreadCreationFailed = 3,
    UnknownError = 99
};

// Helper functions for enum conversion
inline std::string mode_to_string(BenchmarkMode mode) {
    switch (mode) {
        case BenchmarkMode::Mem: return "mem";
        case BenchmarkMode::Stencil: return "stencil";
        case BenchmarkMode::Matmul3D: return "matmul3d";
        case BenchmarkMode::Compute: return "compute";
        case BenchmarkMode::CacheLevel: return "cache";
    }
    return "unknown";
}

inline BenchmarkMode string_to_mode(const std::string& s) {
    if (s == "mem") return BenchmarkMode::Mem;
    if (s == "stencil") return BenchmarkMode::Stencil;
    if (s == "matmul3d") return BenchmarkMode::Matmul3D;
    if (s == "compute") return BenchmarkMode::Compute;
    if (s == "cache") return BenchmarkMode::CacheLevel;
    throw std::invalid_argument("Invalid mode: " + s);
}

inline std::string precision_to_string(Precision p) {
    switch (p) {
        case Precision::Float: return "float";
        case Precision::Double: return "double";
        case Precision::FP64: return "fp64";
        case Precision::FP16: return "fp16";
        case Precision::INT8: return "int8";
        case Precision::FP4: return "fp4";
    }
    return "unknown";
}

inline Precision string_to_precision(const std::string& s) {
    if (s == "float") return Precision::Float;
    if (s == "double") return Precision::Double;
    if (s == "fp64") return Precision::FP64;
    if (s == "fp16") return Precision::FP16;
    if (s == "int8") return Precision::INT8;
    if (s == "fp4") return Precision::FP4;
    throw std::invalid_argument("Invalid precision: " + s + ". Valid values: float, double, fp64, fp16, int8, fp4");
}

// Extended precision string conversion 
inline std::string precision_to_string_extended(Precision p, FP16Mode fp16_mode = FP16Mode::Emulated) {
    switch (p) {
        case Precision::Float: return "float";
        case Precision::Double: return "double";
        case Precision::FP64: return "FP64";
        case Precision::FP16: 
            return fp16_mode == FP16Mode::Native ? "FP16 (native)" : "FP16 (emulated)";
        case Precision::INT8: return "INT8";
        case Precision::FP4: return "FP4 (emulated)";
    }
    return "unknown";
}

// Get precision configuration 
inline PrecisionConfig get_precision_config(Precision p, FP16Mode fp16_mode = FP16Mode::Emulated) {
    switch (p) {
        case Precision::Float:
            return {Precision::Float, "FP32", "32-bit single precision", 4.0, false, false};
        case Precision::Double:
            return {Precision::Double, "double", "64-bit double precision", 8.0, false, false};
        case Precision::FP64:
            return {Precision::FP64, "FP64", "64-bit double precision", 8.0, false, false};
        case Precision::FP16:
            if (fp16_mode == FP16Mode::Native) {
                return {Precision::FP16, "FP16", "16-bit half precision (native)", 2.0, false, false};
            } else {
                return {Precision::FP16, "FP16", "16-bit half precision (emulated)", 2.0, false, true};
            }
        case Precision::INT8:
            return {Precision::INT8, "INT8", "8-bit integer", 1.0, true, false};
        case Precision::FP4:
            return {Precision::FP4, "FP4", "4-bit float (emulated, 2 values/byte)", 0.5, false, true};
    }
    return {p, "unknown", "unknown", 0.0, false, false};
}

// Check if precision is one of the new extended types
inline bool is_extended_precision(Precision p) {
    return p == Precision::FP64 || p == Precision::FP16 || 
           p == Precision::INT8 || p == Precision::FP4;
}

inline std::string output_to_string(OutputFormat f) {
    switch (f) {
        case OutputFormat::Text: return "text";
        case OutputFormat::Json: return "json";
        case OutputFormat::Csv: return "csv";
    }
    return "unknown";
}

inline OutputFormat string_to_output(const std::string& s) {
    if (s == "text") return OutputFormat::Text;
    if (s == "json") return OutputFormat::Json;
    if (s == "csv") return OutputFormat::Csv;
    throw std::invalid_argument("Invalid output format: " + s);
}
