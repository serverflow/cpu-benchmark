#pragma once
// CPU Benchmark - CLI parsing module


#include "types.hpp"
#include "version.hpp"
#include "platform.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <cstring>
#include <thread>
#include <algorithm>

// Print version information ()
inline void print_version() {
    std::cout << version::get_full_version_string() << "\n";
}

// Print usage help 
inline void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --mode=MODE       Benchmark mode: mem, stencil, matmul3d (default: mem)\n";
    std::cout << "  --size=NxNxN      3D array size (default: 128x128x128)\n";
    std::cout << "  --threads=N       Number of threads, 0=auto (default: auto)\n";
    std::cout << "  --time=T          Minimum measurement time in seconds (default: 3.0)\n";
    std::cout << "  --precision=P     Precision: float, double (default: float)\n";
    std::cout << "  --repeats=N       Number of measurement repeats (default: 5)\n";
    std::cout << "  --output=FORMAT   Output format: text, json, csv (default: text)\n";
    std::cout << "  --version         Show version information\n";
    std::cout << "  --help            Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program_name << " --mode=stencil --size=256x256x256\n";
    std::cout << "  " << program_name << " --mode=matmul3d --precision=double --threads=4\n";
    std::cout << "  " << program_name << " --mode=mem --output=json\n";
}

// Parse result structure
struct ParseResult {
    Config config;
    bool success = true;
    bool show_help = false;
    bool show_version = false;  // 
    std::string error_message;
};

// Helper to extract value from --key=value argument
inline bool extract_arg_value(const char* arg, const char* key, std::string& value) {
    size_t key_len = std::strlen(key);
    if (std::strncmp(arg, key, key_len) == 0 && arg[key_len] == '=') {
        value = arg + key_len + 1;
        return true;
    }
    return false;
}

// Validate config 
inline bool validate_config(const Config& config, std::string& error) {
    // Size validation
    if (config.size.Nx == 0 || config.size.Ny == 0 || config.size.Nz == 0) {
        error = "Size dimensions must be greater than 0";
        return false;
    }
    
    // Check for size overflow 
    try {
        config.size.total();
    } catch (const std::overflow_error& e) {
        error = "Size overflow: the specified dimensions are too large. " + std::string(e.what());
        return false;
    }
    
    // Stencil requires minimum 3x3x3
    if (config.mode == BenchmarkMode::Stencil) {
        if (config.size.Nx < 3 || config.size.Ny < 3 || config.size.Nz < 3) {
            error = "Stencil mode requires minimum size 3x3x3";
            return false;
        }
    }

    // Matmul3D requires square matrices (Nx == Ny)
    if (config.mode == BenchmarkMode::Matmul3D) {
        if (config.size.Nx != config.size.Ny) {
            error = "Matmul3D mode requires square matrices (Nx must equal Ny)";
            return false;
        }
    }
    
    // Thread count validation
    unsigned max_threads = get_logical_core_count() * 4;
    if (config.threads > max_threads) {
        error = "Thread count exceeds maximum allowed (" + std::to_string(max_threads) + ")";
        return false;
    }
    
    // Time validation
    if (config.min_time <= 0) {
        error = "Minimum time must be greater than 0";
        return false;
    }
    
    // Repeats validation
    if (config.repeats == 0) {
        error = "Repeats must be greater than 0";
        return false;
    }
    
    return true;
}

// Parse command-line arguments
inline ParseResult parse_args(int argc, char* argv[]) {
    ParseResult result;
    result.config = Config();  // Default values
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string value;
        
        // Check for help
        if (arg == "--help" || arg == "-h") {
            result.show_help = true;
            return result;
        }
        
        // Check for version ()
        if (arg == "--version" || arg == "-v") {
            result.show_version = true;
            return result;
        }
        
        try {
            // --mode 
            if (extract_arg_value(argv[i], "--mode", value)) {
                result.config.mode = string_to_mode(value);
            }
            // --size 
            else if (extract_arg_value(argv[i], "--size", value)) {
                result.config.size = Size3D::parse(value);
            }
            // --threads 
            else if (extract_arg_value(argv[i], "--threads", value)) {
                result.config.threads = static_cast<unsigned>(std::stoul(value));
            }
            // --time 
            else if (extract_arg_value(argv[i], "--time", value)) {
                result.config.min_time = std::stod(value);
            }
            // --precision 
            else if (extract_arg_value(argv[i], "--precision", value)) {
                result.config.precision = string_to_precision(value);
            }
            // --repeats 
            else if (extract_arg_value(argv[i], "--repeats", value)) {
                result.config.repeats = static_cast<unsigned>(std::stoul(value));
            }
            // --output 
            else if (extract_arg_value(argv[i], "--output", value)) {
                result.config.output = string_to_output(value);
            }
            else {
                result.success = false;
                result.error_message = "Unknown argument: " + arg;
                return result;
            }
        } catch (const std::exception& e) {
            result.success = false;
            result.error_message = "Error parsing argument '" + arg + "': " + e.what();
            return result;
        }
    }
    
    // Validate the config
    std::string validation_error;
    if (!validate_config(result.config, validation_error)) {
        result.success = false;
        result.error_message = validation_error;
    }
    
    return result;
}

// Serialize config to command-line arguments (for round-trip testing)
inline std::vector<std::string> config_to_args(const Config& config) {
    std::vector<std::string> args;
    
    args.push_back("--mode=" + mode_to_string(config.mode));
    args.push_back("--size=" + config.size.to_string());
    args.push_back("--threads=" + std::to_string(config.threads));
    args.push_back("--time=" + std::to_string(config.min_time));
    args.push_back("--precision=" + precision_to_string(config.precision));
    args.push_back("--repeats=" + std::to_string(config.repeats));
    args.push_back("--output=" + output_to_string(config.output));
    
    return args;
}

// Compare two configs for equality (for round-trip testing)
inline bool configs_equal(const Config& a, const Config& b) {
    return a.mode == b.mode &&
           a.size.Nx == b.size.Nx &&
           a.size.Ny == b.size.Ny &&
           a.size.Nz == b.size.Nz &&
           a.threads == b.threads &&
           std::abs(a.min_time - b.min_time) < 1e-6 &&
           a.precision == b.precision &&
           a.repeats == b.repeats &&
           a.output == b.output;
}

// ============================================================================
// Extended CLI for Multi-Precision Support
// ============================================================================

// Extended config with new precision options 
// SIMD Optimization: --force-scalar flag
// Progress: --quiet flag
// Auto-size: --auto-size flag
// Warmup: --no-warmup flag
// Thread Affinity: --high-priority flag
// Multi-socket: --socket flag
struct ExtendedConfig {
    Config base;                    // Base configuration
    bool run_all_precisions;        // --precision=all flag
    bool no_color;                  // --no-color flag
    bool force_scalar;              // --force-scalar flag 
    bool quiet;                     // --quiet flag 
    bool auto_size;                 // --auto-size flag 
    bool explicit_size_set;         // True if user specified --size explicitly
    bool enable_warmup;             // --no-warmup flag 
    bool high_priority;             // --high-priority flag 
    bool run_full_test;             // True when running without any flags (full benchmark suite)
    
    // Multi-socket support
    int selected_socket;            // -1 = all sockets, 0+ = specific socket
    
    // Flags to track if user modified default settings (for submission eligibility)
    bool threads_modified;          // True if user specified --threads
    bool time_modified;             // True if user specified --time
    bool repeats_modified;          // True if user specified --repeats
    bool precision_single_modified; // True if user specified single precision (not "all")
    bool mode_modified;             // True if user specified --mode
    
    ExtendedConfig() : base(), run_all_precisions(false), no_color(false), 
                       force_scalar(false), quiet(false), auto_size(true), 
                       explicit_size_set(false), enable_warmup(true), high_priority(false),
                       run_full_test(false), selected_socket(-1),
                       threads_modified(false), time_modified(false), 
                       repeats_modified(false), precision_single_modified(false),
                       mode_modified(false) {}
    
    // Check if this is a "clean" run eligible for server submission
    // Clean run = only --mode or --precision=all, no other modifications
    bool is_submission_eligible() const {
        // Submission is allowed only for:
        // 1. --mode=X (any mode) with default settings
        // 2. --precision=all with default settings
        // 3. Full test mode (no flags)
        
        // Disqualifying modifications:
        // - --threads (custom thread count)
        // - --time (custom test duration)
        // - --repeats (custom repeat count)
        // - --size (custom size - explicit_size_set)
        // - --force-scalar (forces non-optimal path)
        // - --no-warmup (affects results)
        // - Single precision mode (not "all") - only precision=all is allowed for submission
        
        if (threads_modified) return false;
        if (time_modified) return false;
        if (repeats_modified) return false;
        if (explicit_size_set) return false;
        if (force_scalar) return false;
        if (!enable_warmup) return false;
        
        // Full test mode is always eligible
        if (run_full_test) return true;
        
        // For precision mode: only "all" is allowed
        // For other modes (compute, mem, stencil, etc.): single precision is OK
        // But if user explicitly set a single precision, it's not eligible
        if (precision_single_modified) return false;
        
        return true;
    }
    
    // Get reason why submission is not eligible (for user feedback)
    std::string get_submission_ineligibility_reason() const {
        if (threads_modified) return "custom --threads specified";
        if (time_modified) return "custom --time specified";
        if (repeats_modified) return "custom --repeats specified";
        if (explicit_size_set) return "custom --size specified";
        if (force_scalar) return "--force-scalar enabled";
        if (!enable_warmup) return "--no-warmup enabled";
        if (precision_single_modified) return "single precision mode (use --precision=all for submission)";
        return "";
    }
};

// Extended parse result
struct ExtendedParseResult {
    ExtendedConfig config;
    bool success;
    bool show_help;
    bool show_version;  // 
    std::string error_message;
    
    ExtendedParseResult() : config(), success(true), show_help(false), show_version(false), error_message() {}
};

// Print extended usage help 
inline void print_usage_extended(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Running without any options will execute the FULL BENCHMARK SUITE:\n";
    std::cout << "  - Warmup (2 seconds)\n";
    std::cout << "  - Compute test (--mode=compute)\n";
    std::cout << "  - Precision test (--precision=all)\n";
    std::cout << "  - Memory bandwidth test (--mode=mem)\n";
    std::cout << "  - Stencil test (--mode=stencil)\n";
    std::cout << "  - Cache test (--mode=cache)\n";
    std::cout << "  Results will be submitted to the server after completion.\n\n";
    std::cout << "Options:\n";
    std::cout << "  --mode=MODE       Benchmark mode: mem, stencil, matmul3d, compute, cache\n";
    std::cout << "                    (default: mem)\n";
    std::cout << "  --size=NxNxN      3D array size (overrides auto-size)\n";
    std::cout << "  --auto-size       Auto-detect optimal size based on RAM (default: on)\n";
    std::cout << "  --no-auto-size    Disable auto-size, use default 128x128x128\n";
    std::cout << "  --threads=N       Number of threads, 0=auto (default: auto)\n";
    std::cout << "  --time=T          Minimum measurement time in seconds (default: 3.0)\n";
    std::cout << "  --precision=P     Precision: float, double, fp64, fp16, int8, fp4, all\n";
    std::cout << "                    (default: float)\n";
    std::cout << "  --repeats=N       Number of measurement repeats (default: 5)\n";
    std::cout << "  --output=FORMAT   Output format: text, json, csv (default: text)\n";
    std::cout << "  --socket=N        Run benchmark only on CPU socket N (0-based)\n";
    std::cout << "                    By default, all sockets are used\n";
    std::cout << "  --no-color        Disable colored output in comparison table\n";
    std::cout << "  --force-scalar    Force scalar (non-SIMD) kernel implementations\n";
    std::cout << "  --quiet           Suppress progress output, show only final results\n";
    std::cout << "  --no-warmup       Skip CPU warmup phase before measurements\n";
    std::cout << "  --high-priority   Set process to high priority for more accurate results\n";
    std::cout << "  --version         Show version, build date, and compiler information\n";
    std::cout << "  --help            Show this help message\n";
    std::cout << "\nPrecision Types:\n";
    std::cout << "  float   - 32-bit single precision (4 bytes)\n";
    std::cout << "  double  - 64-bit double precision (8 bytes)\n";
    std::cout << "  fp64    - 64-bit double precision (8 bytes)\n";
    std::cout << "  fp16    - 16-bit half precision (2 bytes, native or emulated)\n";
    std::cout << "  int8    - 8-bit integer (1 byte, reports GOPS instead of GFLOPS)\n";
    std::cout << "  fp4     - 4-bit float emulation (0.5 bytes, 2 values per byte)\n";
    std::cout << "  all     - Run benchmarks for all precision types and show comparison\n";
    std::cout << "\nMulti-Socket Systems:\n";
    std::cout << "  By default, benchmarks use all available CPU sockets.\n";
    std::cout << "  Use --socket=N to run on a specific socket only.\n";
    std::cout << "  Example: --socket=0 runs only on the first CPU socket.\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program_name << "                                    # Run full benchmark suite\n";
    std::cout << "  " << program_name << " --mode=stencil --size=256x256x256\n";
    std::cout << "  " << program_name << " --mode=matmul3d --precision=fp16 --threads=4\n";
    std::cout << "  " << program_name << " --precision=all --output=json\n";
    std::cout << "  " << program_name << " --precision=all --no-color\n";
    std::cout << "  " << program_name << " --mode=compute --socket=0          # Test only first CPU\n";
}

// Parse precision string with support for "all" 
// Returns true if "all" was specified, false otherwise
// Throws std::invalid_argument for invalid precision strings
inline bool parse_precision_extended(const std::string& s, Precision& out_precision) {
    // Convert to lowercase for case-insensitive comparison
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "all") {
        return true;  // Indicates --precision=all
    }
    
    // Use existing string_to_precision for other values
    out_precision = string_to_precision(lower);
    return false;
}

// Validate extended config 
inline bool validate_extended_config(const ExtendedConfig& config, std::string& error) {
    // First validate base config
    if (!validate_config(config.base, error)) {
        return false;
    }
    
    // Additional validation for extended config can be added here
    // Currently no additional validation needed
    
    return true;
}

// Parse command-line arguments with extended precision support 

inline ExtendedParseResult parse_args_extended(int argc, char* argv[]) {
    ExtendedParseResult result;
    result.config.base = Config();  // Default values
    result.config.run_all_precisions = false;
    result.config.no_color = false;
    result.config.force_scalar = false;
    result.config.quiet = false;
    result.config.auto_size = true;  // Auto-size enabled by default
    result.config.explicit_size_set = false;
    result.config.enable_warmup = true;  // Warmup enabled by default
    result.config.high_priority = false;  // High priority disabled by default
    result.config.run_full_test = false;  // Full test mode disabled by default
    
    // Track modifications for submission eligibility
    result.config.threads_modified = false;
    result.config.time_modified = false;
    result.config.repeats_modified = false;
    result.config.precision_single_modified = false;
    result.config.mode_modified = false;
    
    // Track if only "neutral" flags were specified (socket, quiet, no-color, etc.)
    // These don't change the test mode, so we should still run full test
    bool only_neutral_flags = true;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string value;
        
        // Check for help
        if (arg == "--help" || arg == "-h") {
            result.show_help = true;
            return result;
        }
        
        // Check for version ()
        if (arg == "--version" || arg == "-v") {
            result.show_version = true;
            return result;
        }
        
        // Check for --no-color flag 
        if (arg == "--no-color") {
            result.config.no_color = true;
            continue;
        }
        
        // Check for --force-scalar flag 
        if (arg == "--force-scalar") {
            result.config.force_scalar = true;
            continue;
        }
        
        // Check for --quiet flag 
        if (arg == "--quiet" || arg == "-q") {
            result.config.quiet = true;
            continue;
        }
        
        // Check for --auto-size flag
        if (arg == "--auto-size") {
            result.config.auto_size = true;
            continue;
        }
        
        // Check for --no-auto-size flag 
        if (arg == "--no-auto-size") {
            result.config.auto_size = false;
            continue;
        }
        
        // Check for --no-warmup flag 
        if (arg == "--no-warmup") {
            result.config.enable_warmup = false;
            continue;
        }
        
        // Check for --high-priority flag 
        if (arg == "--high-priority") {
            result.config.high_priority = true;
            continue;
        }
        
        try {
            // --socket (neutral flag - doesn't change test mode)
            if (extract_arg_value(argv[i], "--socket", value)) {
                result.config.selected_socket = std::stoi(value);
                result.config.base.selected_socket = result.config.selected_socket;  // Also set in base config
                if (result.config.selected_socket < 0) {
                    result.success = false;
                    result.error_message = "Socket number must be non-negative";
                    return result;
                }
                continue;  // Socket is neutral, doesn't affect only_neutral_flags
            }
            // --mode 
            if (extract_arg_value(argv[i], "--mode", value)) {
                result.config.base.mode = string_to_mode(value);
                result.config.mode_modified = true;  // Track modification
                only_neutral_flags = false;  // Mode changes test behavior
            }
            // --size 
            else if (extract_arg_value(argv[i], "--size", value)) {
                result.config.base.size = Size3D::parse(value);
                result.config.explicit_size_set = true;  // User specified size explicitly
                only_neutral_flags = false;  // Size changes test behavior
            }
            // --threads 
            else if (extract_arg_value(argv[i], "--threads", value)) {
                result.config.base.threads = static_cast<unsigned>(std::stoul(value));
                result.config.threads_modified = true;  // Track modification
                only_neutral_flags = false;  // Threads changes test behavior
            }
            // --time 
            else if (extract_arg_value(argv[i], "--time", value)) {
                result.config.base.min_time = std::stod(value);
                result.config.time_modified = true;  // Track modification
                only_neutral_flags = false;  // Time changes test behavior
            }
            // --precision with extended support 
            else if (extract_arg_value(argv[i], "--precision", value)) {
                Precision p;
                bool is_all = parse_precision_extended(value, p);
                if (is_all) {
                    result.config.run_all_precisions = true;
                    // When running all, default to FP64 as the "base" precision
                    result.config.base.precision = Precision::FP64;
                    result.config.precision_single_modified = false;  // "all" is allowed
                } else {
                    result.config.base.precision = p;
                    result.config.run_all_precisions = false;
                    result.config.precision_single_modified = true;  // Single precision = not eligible
                }
                only_neutral_flags = false;  // Precision changes test behavior
            }
            // --repeats 
            else if (extract_arg_value(argv[i], "--repeats", value)) {
                result.config.base.repeats = static_cast<unsigned>(std::stoul(value));
                result.config.repeats_modified = true;  // Track modification
                only_neutral_flags = false;  // Repeats changes test behavior
            }
            // --output 
            else if (extract_arg_value(argv[i], "--output", value)) {
                result.config.base.output = string_to_output(value);
            }
            else {
                result.success = false;
                result.error_message = "Unknown argument: " + arg;
                return result;
            }
        } catch (const std::exception& e) {
            result.success = false;
            result.error_message = "Error parsing argument '" + arg + "': " + e.what();
            return result;
        }
    }
    
    // If only neutral flags were specified (like --socket, --quiet, --no-color),
    // enable full test mode as if no arguments were given
    if (only_neutral_flags && argc > 1) {
        result.config.run_full_test = true;
    }
    
    // Also enable full test mode if no arguments at all
    if (argc == 1) {
        result.config.run_full_test = true;
    }
    
    // Validate the config
    std::string validation_error;
    if (!validate_extended_config(result.config, validation_error)) {
        result.success = false;
        result.error_message = validation_error;
    }
    
    return result;
}

// Serialize extended config to command-line arguments (for round-trip testing)
inline std::vector<std::string> extended_config_to_args(const ExtendedConfig& config) {
    std::vector<std::string> args;
    
    args.push_back("--mode=" + mode_to_string(config.base.mode));
    
    // Only include size if explicitly set
    if (config.explicit_size_set) {
        args.push_back("--size=" + config.base.size.to_string());
    }
    
    args.push_back("--threads=" + std::to_string(config.base.threads));
    args.push_back("--time=" + std::to_string(config.base.min_time));
    
    if (config.run_all_precisions) {
        args.push_back("--precision=all");
    } else {
        args.push_back("--precision=" + precision_to_string(config.base.precision));
    }
    
    args.push_back("--repeats=" + std::to_string(config.base.repeats));
    args.push_back("--output=" + output_to_string(config.base.output));
    
    if (config.no_color) {
        args.push_back("--no-color");
    }
    
    if (config.force_scalar) {
        args.push_back("--force-scalar");
    }
    
    if (config.quiet) {
        args.push_back("--quiet");
    }
    
    if (!config.auto_size) {
        args.push_back("--no-auto-size");
    }
    
    if (!config.enable_warmup) {
        args.push_back("--no-warmup");
    }
    
    if (config.high_priority) {
        args.push_back("--high-priority");
    }
    
    return args;
}

// Compare two extended configs for equality (for round-trip testing)
inline bool extended_configs_equal(const ExtendedConfig& a, const ExtendedConfig& b) {
    // When run_all_precisions is true, the base precision doesn't matter
    // because --precision=all will run all precisions anyway
    bool precision_equal = (a.run_all_precisions && b.run_all_precisions) ||
                           (!a.run_all_precisions && !b.run_all_precisions && 
                            a.base.precision == b.base.precision);
    
    // Size comparison only matters if both have explicit size set
    bool size_equal = (a.explicit_size_set == b.explicit_size_set) &&
                      (!a.explicit_size_set || 
                       (a.base.size.Nx == b.base.size.Nx &&
                        a.base.size.Ny == b.base.size.Ny &&
                        a.base.size.Nz == b.base.size.Nz));
    
    return a.base.mode == b.base.mode &&
           size_equal &&
           a.base.threads == b.base.threads &&
           std::abs(a.base.min_time - b.base.min_time) < 1e-6 &&
           precision_equal &&
           a.base.repeats == b.base.repeats &&
           a.base.output == b.base.output &&
           a.run_all_precisions == b.run_all_precisions &&
           a.no_color == b.no_color &&
           a.force_scalar == b.force_scalar &&
           a.quiet == b.quiet &&
           a.auto_size == b.auto_size &&
           a.enable_warmup == b.enable_warmup &&
           a.high_priority == b.high_priority;
}

// Get list of all precision types for --precision=all mode
inline std::vector<Precision> get_all_precisions() {
    return {
        Precision::FP64,
        Precision::Float,  // FP32
        Precision::FP16,
        Precision::INT8,
        Precision::FP4
    };
}
