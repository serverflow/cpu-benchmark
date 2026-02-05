// CPU Benchmark - Main entry point
// Requirements: 7.1-7.5, 8.1-8.8, 9.1-9.5, 10.1-10.4
// Multi-Precision Requirements: 2.1, 3.3, 3.4, 3.8, 5.1, 6.4, 6.5, 8.3, 9.1, 12.4, 12.5
// SIMD Optimization Requirements: 2.1, 2.2, 2.3, 2.4, 4.3, 10.3, 10.4, 11.1, 11.2, 11.4
// Auto-Size Requirements: 7.1, 7.2, 7.3, 7.4, 7.5

#include "types.hpp"
#include "platform.hpp"
#include "cli.hpp"
#include "benchmark_core.hpp"
#include "math_kernels.hpp"
#include "precision_dispatcher.hpp"
#include "cpu_capabilities.hpp"
#include "comparison_output.hpp"
#include "memory_validator.hpp"
#include "benchmark_output.hpp"
#include "kernel_dispatcher.hpp"
#include "auto_size.hpp"
#include "warmup.hpp"
#include "cache_benchmark.hpp"
#include "compute_benchmark.hpp"
#include "result_submission.hpp"
#include "thread_affinity.hpp"
#include "version.hpp"

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <exception>
#include <random>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {
bool debug_enabled() {
    const char* env = std::getenv("SFBENCH_DEBUG");
    return env && env[0] != '\0' && env[0] != '0';
}

void debug_log(const std::string& msg) {
    if (debug_enabled()) {
        std::cerr << msg << "\n";
    }
}

void signal_handler(int sig) {
    std::cerr << "[fatal] signal " << sig << "\n";
    std::cerr.flush();
    std::abort();
}

void terminate_handler() {
    std::cerr << "[fatal] std::terminate called\n";
    std::cerr.flush();
    std::abort();
}

#ifdef _WIN32
LONG WINAPI unhandled_exception_filter(EXCEPTION_POINTERS* info) {
    if (info && info->ExceptionRecord) {
        std::cerr << "[fatal] unhandled exception 0x"
                  << std::hex << info->ExceptionRecord->ExceptionCode
                  << std::dec << "\n";
    } else {
        std::cerr << "[fatal] unhandled exception (unknown)\n";
    }
    std::cerr.flush();
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

void install_crash_handlers() {
    std::set_terminate(terminate_handler);
    std::signal(SIGABRT, signal_handler);
    std::signal(SIGSEGV, signal_handler);
    std::signal(SIGILL, signal_handler);
    std::signal(SIGFPE, signal_handler);
#ifdef _WIN32
    SetUnhandledExceptionFilter(unhandled_exception_filter);
#endif
}
} // namespace

// Print SFBench ASCII art banner
void print_banner() {
    std::cout << "\n";
    std::cout << R"(   _____ ______ ____                  _     )" << "\n";
    std::cout << R"(  / ____|  ____|  _ \                | |    )" << "\n";
    std::cout << R"( | (___ | |__  | |_) | ___ _ __   ___| |__  )" << "\n";
    std::cout << R"(  \___ \|  __| |  _ < / _ \ '_ \ / __| '_ \ )" << "\n";
    std::cout << R"(  ____) | |    | |_) |  __/ | | | (__| | | |)" << "\n";
    std::cout << R"( |_____/|_|    |____/ \___|_| |_|\___|_| |_|)" << "\n";
    std::cout << "Version: " << version::get_version_string() << "\n";
    std::cout << "\n";
}

// Generate a unique session ID for linking results from same benchmark run
static std::string generate_session_id() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 0xFFFF);
    
    std::ostringstream oss;
    oss << std::hex << millis << "-" << dis(gen) << dis(gen);
    return oss.str();
}

// Helper function to create a row in system info table
std::string sys_row(const std::string& label, const std::string& value, int label_width = 18, int value_width = 35) {
    std::ostringstream oss;
    oss << "| " << std::left << std::setw(label_width) << label 
        << "| " << std::left << std::setw(value_width) << value << "|";
    return oss.str();
}

// Print CPU information (Requirements 7.5, 10.4, SIMD Requirements 2.1, 2.2, 2.3, 10.3)
void print_cpu_info(const CpuInfo& cpu_info) {
    std::vector<std::pair<std::string, std::string>> rows;
    rows.emplace_back("OS", get_os_name());
    rows.emplace_back("Compiler", get_compiler_info());
    rows.emplace_back("Architecture", cpu_info.arch);
    rows.emplace_back("Logical cores", std::to_string(cpu_info.logical_cores));
    rows.emplace_back("Physical cores", std::to_string(cpu_info.physical_cores));

    unsigned socket_count = get_socket_count();
    if (socket_count > 1) {
        rows.emplace_back("CPU Sockets", std::to_string(socket_count) + " (NUMA)");
    }

    rows.emplace_back("Vendor", cpu_info.vendor);
    rows.emplace_back("Model", cpu_info.model);

    std::vector<unsigned> p_cores = get_performance_cores();
    if (!p_cores.empty() && p_cores.size() < cpu_info.logical_cores) {
        std::string p_core_info = std::to_string(p_cores.size()) + " P-threads, " +
                                  std::to_string(cpu_info.logical_cores - p_cores.size()) + " E-threads";
        rows.emplace_back("Core Types", p_core_info);
        rows.emplace_back("Best ST Core", "CPU " + std::to_string(get_best_performance_core()));
    }

    int label_w = 18;
    int value_w = 40;
    for (const auto& row : rows) {
        label_w = (std::max)(label_w, static_cast<int>(row.first.size()));
        value_w = (std::max)(value_w, static_cast<int>(row.second.size()));
    }

    const int total_w = label_w + value_w + 5;

    std::string h_line_top = "+" + std::string(total_w - 2, '-') + "+";
    std::string h_line_bot = "+" + std::string(total_w - 2, '-') + "+";
    std::string h_line_sec = "+" + std::string(total_w - 2, '=') + "+";

    std::string title = " SYSTEM INFORMATION ";
    int pad_left = (total_w - 2 - static_cast<int>(title.size())) / 2;
    int pad_right = total_w - 2 - static_cast<int>(title.size()) - pad_left;

    std::cout << "\n";
    std::cout << h_line_top << "\n";
    std::cout << "|" << std::string(pad_left, ' ') << title << std::string(pad_right, ' ') << "|\n";
    std::cout << h_line_sec << "\n";
    for (const auto& row : rows) {
        std::cout << sys_row(row.first, row.second, label_w, value_w) << "\n";
    }
    std::cout << h_line_bot << "\n";
    
    // Get SIMD capabilities 
    const CpuCapabilities& caps = CpuCapabilities::get();
    
    // Print SIMD capabilities section
    std::cout << "\nSIMD Capabilities:\n";
    std::cout << format_simd_capabilities(caps) << "\n";
    
    // Print selected SIMD optimization level
    const KernelDispatcher& dispatcher = KernelDispatcher::instance();
    SimdLevel simd_level = dispatcher.get_selected_level();
    std::cout << "  Active Level:  " << simd_level_to_string(simd_level);
    if (dispatcher.is_force_scalar()) {
        std::cout << " (forced scalar)";
    }
    std::cout << "\n";
    
    // Log kernel selection info 
    std::cout << "  Float Kernel:  " << dispatcher.get_kernel_name_float() << "\n";
    std::cout << "  Double Kernel: " << dispatcher.get_kernel_name_double() << "\n";
    
    // Print cache information section 
    std::cout << "\nCache Information:\n";
    std::cout << format_cache_info(cpu_info.cache, cpu_info.physical_cores) << "\n";
}

// Run benchmark with specified precision (Requirements 1.4, 1.5)
template<typename T>
std::string run_benchmark_typed(const Config& config, const CpuInfo& cpu_info) {
    Benchmark<T> benchmark(config, cpu_info);
    benchmark.run();
    
    // Format output based on config (Requirements 9.1-9.3)
    switch (config.output) {
        case OutputFormat::Text:
            return benchmark.format_text();
        case OutputFormat::Json:
            return benchmark.format_json();
        case OutputFormat::Csv:
            return benchmark.format_csv();
    }
    return "";
}

int main(int argc, char* argv[]) {
    install_crash_handlers();
    try {
        // Parse command-line arguments with extended precision support
      
        ExtendedParseResult parse_result = parse_args_extended(argc, argv);
        
        // Show version if requested 
        if (parse_result.show_version) {
            print_version();
            return static_cast<int>(ErrorCode::Success);
        }
        
        // Show help if requested
        if (parse_result.show_help) {
            print_usage_extended(argv[0]);
            return static_cast<int>(ErrorCode::Success);
        }
        
        // Check for parsing errors 
        if (!parse_result.success) {
            std::cerr << "Error: " << parse_result.error_message << std::endl;
            std::cerr << std::endl;
            print_usage_extended(argv[0]);
            return static_cast<int>(ErrorCode::InvalidArguments);
        }
        
        ExtendedConfig ext_config = parse_result.config;
        
        // Print banner for text output
        if (ext_config.base.output == OutputFormat::Text) {
            print_banner();
        }
        
        // Get CPU information 
        CpuInfo cpu_info = get_cpu_info();
        
        // Handle full benchmark mode (no arguments)
        if (ext_config.run_full_test) {
            // Print warning and ask for confirmation
            std::cout << "\n";
            std::cout << "========================================================================\n";
            std::cout << "                    CPU BENCHMARK SUITE\n";
            std::cout << "========================================================================\n";
            std::cout << "\n";
            std::cout << "This will run the benchmark suite including:\n";
            std::cout << "  1. CPU Warmup (2 seconds)\n";
            std::cout << "  2. Compute test (--mode=compute) - CPU peak performance\n";
            std::cout << "  3. Precision test (--precision=all) - All data types\n";
            std::cout << "\n";
            std::cout << "Do you want to run the benchmark? (y/n): ";
            std::cout.flush();
            
            std::string input;
            if (!std::getline(std::cin, input)) {
                return static_cast<int>(ErrorCode::Success);
            }
            
            // Trim and check response
            size_t start = input.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) {
                std::cout << "Benchmark cancelled.\n";
                return static_cast<int>(ErrorCode::Success);
            }
            size_t end = input.find_last_not_of(" \t\r\n");
            input = input.substr(start, end - start + 1);
            
            if (input != "y" && input != "Y" && input != "yes" && input != "Yes" && input != "YES") {
                std::cout << "Benchmark cancelled.\n";
                return static_cast<int>(ErrorCode::Success);
            }
            
            std::cout << "\nStarting benchmark suite...\n";
            
            // Print CPU info
            print_cpu_info(cpu_info);
            
            // Get OS version once at start
            std::string os_version = get_os_version();
            std::cout << "OS Version: " << os_version << "\n";
            
            // Handle socket selection
            if (ext_config.selected_socket >= 0) {
                unsigned socket_count = get_socket_count();
                if (static_cast<unsigned>(ext_config.selected_socket) >= socket_count) {
                    std::cerr << "Error: Socket " << ext_config.selected_socket 
                              << " does not exist. System has " << socket_count << " socket(s).\n";
                    return static_cast<int>(ErrorCode::InvalidArguments);
                }
                
                // Sync selected_socket to base config for thread pool configuration
                ext_config.base.selected_socket = ext_config.selected_socket;
                
                std::cout << "Restricting benchmark to socket " << ext_config.selected_socket << "\n";
                AffinityResult affinity_result = ThreadAffinityManager::set_process_socket_affinity(
                    static_cast<unsigned>(ext_config.selected_socket));
                if (affinity_result != AffinityResult::Success) {
                    std::cerr << "Warning: Failed to set socket affinity: " 
                              << affinity_result_to_string(affinity_result) << "\n";
                }
            }
            
            // Auto-detect optimal size
            TestSize auto_detected = AutoSizeDetector::detect_optimal_size(Precision::Float);
            ext_config.base.size = auto_detected.size;
            std::cout << "\nAuto-size: " << auto_detected.size.to_string() 
                      << " (" << auto_detected.description << ", "
                      << (auto_detected.memory_bytes / (1024*1024)) << " MB)\n";
            
            // 1. Warmup phase (2 seconds - same as individual tests)
            std::cout << "\n[1/3] Warming up CPU (2 seconds)...\n";
            WarmupConfig warmup_config;
            warmup_config.enabled = true;
            warmup_config.duration = std::chrono::seconds(2);
            warmup_config.wait_for_stable_frequency = true;
            WarmupResult warmup_result = WarmupManager::perform_warmup(warmup_config);
            std::cout << WarmupManager::format_result(warmup_result) << "\n";
            
            // Generate session ID to link compute and precision results
            std::string session_id = generate_session_id();
            
            // 2. Compute test
            std::cout << "\n[2/3] Running compute benchmark (ST + MT)...\n";
            ComputeSubmissionData compute_data;
            {
                debug_log("[main] compute run start");
                ComputeBenchmark compute_bench(0, false, ext_config.selected_socket);  // auto threads, no high priority, socket selection
                ComputeBenchmarkResults compute_results = compute_bench.run(3.0, 10.0);
                debug_log("[main] compute run done");
                std::cout << ComputeBenchmark::format_text(compute_results);
                std::cout.flush();
                debug_log("[main] compute results printed");
                
                // Store results for submission
                compute_data.st_time_sec = compute_results.single_thread.time_sec;
                compute_data.st_gflops = compute_results.single_thread.gflops;
                compute_data.st_score = compute_results.st_score;
                compute_data.st_threads = compute_results.single_thread.threads;
                compute_data.mt_time_sec = compute_results.multi_thread.time_sec;
                compute_data.mt_gflops = compute_results.multi_thread.gflops;
                compute_data.mt_score = compute_results.mt_score;
                compute_data.mt_threads = compute_results.multi_thread.threads;
                compute_data.overall_score = compute_results.overall_score;
                compute_data.simd_level = "scalar_fp64";
                compute_data.socket_count = static_cast<int>(get_socket_count());
                compute_data.selected_socket = ext_config.selected_socket;
                compute_data.os_version = os_version;
                compute_data.session_id = session_id;
                
                // Store frequency data
                if (compute_results.frequency.available) {
                    compute_data.frequency.min_mhz = compute_results.frequency.min_mhz;
                    compute_data.frequency.max_mhz = compute_results.frequency.max_mhz;
                    compute_data.frequency.avg_mhz = compute_results.frequency.avg_mhz;
                    compute_data.frequency.available = true;
                }
            }
            
            // 3. Precision test (--precision=all)
            std::cout << "\n[3/3] Running precision comparison benchmark...\n";
            PrecisionAllSubmissionData precision_data;
            {
                Config precision_config = ext_config.base;
                precision_config.mode = BenchmarkMode::Mem;
                precision_config.precision = Precision::FP64;
                precision_config.selected_socket = ext_config.selected_socket;  // Pass socket selection for NUMA
                
                ComparisonTable table = run_all_precision_benchmarks(precision_config, cpu_info, true);
                std::cout << table.format_text(true);
                
                // Store results for submission
                for (const auto& r : table.results()) {
                    PrecisionResultData prd;
                    prd.precision_name = precision_to_string(r.precision);
                    prd.fp16_mode = (r.fp16_mode == FP16Mode::Native) ? "native" : "emulated";
                    prd.bytes_per_element = r.config.bytes_per_element;
                    prd.is_integer = r.config.is_integer;
                    prd.is_emulated = r.config.is_emulated;
                    prd.time_min_sec = r.result.time_min_sec;
                    prd.time_avg_sec = r.result.time_avg_sec;
                    prd.time_stddev_sec = r.result.time_stddev_sec;
                    prd.gflops_avg = r.result.gflops_avg;
                    prd.gflops_max = r.result.gflops_max;
                    prd.total_flops = r.result.total_flops;
                    prd.iterations = r.result.iterations;
                    precision_data.results.push_back(prd);
                }
                precision_data.socket_count = static_cast<int>(get_socket_count());
                precision_data.selected_socket = ext_config.selected_socket;
                precision_data.os_version = os_version;
                precision_data.session_id = session_id;
            }
            
            // Print summary with colors
            std::cout << "\n";
            std::cout << "========================================================================\n";
            std::cout << "                    BENCHMARK COMPLETE\n";
            std::cout << "========================================================================\n";
            std::cout << "\n";
            
            // Color codes for Windows console
            #ifdef _WIN32
            HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
            CONSOLE_SCREEN_BUFFER_INFO consoleInfo;
            GetConsoleScreenBufferInfo(hConsole, &consoleInfo);
            WORD savedAttributes = consoleInfo.wAttributes;
            #endif
            
            std::cout << "+---------------------------+------------------+\n";
            std::cout << "|         SCORE             |      VALUE       |\n";
            std::cout << "+---------------------------+------------------+\n";
            
            // ST Score - Cyan color
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
            #else
            std::cout << "\033[1;36m";  // Cyan
            #endif
            std::cout << "|  Single-Thread Score      |";
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
            #else
            std::cout << "\033[1;32m";  // Green
            #endif
            std::cout << std::right << std::setw(14) << compute_data.st_score << "    ";
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, savedAttributes);
            #else
            std::cout << "\033[0m";
            #endif
            std::cout << "|\n";
            
            // MT Score - Yellow color
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
            #else
            std::cout << "\033[1;33m";  // Yellow
            #endif
            std::cout << "|  Multi-Thread Score       |";
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
            #else
            std::cout << "\033[1;32m";  // Green
            #endif
            std::cout << std::right << std::setw(14) << compute_data.mt_score << "    ";
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, savedAttributes);
            #else
            std::cout << "\033[0m";
            #endif
            std::cout << "|\n";
            
            std::cout << "+---------------------------+------------------+\n";
            std::cout << "\n";
            
            // Ask once for submission
            std::cout << "Would you like to submit your results to the benchmark server? (y/n): ";
            std::cout.flush();
            
            std::string submit_input;
            if (!std::getline(std::cin, submit_input)) {
                return static_cast<int>(ErrorCode::Success);
            }
            
            // Trim input
            size_t s_start = submit_input.find_first_not_of(" \t\r\n");
            if (s_start == std::string::npos) {
                return static_cast<int>(ErrorCode::Success);
            }
            size_t s_end = submit_input.find_last_not_of(" \t\r\n");
            submit_input = submit_input.substr(s_start, s_end - s_start + 1);
            
            if (submit_input != "y" && submit_input != "Y" && submit_input != "yes" && submit_input != "Yes" && submit_input != "YES") {
                std::cout << "Results not submitted.\n";
                return static_cast<int>(ErrorCode::Success);
            }
            
            // Get nickname once
            std::cout << "Enter your nickname (press Enter for Anonymous): ";
            std::cout.flush();
            std::string nickname;
            if (!std::getline(std::cin, nickname)) {
                nickname = "Anonymous";
            }
            // Trim nickname
            size_t n_start = nickname.find_first_not_of(" \t\r\n");
            if (n_start == std::string::npos) {
                nickname = "Anonymous";
            } else {
                size_t n_end = nickname.find_last_not_of(" \t\r\n");
                nickname = nickname.substr(n_start, n_end - n_start + 1);
                if (nickname.empty()) nickname = "Anonymous";
            }
            
            // Submit results
            Config submission_config = ext_config.base;
            submission_config.threads = submission_config.threads == 0 
                ? get_logical_core_count()
                : submission_config.threads;
            
            // Submit compute results
            std::cout << "Submitting compute results as \"" << nickname << "\"..." << std::flush;
            bool compute_ok = submit_compute_results(compute_data, submission_config, cpu_info, nickname);
            if (compute_ok) {
                std::cout << " Done!\n";
            } else {
                std::cout << " Failed!\n";
            }
            
            // Submit precision_all results
            std::cout << "Submitting precision results as \"" << nickname << "\"..." << std::flush;
            bool precision_ok = submit_precision_all_results(precision_data, submission_config, cpu_info, nickname);
            if (precision_ok) {
                std::cout << " Done!\n";
            } else {
                std::cout << " Failed!\n";
            }
            
            return static_cast<int>(ErrorCode::Success);
        }
        
        if (ext_config.auto_size && !ext_config.explicit_size_set) {
            TestSize auto_detected = AutoSizeDetector::detect_optimal_size(
                ext_config.base.precision);
            ext_config.base.size = auto_detected.size;
            
            if (ext_config.base.output == OutputFormat::Text && !ext_config.quiet) {
                std::cout << "\nAuto-size: " << auto_detected.size.to_string() 
                          << " (" << auto_detected.description << ", "
                          << (auto_detected.memory_bytes / (1024*1024)) << " MB)\n";
            }
        }
        
        const Config& config = ext_config.base;
        
        // Handle socket selection for non-full_test modes
        if (ext_config.selected_socket >= 0) {
            unsigned socket_count = get_socket_count();
            if (static_cast<unsigned>(ext_config.selected_socket) >= socket_count) {
                std::cerr << "Error: Socket " << ext_config.selected_socket 
                          << " does not exist. System has " << socket_count << " socket(s).\n";
                return static_cast<int>(ErrorCode::InvalidArguments);
            }
            
            if (config.output == OutputFormat::Text && !ext_config.quiet) {
                std::cout << "\nRestricting benchmark to socket " << ext_config.selected_socket << "\n";
            }
            AffinityResult affinity_result = ThreadAffinityManager::set_process_socket_affinity(
                static_cast<unsigned>(ext_config.selected_socket));
            if (affinity_result != AffinityResult::Success && config.output == OutputFormat::Text) {
                std::cerr << "Warning: Failed to set socket affinity: " 
                          << affinity_result_to_string(affinity_result) << "\n";
            }
        }
        
        // Configure KernelDispatcher with force_scalar flag 
        KernelDispatcher& dispatcher = get_kernel_dispatcher();
        dispatcher.set_force_scalar(ext_config.force_scalar);
        
        // Print CPU info for text output (includes SIMD capabilities and cache info)
        if (config.output == OutputFormat::Text) {
            print_cpu_info(cpu_info);
        }
        
        // Perform CPU warmup phase (Requirements 8.1, 8.2, 8.3, 8.4, 8.5)
        WarmupResult warmup_result;
        if (ext_config.enable_warmup) {
            if (config.output == OutputFormat::Text && !ext_config.quiet) {
                std::cout << "\nWarming up CPU...\n";
            }
            
            WarmupConfig warmup_config;
            warmup_config.enabled = true;
            warmup_config.duration = std::chrono::seconds(2);
            warmup_config.wait_for_stable_frequency = true;
            
            warmup_result = WarmupManager::perform_warmup(warmup_config);
            
            if (config.output == OutputFormat::Text && !ext_config.quiet) {
                std::cout << WarmupManager::format_result(warmup_result) << "\n";
            }
        } else {
            // Warmup disabled via --no-warmup flag 
            WarmupConfig warmup_config;
            warmup_config.enabled = false;
            warmup_result = WarmupManager::perform_warmup(warmup_config);
            
            if (config.output == OutputFormat::Text && !ext_config.quiet) {
                std::cout << "\nWarmup: Skipped (--no-warmup)\n";
            }
        }
        
        
        // For --precision=all mode, check memory for the largest precision (FP64)
        Precision mem_check_precision = ext_config.run_all_precisions 
            ? Precision::FP64 
            : config.precision;
        
        MemoryRequirement mem_req = MemoryValidator::validate(
            config.size, mem_check_precision);
        
        if (!mem_req.sufficient) {
            std::cerr << "Error: Insufficient memory for benchmark.\n"
                      << "  Required: " << MemoryValidator::format_bytes(mem_req.required_bytes) << "\n"
                      << "  Available: " << MemoryValidator::format_bytes(mem_req.available_bytes) << "\n";
            return static_cast<int>(ErrorCode::OutOfMemory);
        }
        
        
        // Special handling: 10s warmup, ST test, MT test, results table
        if (config.mode == BenchmarkMode::Compute) {
            // Determine warmup duration
            double warmup_sec = ext_config.enable_warmup ? 3.0 : 0.0;
            // Compute mode uses longer test duration for stability
            double compute_test_sec = 10.0;
            
            if (config.output == OutputFormat::Text && !ext_config.quiet) {
                std::cout << "\nRunning compute benchmark (ST + MT)...\n";
                if (warmup_sec > 0) {
                    std::cout << "Warmup: " << static_cast<int>(warmup_sec) << " seconds\n";
                } else {
                    std::cout << "Warmup: Skipped\n";
                }
            }
            
            // Create compute benchmark with thread count, high priority setting, and socket selection
            ComputeBenchmark compute_bench(config.threads, ext_config.high_priority, ext_config.selected_socket);
            // New API: run(warmup_seconds, test_seconds) - fixed time measurement
            ComputeBenchmarkResults compute_results = compute_bench.run(
                warmup_sec, compute_test_sec);  // 10 seconds per test for compute mode
            
            // Output results based on format
            std::string output;
            switch (config.output) {
                case OutputFormat::Text:
                    output = ComputeBenchmark::format_text(compute_results);
                    break;
                case OutputFormat::Json:
                    output = ComputeBenchmark::format_json(compute_results);
                    break;
                case OutputFormat::Csv:
                    output = "test,type,threads,time_sec,gflops,score\n";
                    output += "single_core,scalar_fp64," + std::to_string(compute_results.single_thread.threads) + ",";
                    output += std::to_string(compute_results.single_thread.time_sec) + ",";
                    output += std::to_string(compute_results.single_thread.gflops) + ",";
                    output += std::to_string(compute_results.st_score) + "\n";
                    output += "all_cores,scalar_fp64," + std::to_string(compute_results.multi_thread.threads) + ",";
                    output += std::to_string(compute_results.multi_thread.time_sec) + ",";
                    output += std::to_string(compute_results.multi_thread.gflops) + ",";
                    output += std::to_string(compute_results.mt_score) + "\n";
                    if (compute_results.simd_available) {
                        output += "simd_single,simd_fp32," + std::to_string(compute_results.simd_single_thread.threads) + ",";
                        output += std::to_string(compute_results.simd_single_thread.time_sec) + ",";
                        output += std::to_string(compute_results.simd_single_thread.gflops) + ",-\n";
                        output += "simd_multi,simd_fp32," + std::to_string(compute_results.simd_multi_thread.threads) + ",";
                        output += std::to_string(compute_results.simd_multi_thread.time_sec) + ",";
                        output += std::to_string(compute_results.simd_multi_thread.gflops) + ",-\n";
                    }
                    break;
            }
            std::cout << output;
            
            // Offer result submission in text mode for compute benchmark (only for clean runs)
            if (config.output == OutputFormat::Text) {
                if (ext_config.is_submission_eligible()) {
                    // Create ComputeSubmissionData with ST/MT results (scalar FP64 baseline)
                    ComputeSubmissionData compute_data;
                    compute_data.st_time_sec = compute_results.single_thread.time_sec;
                    compute_data.st_gflops = compute_results.single_thread.gflops;
                    compute_data.st_score = compute_results.st_score;
                    compute_data.st_threads = compute_results.single_thread.threads;
                    
                    compute_data.mt_time_sec = compute_results.multi_thread.time_sec;
                    compute_data.mt_gflops = compute_results.multi_thread.gflops;
                    compute_data.mt_score = compute_results.mt_score;
                    compute_data.mt_threads = compute_results.multi_thread.threads;
                    
                    compute_data.overall_score = compute_results.overall_score;
                    compute_data.simd_level = "scalar_fp64";  // Score basis
                    
                    // Add socket and OS info
                    compute_data.socket_count = static_cast<int>(get_socket_count());
                    compute_data.selected_socket = ext_config.selected_socket;
                    compute_data.os_version = get_os_version();
                    
                    // Add frequency data
                    if (compute_results.frequency.available) {
                        compute_data.frequency.min_mhz = compute_results.frequency.min_mhz;
                        compute_data.frequency.max_mhz = compute_results.frequency.max_mhz;
                        compute_data.frequency.avg_mhz = compute_results.frequency.avg_mhz;
                        compute_data.frequency.available = true;
                    }
                    
                    submit_compute_results_interactive(compute_data, config, cpu_info);
                } else {
                    std::cout << "\n[Submission disabled: " << ext_config.get_submission_ineligibility_reason() << "]\n";
                }
            }
            
            return static_cast<int>(ErrorCode::Success);
        }
        
        // Handle cache-level benchmark mode (Requirements 17.3, 17.5)
        if (config.mode == BenchmarkMode::CacheLevel) {
            if (config.output == OutputFormat::Text && !ext_config.quiet) {
                std::cout << "\nRunning cache-level benchmarks...\n";
            }
            
            // Run cache-level benchmarks for float precision
            CacheBenchmark<float> cache_bench(config.threads, config.repeats);
            CacheBenchmarkResults cache_results = cache_bench.run_all_levels();
            cache_results.warmup_performed = warmup_result.performed;
            
            // Output results based on format
            std::string output;
            switch (config.output) {
                case OutputFormat::Text:
                    output = CacheBenchmark<float>::format_text(cache_results);
                    break;
                case OutputFormat::Json:
                    output = CacheBenchmark<float>::format_json(cache_results);
                    break;
                case OutputFormat::Csv:
                    // CSV format: simple table
                    output = "level,data_size_bytes,time_min_sec,bandwidth_gbs,gflops,fits_in_cache\n";
                    for (const auto& r : cache_results.level_results) {
                        output += cache_level_to_string(r.level) + ",";
                        output += std::to_string(r.data_size_bytes) + ",";
                        output += std::to_string(r.time_min_sec) + ",";
                        output += std::to_string(r.bandwidth_gbs) + ",";
                        output += std::to_string(r.gflops) + ",";
                        output += (r.fits_in_cache ? "true" : "false") + std::string("\n");
                    }
                    break;
            }
            std::cout << output;
            
            // Offer result submission in text mode for cache benchmark (only for clean runs)
            if (config.output == OutputFormat::Text && !cache_results.level_results.empty()) {
                if (ext_config.is_submission_eligible()) {
                    // Use the best result (usually L1 cache level) for submission
                    const auto& best = cache_results.level_results[0];
                    BenchmarkResult cache_result;
                    cache_result.time_avg_sec = best.time_min_sec;
                    cache_result.time_min_sec = best.time_min_sec;
                    cache_result.time_stddev_sec = 0.0;
                    cache_result.gflops_avg = best.gflops;
                    cache_result.gflops_max = best.gflops;
                    cache_result.total_flops = static_cast<uint64_t>(best.gflops * 1e9 * best.time_min_sec);
                    cache_result.iterations = 1;
                    
                    Config submission_config = config;
                    submission_config.threads = config.threads == 0 
                        ? get_logical_core_count()
                        : config.threads;
                    submit_results_interactive(cache_result, submission_config, cpu_info);
                } else {
                    std::cout << "\n[Submission disabled: " << ext_config.get_submission_ineligibility_reason() << "]\n";
                }
            }
            
            return static_cast<int>(ErrorCode::Success);
        }
        
       
        if (ext_config.run_all_precisions) {
            // Run benchmarks for all precision types
            ComparisonTable table = run_all_precision_benchmarks(config, cpu_info, 
                config.output == OutputFormat::Text);
            
            // Set color preference based on --no-color flag
            table.set_use_colors(!ext_config.no_color);
            
            // Output results based on format
            std::string output;
            switch (config.output) {
                case OutputFormat::Text:
                    output = table.format_text(!ext_config.no_color);
                    break;
                case OutputFormat::Json:
                    output = table.format_json();
                    break;
                case OutputFormat::Csv:
                    output = table.format_csv();
                    break;
            }
            std::cout << output;
            
            // Offer result submission in text mode for precision=all (only for clean runs)
            if (config.output == OutputFormat::Text && !table.results().empty()) {
                if (ext_config.is_submission_eligible()) {
                    // Build submission data with ALL precision results
                    PrecisionAllSubmissionData precision_data;
                    for (const auto& r : table.results()) {
                        PrecisionResultData prd;
                        prd.precision_name = precision_to_string(r.precision);
                        prd.fp16_mode = (r.fp16_mode == FP16Mode::Native) ? "native" : "emulated";
                        prd.bytes_per_element = r.config.bytes_per_element;
                        prd.is_integer = r.config.is_integer;
                        prd.is_emulated = r.config.is_emulated;
                        prd.time_min_sec = r.result.time_min_sec;
                        prd.time_avg_sec = r.result.time_avg_sec;
                        prd.time_stddev_sec = r.result.time_stddev_sec;
                        prd.gflops_avg = r.result.gflops_avg;
                        prd.gflops_max = r.result.gflops_max;
                        prd.total_flops = r.result.total_flops;
                        prd.iterations = r.result.iterations;
                        precision_data.results.push_back(prd);
                    }
                    
                    // Create config for submission
                    Config submission_config = config;
                    submission_config.threads = config.threads == 0 
                        ? get_logical_core_count()
                        : config.threads;
                    
                    // Submit all precision results
                    submit_precision_all_results_interactive(precision_data, submission_config, cpu_info);
                } else {
                    std::cout << "\n[Submission disabled: " << ext_config.get_submission_ineligibility_reason() << "]\n";
                }
            }
        } else {
            // Single precision mode - use precision dispatcher
            // (Multi-Precision Requirements 2.1, 3.3, 3.4, 5.1, 6.4)
            PrecisionResult result = run_benchmark_for_precision(
                config.precision, config, cpu_info);
            
            // Get thread count for output
            unsigned thread_count = config.threads == 0 
                ? get_logical_core_count()
                : config.threads;
            
            // Format and output results
            std::string output = format_single_precision_output(
                result, config, cpu_info, thread_count);
            std::cout << output;
            
            // Single precision mode - check submission eligibility
            // Submission is NOT allowed for explicit single precision (--precision=fp32, etc.)
            // But IS allowed for default mode runs without --precision flag
            if (config.output == OutputFormat::Text) {
                if (ext_config.is_submission_eligible()) {
                    // Create a copy of config with actual thread count for submission
                    Config submission_config = config;
                    submission_config.threads = thread_count;
                    submit_results_interactive(result.result, submission_config, cpu_info);
                } else {
                    std::cout << "\n[Submission disabled: " << ext_config.get_submission_ineligibility_reason() << "]\n";
                }
            }
        }
        
        return static_cast<int>(ErrorCode::Success);
        
    } catch (const std::bad_alloc& e) {
        std::cerr << "Error: Out of memory - " << e.what() << std::endl;
        return static_cast<int>(ErrorCode::OutOfMemory);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return static_cast<int>(ErrorCode::UnknownError);
    }
}
