#pragma once
// CPU Benchmark - Comparison Table Output


#include "types.hpp"
#include "cpu_capabilities.hpp"
#include "platform.hpp"
#include "warmup.hpp"
#include "score_calculator.hpp"
#include "reference_comparison.hpp"
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

// Result for single precision run (Requirement 9.1)
struct PrecisionResult {
    Precision precision;
    FP16Mode fp16_mode;         // Only relevant for FP16
    BenchmarkResult result;
    PrecisionConfig config;
    BenchmarkMode mode;         // Benchmark mode for MLUP/s display
    
    PrecisionResult() 
        : precision(Precision::Float)
        , fp16_mode(FP16Mode::Emulated)
        , result()
        , config()
        , mode(BenchmarkMode::Mem)
    {}
    
    PrecisionResult(Precision p, FP16Mode fp16m, const BenchmarkResult& res, 
                    const PrecisionConfig& cfg, BenchmarkMode m)
        : precision(p)
        , fp16_mode(fp16m)
        , result(res)
        , config(cfg)
        , mode(m)
    {}
};

// Terminal color support detection (Requirement 10.4)
inline bool terminal_supports_colors() {
#ifdef _WIN32
    // Windows: Check if output is a console and enable ANSI support
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) {
        return false;
    }
    
    DWORD dwMode = 0;
    if (!GetConsoleMode(hOut, &dwMode)) {
        return false;
    }

    // Try to enable virtual terminal processing for ANSI colors
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    if (SetConsoleMode(hOut, dwMode)) {
        return true;
    }
    
    // Fallback: check for common environment variables
    const char* term = std::getenv("TERM");
    const char* colorterm = std::getenv("COLORTERM");
    const char* wt_session = std::getenv("WT_SESSION");  // Windows Terminal
    
    if (wt_session != nullptr) return true;
    if (colorterm != nullptr) return true;
    if (term != nullptr && std::string(term) != "dumb") return true;
    
    return false;
#else
    // Unix: Check if stdout is a TTY and TERM is set
    if (!isatty(STDOUT_FILENO)) {
        return false;
    }
    
    const char* term = std::getenv("TERM");
    if (term == nullptr || std::string(term) == "dumb") {
        return false;
    }
    
    return true;
#endif
}

// Comparison table formatter (Requirements 9.1-9.6, 10.1-10.4)
class ComparisonTable {
public:
    ComparisonTable() : use_colors_(true) {}
    
    // Add a result to the table
    void add_result(const PrecisionResult& result) {
        results_.push_back(result);
    }
    
    // Set whether to use colors
    void set_use_colors(bool use_colors) {
        use_colors_ = use_colors;
    }
    
    // Format as ASCII table (Requirements 9.1-9.6, 10.1-10.3)
    // Score display Requirements: 9.1, 9.3
    std::string format_text(bool use_colors = true, bool show_score = true) const {
        if (results_.empty()) {
            return "No results to display.\n";
        }
        
        bool colors_enabled = use_colors && use_colors_ && terminal_supports_colors();
        
        std::ostringstream oss;
        
        // Check if we need MLUP/s column (stencil mode)
        bool show_mlups = false;
        for (const auto& r : results_) {
            if (r.mode == BenchmarkMode::Stencil) {
                show_mlups = true;
                break;
            }
        }

        struct RowStrings {
            std::string type;
            std::string bytes;
            std::string time_min;
            std::string time_avg;
            std::string perf;
            std::string mlups;
        };

        std::vector<RowStrings> rows;
        rows.reserve(results_.size());

        size_t max_type = std::string("Type").size();
        size_t max_bytes = std::string("Bytes").size();
        size_t max_time_min = std::string("Time min").size();
        size_t max_time_avg = std::string("Time avg").size();
        size_t max_perf = std::string("GFLOPS/GOPS").size();
        size_t max_mlups = std::string("MLUP/s").size();

        for (const auto& r : results_) {
            RowStrings row;

            row.type = r.config.name;
            if (r.precision == Precision::FP16) {
                row.type += r.fp16_mode == FP16Mode::Native ? " (native)" : " (emulated)";
            } else if (r.precision == Precision::FP4) {
                row.type += " (emulated)";
            }

            if (r.precision == Precision::FP4) {
                row.bytes = "0.5*";
            } else {
                std::ostringstream bs;
                bs << std::fixed << std::setprecision(0) << r.config.bytes_per_element;
                row.bytes = bs.str();
            }

            std::ostringstream time_min_ss, time_avg_ss;
            time_min_ss << std::fixed << std::setprecision(3) << r.result.time_min_sec * 1000.0 << " ms";
            time_avg_ss << std::fixed << std::setprecision(3) << r.result.time_avg_sec * 1000.0 << " ms";
            row.time_min = time_min_ss.str();
            row.time_avg = time_avg_ss.str();

            std::ostringstream perf_ss;
            if (r.config.is_integer) {
                perf_ss << std::fixed << std::setprecision(2) << r.result.gflops_avg << " (GOPS)";
            } else {
                perf_ss << std::fixed << std::setprecision(2) << r.result.gflops_avg;
            }
            row.perf = perf_ss.str();

            if (show_mlups) {
                std::ostringstream mlups_ss;
                mlups_ss << std::fixed << std::setprecision(2) << r.result.mlups_avg;
                row.mlups = mlups_ss.str();
            }

            max_type = (std::max)(max_type, row.type.size());
            max_bytes = (std::max)(max_bytes, row.bytes.size());
            max_time_min = (std::max)(max_time_min, row.time_min.size());
            max_time_avg = (std::max)(max_time_avg, row.time_avg.size());
            max_perf = (std::max)(max_perf, row.perf.size());
            if (show_mlups) {
                max_mlups = (std::max)(max_mlups, row.mlups.size());
            }

            rows.push_back(std::move(row));
        }

        // Column widths (minimums keep layout readable)
        const int type_w = (std::max)(18, static_cast<int>(max_type));
        const int bytes_w = (std::max)(6, static_cast<int>(max_bytes));
        const int time_min_w = (std::max)(12, static_cast<int>(max_time_min));
        const int time_avg_w = (std::max)(12, static_cast<int>(max_time_avg));
        const int perf_w = (std::max)(14, static_cast<int>(max_perf));
        const int mlups_w = show_mlups ? (std::max)(10, static_cast<int>(max_mlups)) : 0;
        
        // Build horizontal lines
        std::string h_line = "+";
        h_line += std::string(type_w + 2, '-') + "+";
        h_line += std::string(bytes_w + 2, '-') + "+";
        h_line += std::string(time_min_w + 2, '-') + "+";
        h_line += std::string(time_avg_w + 2, '-') + "+";
        h_line += std::string(perf_w + 2, '-') + "+";
        if (show_mlups) {
            h_line += std::string(mlups_w + 2, '-') + "+";
        }
        
        // Find best performance for highlighting
        size_t best_idx = find_best_performance();
        
        // Title
        oss << "\n";
        oss << h_line << "\n";
        oss << "|" << center_text("PRECISION COMPARISON", h_line.length() - 2) << "|\n";
        oss << h_line << "\n";
        
        // Header row
        oss << "| " << std::left << std::setw(type_w) << "Type"
            << " | " << std::right << std::setw(bytes_w) << "Bytes"
            << " | " << std::setw(time_min_w) << "Time min"
            << " | " << std::setw(time_avg_w) << "Time avg"
            << " | " << std::setw(perf_w) << "GFLOPS/GOPS";
        if (show_mlups) {
            oss << " | " << std::setw(mlups_w) << "MLUP/s";
        }
        oss << " |\n";
        oss << h_line << "\n";

        // Data rows
        for (size_t i = 0; i < results_.size(); ++i) {
            const auto& row = rows[i];

            std::string perf_str = row.perf;
            if (i == best_idx && colors_enabled) {
                perf_str = color_green(perf_str, true);
            }

            oss << "| " << std::left << std::setw(type_w) << row.type
                << " | " << std::right << std::setw(bytes_w) << row.bytes
                << " | " << std::setw(time_min_w) << row.time_min
                << " | " << std::setw(time_avg_w) << row.time_avg
                << " | ";

            if (i == best_idx && colors_enabled) {
                size_t visible_len = row.perf.length();
                if (visible_len < static_cast<size_t>(perf_w)) {
                    oss << std::string(perf_w - visible_len, ' ');
                }
                oss << perf_str;
            } else {
                oss << std::setw(perf_w) << perf_str;
            }

            if (show_mlups) {
                oss << " | " << std::setw(mlups_w) << row.mlups;
            }
            oss << " |\n";
        }
        
        oss << h_line << "\n";
        
        // Footnote for FP4 (Requirement 9.3)
        bool has_fp4 = false;
        for (const auto& r : results_) {
            if (r.precision == Precision::FP4) {
                has_fp4 = true;
                break;
            }
        }
        if (has_fp4) {
            oss << "* FP4 stores 2 values per byte (4 bits each)\n";
        }
        
        // Notes about emulation overhead (Requirements 12.4, 12.5)
        bool has_fp16_emulated = false;
        bool has_fp4_any = false;
        for (const auto& r : results_) {
            if (r.precision == Precision::FP16 && r.fp16_mode == FP16Mode::Emulated) {
                has_fp16_emulated = true;
            }
            if (r.precision == Precision::FP4) {
                has_fp4_any = true;
            }
        }
        if (has_fp16_emulated) {
            oss << "Note: FP16 (emulated) includes conversion overhead (store as half, compute as float)\n";
        }
        if (has_fp4_any) {
            oss << "Note: FP4 is CPU emulation with significant conversion overhead\n";
        }
        
        // Score section removed - no longer showing reference comparison
        (void)show_score; // Suppress unused parameter warning
        
        oss << "\n";
        return oss.str();
    }

    // Format as JSON array (Requirement 11.1, 11.3, 11.5)
    std::string format_json() const {
        if (results_.empty()) {
            return "[]";
        }
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(9);
        oss << "[";
        
        for (size_t i = 0; i < results_.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "\n  ";
            oss << format_single_json(results_[i]);
        }
        
        oss << "\n]";
        return oss.str();
    }
    
    // Format as CSV (Requirement 11.2, 11.4)
    std::string format_csv() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        
        // Header with precision column
        oss << "precision,fp16_mode,bytes_per_element,time_min_ms,time_avg_ms,";
        oss << "time_stddev_ms,gflops_avg,gflops_max,mlups_avg\n";
        
        // Data rows
        for (const auto& r : results_) {
            oss << precision_to_string(r.precision) << ",";
            
            // FP16 mode (only meaningful for FP16)
            if (r.precision == Precision::FP16) {
                oss << (r.fp16_mode == FP16Mode::Native ? "native" : "emulated");
            } else {
                oss << "n/a";
            }
            oss << ",";
            
            oss << r.config.bytes_per_element << ",";
            oss << r.result.time_min_sec * 1000.0 << ",";
            oss << r.result.time_avg_sec * 1000.0 << ",";
            oss << r.result.time_stddev_sec * 1000.0 << ",";
            oss << r.result.gflops_avg << ",";
            oss << r.result.gflops_max << ",";
            oss << r.result.mlups_avg << "\n";
        }
        
        return oss.str();
    }
    
    // Get results vector
    const std::vector<PrecisionResult>& results() const { return results_; }
    
    // Clear all results
    void clear() { results_.clear(); }
    
private:
    std::vector<PrecisionResult> results_;
    bool use_colors_;
    
    // Find index of best GFLOPS/GOPS result (Requirement 10.1)
    size_t find_best_performance() const {
        if (results_.empty()) return 0;
        
        size_t best_idx = 0;
        double best_perf = results_[0].result.gflops_avg;
        
        for (size_t i = 1; i < results_.size(); ++i) {
            if (results_[i].result.gflops_avg > best_perf) {
                best_perf = results_[i].result.gflops_avg;
                best_idx = i;
            }
        }
        
        return best_idx;
    }
    
    // Calculate score from results (Requirements 9.1, 9.3)
    BenchmarkScore calculate_score_from_results() const {
        ScoreCalculator calculator;
        
        for (const auto& r : results_) {
            TestResultEntry entry;
            entry.name = precision_to_string(r.precision);
            entry.gflops = r.result.gflops_avg;
            entry.bandwidth_gbps = r.result.bandwidth_gbs;
            entry.is_single_thread = true;  // Default to ST for comparison mode
            
            // Determine test type based on mode
            switch (r.mode) {
                case BenchmarkMode::Mem:
                case BenchmarkMode::CacheLevel:
                    entry.type = TestResultEntry::TestType::Memory;
                    break;
                case BenchmarkMode::Stencil:
                    entry.type = TestResultEntry::TestType::Mixed;
                    break;
                case BenchmarkMode::Matmul3D:
                case BenchmarkMode::Compute:
                    entry.type = TestResultEntry::TestType::Compute;
                    break;
            }
            calculator.add_result(entry);
        }
        
        return calculator.calculate_score();
    }
    
    // ANSI color helpers (Requirement 10.1)
    std::string color_green(const std::string& s, bool enabled) const {
        if (!enabled) return s;
        return "\033[32m" + s + "\033[0m";
    }
    
    std::string color_reset(bool enabled) const {
        if (!enabled) return "";
        return "\033[0m";
    }
    
    // Center text in a given width
    std::string center_text(const std::string& text, size_t width) const {
        if (text.length() >= width) return text;
        size_t padding = width - text.length();
        size_t left_pad = padding / 2;
        size_t right_pad = padding - left_pad;
        return std::string(left_pad, ' ') + text + std::string(right_pad, ' ');
    }
    
    // Format single result as JSON object (Requirement 11.3, 11.5)
    std::string format_single_json(const PrecisionResult& r) const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(9);
        
        oss << "{";
        oss << "\"precision\":\"" << precision_to_string(r.precision) << "\",";
        
        // FP16 mode field (Requirement 11.5)
        if (r.precision == Precision::FP16) {
            oss << "\"fp16_mode\":\"" << (r.fp16_mode == FP16Mode::Native ? "native" : "emulated") << "\",";
        }
        
        oss << "\"bytes_per_element\":" << r.config.bytes_per_element << ",";
        oss << "\"is_integer\":" << (r.config.is_integer ? "true" : "false") << ",";
        oss << "\"is_emulated\":" << (r.config.is_emulated ? "true" : "false") << ",";
        
        // Times array
        oss << "\"times\":[";
        for (size_t i = 0; i < r.result.times_sec.size(); ++i) {
            if (i > 0) oss << ",";
            oss << r.result.times_sec[i];
        }
        oss << "],";
        
        oss << "\"time_avg\":" << r.result.time_avg_sec << ",";
        oss << "\"time_min\":" << r.result.time_min_sec << ",";
        oss << "\"time_stddev\":" << r.result.time_stddev_sec << ",";
        oss << "\"gflops_avg\":" << r.result.gflops_avg << ",";
        oss << "\"gflops_max\":" << r.result.gflops_max << ",";
        oss << "\"mlups_avg\":" << r.result.mlups_avg << ",";
        oss << "\"total_flops\":" << r.result.total_flops << ",";
        oss << "\"iterations\":" << r.result.iterations;
        oss << "}";
        
        return oss.str();
    }
};

// Helper function to create PrecisionResult from benchmark run
inline PrecisionResult make_precision_result(
    Precision precision,
    FP16Mode fp16_mode,
    const BenchmarkResult& result,
    BenchmarkMode mode
) {
    PrecisionConfig config = get_precision_config(precision, fp16_mode);
    return PrecisionResult(precision, fp16_mode, result, config, mode);
}

// Format JSON with full system information including SIMD capabilities (Requirement 2.4, 4.3, 10.4)
inline std::string format_json_with_system_info(
    const std::vector<PrecisionResult>& results,
    const CpuInfo& cpu_info
) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(9);
    
    const CpuCapabilities& caps = CpuCapabilities::get();
    
    oss << "{\n";
    
    // CPU info with cache (Requirement 10.4)
    oss << "  \"cpu\": {\n";
    oss << "    \"arch\": \"" << cpu_info.arch << "\",\n";
    oss << "    \"logical_cores\": " << cpu_info.logical_cores << ",\n";
    oss << "    \"physical_cores\": " << cpu_info.physical_cores << ",\n";
    oss << "    \"vendor\": \"" << cpu_info.vendor << "\",\n";
    oss << "    \"model\": \"" << cpu_info.model << "\",\n";
    
    // Cache information (Requirement 10.4)
    oss << "    \"cache\": {\n";
    oss << "      \"l1_data_kb\": " << (cpu_info.cache.l1_available ? std::to_string(cpu_info.cache.l1_data_size / 1024) : "null") << ",\n";
    oss << "      \"l1_inst_kb\": " << (cpu_info.cache.l1_available ? std::to_string(cpu_info.cache.l1_inst_size / 1024) : "null") << ",\n";
    oss << "      \"l2_kb\": " << (cpu_info.cache.l2_available ? std::to_string(cpu_info.cache.l2_size / 1024) : "null") << ",\n";
    oss << "      \"l3_kb\": " << (cpu_info.cache.l3_available ? std::to_string(cpu_info.cache.l3_size / 1024) : "null") << ",\n";
    oss << "      \"line_size\": " << cpu_info.cache.cache_line_size << "\n";
    oss << "    },\n";
    
    // SIMD capabilities (Requirement 2.4)
    oss << "    \"simd_capabilities\": {\n";
    oss << "      \"sse2\": " << (caps.has_sse2 ? "true" : "false") << ",\n";
    oss << "      \"sse4_2\": " << (caps.has_sse4_2 ? "true" : "false") << ",\n";
    oss << "      \"avx\": " << (caps.has_avx ? "true" : "false") << ",\n";
    oss << "      \"avx2\": " << (caps.has_avx2 ? "true" : "false") << ",\n";
    oss << "      \"avx512f\": " << (caps.has_avx512f ? "true" : "false") << ",\n";
    oss << "      \"avx512_fp16\": " << (caps.has_avx512_fp16 ? "true" : "false") << ",\n";
    oss << "      \"avx512_vnni\": " << (caps.has_avx512_vnni ? "true" : "false") << ",\n";
    oss << "      \"arm_neon\": " << (caps.has_arm_neon ? "true" : "false") << ",\n";
    oss << "      \"arm_neon_fp16\": " << (caps.has_arm_neon_fp16 ? "true" : "false") << "\n";
    oss << "    }\n";
    oss << "  },\n";
    
    // Benchmark SIMD info (Requirement 4.3, 11.2)
    oss << "  \"benchmark\": {\n";
    oss << "    \"simd_level\": \"" << simd_level_to_string(caps.get_simd_level()) << "\",\n";
    
    // FP16 mode (Requirement 4.3)
    std::string fp16_mode;
    if (caps.has_avx512_fp16) {
        fp16_mode = "native (AVX-512 FP16)";
    } else if (caps.has_arm_neon_fp16) {
        fp16_mode = "native (ARM NEON FP16)";
    } else {
        fp16_mode = "emulated";
    }
    oss << "    \"fp16_mode\": \"" << fp16_mode << "\",\n";
    
    // Warmup status (Requirement 8.5)
    oss << "    \"warmup_performed\": " << (WarmupManager::was_warmup_performed() ? "true" : "false") << "\n";
    oss << "  },\n";
    
    // Results array
    oss << "  \"results\": [";
    for (size_t i = 0; i < results.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "\n    ";
        
        const auto& r = results[i];
        oss << "{";
        oss << "\"precision\":\"" << precision_to_string(r.precision) << "\",";
        
        if (r.precision == Precision::FP16) {
            oss << "\"fp16_mode\":\"" << (r.fp16_mode == FP16Mode::Native ? "native" : "emulated") << "\",";
        }
        
        oss << "\"bytes_per_element\":" << r.config.bytes_per_element << ",";
        oss << "\"is_integer\":" << (r.config.is_integer ? "true" : "false") << ",";
        oss << "\"is_emulated\":" << (r.config.is_emulated ? "true" : "false") << ",";
        oss << "\"time_avg\":" << r.result.time_avg_sec << ",";
        oss << "\"time_min\":" << r.result.time_min_sec << ",";
        oss << "\"time_stddev\":" << r.result.time_stddev_sec << ",";
        oss << "\"gflops_avg\":" << r.result.gflops_avg << ",";
        oss << "\"gflops_max\":" << r.result.gflops_max << ",";
        oss << "\"mlups_avg\":" << r.result.mlups_avg << ",";
        oss << "\"total_flops\":" << r.result.total_flops << ",";
        oss << "\"iterations\":" << r.result.iterations;
        oss << "}";
    }
    oss << "\n  ],\n";
    
    // Score section (Requirements 9.1, 9.2, 9.3, 9.4)
    // Calculate score from results
    ScoreCalculator calculator;
    for (const auto& r : results) {
        TestResultEntry entry;
        entry.name = precision_to_string(r.precision);
        entry.gflops = r.result.gflops_avg;
        entry.bandwidth_gbps = r.result.bandwidth_gbs;
        entry.is_single_thread = true;  // Default to ST for comparison mode
        
        // Determine test type based on mode
        switch (r.mode) {
            case BenchmarkMode::Mem:
            case BenchmarkMode::CacheLevel:
                entry.type = TestResultEntry::TestType::Memory;
                break;
            case BenchmarkMode::Stencil:
                entry.type = TestResultEntry::TestType::Mixed;
                break;
            case BenchmarkMode::Matmul3D:
            case BenchmarkMode::Compute:
                entry.type = TestResultEntry::TestType::Compute;
                break;
        }
        calculator.add_result(entry);
    }
    
    BenchmarkScore score = calculator.calculate_score();
    
    oss << "  \"score\": {\n";
    oss << "    \"overall\": " << score.overall_score << ",\n";
    oss << "    \"single_thread\": " << score.single_thread_score << ",\n";
    oss << "    \"multi_thread\": " << score.multi_thread_score << ",\n";
    oss << "    \"compute\": " << score.compute_score << ",\n";
    oss << "    \"memory\": " << score.memory_score << ",\n";
    oss << "    \"vs_reference_percent\": " << std::fixed << std::setprecision(2) << score.vs_reference_percent << ",\n";
    oss << "    \"reference_score\": " << ScoreCalculator::REFERENCE_SCORE << "\n";
    oss << "  },\n";
    
    // Reference comparison section (Requirements 11.1, 11.2, 11.3, 11.4)
    ReferenceComparer comparer;
    ReferenceComparison comparison = comparer.compare_to_baseline(score.overall_score);
    
    oss << "  \"reference_comparison\": {\n";
    oss << "    \"available\": " << (comparison.available ? "true" : "false") << ",\n";
    if (comparison.available) {
        oss << "    \"reference_name\": \"" << comparison.reference_name << "\",\n";
        oss << "    \"reference_score\": " << comparison.reference_score << ",\n";
        oss << "    \"user_score\": " << comparison.user_score << ",\n";
        oss << "    \"percentage_difference\": " << std::fixed << std::setprecision(2) << comparison.percentage_difference << ",\n";
        oss << "    \"status\": \"" << ReferenceComparer::status_to_string(comparison.status) << "\",\n";
        oss << "    \"message\": \"" << comparison.formatted_message << "\"\n";
    } else {
        oss << "    \"message\": \"Reference comparison unavailable\"\n";
    }
    oss << "  }\n";
    
    oss << "}";
    
    return oss.str();
}
