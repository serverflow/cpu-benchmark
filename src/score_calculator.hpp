#pragma once
// CPU Benchmark - Score Calculator
// **Feature: production-ready, Properties 14-17**

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>

// Test result for score calculation
struct TestResultEntry {
    std::string name;
    double gflops = 0.0;          // For compute tests
    double bandwidth_gbps = 0.0;  // For memory tests
    double latency_ns = 0.0;      // For latency tests
    bool is_single_thread = true; // ST vs MT test
    
    enum class TestType {
        Compute,
        Memory,
        Mixed
    };
    TestType type = TestType::Mixed;
};

// Benchmark score result
struct BenchmarkScore {
    int single_thread_score = 0;
    int multi_thread_score = 0;
    int compute_score = 0;
    int memory_score = 0;
    int overall_score = 0;
    
    // Comparison with reference
    double vs_reference_percent = 0.0;
    
    // Validity flags
    bool has_single_thread = false;
    bool has_multi_thread = false;
};

// Score Calculator class

class ScoreCalculator {
public:
    // Reference score (AMD Ryzen 5 5600 = 1000)
    
    static constexpr int REFERENCE_SCORE = 1000;
    
    // Weights for different test types

    static constexpr double COMPUTE_WEIGHT = 0.5;
    static constexpr double MEMORY_WEIGHT = 0.3;
    static constexpr double MIXED_WEIGHT = 0.2;
    
    // Reference values for AMD Ryzen 5 5600 (baseline system)
    // ST: 1667 GFLOPS = 1000 score, MT: 1697 GFLOPS = 1000 score
    static constexpr double REFERENCE_GFLOPS_ST = 83.35;   // Single-thread GFLOPS (1667/20 for score 1000)
    static constexpr double REFERENCE_GFLOPS_MT = 141.42;  // Multi-thread GFLOPS (1697/12 for score 1000)
    static constexpr double REFERENCE_BANDWIDTH = 40.0;    // GB/s memory bandwidth
    
    ScoreCalculator() = default;
    
    // Add a test result
    void add_result(const TestResultEntry& result) {
        results_.push_back(result);
    }
    
    // Add result with simplified parameters
    void add_result(const std::string& name, double gflops, double bandwidth_gbps,
                    bool is_single_thread, TestResultEntry::TestType type) {
        TestResultEntry entry;
        entry.name = name;
        entry.gflops = gflops;
        entry.bandwidth_gbps = bandwidth_gbps;
        entry.is_single_thread = is_single_thread;
        entry.type = type;
        results_.push_back(entry);
    }
    
    // Clear all results
    void clear() {
        results_.clear();
    }
    
    // Get number of results
    size_t result_count() const {
        return results_.size();
    }
    
    // Calculate the benchmark score
 
    BenchmarkScore calculate_score() const {
        BenchmarkScore score;
        
        if (results_.empty()) {
            return score;
        }
        
        // Separate results by thread type and test type
        std::vector<double> st_compute_scores;
        std::vector<double> st_memory_scores;
        std::vector<double> st_mixed_scores;
        std::vector<double> mt_compute_scores;
        std::vector<double> mt_memory_scores;
        std::vector<double> mt_mixed_scores;
        
        for (const auto& result : results_) {
            double normalized_score = normalize_result(result);
            
            if (result.is_single_thread) {
                score.has_single_thread = true;
                switch (result.type) {
                    case TestResultEntry::TestType::Compute:
                        st_compute_scores.push_back(normalized_score);
                        break;
                    case TestResultEntry::TestType::Memory:
                        st_memory_scores.push_back(normalized_score);
                        break;
                    case TestResultEntry::TestType::Mixed:
                        st_mixed_scores.push_back(normalized_score);
                        break;
                }
            } else {
                score.has_multi_thread = true;
                switch (result.type) {
                    case TestResultEntry::TestType::Compute:
                        mt_compute_scores.push_back(normalized_score);
                        break;
                    case TestResultEntry::TestType::Memory:
                        mt_memory_scores.push_back(normalized_score);
                        break;
                    case TestResultEntry::TestType::Mixed:
                        mt_mixed_scores.push_back(normalized_score);
                        break;
                }
            }
        }
        
        // Calculate component scores (average of each category)
        double st_compute = average(st_compute_scores);
        double st_memory = average(st_memory_scores);
        double st_mixed = average(st_mixed_scores);
        
        double mt_compute = average(mt_compute_scores);
        double mt_memory = average(mt_memory_scores);
        double mt_mixed = average(mt_mixed_scores);
        
        // Calculate weighted single-thread score
        if (score.has_single_thread) {
            double st_weighted = calculate_weighted_score(st_compute, st_memory, st_mixed);
            score.single_thread_score = static_cast<int>(std::round(st_weighted));
        }
        
        // Calculate weighted multi-thread score
        if (score.has_multi_thread) {
            double mt_weighted = calculate_weighted_score(mt_compute, mt_memory, mt_mixed);
            score.multi_thread_score = static_cast<int>(std::round(mt_weighted));
        }
        
        // Calculate overall compute and memory scores
        std::vector<double> all_compute;
        all_compute.insert(all_compute.end(), st_compute_scores.begin(), st_compute_scores.end());
        all_compute.insert(all_compute.end(), mt_compute_scores.begin(), mt_compute_scores.end());
        score.compute_score = static_cast<int>(std::round(average(all_compute)));
        
        std::vector<double> all_memory;
        all_memory.insert(all_memory.end(), st_memory_scores.begin(), st_memory_scores.end());
        all_memory.insert(all_memory.end(), mt_memory_scores.begin(), mt_memory_scores.end());
        score.memory_score = static_cast<int>(std::round(average(all_memory)));
        
        // Calculate overall score (average of ST and MT if both present)
        if (score.has_single_thread && score.has_multi_thread) {
            score.overall_score = (score.single_thread_score + score.multi_thread_score) / 2;
        } else if (score.has_single_thread) {
            score.overall_score = score.single_thread_score;
        } else if (score.has_multi_thread) {
            score.overall_score = score.multi_thread_score;
        }
        
        // Calculate percentage vs reference
       
        if (score.overall_score > 0) {
            score.vs_reference_percent = 
                ((static_cast<double>(score.overall_score) - REFERENCE_SCORE) / REFERENCE_SCORE) * 100.0;
        }
        
        return score;
    }
    
    // Format score for display
    static std::string format_score(const BenchmarkScore& score) {
        std::string result;
        result += "Overall Score: " + std::to_string(score.overall_score) + "\n";
        
        if (score.has_single_thread) {
            result += "Single-Thread: " + std::to_string(score.single_thread_score) + "\n";
        }
        if (score.has_multi_thread) {
            result += "Multi-Thread:  " + std::to_string(score.multi_thread_score) + "\n";
        }
        
        result += "Compute Score: " + std::to_string(score.compute_score) + "\n";
        result += "Memory Score:  " + std::to_string(score.memory_score) + "\n";
        
        // Reference comparison
        if (score.vs_reference_percent >= 0) {
            result += "vs Reference:  +" + format_percent(score.vs_reference_percent) + "% faster\n";
        } else {
            result += "vs Reference:  " + format_percent(score.vs_reference_percent) + "% slower\n";
        }
        
        return result;
    }
    
    // Get the results
    const std::vector<TestResultEntry>& results() const {
        return results_;
    }

private:
    std::vector<TestResultEntry> results_;
    
    // Normalize a result to the reference scale (1000 = reference)
    double normalize_result(const TestResultEntry& result) const {
        double reference_value = 0.0;
        double measured_value = 0.0;
        
        switch (result.type) {
            case TestResultEntry::TestType::Compute:
                reference_value = result.is_single_thread ? REFERENCE_GFLOPS_ST : REFERENCE_GFLOPS_MT;
                measured_value = result.gflops;
                break;
            case TestResultEntry::TestType::Memory:
                reference_value = REFERENCE_BANDWIDTH;
                measured_value = result.bandwidth_gbps;
                break;
            case TestResultEntry::TestType::Mixed:
                // For mixed tests, use GFLOPS as primary metric
                reference_value = result.is_single_thread ? REFERENCE_GFLOPS_ST : REFERENCE_GFLOPS_MT;
                measured_value = result.gflops;
                break;
        }
        
        if (reference_value <= 0.0) {
            return 0.0;
        }
        
        // Scale to reference score (1000 = reference system)
        return (measured_value / reference_value) * REFERENCE_SCORE;
    }
    
    // Calculate weighted score from component scores
    //  compute=0.5, memory=0.3, mixed=0.2
    double calculate_weighted_score(double compute, double memory, double mixed) const {
        // Count how many components have valid scores
        int valid_count = 0;
        double total_weight = 0.0;
        double weighted_sum = 0.0;
        
        if (compute > 0) {
            weighted_sum += compute * COMPUTE_WEIGHT;
            total_weight += COMPUTE_WEIGHT;
            valid_count++;
        }
        if (memory > 0) {
            weighted_sum += memory * MEMORY_WEIGHT;
            total_weight += MEMORY_WEIGHT;
            valid_count++;
        }
        if (mixed > 0) {
            weighted_sum += mixed * MIXED_WEIGHT;
            total_weight += MIXED_WEIGHT;
            valid_count++;
        }
        
        // If no valid scores, return 0
        if (valid_count == 0 || total_weight <= 0.0) {
            return 0.0;
        }
        
        // Normalize by actual weight used
        return weighted_sum / total_weight * (COMPUTE_WEIGHT + MEMORY_WEIGHT + MIXED_WEIGHT);
    }
    
    // Calculate average of a vector
    static double average(const std::vector<double>& values) {
        if (values.empty()) {
            return 0.0;
        }
        double sum = std::accumulate(values.begin(), values.end(), 0.0);
        return sum / static_cast<double>(values.size());
    }
    
    // Format percentage with 1 decimal place
    static std::string format_percent(double percent) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.1f", std::abs(percent));
        return std::string(buf);
    }
};
