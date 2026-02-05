#pragma once
// CPU Benchmark - Reference Comparison Logic
// **Feature: production-ready, Properties 21-22**

#include "reference_data.hpp"
#include "score_calculator.hpp"
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>

// Reference Comparison Calculator
class ReferenceComparer {
public:
    // Tolerance for "equal" comparison (within 1%)
    static constexpr double EQUAL_TOLERANCE_PERCENT = 1.0;
    
    ReferenceComparer() = default;
    
    // Calculate percentage difference from reference
  
    // Formula: ((score - reference) / reference) * 100
    static double calculate_percentage_difference(int user_score, int reference_score) {
        if (reference_score == 0) {
            return 0.0;
        }
        return (static_cast<double>(user_score - reference_score) / 
                static_cast<double>(reference_score)) * 100.0;
    }
    
    // Determine comparison status (above/below/equal)
  
    static ComparisonStatus determine_status(double percentage_difference) {
        if (percentage_difference > EQUAL_TOLERANCE_PERCENT) {
            return ComparisonStatus::Above;
        } else if (percentage_difference < -EQUAL_TOLERANCE_PERCENT) {
            return ComparisonStatus::Below;
        } else {
            return ComparisonStatus::Equal;
        }
    }
    
    // Compare user score against baseline reference

    ReferenceComparison compare_to_baseline(int user_score) const {
        const auto& baseline = ReferenceDatabase::instance().get_baseline();
        return compare_to_reference(user_score, baseline);
    }
    
    // Compare user score against specific reference CPU
    ReferenceComparison compare_to_reference(int user_score, 
                                              const ReferenceCpuData& reference) const {
        ReferenceComparison result;
        result.available = true;
        result.reference_name = reference.name;
        result.reference_score = reference.score;
        result.user_score = user_score;
        
        // Calculate percentage difference
   
        result.percentage_difference = calculate_percentage_difference(
            user_score, reference.score);
        
        // Determine status
       
        result.status = determine_status(result.percentage_difference);
        
        // Generate formatted message
        result.formatted_message = format_comparison_message(result);
        
        return result;
    }
    
    // Compare user score against closest reference CPU
    ReferenceComparison compare_to_closest(int user_score) const {
        auto closest = ReferenceDatabase::instance().find_closest_by_score(user_score);
        
        if (!closest.has_value()) {
            ReferenceComparison result;
            result.available = false;
            result.formatted_message = "No reference data available";
            return result;
        }
        
        return compare_to_reference(user_score, closest.value());
    }
    
    // Format comparison message for display
   
    static std::string format_comparison_message(const ReferenceComparison& comparison) {
        if (!comparison.available) {
            return "Reference comparison unavailable";
        }
        
        std::ostringstream oss;
        double abs_percent = std::abs(comparison.percentage_difference);
        
        switch (comparison.status) {
            case ComparisonStatus::Above:
                oss << std::fixed << std::setprecision(1) << abs_percent 
                    << "% faster than " << comparison.reference_name;
                break;
            case ComparisonStatus::Below:
                oss << std::fixed << std::setprecision(1) << abs_percent 
                    << "% slower than " << comparison.reference_name;
                break;
            case ComparisonStatus::Equal:
                oss << "Equal to " << comparison.reference_name;
                break;
        }
        
        return oss.str();
    }
    
    // Format short comparison (just percentage)
    static std::string format_short_comparison(const ReferenceComparison& comparison) {
        if (!comparison.available) {
            return "N/A";
        }
        
        std::ostringstream oss;
        double abs_percent = std::abs(comparison.percentage_difference);
        
        switch (comparison.status) {
            case ComparisonStatus::Above:
                oss << "+" << std::fixed << std::setprecision(1) << abs_percent << "%";
                break;
            case ComparisonStatus::Below:
                oss << "-" << std::fixed << std::setprecision(1) << abs_percent << "%";
                break;
            case ComparisonStatus::Equal:
                oss << "0%";
                break;
        }
        
        return oss.str();
    }
    
    // Get status as string
    static std::string status_to_string(ComparisonStatus status) {
        switch (status) {
            case ComparisonStatus::Above: return "above";
            case ComparisonStatus::Below: return "below";
            case ComparisonStatus::Equal: return "equal";
            default: return "unknown";
        }
    }
    
    // Create unavailable comparison result
   
    static ReferenceComparison create_unavailable() {
        ReferenceComparison result;
        result.available = false;
        result.formatted_message = "Reference comparison unavailable";
        return result;
    }
};

// Helper function to compare benchmark score to baseline
inline ReferenceComparison compare_score_to_baseline(const BenchmarkScore& score) {
    ReferenceComparer comparer;
    return comparer.compare_to_baseline(score.overall_score);
}

// Helper function to format comparison for console output
inline std::string format_reference_comparison_text(const ReferenceComparison& comparison,
                                                     bool use_colors = false) {
    if (!comparison.available) {
        return "";
    }
    
    std::ostringstream oss;
    
    // Color codes
    const char* green = use_colors ? "\033[32m" : "";
    const char* red = use_colors ? "\033[31m" : "";
    const char* reset = use_colors ? "\033[0m" : "";
    
    oss << "vs " << comparison.reference_name << ": ";
    
    switch (comparison.status) {
        case ComparisonStatus::Above:
            oss << green << "+" << std::fixed << std::setprecision(1) 
                << std::abs(comparison.percentage_difference) << "% faster" << reset;
            break;
        case ComparisonStatus::Below:
            oss << red << "-" << std::fixed << std::setprecision(1) 
                << std::abs(comparison.percentage_difference) << "% slower" << reset;
            break;
        case ComparisonStatus::Equal:
            oss << "Equal performance";
            break;
    }
    
    return oss.str();
}

// Helper function to format comparison for JSON output
inline std::string format_reference_comparison_json(const ReferenceComparison& comparison) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    
    oss << "{\n";
    oss << "    \"available\": " << (comparison.available ? "true" : "false") << ",\n";
    
    if (comparison.available) {
        oss << "    \"reference_name\": \"" << comparison.reference_name << "\",\n";
        oss << "    \"reference_score\": " << comparison.reference_score << ",\n";
        oss << "    \"user_score\": " << comparison.user_score << ",\n";
        oss << "    \"percentage_difference\": " << comparison.percentage_difference << ",\n";
        oss << "    \"status\": \"" << ReferenceComparer::status_to_string(comparison.status) << "\",\n";
        oss << "    \"message\": \"" << comparison.formatted_message << "\"\n";
    } else {
        oss << "    \"message\": \"Reference comparison unavailable\"\n";
    }
    
    oss << "  }";
    return oss.str();
}

