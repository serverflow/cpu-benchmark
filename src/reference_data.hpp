#pragma once
// CPU Benchmark - Reference Data for CPU Comparison

// **Feature: production-ready, Properties 21-22**

#include <string>
#include <vector>
#include <optional>
#include <cmath>
#include <algorithm>

// Reference CPU data entry
struct ReferenceCpuData {
    std::string name;           // CPU name (e.g., "Intel Core i5-10400")
    std::string vendor;         // Vendor (Intel, AMD, Apple)
    int cores;                  // Physical cores
    int threads;                // Logical threads
    int score;                  // Reference benchmark score
    double gflops_st;           // Single-thread GFLOPS
    double gflops_mt;           // Multi-thread GFLOPS
    double bandwidth_gbps;      // Memory bandwidth GB/s
    
    ReferenceCpuData() 
        : cores(0), threads(0), score(0)
        , gflops_st(0.0), gflops_mt(0.0), bandwidth_gbps(0.0) {}
    
    ReferenceCpuData(const std::string& n, const std::string& v, 
                     int c, int t, int s, double gst, double gmt, double bw)
        : name(n), vendor(v), cores(c), threads(t), score(s)
        , gflops_st(gst), gflops_mt(gmt), bandwidth_gbps(bw) {}
};

// Comparison status indicator
//  Indicate whether performance is above or below reference
enum class ComparisonStatus {
    Above,      // Score is above reference
    Below,      // Score is below reference
    Equal       // Score equals reference (within tolerance)
};

// Reference comparison result

struct ReferenceComparison {
    bool available = false;                     // Whether comparison is available
    std::string reference_name;                 // Name of reference CPU
    int reference_score = 0;                    // Reference score
    int user_score = 0;                         // User's score
    double percentage_difference = 0.0;         // Percentage difference from reference
    ComparisonStatus status = ComparisonStatus::Equal;  // Above/below/equal indicator
    std::string formatted_message;              // Human-readable comparison message
    
    ReferenceComparison() = default;
};

// Reference data database with embedded CPU data

class ReferenceDatabase {
public:
    // Get singleton instance
    static ReferenceDatabase& instance() {
        static ReferenceDatabase db;
        return db;
    }
    
    // Get the baseline reference CPU (Ryzen 5 5600 = 1000)
    const ReferenceCpuData& get_baseline() const {
        return baseline_;
    }
    
    // Get all reference CPUs
    const std::vector<ReferenceCpuData>& get_all_references() const {
        return references_;
    }
    
    // Find reference by name (partial match)
    std::optional<ReferenceCpuData> find_by_name(const std::string& name) const {
        std::string lower_name = to_lower(name);
        
        for (const auto& ref : references_) {
            if (to_lower(ref.name).find(lower_name) != std::string::npos) {
                return ref;
            }
        }
        return std::nullopt;
    }
    
    // Find closest reference by score
    std::optional<ReferenceCpuData> find_closest_by_score(int score) const {
        if (references_.empty()) {
            return std::nullopt;
        }
        
        auto closest = std::min_element(references_.begin(), references_.end(),
            [score](const ReferenceCpuData& a, const ReferenceCpuData& b) {
                return std::abs(a.score - score) < std::abs(b.score - score);
            });
        
        return *closest;
    }
    
    // Check if database has any references
    bool has_references() const {
        return !references_.empty();
    }
    
    // Get reference count
    size_t reference_count() const {
        return references_.size();
    }

private:
    ReferenceCpuData baseline_;
    std::vector<ReferenceCpuData> references_;
    
    ReferenceDatabase() {
        initialize_references();
    }
    
    // Initialize embedded reference data
    // Add references for popular CPUs
    void initialize_references() {
        // Baseline reference: AMD Ryzen 5 5600 = 1000
        // ST: 1667 GFLOPS, MT: 1697 GFLOPS (normalized to 1000 score)
        baseline_ = ReferenceCpuData(
            "AMD Ryzen 5 5600", "AMD",
            6, 12, 1000,
            83.35, 141.42, 40.0  // ST GFLOPS for 1000 score, MT GFLOPS for 1000 score
        );
        
        // Intel Desktop CPUs
        references_.push_back(baseline_);
        
        references_.push_back(ReferenceCpuData(
            "Intel Core i3-10100", "Intel",
            4, 8, 720,
            45.0, 160.0, 35.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Intel Core i5-12400", "Intel",
            6, 12, 1350,
            65.0, 340.0, 45.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Intel Core i5-13400", "Intel",
            10, 16, 1550,
            70.0, 420.0, 50.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Intel Core i7-10700", "Intel",
            8, 16, 1280,
            52.0, 380.0, 38.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Intel Core i7-12700", "Intel",
            12, 20, 1850,
            75.0, 580.0, 55.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Intel Core i7-13700", "Intel",
            16, 24, 2100,
            80.0, 720.0, 60.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Intel Core i9-10900K", "Intel",
            10, 20, 1650,
            58.0, 520.0, 42.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Intel Core i9-12900K", "Intel",
            16, 24, 2350,
            85.0, 850.0, 65.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Intel Core i9-13900K", "Intel",
            24, 32, 2800,
            90.0, 1100.0, 70.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Intel Core i9-14900K", "Intel",
            24, 32, 3000,
            95.0, 1200.0, 75.0
        ));
        
        // AMD Desktop CPUs
        references_.push_back(ReferenceCpuData(
            "AMD Ryzen 5 3600", "AMD",
            6, 12, 950,
            48.0, 240.0, 40.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "AMD Ryzen 5 5600X", "AMD",
            6, 12, 1250,
            62.0, 320.0, 45.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "AMD Ryzen 5 7600X", "AMD",
            6, 12, 1500,
            72.0, 380.0, 55.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "AMD Ryzen 7 3700X", "AMD",
            8, 16, 1150,
            50.0, 350.0, 42.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "AMD Ryzen 7 5800X", "AMD",
            8, 16, 1450,
            65.0, 450.0, 48.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "AMD Ryzen 7 7800X3D", "AMD",
            8, 16, 1700,
            75.0, 520.0, 55.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "AMD Ryzen 9 5900X", "AMD",
            12, 24, 1750,
            68.0, 650.0, 50.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "AMD Ryzen 9 5950X", "AMD",
            16, 32, 2000,
            70.0, 850.0, 52.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "AMD Ryzen 9 7900X", "AMD",
            12, 24, 2200,
            82.0, 800.0, 65.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "AMD Ryzen 9 7950X", "AMD",
            16, 32, 2650,
            85.0, 1050.0, 70.0
        ));
        
        // Apple Silicon
        references_.push_back(ReferenceCpuData(
            "Apple M1", "Apple",
            8, 8, 1100,
            55.0, 280.0, 60.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Apple M1 Pro", "Apple",
            10, 10, 1450,
            60.0, 420.0, 180.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Apple M1 Max", "Apple",
            10, 10, 1550,
            62.0, 480.0, 380.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Apple M2", "Apple",
            8, 8, 1250,
            62.0, 320.0, 90.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Apple M2 Pro", "Apple",
            12, 12, 1650,
            68.0, 550.0, 180.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Apple M2 Max", "Apple",
            12, 12, 1800,
            70.0, 650.0, 380.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Apple M3", "Apple",
            8, 8, 1400,
            70.0, 380.0, 100.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Apple M3 Pro", "Apple",
            12, 12, 1850,
            75.0, 650.0, 150.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Apple M3 Max", "Apple",
            16, 16, 2200,
            80.0, 900.0, 400.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Apple M4", "Apple",
            10, 10, 1600,
            78.0, 480.0, 120.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Apple M4 Pro", "Apple",
            14, 14, 2100,
            85.0, 800.0, 270.0
        ));
        
        // Server/Workstation CPUs
        references_.push_back(ReferenceCpuData(
            "Intel Xeon W-2255", "Intel",
            10, 20, 1400,
            55.0, 480.0, 90.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "AMD EPYC 7302", "AMD",
            16, 32, 1600,
            45.0, 600.0, 140.0
        ));
        
        // Laptop CPUs
        references_.push_back(ReferenceCpuData(
            "Intel Core i5-1135G7", "Intel",
            4, 8, 850,
            48.0, 180.0, 35.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Intel Core i7-1165G7", "Intel",
            4, 8, 950,
            52.0, 200.0, 38.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Intel Core i5-12500H", "Intel",
            12, 16, 1400,
            65.0, 420.0, 45.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "Intel Core i7-12700H", "Intel",
            14, 20, 1650,
            72.0, 550.0, 50.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "AMD Ryzen 5 5600H", "AMD",
            6, 12, 1100,
            55.0, 280.0, 42.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "AMD Ryzen 7 5800H", "AMD",
            8, 16, 1300,
            60.0, 400.0, 45.0
        ));
        
        references_.push_back(ReferenceCpuData(
            "AMD Ryzen 7 6800H", "AMD",
            8, 16, 1450,
            68.0, 450.0, 50.0
        ));
    }
    
    // Convert string to lowercase for case-insensitive comparison
    static std::string to_lower(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return result;
    }
};

