#pragma once
// CPU Benchmark - Progress Reporter


#include <string>
#include <chrono>
#include <functional>
#include <mutex>
#include <atomic>

// Progress update callback type
// Parameters: current_step, total_steps, percentage, eta_seconds
using ProgressCallback = std::function<void(size_t, size_t, double, double)>;

// Progress reporter styles
enum class ProgressStyle {
    None,       // No output
    Simple,     // Only percentages: "50%"
    Bar,        // Progress bar: [=====>    ] 50%
    Detailed    // With ETA and speed: [=====>    ] 50% ETA: 00:30
};

// Convert style to string
inline std::string progress_style_to_string(ProgressStyle style) {
    switch (style) {
        case ProgressStyle::None: return "none";
        case ProgressStyle::Simple: return "simple";
        case ProgressStyle::Bar: return "bar";
        case ProgressStyle::Detailed: return "detailed";
        default: return "unknown";
    }
}

// Convert string to style
inline ProgressStyle string_to_progress_style(const std::string& s) {
    if (s == "none") return ProgressStyle::None;
    if (s == "simple") return ProgressStyle::Simple;
    if (s == "bar") return ProgressStyle::Bar;
    if (s == "detailed") return ProgressStyle::Detailed;
    return ProgressStyle::Bar;  // Default
}

// Progress Reporter class 
class ProgressReporter {
public:
    // Constructor with style selection
    explicit ProgressReporter(ProgressStyle style = ProgressStyle::Bar);
    
    // Destructor
    ~ProgressReporter();
    
    // Start a new phase with name and total steps
    void start_phase(const std::string& name, size_t total_steps);
    
    // Update progress to current step
    void update(size_t current_step);
    
    // Finish current phase
    void finish_phase();
    
    // Set quiet mode 
    void set_quiet(bool quiet);
    
    // Check if quiet mode is enabled
    bool is_quiet() const;
    
    // Set progress callback for external consumers
    void set_callback(ProgressCallback callback);
    
    // Get current style
    ProgressStyle get_style() const;
    
    // Set style
    void set_style(ProgressStyle style);
    
    // Get current phase name
    std::string get_current_phase() const;
    
    // Get current percentage (0-100)
    double get_percentage() const;
    
    // Get estimated time remaining in seconds
    double get_eta_seconds() const;
    
    // Get elapsed time in seconds
    double get_elapsed_seconds() const;
    
    // Get last update time point
    std::chrono::steady_clock::time_point get_last_update_time() const;
    
    // Get minimum update interval in seconds 
    static constexpr double MIN_UPDATE_INTERVAL_SEC = 2.0;
    
    // Format progress string based on style
    std::string format_progress() const;
    
    // Format ETA as HH:MM:SS or MM:SS
    static std::string format_eta(double seconds);
    
private:
    // Render progress to console
    void render();
    
    // Check if enough time has passed since last update
    bool should_update() const;
    
    // Clear current line in console
    void clear_line();
    
    ProgressStyle style_;
    bool quiet_;
    std::string current_phase_;
    size_t total_steps_;
    size_t current_step_;
    
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point last_update_time_;
    std::chrono::steady_clock::time_point last_render_time_;
    
    ProgressCallback callback_;
    mutable std::mutex mutex_;
    
    // Progress bar configuration
    static constexpr int BAR_WIDTH = 30;
};

