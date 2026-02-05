// CPU Benchmark - Progress Reporter Implementation


#include "progress_reporter.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

// Constructor
ProgressReporter::ProgressReporter(ProgressStyle style)
    : style_(style)
    , quiet_(false)
    , current_phase_("")
    , total_steps_(0)
    , current_step_(0)
    , callback_(nullptr)
{
}

// Destructor
ProgressReporter::~ProgressReporter() {
    // Ensure we finish any ongoing phase
    if (total_steps_ > 0 && current_step_ < total_steps_) {
        finish_phase();
    }
}

// Start a new phase 
void ProgressReporter::start_phase(const std::string& name, size_t total_steps) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    current_phase_ = name;
    total_steps_ = total_steps;
    current_step_ = 0;
    start_time_ = std::chrono::steady_clock::now();
    last_update_time_ = start_time_;
    last_render_time_ = start_time_;
    
    // Initial render
    if (!quiet_ && style_ != ProgressStyle::None) {
        std::cout << current_phase_ << ": ";
        std::cout.flush();
    }
}

// Update progress
void ProgressReporter::update(size_t current_step) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    current_step_ = current_step;
    auto now = std::chrono::steady_clock::now();
    
    // Calculate percentage and ETA
    double percentage = (total_steps_ > 0) 
        ? (static_cast<double>(current_step_) / static_cast<double>(total_steps_)) * 100.0 
        : 0.0;
    
    double elapsed = std::chrono::duration<double>(now - start_time_).count();
    double eta = 0.0;
    if (current_step_ > 0 && current_step_ < total_steps_) {
        double rate = static_cast<double>(current_step_) / elapsed;
        size_t remaining = total_steps_ - current_step_;
        eta = static_cast<double>(remaining) / rate;
    }
    
    // Call callback if set
    if (callback_) {
        callback_(current_step_, total_steps_, percentage, eta);
    }
    
    // Check if we should render 
    double since_last_render = std::chrono::duration<double>(now - last_render_time_).count();
    if (since_last_render >= MIN_UPDATE_INTERVAL_SEC || current_step_ == total_steps_) {
        last_render_time_ = now;
        last_update_time_ = now;
        
        if (!quiet_ && style_ != ProgressStyle::None) {
            render();
        }
    }
}

// Finish current phase
void ProgressReporter::finish_phase() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    current_step_ = total_steps_;
    
    if (!quiet_ && style_ != ProgressStyle::None) {
        // Final render
        render();
        std::cout << std::endl;
    }
    
    // Reset state
    current_phase_ = "";
    total_steps_ = 0;
    current_step_ = 0;
}

// Set quiet mode 
void ProgressReporter::set_quiet(bool quiet) {
    std::lock_guard<std::mutex> lock(mutex_);
    quiet_ = quiet;
}

// Check if quiet mode is enabled
bool ProgressReporter::is_quiet() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return quiet_;
}

// Set callback
void ProgressReporter::set_callback(ProgressCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    callback_ = callback;
}

// Get current style
ProgressStyle ProgressReporter::get_style() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return style_;
}

// Set style
void ProgressReporter::set_style(ProgressStyle style) {
    std::lock_guard<std::mutex> lock(mutex_);
    style_ = style;
}

// Get current phase name
std::string ProgressReporter::get_current_phase() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_phase_;
}

// Get current percentage
double ProgressReporter::get_percentage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (total_steps_ == 0) return 0.0;
    return (static_cast<double>(current_step_) / static_cast<double>(total_steps_)) * 100.0;
}

// Get ETA in seconds
double ProgressReporter::get_eta_seconds() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (current_step_ == 0 || current_step_ >= total_steps_) {
        return 0.0;
    }
    
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time_).count();
    double rate = static_cast<double>(current_step_) / elapsed;
    size_t remaining = total_steps_ - current_step_;
    
    return static_cast<double>(remaining) / rate;
}

// Get elapsed time in seconds
double ProgressReporter::get_elapsed_seconds() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_time_).count();
}

// Get last update time
std::chrono::steady_clock::time_point ProgressReporter::get_last_update_time() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_update_time_;
}

// Format ETA as MM:SS or HH:MM:SS
std::string ProgressReporter::format_eta(double seconds) {
    if (seconds < 0 || !std::isfinite(seconds)) {
        return "--:--";
    }
    
    int total_seconds = static_cast<int>(std::round(seconds));
    int hours = total_seconds / 3600;
    int minutes = (total_seconds % 3600) / 60;
    int secs = total_seconds % 60;
    
    std::ostringstream oss;
    if (hours > 0) {
        oss << std::setfill('0') << std::setw(2) << hours << ":"
            << std::setfill('0') << std::setw(2) << minutes << ":"
            << std::setfill('0') << std::setw(2) << secs;
    } else {
        oss << std::setfill('0') << std::setw(2) << minutes << ":"
            << std::setfill('0') << std::setw(2) << secs;
    }
    
    return oss.str();
}

// Format progress string based on style
std::string ProgressReporter::format_progress() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    double percentage = (total_steps_ > 0) 
        ? (static_cast<double>(current_step_) / static_cast<double>(total_steps_)) * 100.0 
        : 0.0;
    
    std::ostringstream oss;
    
    switch (style_) {
        case ProgressStyle::None:
            break;
            
        case ProgressStyle::Simple:
            oss << std::fixed << std::setprecision(0) << percentage << "%";
            break;
            
        case ProgressStyle::Bar: {
            int filled = static_cast<int>((percentage / 100.0) * BAR_WIDTH);
            oss << "[";
            for (int i = 0; i < BAR_WIDTH; ++i) {
                if (i < filled) {
                    oss << "=";
                } else if (i == filled) {
                    oss << ">";
                } else {
                    oss << " ";
                }
            }
            oss << "] " << std::fixed << std::setprecision(0) << percentage << "%";
            break;
        }
            
        case ProgressStyle::Detailed: {
            int filled = static_cast<int>((percentage / 100.0) * BAR_WIDTH);
            oss << "[";
            for (int i = 0; i < BAR_WIDTH; ++i) {
                if (i < filled) {
                    oss << "=";
                } else if (i == filled) {
                    oss << ">";
                } else {
                    oss << " ";
                }
            }
            oss << "] " << std::fixed << std::setprecision(0) << percentage << "%";
            
            // Add ETA
            double eta = 0.0;
            if (current_step_ > 0 && current_step_ < total_steps_) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time_).count();
                double rate = static_cast<double>(current_step_) / elapsed;
                size_t remaining = total_steps_ - current_step_;
                eta = static_cast<double>(remaining) / rate;
            }
            oss << " ETA: " << format_eta(eta);
            break;
        }
    }
    
    return oss.str();
}

// Render progress to console
void ProgressReporter::render() {
    // Note: mutex is already held by caller
    
    double percentage = (total_steps_ > 0) 
        ? (static_cast<double>(current_step_) / static_cast<double>(total_steps_)) * 100.0 
        : 0.0;
    
    // Clear line and return to beginning
    std::cout << "\r";
    
    // Print phase name
    std::cout << current_phase_ << ": ";
    
    switch (style_) {
        case ProgressStyle::None:
            break;
            
        case ProgressStyle::Simple:
            std::cout << std::fixed << std::setprecision(0) << percentage << "%";
            break;
            
        case ProgressStyle::Bar: {
            int filled = static_cast<int>((percentage / 100.0) * BAR_WIDTH);
            std::cout << "[";
            for (int i = 0; i < BAR_WIDTH; ++i) {
                if (i < filled) {
                    std::cout << "=";
                } else if (i == filled) {
                    std::cout << ">";
                } else {
                    std::cout << " ";
                }
            }
            std::cout << "] " << std::fixed << std::setprecision(0) << percentage << "%";
            break;
        }
            
        case ProgressStyle::Detailed: {
            int filled = static_cast<int>((percentage / 100.0) * BAR_WIDTH);
            std::cout << "[";
            for (int i = 0; i < BAR_WIDTH; ++i) {
                if (i < filled) {
                    std::cout << "=";
                } else if (i == filled) {
                    std::cout << ">";
                } else {
                    std::cout << " ";
                }
            }
            std::cout << "] " << std::fixed << std::setprecision(0) << percentage << "%";
            
            // Add ETA
            double eta = 0.0;
            if (current_step_ > 0 && current_step_ < total_steps_) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time_).count();
                double rate = static_cast<double>(current_step_) / elapsed;
                size_t remaining = total_steps_ - current_step_;
                eta = static_cast<double>(remaining) / rate;
            }
            std::cout << " ETA: " << format_eta(eta);
            break;
        }
    }
    
    // Pad with spaces to clear any previous longer output
    std::cout << "          ";
    std::cout.flush();
}

// Clear current line
void ProgressReporter::clear_line() {
    std::cout << "\r" << std::string(80, ' ') << "\r";
    std::cout.flush();
}

