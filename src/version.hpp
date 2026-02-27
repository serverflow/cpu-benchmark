#pragma once
// CPU Benchmark - Version information


#include <string>
#include <sstream>

namespace version {

// Version numbers
constexpr int MAJOR = 0;
constexpr int MINOR = 42;
constexpr int PATCH = 0;

// Version string in format "beta X.Y"
constexpr const char* VERSION_STRING = "beta 0.44";

// Build date and time 
constexpr const char* BUILD_DATE = __DATE__;
constexpr const char* BUILD_TIME = __TIME__;

// Get full version string 
// Returns version in format "X.Y.Z"
inline std::string get_version_string() {
    return VERSION_STRING;
}

// Get compiler information for version output 
inline std::string get_compiler_info() {
#if defined(_MSC_VER)
    std::ostringstream oss;
    oss << "MSVC " << _MSC_VER;
    #if defined(_MSC_FULL_VER)
    oss << " (" << _MSC_FULL_VER << ")";
    #endif
    return oss.str();
#elif defined(__clang__)
    std::ostringstream oss;
    oss << "Clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
    return oss.str();
#elif defined(__GNUC__)
    std::ostringstream oss;
    oss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
    return oss.str();
#else
    return "Unknown compiler";
#endif
}

// Get build date string
inline std::string get_build_date() {
    return BUILD_DATE;
}

// Get build time string
inline std::string get_build_time() {
    return BUILD_TIME;
}

// Get full version string with all details 
// Format: "CPU Benchmark vX.Y.Z (built DATE TIME with COMPILER)"
inline std::string get_full_version_string() {
    std::ostringstream oss;
    oss << "SFBench " << VERSION_STRING
        << " (built " << BUILD_DATE << " " << BUILD_TIME
        << " with " << get_compiler_info() << ")";
    return oss.str();
}

// Get version info for export/reports 
// Returns just the version number for embedding in JSON/HTML
inline std::string get_version_for_export() {
    return VERSION_STRING;
}

} 
