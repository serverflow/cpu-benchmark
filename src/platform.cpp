// CPU Benchmark - Platform detection implementation


#include "platform.hpp"
#include <thread>
#include <cstring>
#include <set>
#include <algorithm>

// Platform-specific includes
#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <intrin.h>
    #include <pdh.h>
    #include <pdhmsg.h>
    #include <vector>
    #include <mutex>
#elif defined(__linux__)
    #include <fstream>
    #include <sstream>
    #include <unistd.h>
#elif defined(__APPLE__)
    #include <sys/sysctl.h>
    #include <mach/mach.h>
    #include <mach/mach_host.h>
#endif

// x86-64 CPUID includes
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifndef _WIN32
        #include <cpuid.h>
    #endif
#endif

// Get fallback CPU name when detection fails
// Returns string in format "{arch} {cores}C/{threads}T"
std::string get_fallback_cpu_name(const std::string& arch, unsigned physical_cores, unsigned logical_cores) {
    return arch + " " + std::to_string(physical_cores) + "C/" + std::to_string(logical_cores) + "T";
}

// Get architecture string using preprocessor macros 
std::string get_arch_string() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#elif defined(__arm__) || defined(_M_ARM)
    return "arm";
#else
    return "unknown";
#endif
}

// Get operating system name 
std::string get_os_name() {
#ifdef _WIN32
    return "Windows";
#elif defined(__linux__)
    return "Linux";
#elif defined(__APPLE__)
    return "macOS";
#else
    return "Unknown";
#endif
}

// Get compiler information
std::string get_compiler_info() {
#if defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
    return "Clang " + std::to_string(__clang_major__) + "." + 
           std::to_string(__clang_minor__) + "." + 
           std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
    return "GCC " + std::to_string(__GNUC__) + "." + 
           std::to_string(__GNUC_MINOR__) + "." + 
           std::to_string(__GNUC_PATCHLEVEL__);
#else
    return "Unknown compiler";
#endif
}

// Get total logical core count (handles Windows processor groups)
unsigned get_logical_core_count() {
#ifdef _WIN32
    constexpr WORD kAllGroups = 0xFFFF;
    typedef DWORD (WINAPI *GetActiveProcessorCountFunc)(WORD);
    typedef DWORD (WINAPI *GetMaximumProcessorCountFunc)(WORD);
    static GetActiveProcessorCountFunc pGetActiveProcessorCount = nullptr;
    static GetMaximumProcessorCountFunc pGetMaximumProcessorCount = nullptr;
    static bool initialized = false;

    if (!initialized) {
        HMODULE hKernel32 = GetModuleHandleA("kernel32.dll");
        if (hKernel32) {
            pGetActiveProcessorCount = reinterpret_cast<GetActiveProcessorCountFunc>(
                GetProcAddress(hKernel32, "GetActiveProcessorCount"));
            pGetMaximumProcessorCount = reinterpret_cast<GetMaximumProcessorCountFunc>(
                GetProcAddress(hKernel32, "GetMaximumProcessorCount"));
        }
        initialized = true;
    }

    // IMPORTANT: prefer active processor count for real topology size.
    // On multi-group systems (e.g. 2x Xeon with 80 logical CPUs),
    // GetMaximumProcessorCount(ALL_GROUPS) can report the group capacity
    // (64 + 64 = 128) instead of the actual active logical CPUs.
    if (pGetActiveProcessorCount) {
        DWORD count = pGetActiveProcessorCount(kAllGroups);
        if (count > 0) return static_cast<unsigned>(count);
    }

    // Keep maximum count only as a legacy fallback when active API is absent.
    if (!pGetActiveProcessorCount && pGetMaximumProcessorCount) {
        DWORD count = pGetMaximumProcessorCount(kAllGroups);
        if (count > 0) return static_cast<unsigned>(count);
    }

    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    if (sysInfo.dwNumberOfProcessors > 0) {
        return static_cast<unsigned>(sysInfo.dwNumberOfProcessors);
    }
#endif

    unsigned count = std::thread::hardware_concurrency();
    return count > 0 ? count : 1;
}

// ============================================================================
// OS Version Detection
// ============================================================================

std::string get_os_version() {
#ifdef _WIN32
    // Windows: use registry to get version info
    std::string version = "Windows";
    std::string product_name;
    std::string build_number;
    
    // Try to get detailed version from registry
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, 
                      "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", 
                      0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        char buffer[256];
        DWORD bufferSize = sizeof(buffer);
        DWORD type;
        
        // Get ProductName (e.g., "Windows 10 Pro", "Windows 11 Home")
        if (RegQueryValueExA(hKey, "ProductName", nullptr, &type, 
                            reinterpret_cast<LPBYTE>(buffer), &bufferSize) == ERROR_SUCCESS) {
            product_name = buffer;
        }
        
        // Get CurrentBuild
        bufferSize = sizeof(buffer);
        if (RegQueryValueExA(hKey, "CurrentBuild", nullptr, &type,
                            reinterpret_cast<LPBYTE>(buffer), &bufferSize) == ERROR_SUCCESS) {
            build_number = buffer;
        }
        
        RegCloseKey(hKey);
    }
    
    // Windows 11 detection: builds 22000+ are Windows 11
    // Registry ProductName may still say "Windows 10" on Windows 11 systems
    if (!build_number.empty()) {
        try {
            int build = std::stoi(build_number);
            if (build >= 22000) {
                // This is Windows 11, fix the product name if it says Windows 10
                if (product_name.find("Windows 10") != std::string::npos) {
                    // Replace "Windows 10" with "Windows 11"
                    size_t pos = product_name.find("Windows 10");
                    if (pos != std::string::npos) {
                        product_name.replace(pos, 10, "Windows 11");
                    }
                }
            }
        } catch (...) {
            // Ignore parsing errors
        }
    }
    
    if (!product_name.empty()) {
        version = product_name;
    }
    
    if (!build_number.empty()) {
        version += " Build " + build_number;
    }
    
    return version;
    
#elif defined(__linux__)
    // Linux: try /etc/os-release first
    std::ifstream os_release("/etc/os-release");
    if (os_release.is_open()) {
        std::string line;
        std::string pretty_name;
        
        while (std::getline(os_release, line)) {
            if (line.find("PRETTY_NAME=") == 0) {
                pretty_name = line.substr(12);
                // Remove quotes
                if (!pretty_name.empty() && pretty_name.front() == '"') {
                    pretty_name = pretty_name.substr(1);
                }
                if (!pretty_name.empty() && pretty_name.back() == '"') {
                    pretty_name.pop_back();
                }
                return pretty_name;
            }
        }
    }
    
    // Fallback: try uname
    std::ifstream version_file("/proc/version");
    if (version_file.is_open()) {
        std::string version;
        std::getline(version_file, version);
        // Extract kernel version
        size_t pos = version.find("Linux version ");
        if (pos != std::string::npos) {
            version = version.substr(pos + 14);
            pos = version.find(' ');
            if (pos != std::string::npos) {
                return "Linux " + version.substr(0, pos);
            }
        }
    }
    
    return "Linux";
    
#elif defined(__APPLE__)
    // macOS: use sysctl for version info
    char version[256] = {0};
    size_t len = sizeof(version);
    
    if (sysctlbyname("kern.osproductversion", version, &len, nullptr, 0) == 0) {
        return "macOS " + std::string(version);
    }
    
    // Fallback
    len = sizeof(version);
    if (sysctlbyname("kern.osrelease", version, &len, nullptr, 0) == 0) {
        return "macOS (Darwin " + std::string(version) + ")";
    }
    
    return "macOS";
    
#else
    return "Unknown OS";
#endif
}

// ============================================================================
// CPU Frequency Detection
// ============================================================================

#ifdef _WIN32
namespace {

// ------------------------------
// Windows helpers
// ------------------------------

// Best-effort processor count (handles >64 logical CPUs on Windows)
static DWORD get_active_processor_count_all_groups() {
    // ALL_PROCESSOR_GROUPS is defined as 0xFFFF in WinAPI, but avoid relying on headers.
    constexpr WORD kAllGroups = 0xFFFF;

    typedef DWORD (WINAPI *GetActiveProcessorCountFunc)(WORD);
    static GetActiveProcessorCountFunc pGetActiveProcessorCount = nullptr;
    static bool initialized = false;

    if (!initialized) {
        HMODULE hKernel32 = GetModuleHandleA("kernel32.dll");
        if (hKernel32) {
            pGetActiveProcessorCount = reinterpret_cast<GetActiveProcessorCountFunc>(
                GetProcAddress(hKernel32, "GetActiveProcessorCount"));
        }
        initialized = true;
    }

    if (pGetActiveProcessorCount) {
        DWORD count = pGetActiveProcessorCount(kAllGroups);
        if (count > 0) return count;
    }

    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return (sysInfo.dwNumberOfProcessors > 0) ? sysInfo.dwNumberOfProcessors : 1;
}

// Query current/max MHz via CallNtPowerInformation.
// Returns true on success and fills out_current_mhz/out_max_mhz with the maximum across CPUs.
static bool query_power_information_mhz(ULONG& out_current_mhz, ULONG& out_max_mhz) {
    typedef struct _PROCESSOR_POWER_INFORMATION {
        ULONG Number;
        ULONG MaxMhz;
        ULONG CurrentMhz;
        ULONG MhzLimit;
        ULONG MaxIdleState;
        ULONG CurrentIdleState;
    } PROCESSOR_POWER_INFORMATION;

    typedef LONG (WINAPI *CallNtPowerInformationFunc)(
        ULONG InformationLevel,
        PVOID InputBuffer,
        ULONG InputBufferLength,
        PVOID OutputBuffer,
        ULONG OutputBufferLength
    );

    static CallNtPowerInformationFunc pCallNtPowerInformation = nullptr;
    static bool initialized = false;

    if (!initialized) {
        HMODULE hPowrprof = LoadLibraryA("powrprof.dll");
        if (hPowrprof) {
            pCallNtPowerInformation = reinterpret_cast<CallNtPowerInformationFunc>(
                GetProcAddress(hPowrprof, "CallNtPowerInformation"));
        }
        initialized = true;
    }

    if (!pCallNtPowerInformation) {
        return false;
    }

    // ProcessorInformation is 11 in POWER_INFORMATION_LEVEL.
    constexpr ULONG kProcessorInformation = 11;

    // Some systems (esp. >64 logical CPUs) may require a larger buffer. Retry with growth.
    DWORD count = get_active_processor_count_all_groups();
    DWORD capacity = (count > 0) ? count : 1;
    for (int attempt = 0; attempt < 4; ++attempt) {
        std::vector<PROCESSOR_POWER_INFORMATION> powerInfo(capacity);

        LONG status = pCallNtPowerInformation(
            kProcessorInformation,
            nullptr, 0,
            powerInfo.data(),
            static_cast<ULONG>(powerInfo.size() * sizeof(PROCESSOR_POWER_INFORMATION)));

        if (status == 0) { // STATUS_SUCCESS
            ULONG maxCurrent = 0;
            ULONG maxMax = 0;
            for (size_t i = 0; i < powerInfo.size(); ++i) {
                if (powerInfo[i].CurrentMhz > maxCurrent) maxCurrent = powerInfo[i].CurrentMhz;
                if (powerInfo[i].MaxMhz > maxMax) maxMax = powerInfo[i].MaxMhz;
            }
            if (maxCurrent > 0 || maxMax > 0) {
                out_current_mhz = maxCurrent;
                out_max_mhz = maxMax;
                return true;
            }
            return false;
        }

        // STATUS_INFO_LENGTH_MISMATCH is commonly returned if buffer size is wrong.
        // Grow buffer and retry.
        capacity *= 2;
    }
    return false;
}

// Query real-time CPU frequency (MHz) via PDH counter "Processor Frequency".
// This is the most reliable approach on modern Windows and reflects turbo/boost.
static double query_pdh_frequency_mhz() {
    struct PdhState {
        PDH_HQUERY query = nullptr;
        PDH_HCOUNTER counter = nullptr;
        bool ok = false;
        std::mutex mtx;
    };

    static PdhState state;
    static std::once_flag init_flag;

    std::call_once(init_flag, []() {
        if (PdhOpenQueryA(nullptr, 0, &state.query) != ERROR_SUCCESS) {
            state.ok = false;
            return;
        }

        // Prefer English counter path (works on non-English Windows).
        typedef PDH_STATUS (WINAPI *PdhAddEnglishCounterAFunc)(PDH_HQUERY, LPCSTR, DWORD_PTR, PDH_HCOUNTER*);
        static PdhAddEnglishCounterAFunc pAddEnglish = nullptr;
        {
            HMODULE hPdh = LoadLibraryA("pdh.dll");
            if (hPdh) {
                pAddEnglish = reinterpret_cast<PdhAddEnglishCounterAFunc>(
                    GetProcAddress(hPdh, "PdhAddEnglishCounterA"));
            }
        }

        const char* paths[] = {
            "\\Processor Information(_Total)\\Processor Frequency",
            "\\Processor(_Total)\\Processor Frequency"
        };

        PDH_STATUS st = PDH_CSTATUS_NO_OBJECT;
        for (const char* path : paths) {
            if (pAddEnglish) {
                st = pAddEnglish(state.query, path, 0, &state.counter);
            } else {
                st = PdhAddCounterA(state.query, path, 0, &state.counter);
            }
            if (st == ERROR_SUCCESS) {
                state.ok = true;
                // Prime the query.
                PdhCollectQueryData(state.query);
                return;
            }
        }

        state.ok = false;
    });

    if (!state.ok || !state.query || !state.counter) {
        return 0.0;
    }

    std::lock_guard<std::mutex> lock(state.mtx);
    if (PdhCollectQueryData(state.query) != ERROR_SUCCESS) {
        return 0.0;
    }

    PDH_FMT_COUNTERVALUE value;
    DWORD type = 0;
    if (PdhGetFormattedCounterValue(state.counter, PDH_FMT_DOUBLE, &type, &value) != ERROR_SUCCESS) {
        return 0.0;
    }
    if (value.CStatus != ERROR_SUCCESS) {
        return 0.0;
    }
    return (value.doubleValue > 0.0) ? value.doubleValue : 0.0;
}

} // namespace
#endif

CpuFrequencyInfo get_cpu_frequency() {
    CpuFrequencyInfo info;
    
#ifdef _WIN32
    // Windows: registry provides base/nominal only; use real-time sampler for current
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        DWORD mhz = 0;
        DWORD size = sizeof(mhz);
        
        if (RegQueryValueExA(hKey, "~MHz", nullptr, nullptr,
                            reinterpret_cast<LPBYTE>(&mhz), &size) == ERROR_SUCCESS) {
            info.base_mhz = static_cast<double>(mhz);
            info.min_mhz = info.base_mhz * 0.3;  // Estimate min as 30% of base
        }
        
        RegCloseKey(hKey);
    }

    // Try to get max/current via power information (more accurate than registry)
    ULONG cur = 0, mx = 0;
    if (query_power_information_mhz(cur, mx)) {
        if (mx > 0) info.max_mhz = static_cast<double>(mx);
        if (cur > 0) info.current_mhz = static_cast<double>(cur);
    }

    // Always try PDH for a real-time current frequency
    double pdh_cur = query_pdh_frequency_mhz();
    if (pdh_cur > 0.0) {
        info.current_mhz = pdh_cur;
        if (info.max_mhz <= 0.0 && info.base_mhz > 0.0) {
            // Conservative estimate if max is unknown
            info.max_mhz = info.base_mhz;
        }
    }

    // Final fallback for current/max
    if (info.current_mhz <= 0.0) info.current_mhz = info.base_mhz;
    if (info.max_mhz <= 0.0) info.max_mhz = info.base_mhz;

    info.available = (info.base_mhz > 0.0 || info.current_mhz > 0.0 || info.max_mhz > 0.0);
    
#elif defined(__linux__)
    // Linux: read from /sys/devices/system/cpu/cpu0/cpufreq/
    auto read_freq_file = [](const std::string& path) -> double {
        std::ifstream file(path);
        if (file.is_open()) {
            double khz;
            file >> khz;
            return khz / 1000.0;  // Convert kHz to MHz
        }
        return 0.0;
    };
    
    info.current_mhz = read_freq_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
    info.min_mhz = read_freq_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq");
    info.max_mhz = read_freq_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq");
    info.base_mhz = read_freq_file("/sys/devices/system/cpu/cpu0/cpufreq/base_frequency");
    
    if (info.base_mhz == 0.0) {
        info.base_mhz = info.max_mhz;  // Use max as base if base not available
    }
    
    info.available = (info.current_mhz > 0.0 || info.max_mhz > 0.0);
    
    // Fallback: try /proc/cpuinfo
    if (!info.available) {
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("cpu MHz") == 0) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    info.current_mhz = std::stod(line.substr(colon + 1));
                    info.base_mhz = info.current_mhz;
                    info.max_mhz = info.current_mhz;
                    info.min_mhz = info.current_mhz * 0.3;
                    info.available = true;
                    break;
                }
            }
        }
    }
    
#elif defined(__APPLE__)
    // macOS: use sysctl
    uint64_t freq = 0;
    size_t len = sizeof(freq);
    
    // Try to get CPU frequency (works on Intel Macs, fails on Apple Silicon)
    if (sysctlbyname("hw.cpufrequency", &freq, &len, nullptr, 0) == 0 && freq > 0) {
        info.base_mhz = static_cast<double>(freq) / 1e6;
        info.current_mhz = info.base_mhz;
        info.max_mhz = info.base_mhz;
        info.min_mhz = info.base_mhz * 0.3;
        info.available = true;
    }
    
    // Try max frequency
    len = sizeof(freq);
    if (sysctlbyname("hw.cpufrequency_max", &freq, &len, nullptr, 0) == 0 && freq > 0) {
        info.max_mhz = static_cast<double>(freq) / 1e6;
        if (!info.available) {
            info.base_mhz = info.max_mhz;
            info.current_mhz = info.max_mhz;
            info.available = true;
        }
    }
    
    // Try min frequency
    len = sizeof(freq);
    if (sysctlbyname("hw.cpufrequency_min", &freq, &len, nullptr, 0) == 0 && freq > 0) {
        info.min_mhz = static_cast<double>(freq) / 1e6;
    }
    
    // Apple Silicon fallback: hw.cpufrequency doesn't work on M1/M2/M3
    // Try to detect Apple Silicon and use known frequencies
    if (!info.available) {
        char brand[256] = {0};
        len = sizeof(brand);
        if (sysctlbyname("machdep.cpu.brand_string", brand, &len, nullptr, 0) == 0) {
            std::string brand_str(brand);
            
            // Apple Silicon chips have known P-core frequencies
            if (brand_str.find("Apple M3 Max") != std::string::npos ||
                brand_str.find("Apple M3 Pro") != std::string::npos) {
                info.max_mhz = 4050.0;  // M3 Pro/Max P-cores
                info.base_mhz = 4050.0;
                info.current_mhz = 4050.0;
                info.min_mhz = 600.0;
                info.available = true;
            } else if (brand_str.find("Apple M3") != std::string::npos) {
                info.max_mhz = 4050.0;  // M3 P-cores
                info.base_mhz = 4050.0;
                info.current_mhz = 4050.0;
                info.min_mhz = 600.0;
                info.available = true;
            } else if (brand_str.find("Apple M2 Max") != std::string::npos ||
                       brand_str.find("Apple M2 Pro") != std::string::npos) {
                info.max_mhz = 3500.0;  // M2 Pro/Max P-cores
                info.base_mhz = 3500.0;
                info.current_mhz = 3500.0;
                info.min_mhz = 600.0;
                info.available = true;
            } else if (brand_str.find("Apple M2") != std::string::npos) {
                info.max_mhz = 3500.0;  // M2 P-cores
                info.base_mhz = 3500.0;
                info.current_mhz = 3500.0;
                info.min_mhz = 600.0;
                info.available = true;
            } else if (brand_str.find("Apple M1 Max") != std::string::npos ||
                       brand_str.find("Apple M1 Pro") != std::string::npos) {
                info.max_mhz = 3228.0;  // M1 Pro/Max P-cores
                info.base_mhz = 3228.0;
                info.current_mhz = 3228.0;
                info.min_mhz = 600.0;
                info.available = true;
            } else if (brand_str.find("Apple M1") != std::string::npos) {
                info.max_mhz = 3200.0;  // M1 P-cores
                info.base_mhz = 3200.0;
                info.current_mhz = 3200.0;
                info.min_mhz = 600.0;
                info.available = true;
            }
        }
    }
#endif
    
    return info;
}

double sample_current_frequency() {
#ifdef _WIN32
    // Windows: Prefer PDH "Processor Frequency" (real-time, reflects turbo)
    double pdh_freq = query_pdh_frequency_mhz();
    if (pdh_freq > 0.0) {
        return pdh_freq;
    }

    // Fallback: CallNtPowerInformation (may fail on some configurations)
    ULONG cur = 0, mx = 0;
    if (query_power_information_mhz(cur, mx) && cur > 0) {
        return static_cast<double>(cur);
    }
    
    // Fallback to registry (base frequency only)
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        DWORD mhz = 0;
        DWORD size = sizeof(mhz);
        
        if (RegQueryValueExA(hKey, "~MHz", nullptr, nullptr,
                            reinterpret_cast<LPBYTE>(&mhz), &size) == ERROR_SUCCESS) {
            RegCloseKey(hKey);
            return static_cast<double>(mhz);
        }
        RegCloseKey(hKey);
    }
    return 0.0;
    
#elif defined(__linux__)
    // Try scaling_cur_freq first (most accurate for current frequency)
    // Try multiple CPUs as frequency may vary per core
    for (int cpu = 0; cpu < 8; ++cpu) {
        std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/scaling_cur_freq";
        std::ifstream file(path);
        if (file.is_open()) {
            double khz;
            file >> khz;
            if (khz > 0) {
                return khz / 1000.0;
            }
        }
    }
    
    // Fallback: try /proc/cpuinfo (less accurate but more widely available)
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("cpu MHz") == 0) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    try {
                        return std::stod(line.substr(colon + 1));
                    } catch (...) {
                        return 0.0;
                    }
                }
            }
        }
    }
    return 0.0;
    
#elif defined(__APPLE__)
    // macOS: hw.cpufrequency works on Intel, fails on Apple Silicon
    uint64_t freq = 0;
    size_t len = sizeof(freq);
    if (sysctlbyname("hw.cpufrequency", &freq, &len, nullptr, 0) == 0 && freq > 0) {
        return static_cast<double>(freq) / 1e6;
    }
    
    // Apple Silicon fallback: return known P-core frequencies
    // Real-time frequency monitoring is not available through public APIs
    char brand[256] = {0};
    len = sizeof(brand);
    if (sysctlbyname("machdep.cpu.brand_string", brand, &len, nullptr, 0) == 0) {
        std::string brand_str(brand);
        
        if (brand_str.find("Apple M3") != std::string::npos) {
            return 4050.0;  // M3 family P-cores
        } else if (brand_str.find("Apple M2") != std::string::npos) {
            return 3500.0;  // M2 family P-cores
        } else if (brand_str.find("Apple M1") != std::string::npos) {
            return 3200.0;  // M1 family P-cores
        }
    }
    return 0.0;
    
#else
    return 0.0;
#endif
}

// ============================================================================
// Multi-Socket Detection
// ============================================================================

unsigned get_socket_count() {
#ifdef _WIN32
    // Windows: use GetLogicalProcessorInformationEx
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorPackage, nullptr, &length);
    
    if (length == 0) {
        return 1;
    }
    
    std::vector<char> buffer(length);
    if (!GetLogicalProcessorInformationEx(RelationProcessorPackage,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
            &length)) {
        return 1;
    }
    
    unsigned socket_count = 0;
    char* ptr = buffer.data();
    char* end = ptr + length;
    
    while (ptr < end) {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr);
        if (!info || info->Size == 0 || ptr + info->Size > end) {
            break;
        }
        if (info->Relationship == RelationProcessorPackage) {
            socket_count++;
        }
        ptr += info->Size;
    }
    
    return socket_count > 0 ? socket_count : 1;
    
#elif defined(__linux__)
    // Linux: count unique physical_package_id values
    std::vector<int> packages;
    
    for (unsigned cpu = 0; cpu < get_logical_core_count(); ++cpu) {
        std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + 
                          "/topology/physical_package_id";
        std::ifstream file(path);
        if (file.is_open()) {
            int package_id;
            file >> package_id;
            
            bool found = false;
            for (int p : packages) {
                if (p == package_id) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                packages.push_back(package_id);
            }
        }
    }
    
    return packages.empty() ? 1 : static_cast<unsigned>(packages.size());
    
#elif defined(__APPLE__)
    // macOS: typically single socket, but check sysctl
    int packages = 0;
    size_t len = sizeof(packages);
    
    if (sysctlbyname("hw.packages", &packages, &len, nullptr, 0) == 0 && packages > 0) {
        return static_cast<unsigned>(packages);
    }
    
    return 1;
    
#else
    return 1;
#endif
}

std::vector<unsigned> get_cores_for_socket(unsigned socket_id) {
    std::vector<unsigned> cores;
    
#ifdef _WIN32
    // Windows: use GetLogicalProcessorInformationEx
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorPackage, nullptr, &length);
    
    if (length == 0) {
        // Fallback: return all cores for socket 0
        if (socket_id == 0) {
            for (unsigned i = 0; i < get_logical_core_count(); ++i) {
                cores.push_back(i);
            }
        }
        return cores;
    }
    
    std::vector<char> buffer(length);
    if (!GetLogicalProcessorInformationEx(RelationProcessorPackage,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
            &length)) {
        return cores;
    }
    
    unsigned current_socket = 0;
    char* ptr = buffer.data();
    char* end = ptr + length;
    
    while (ptr < end) {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr);
        if (!info || info->Size == 0 || ptr + info->Size > end) {
            break;
        }
        if (info->Relationship == RelationProcessorPackage) {
            if (current_socket == socket_id) {
                // Extract cores from processor mask
                for (WORD g = 0; g < info->Processor.GroupCount; ++g) {
                    KAFFINITY mask = info->Processor.GroupMask[g].Mask;
                    WORD group = info->Processor.GroupMask[g].Group;
                    for (unsigned bit = 0; bit < 64; ++bit) {
                        if (mask & (1ULL << bit)) {
                            cores.push_back(bit + static_cast<unsigned>(group) * 64);
                        }
                    }
                }
                break;
            }
            current_socket++;
        }
        ptr += info->Size;
    }
    
#elif defined(__linux__)
    // Linux: find cores with matching physical_package_id
    for (unsigned cpu = 0; cpu < get_logical_core_count(); ++cpu) {
        std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + 
                          "/topology/physical_package_id";
        std::ifstream file(path);
        if (file.is_open()) {
            unsigned package_id;
            file >> package_id;
            if (package_id == socket_id) {
                cores.push_back(cpu);
            }
        }
    }
    
#elif defined(__APPLE__)
    // macOS: typically single socket, return all cores for socket 0
    if (socket_id == 0) {
        for (unsigned i = 0; i < get_logical_core_count(); ++i) {
            cores.push_back(i);
        }
    }
#endif
    
    return cores;
}

// ============================================================================
// Performance Core Detection (for Hybrid Architectures)
// ============================================================================


std::vector<unsigned> get_performance_cores() {
    // Returns a list of logical CPU indices that belong to the *performance* core type.
    // IMPORTANT: this function returns **logical processors**, not physical cores.
    // On Intel Alder/Raptor/Meteor Lake with SMT enabled, P-cores typically contribute 2
    // logical processors each, E-cores contribute 1.

    unsigned total = get_logical_core_count();
    if (total == 0) total = 1;

    // Helper: return all logical CPUs
    auto all_cores = [total]() {
        std::vector<unsigned> v;
        v.reserve(total);
        for (unsigned i = 0; i < total; ++i) v.push_back(i);
        return v;
    };

#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0) {
        return all_cores();
    }

    std::vector<uint8_t> buffer(len);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
            &len)) {
        return all_cores();
    }

    struct CoreEntry {
        std::vector<unsigned> logical;
        uint8_t efficiency_class = 0;
        bool smt = false;            // SMT present on this core (usually indicates a P-core on Intel hybrids)
        unsigned logical_count = 0;  // total logical processors for this core
    };

    auto popcount64 = [](unsigned long long x) -> unsigned {
#if defined(_MSC_VER)
        return static_cast<unsigned>(__popcnt64(x));
#elif defined(__GNUC__) || defined(__clang__)
        return static_cast<unsigned>(__builtin_popcountll(x));
#else
        unsigned c = 0;
        while (x) { c += (x & 1ULL); x >>= 1ULL; }
        return c;
#endif
    };

    std::vector<CoreEntry> cores;
    cores.reserve(total);

    bool has_nonzero_eff = false;
    uint8_t max_eff = 0;
    bool has_smt = false;
    bool has_no_smt = false;

    size_t offset = 0;
    size_t limit = static_cast<size_t>(len);
    while (offset + sizeof(DWORD) * 2 <= limit) {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
        if (!info || info->Size == 0 || offset + info->Size > limit) break;

        if (info->Relationship == RelationProcessorCore) {
            CoreEntry entry;
            entry.efficiency_class = info->Processor.EfficiencyClass;

            // Collect logical processors for this core across processor groups
            unsigned logical_count = 0;
            for (WORD g = 0; g < info->Processor.GroupCount; ++g) {
                const GROUP_AFFINITY& ga = info->Processor.GroupMask[g];
                unsigned long long mask = static_cast<unsigned long long>(ga.Mask);
                logical_count += popcount64(mask);

                for (unsigned bit = 0; bit < 64; ++bit) {
                    if (mask & (1ULL << bit)) {
                        unsigned lp = static_cast<unsigned>(ga.Group) * 64u + bit;
                        if (lp < total) entry.logical.push_back(lp);
                    }
                }
            }

            entry.logical_count = logical_count;
            entry.smt = (info->Processor.Flags != 0) || (logical_count > 1);

            if (entry.efficiency_class != 0) {
                has_nonzero_eff = true;
                if (entry.efficiency_class > max_eff) max_eff = entry.efficiency_class;
            }

            if (entry.smt) has_smt = true;
            else has_no_smt = true;

            if (!entry.logical.empty()) {
                cores.push_back(std::move(entry));
            }
        }

        offset += info->Size;
    }

    if (cores.empty()) {
        return all_cores();
    }

    std::vector<unsigned> p;

    // Heuristic #1 (very reliable on Intel 12/13/14 gen): P-cores usually have SMT, E-cores don't.
    // If we see a mix, treat SMT cores as performance cores.
    if (has_smt && has_no_smt) {
        for (const auto& c : cores) {
            if (c.logical_count > 1) {
                p.insert(p.end(), c.logical.begin(), c.logical.end());
            }
        }
    }
    // Heuristic #2: Use EfficiencyClass. According to Microsoft docs, a *higher* value means
    // greater performance (and lower efficiency). Select cores with the maximum class.
    // (EfficiencyClass is typically nonzero only on heterogeneous systems.)
    else if (has_nonzero_eff && max_eff > 0) {
        for (const auto& c : cores) {
            if (c.efficiency_class == max_eff) {
                p.insert(p.end(), c.logical.begin(), c.logical.end());
            }
        }
    }

    if (p.empty()) {
        return all_cores();
    }

    // Deduplicate + sort
    std::sort(p.begin(), p.end());
    p.erase(std::unique(p.begin(), p.end()), p.end());
    return p;

#elif defined(__linux__)
    // Linux: prefer kernel-exposed core_type when available (maps to CPUID leaf 0x1A on x86).
    // Typically: 1 = "Atom" (E-core), 2 = "Core" (P-core).
    std::vector<int> core_type(total, -1);
    int max_type = -1;
    bool any_core_type = false;

    for (unsigned cpu = 0; cpu < total; ++cpu) {
        std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/core_type";
        std::ifstream f(path);
        if (!f.is_open()) continue;

        int t = -1;
        f >> t;
        if (t >= 0) {
            core_type[cpu] = t;
            any_core_type = true;
            if (t > max_type) max_type = t;
        }
    }

    if (any_core_type && max_type > 0) {
        std::vector<unsigned> p;
        for (unsigned cpu = 0; cpu < total; ++cpu) {
            if (core_type[cpu] == max_type) p.push_back(cpu);
        }
        if (!p.empty() && p.size() < total) return p;
    }

    // Fallback #1: SMT heuristic using thread_siblings_list (P-cores usually have 2 siblings).
    
auto parse_cpu_list_count = [](const std::string& s) -> unsigned {
    // Accept formats like "0", "0,8", "0-3", "0-1,4-5".
    unsigned count = 0;
    size_t i = 0;
    auto skip_seps = [&](void) {
        while (i < s.size() && (s[i] == ',' || s[i] == ' ' || s[i] == '\n' || s[i] == '\t')) ++i;
    };

    while (i < s.size()) {
        skip_seps();
        if (i >= s.size()) break;

        // parse start number
        unsigned start = 0;
        bool have = false;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
            have = true;
            start = start * 10 + static_cast<unsigned>(s[i] - '0');
            ++i;
        }
        if (!have) break;

        skip_seps();
        if (i < s.size() && s[i] == '-') {
            ++i;
            // parse end number
            unsigned end = 0;
            have = false;
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
                have = true;
                end = end * 10 + static_cast<unsigned>(s[i] - '0');
                ++i;
            }
            if (have && end >= start) count += (end - start + 1);
            else count += 1;
        } else {
            count += 1;
        }

        // move to next token
        while (i < s.size() && s[i] != ',') ++i;
        if (i < s.size() && s[i] == ',') ++i;
    }

    return count;
};

// Try SMT heuristic if thread_siblings_list exists
bool any_siblings = false;
bool has_smt = false;
bool has_no_smt = false;
std::vector<unsigned> smt_cpus;
smt_cpus.reserve(total);

for (unsigned cpu = 0; cpu < total; ++cpu) {
    std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list";
    std::ifstream f(path);
    if (!f.is_open()) continue;
    std::string line;
    std::getline(f, line);
    if (line.empty()) continue;

    unsigned siblings = parse_cpu_list_count(line);
    any_siblings = true;
    if (siblings > 1) {
        has_smt = true;
        smt_cpus.push_back(cpu);
    } else {
        has_no_smt = true;
    }
}

if (any_siblings && has_smt && has_no_smt) {
    // P-cores usually have SMT -> take those logical CPUs
    std::sort(smt_cpus.begin(), smt_cpus.end());
    smt_cpus.erase(std::unique(smt_cpus.begin(), smt_cpus.end()), smt_cpus.end());
    if (!smt_cpus.empty() && smt_cpus.size() < total) return smt_cpus;
}

    std::vector<unsigned> p_cores;
    p_cores.reserve(total);

    // Existing heuristic: use base/max frequency differences
    std::vector<unsigned long long> max_freqs(total, 0);
    unsigned long long max_freq = 0;

    for (unsigned i = 0; i < total; ++i) {
        std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/cpuinfo_max_freq";
        std::ifstream file(path);
        if (!file.is_open()) continue;

        unsigned long long freq = 0;
        file >> freq;
        max_freqs[i] = freq;
        if (freq > max_freq) max_freq = freq;
    }

    if (max_freq == 0) {
        return all_cores();
    }

    // Consider cores within 90% of the maximum as performance cores
    unsigned long long threshold = max_freq * 9 / 10;
    for (unsigned i = 0; i < total; ++i) {
        if (max_freqs[i] >= threshold && max_freqs[i] > 0) {
            p_cores.push_back(i);
        }
    }

    if (p_cores.empty() || p_cores.size() == total) {
        return all_cores();
    }
    return p_cores;

#elif defined(__APPLE__)
    // macOS: treat "performance" cores as those contributing to hw.perflevel0.
    // The current implementation is a conservative fallback.
    return all_cores();

#else
    return all_cores();
#endif
}


unsigned get_best_performance_core() {
    auto p = get_performance_cores();
    if (p.empty()) return 0;
    // Prefer the lowest logical index among performance cores (stable choice).
    return *std::min_element(p.begin(), p.end());
}



// ============================================================================
// Cache Detection Implementation 
// ============================================================================

#ifdef _WIN32
// Windows cache detection using GetLogicalProcessorInformation
static CacheInfo get_cache_info_windows() {
    CacheInfo cache;

    auto finalize_cache = [&](size_t l1_data_max,
                              size_t l1_inst_max,
                              size_t l2_max,
                              size_t l3_total) {
        cache.l1_data_size = l1_data_max;
        cache.l1_inst_size = l1_inst_max;
        cache.l2_size = l2_max;
        cache.l3_size = l3_total;
        cache.l1_available = (l1_data_max > 0 || l1_inst_max > 0);
        cache.l2_available = (l2_max > 0);
        cache.l3_available = (l3_total > 0);
    };

    DWORD buffer_size = 0;
    if (!GetLogicalProcessorInformationEx(RelationCache, nullptr, &buffer_size) &&
        GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        buffer_size = 0;
    }

    if (buffer_size > 0) {
        std::vector<unsigned char> buffer(buffer_size);
        if (GetLogicalProcessorInformationEx(RelationCache,
                reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
                &buffer_size)) {
            size_t offset = 0;
            size_t l1_data_max = 0;
            size_t l1_inst_max = 0;
            size_t l2_max = 0;
            size_t l3_total = 0;

            while (offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) <= buffer_size) {
                auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
                if (info->Relationship == RelationCache) {
                    const CACHE_RELATIONSHIP& cd = info->Cache;

                    if (cache.cache_line_size == 0 && cd.LineSize > 0) {
                        cache.cache_line_size = cd.LineSize;
                    }

                    switch (cd.Level) {
                        case 1:
                            if (cd.Type == CacheData || cd.Type == CacheUnified) {
                                l1_data_max = (std::max)(l1_data_max, static_cast<size_t>(cd.CacheSize));
                            } else if (cd.Type == CacheInstruction) {
                                l1_inst_max = (std::max)(l1_inst_max, static_cast<size_t>(cd.CacheSize));
                            }
                            break;
                        case 2:
                            l2_max = (std::max)(l2_max, static_cast<size_t>(cd.CacheSize));
                            break;
                        case 3:
                            l3_total += static_cast<size_t>(cd.CacheSize);
                            break;
                    }
                }

                if (info->Size == 0) {
                    break;
                }
                offset += info->Size;
            }

            finalize_cache(l1_data_max, l1_inst_max, l2_max, l3_total);
        }
    }

    if (!cache.l1_available && !cache.l2_available && !cache.l3_available) {
        DWORD legacy_size = 0;
        GetLogicalProcessorInformation(nullptr, &legacy_size);

        if (legacy_size > 0) {
            std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
                legacy_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));

            if (GetLogicalProcessorInformation(buffer.data(), &legacy_size)) {
                size_t l1_data_max = 0;
                size_t l1_inst_max = 0;
                size_t l2_max = 0;
                size_t l3_total = 0;

                for (const auto& info : buffer) {
                    if (info.Relationship == RelationCache) {
                        const CACHE_DESCRIPTOR& cd = info.Cache;

                        if (cache.cache_line_size == 0 && cd.LineSize > 0) {
                            cache.cache_line_size = cd.LineSize;
                        }

                        switch (cd.Level) {
                            case 1:
                                if (cd.Type == CacheData || cd.Type == CacheUnified) {
                                    l1_data_max = (std::max)(l1_data_max, static_cast<size_t>(cd.Size));
                                } else if (cd.Type == CacheInstruction) {
                                    l1_inst_max = (std::max)(l1_inst_max, static_cast<size_t>(cd.Size));
                                }
                                break;
                            case 2:
                                l2_max = (std::max)(l2_max, static_cast<size_t>(cd.Size));
                                break;
                            case 3:
                                l3_total += static_cast<size_t>(cd.Size);
                                break;
                        }
                    }
                }

                finalize_cache(l1_data_max, l1_inst_max, l2_max, l3_total);
            }
        }
    }

    if (cache.cache_line_size == 0) {
        cache.cache_line_size = 64;
    }

    return cache;
}
#endif // _WIN32

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && !defined(_WIN32)
// x86-64 cache detection using CPUID leaf 4 (Linux/macOS)

static void cpuid_ex(int info[4], int function_id, int sub_function) {
    __cpuid_count(function_id, sub_function, info[0], info[1], info[2], info[3]);
}

#if defined(__linux__)
// Linux x86: fallback to sysfs if CPUID doesn't work
static size_t read_sysfs_cache_size_x86(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return 0;
    }
    
    std::string value;
    file >> value;
    
    if (value.empty()) {
        return 0;
    }
    
    // Parse size with suffix (K, M, G)
    size_t size = 0;
    char suffix = 0;
    
    // Remove trailing 'K', 'M', 'G' if present
    if (!value.empty() && (value.back() == 'K' || value.back() == 'M' || value.back() == 'G')) {
        suffix = value.back();
        value.pop_back();
    }
    
    try {
        size = std::stoull(value);
    } catch (...) {
        return 0;
    }
    
    // Apply multiplier
    switch (suffix) {
        case 'K': size *= 1024; break;
        case 'M': size *= 1024 * 1024; break;
        case 'G': size *= 1024 * 1024 * 1024; break;
    }
    
    return size;
}

// Read shared_cpu_list to determine how many CPUs share this cache
static int count_cpus_in_shared_list(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return 1;
    }
    
    std::string list;
    std::getline(file, list);
    
    if (list.empty()) {
        return 1;
    }
    
    // Parse CPU list format: "0-3,8-11" or "0,1,2,3"
    int count = 0;
    size_t pos = 0;
    while (pos < list.size()) {
        // Find next number
        size_t start = pos;
        while (pos < list.size() && list[pos] != ',' && list[pos] != '-') {
            pos++;
        }
        
        int first_num = 0;
        try {
            first_num = std::stoi(list.substr(start, pos - start));
        } catch (...) {
            break;
        }
        
        if (pos < list.size() && list[pos] == '-') {
            // Range: "0-3"
            pos++; // skip '-'
            start = pos;
            while (pos < list.size() && list[pos] != ',') {
                pos++;
            }
            int second_num = 0;
            try {
                second_num = std::stoi(list.substr(start, pos - start));
            } catch (...) {
                break;
            }
            count += (second_num - first_num + 1);
        } else {
            // Single number
            count++;
        }
        
        if (pos < list.size() && list[pos] == ',') {
            pos++; // skip ','
        }
    }
    
    return count > 0 ? count : 1;
}

static CacheInfo get_cache_info_x86_sysfs() {
    CacheInfo cache;
    
    // For L3 cache on multi-CCX systems (like AMD EPYC), we need to count
    // unique L3 caches across all CPUs. Each CCX has its own L3.
    // We'll scan all CPUs and count unique L3 caches by their shared_cpu_list.
    
    std::set<std::string> unique_l3_caches;
    size_t l3_size_per_ccx = 0;
    
    // First, get basic cache info from cpu0
    const std::string base_path = "/sys/devices/system/cpu/cpu0/cache/";
    
    // Enumerate cache indices (index0, index1, index2, ...)
    for (int idx = 0; idx < 10; ++idx) {
        std::string index_path = base_path + "index" + std::to_string(idx) + "/";
        
        // Read cache level
        std::ifstream level_file(index_path + "level");
        if (!level_file.is_open()) {
            break; // No more cache indices
        }
        
        int level = 0;
        level_file >> level;
        
        // Read cache type
        std::ifstream type_file(index_path + "type");
        std::string type;
        if (type_file.is_open()) {
            type_file >> type;
        }
        
        // Read cache size
        size_t size = read_sysfs_cache_size_x86(index_path + "size");
        
        // Read coherency line size (cache line size)
        std::ifstream line_file(index_path + "coherency_line_size");
        if (line_file.is_open() && cache.cache_line_size == 0) {
            line_file >> cache.cache_line_size;
        }
        
        // Store based on level and type
        switch (level) {
            case 1:
                if (type == "Data") {
                    cache.l1_data_size = size;
                    cache.l1_available = true;
                } else if (type == "Instruction") {
                    cache.l1_inst_size = size;
                    cache.l1_available = true;
                } else { // Unified or unknown
                    cache.l1_data_size = size;
                    cache.l1_available = true;
                }
                break;
            case 2:
                cache.l2_size = size;
                cache.l2_available = true;
                break;
            case 3:
                l3_size_per_ccx = size;
                cache.l3_available = true;
                break;
        }
    }
    
    // Now count unique L3 caches across all CPUs
    // Each L3 cache has a unique shared_cpu_list
    if (cache.l3_available && l3_size_per_ccx > 0) {
        unsigned num_cpus = get_logical_core_count();
        
        for (unsigned cpu = 0; cpu < num_cpus; ++cpu) {
            std::string cpu_cache_path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cache/";
            
            // Find L3 cache index for this CPU
            for (int idx = 0; idx < 10; ++idx) {
                std::string index_path = cpu_cache_path + "index" + std::to_string(idx) + "/";
                
                std::ifstream level_file(index_path + "level");
                if (!level_file.is_open()) {
                    break;
                }
                
                int level = 0;
                level_file >> level;
                
                if (level == 3) {
                    // Read shared_cpu_list to identify this L3 cache uniquely
                    std::ifstream shared_file(index_path + "shared_cpu_list");
                    if (shared_file.is_open()) {
                        std::string shared_list;
                        std::getline(shared_file, shared_list);
                        unique_l3_caches.insert(shared_list);
                    }
                    break;
                }
            }
        }
        
        // Total L3 = size per CCX * number of unique L3 caches
        if (!unique_l3_caches.empty()) {
            cache.l3_size = l3_size_per_ccx * unique_l3_caches.size();
        } else {
            cache.l3_size = l3_size_per_ccx;
        }
    }
    
    // Default cache line size if not detected
    if (cache.cache_line_size == 0) {
        cache.cache_line_size = 64;
    }
    
    return cache;
}
#endif // __linux__

static CacheInfo get_cache_info_x86_cpuid() {
    CacheInfo cache;
    
    // Check if CPUID leaf 4 is supported
    int info[4] = {0};
    __cpuid(0, info[0], info[1], info[2], info[3]);
    
    int max_leaf = info[0];
    if (max_leaf < 4) {
        // CPUID leaf 4 not supported, try to get cache line size from leaf 1
        __cpuid(1, info[0], info[1], info[2], info[3]);
        // EBX[15:8] contains CLFLUSH line size in 8-byte units
        int clflush_size = ((info[1] >> 8) & 0xFF) * 8;
        if (clflush_size > 0) {
            cache.cache_line_size = static_cast<size_t>(clflush_size);
        } else {
            cache.cache_line_size = 64; // Default
        }
        return cache;
    }
    
    // Enumerate cache levels using CPUID leaf 4
    for (int sub_leaf = 0; sub_leaf < 32; ++sub_leaf) {
        cpuid_ex(info, 4, sub_leaf);
        
        // EAX[4:0] = cache type (0 = null, 1 = data, 2 = instruction, 3 = unified)
        int cache_type = info[0] & 0x1F;
        if (cache_type == 0) {
            break; // No more caches
        }
        
        // EAX[7:5] = cache level (1, 2, or 3)
        int cache_level = (info[0] >> 5) & 0x7;
        
        // Calculate cache size:
        // EBX[31:22] = Ways - 1
        // EBX[21:12] = Partitions - 1
        // EBX[11:0] = Line Size - 1
        // ECX = Sets - 1
        int ways = ((info[1] >> 22) & 0x3FF) + 1;
        int partitions = ((info[1] >> 12) & 0x3FF) + 1;
        int line_size = (info[1] & 0xFFF) + 1;
        int sets = info[2] + 1;
        
        size_t cache_size = static_cast<size_t>(ways) * partitions * line_size * sets;
        
        // Store cache line size (should be same for all levels)
        if (cache.cache_line_size == 0) {
            cache.cache_line_size = static_cast<size_t>(line_size);
        }
        
        // Store cache size based on level and type
        switch (cache_level) {
            case 1:
                if (cache_type == 1) { // Data cache
                    cache.l1_data_size = cache_size;
                    cache.l1_available = true;
                } else if (cache_type == 2) { // Instruction cache
                    cache.l1_inst_size = cache_size;
                    cache.l1_available = true;
                } else if (cache_type == 3) { // Unified
                    cache.l1_data_size = cache_size;
                    cache.l1_available = true;
                }
                break;
            case 2:
                cache.l2_size = cache_size;
                cache.l2_available = true;
                break;
            case 3:
                cache.l3_size = cache_size;
                cache.l3_available = true;
                break;
        }
    }
    
    // Default cache line size if not detected
    if (cache.cache_line_size == 0) {
        cache.cache_line_size = 64;
    }
    
    return cache;
}
#endif // x86-64 non-Windows

#if defined(__aarch64__) || defined(_M_ARM64)
// ARM64 cache detection

#if defined(__linux__)
// Linux ARM64: read from sysfs
static size_t read_sysfs_cache_size(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return 0;
    }
    
    std::string value;
    file >> value;
    
    if (value.empty()) {
        return 0;
    }
    
    // Parse size with suffix (K, M, G)
    size_t size = 0;
    char suffix = 0;
    
    // Remove trailing 'K', 'M', 'G' if present
    if (!value.empty() && (value.back() == 'K' || value.back() == 'M' || value.back() == 'G')) {
        suffix = value.back();
        value.pop_back();
    }
    
    try {
        size = std::stoull(value);
    } catch (...) {
        return 0;
    }
    
    // Apply multiplier
    switch (suffix) {
        case 'K': size *= 1024; break;
        case 'M': size *= 1024 * 1024; break;
        case 'G': size *= 1024 * 1024 * 1024; break;
    }
    
    return size;
}

static CacheInfo get_cache_info_arm64_linux() {
    CacheInfo cache;
    
    // Read cache information from sysfs
    const std::string base_path = "/sys/devices/system/cpu/cpu0/cache/";
    
    // Enumerate cache indices (index0, index1, index2, ...)
    for (int idx = 0; idx < 10; ++idx) {
        std::string index_path = base_path + "index" + std::to_string(idx) + "/";
        
        // Read cache level
        std::ifstream level_file(index_path + "level");
        if (!level_file.is_open()) {
            break; // No more cache indices
        }
        
        int level = 0;
        level_file >> level;
        
        // Read cache type
        std::ifstream type_file(index_path + "type");
        std::string type;
        if (type_file.is_open()) {
            type_file >> type;
        }
        
        // Read cache size
        size_t size = read_sysfs_cache_size(index_path + "size");
        
        // Read coherency line size (cache line size)
        std::ifstream line_file(index_path + "coherency_line_size");
        if (line_file.is_open() && cache.cache_line_size == 0) {
            line_file >> cache.cache_line_size;
        }
        
        // Store based on level and type
        switch (level) {
            case 1:
                if (type == "Data") {
                    cache.l1_data_size = size;
                    cache.l1_available = true;
                } else if (type == "Instruction") {
                    cache.l1_inst_size = size;
                    cache.l1_available = true;
                } else { // Unified or unknown
                    cache.l1_data_size = size;
                    cache.l1_available = true;
                }
                break;
            case 2:
                cache.l2_size = size;
                cache.l2_available = true;
                break;
            case 3:
                cache.l3_size = size;
                cache.l3_available = true;
                break;
        }
    }
    
    // Default cache line size for ARM64 if not detected
    if (cache.cache_line_size == 0) {
        cache.cache_line_size = 64; // Common default for ARM64
    }
    
    return cache;
}
#endif // __linux__

#if defined(__APPLE__)
// macOS ARM64: use sysctl
static CacheInfo get_cache_info_arm64_macos() {
    CacheInfo cache;
    
    size_t size;
    size_t len = sizeof(size);
    
    // L1 data cache
    if (sysctlbyname("hw.l1dcachesize", &size, &len, nullptr, 0) == 0) {
        cache.l1_data_size = size;
        cache.l1_available = true;
    }
    
    // L1 instruction cache
    len = sizeof(size);
    if (sysctlbyname("hw.l1icachesize", &size, &len, nullptr, 0) == 0) {
        cache.l1_inst_size = size;
        cache.l1_available = true;
    }
    
    // L2 cache
    len = sizeof(size);
    if (sysctlbyname("hw.l2cachesize", &size, &len, nullptr, 0) == 0) {
        cache.l2_size = size;
        cache.l2_available = true;
    }
    
    // L3 cache (may not exist on all Apple Silicon)
    len = sizeof(size);
    if (sysctlbyname("hw.l3cachesize", &size, &len, nullptr, 0) == 0 && size > 0) {
        cache.l3_size = size;
        cache.l3_available = true;
    }

    // Apple Silicon often exposes System Level Cache (SLC) instead of a classic L3
    if (!cache.l3_available) {
        len = sizeof(size);
        if (sysctlbyname("hw.systemcachesize", &size, &len, nullptr, 0) == 0 && size > 0) {
            cache.l3_size = size;
            cache.l3_available = true;
        }
    }
    
    // Cache line size
    len = sizeof(size);
    if (sysctlbyname("hw.cachelinesize", &size, &len, nullptr, 0) == 0) {
        cache.cache_line_size = size;
    } else {
        cache.cache_line_size = 64; // Default for Apple Silicon
    }
    
    return cache;
}
#endif // __APPLE__

#endif // ARM64

// Main cache detection function
CacheInfo get_cache_info() {
#ifdef _WIN32
    // Windows: use GetLogicalProcessorInformation API
    return get_cache_info_windows();
#elif defined(__x86_64__) || defined(__i386__)
    #if defined(__linux__)
        // Linux x86: prefer sysfs (more accurate for multi-CCX/multi-socket)
        CacheInfo cache = get_cache_info_x86_sysfs();
        if (!cache.l1_available && !cache.l2_available && !cache.l3_available) {
            cache = get_cache_info_x86_cpuid();
        }
        return cache;
    #else
        // macOS/other x86: use CPUID
        return get_cache_info_x86_cpuid();
    #endif
#elif defined(__aarch64__)
    #if defined(__linux__)
        return get_cache_info_arm64_linux();
    #elif defined(__APPLE__)
        return get_cache_info_arm64_macos();
    #else
        // Other ARM64 - return default cache info
        CacheInfo cache;
        cache.cache_line_size = 64; // Common default
        return cache;
    #endif
#else
    // Unknown platform - return empty cache info with default line size
    CacheInfo cache;
    cache.cache_line_size = 64;
    return cache;
#endif
}

// ============================================================================
// CPU Info Implementation
// ============================================================================

#ifdef _WIN32
// Windows implementation
static void get_cpuid(int info[4], int function_id) {
    __cpuid(info, function_id);
}

static std::string get_cpu_vendor_windows() {
    int info[4];
    get_cpuid(info, 0);
    
    char vendor[13];
    memcpy(vendor, &info[1], 4);     // EBX
    memcpy(vendor + 4, &info[3], 4); // EDX
    memcpy(vendor + 8, &info[2], 4); // ECX
    vendor[12] = '\0';
    
    return std::string(vendor);
}

static std::string get_cpu_model_windows() {
    int info[4];
    char brand[49];
    
    // Check if extended CPUID is supported
    get_cpuid(info, 0x80000000);
    unsigned max_extended = static_cast<unsigned>(info[0]);
    
    if (max_extended >= 0x80000004) {
        get_cpuid(info, 0x80000002);
        memcpy(brand, info, 16);
        get_cpuid(info, 0x80000003);
        memcpy(brand + 16, info, 16);
        get_cpuid(info, 0x80000004);
        memcpy(brand + 32, info, 16);
        brand[48] = '\0';
        
        // Trim leading spaces
        std::string result(brand);
        size_t start = result.find_first_not_of(' ');
        if (start != std::string::npos) {
            return result.substr(start);
        }
        return result;
    }
    
    // Fallback: return arch with core count 
    unsigned logical = get_logical_core_count();
    // Can't call get_physical_cores_windows() here as it's not defined yet
    // Use logical cores as fallback
    return get_fallback_cpu_name(get_arch_string(), logical, logical);
}

static unsigned get_physical_cores_windows() {
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || length == 0) {
        return get_logical_core_count();
    }

    std::vector<uint8_t> buffer(length);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
            &length)) {
        return get_logical_core_count();
    }

    unsigned physical_cores = 0;
    size_t offset = 0;
    while (offset + sizeof(DWORD) * 2 <= length) {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
        if (!info || info->Size == 0 || offset + info->Size > length) {
            break;
        }
        if (info->Relationship == RelationProcessorCore) {
            ++physical_cores;
        }
        offset += info->Size;
    }

    return physical_cores > 0 ? physical_cores : get_logical_core_count();
}

CpuInfo get_cpu_info() {
    CpuInfo info;
    
    info.arch = get_arch_string();
    info.logical_cores = get_logical_core_count();
    info.physical_cores = get_physical_cores_windows();
    info.vendor = get_cpu_vendor_windows();
    info.model = get_cpu_model_windows();
    info.cache = get_cache_info();
    
    // Multi-socket detection
    info.socket_count = get_socket_count();
    
    if (info.socket_count > 1) {
        // For multi-socket systems, collect per-socket info
        std::vector<std::string> socket_models;
        
        for (unsigned s = 0; s < info.socket_count; ++s) {
            SocketInfo socket;
            socket.socket_id = s;
            socket.core_ids = get_cores_for_socket(s);
            socket.logical_cores = static_cast<unsigned>(socket.core_ids.size());
            socket.physical_cores = socket.logical_cores / 2;  // Assume HT
            if (socket.physical_cores == 0) socket.physical_cores = socket.logical_cores;
            socket.model = info.model;  // Same model for now
            info.sockets.push_back(socket);
            socket_models.push_back(socket.model);
        }
        
        // Combine model names if different
        bool all_same = true;
        for (size_t i = 1; i < socket_models.size(); ++i) {
            if (socket_models[i] != socket_models[0]) {
                all_same = false;
                break;
            }
        }
        
        if (!all_same) {
            info.model = "";
            for (size_t i = 0; i < socket_models.size(); ++i) {
                if (i > 0) info.model += "; ";
                info.model += socket_models[i];
            }
        }
    } else {
        // Single socket system
        SocketInfo socket;
        socket.socket_id = 0;
        socket.model = info.model;
        socket.logical_cores = info.logical_cores;
        socket.physical_cores = info.physical_cores;
        for (unsigned i = 0; i < info.logical_cores; ++i) {
            socket.core_ids.push_back(i);
        }
        info.sockets.push_back(socket);
    }
    
    return info;
}

#elif defined(__linux__)
// Linux implementation

static std::string read_proc_cpuinfo_field(const std::string& field) {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    
    while (std::getline(cpuinfo, line)) {
        if (line.find(field) == 0) {
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                std::string value = line.substr(colon_pos + 1);
                // Trim leading whitespace
                size_t start = value.find_first_not_of(" \t");
                if (start != std::string::npos) {
                    return value.substr(start);
                }
                return value;
            }
        }
    }
    
    return "";
}

static unsigned get_physical_cores_linux() {
    // Try to read from /sys/devices/system/cpu
    std::ifstream core_count("/sys/devices/system/cpu/cpu0/topology/core_siblings_list");
    if (core_count.is_open()) {
        // Count unique physical IDs
        std::ifstream online("/sys/devices/system/cpu/online");
        if (online.is_open()) {
            // Parse the online CPUs and count unique core_ids
            unsigned physical = 0;
            std::string core_id_prev = "";
            
            for (unsigned i = 0; i < get_logical_core_count(); ++i) {
                std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/core_id";
                std::ifstream core_id_file(path);
                if (core_id_file.is_open()) {
                    std::string core_id;
                    core_id_file >> core_id;
                    // Simple heuristic: count unique core IDs
                    // This is a simplification; proper implementation would track all unique IDs
                }
            }
        }
    }
    
    // Fallback: try to parse from /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    unsigned cpu_cores = 0;
    unsigned siblings = 0;
    
    while (std::getline(cpuinfo, line)) {
        if (line.find("cpu cores") == 0) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                cpu_cores = std::stoul(line.substr(colon + 1));
            }
        }
        if (line.find("siblings") == 0) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                siblings = std::stoul(line.substr(colon + 1));
            }
        }
    }
    
    if (cpu_cores > 0 && siblings > 0) {
        // Calculate physical cores based on logical cores and hyperthreading ratio
        unsigned logical = get_logical_core_count();
        unsigned ht_ratio = siblings / cpu_cores;
        if (ht_ratio > 0) {
            return logical / ht_ratio;
        }
    }
    
    // Final fallback: assume no hyperthreading
    return get_logical_core_count();
}

CpuInfo get_cpu_info() {
    CpuInfo info;
    
    info.arch = get_arch_string();
    info.logical_cores = get_logical_core_count();
    info.physical_cores = get_physical_cores_linux();
    
    // Read vendor and model from /proc/cpuinfo
    std::string vendor_id = read_proc_cpuinfo_field("vendor_id");
    std::string model_name = read_proc_cpuinfo_field("model name");
    
    // ARM processors may use different field names
    if (vendor_id.empty()) {
        vendor_id = read_proc_cpuinfo_field("CPU implementer");
        if (vendor_id.empty()) {
            vendor_id = "Unknown";
        }
    }
    
    if (model_name.empty()) {
        model_name = read_proc_cpuinfo_field("Hardware");
        if (model_name.empty()) {
            model_name = read_proc_cpuinfo_field("Processor");
            if (model_name.empty()) {
                // Fallback: return arch with core count
                unsigned logical = get_logical_core_count();
                unsigned physical = get_physical_cores_linux();
                model_name = get_fallback_cpu_name(get_arch_string(), physical, logical);
            }
        }
    }
    
    info.vendor = vendor_id;
    info.model = model_name;
    info.cache = get_cache_info();
    
    // Multi-socket detection
    info.socket_count = get_socket_count();
    
    if (info.socket_count > 1) {
        // For multi-socket systems, collect per-socket info
        std::vector<std::string> socket_models;
        
        for (unsigned s = 0; s < info.socket_count; ++s) {
            SocketInfo socket;
            socket.socket_id = s;
            socket.core_ids = get_cores_for_socket(s);
            socket.logical_cores = static_cast<unsigned>(socket.core_ids.size());
            socket.physical_cores = socket.logical_cores / 2;  // Assume HT
            if (socket.physical_cores == 0) socket.physical_cores = socket.logical_cores;
            socket.model = model_name;  // Same model for now
            info.sockets.push_back(socket);
            socket_models.push_back(socket.model);
        }
        
        // Combine model names if different
        bool all_same = true;
        for (size_t i = 1; i < socket_models.size(); ++i) {
            if (socket_models[i] != socket_models[0]) {
                all_same = false;
                break;
            }
        }
        
        if (!all_same) {
            info.model = "";
            for (size_t i = 0; i < socket_models.size(); ++i) {
                if (i > 0) info.model += "; ";
                info.model += socket_models[i];
            }
        }
    } else {
        // Single socket system
        SocketInfo socket;
        socket.socket_id = 0;
        socket.model = info.model;
        socket.logical_cores = info.logical_cores;
        socket.physical_cores = info.physical_cores;
        for (unsigned i = 0; i < info.logical_cores; ++i) {
            socket.core_ids.push_back(i);
        }
        info.sockets.push_back(socket);
    }
    
    return info;
}

#elif defined(__APPLE__)
// macOS implementation

#include <sys/sysctl.h>

// Forward declarations for macOS
static unsigned get_physical_cores_macos();

// Apple Silicon model identifier to human-readable name mapping
// Sources: Apple Support, EveryMac, Geekbench Browser
static std::string map_apple_silicon_model(const std::string& hw_model) {
    // === Mac mini ===
    if (hw_model == "Macmini9,1") return "Apple M1";
    if (hw_model == "Mac14,3") return "Apple M2";
    if (hw_model == "Mac14,12") return "Apple M2 Pro";
    if (hw_model == "Mac16,10") return "Apple M4";       // Mac mini (2024)
    if (hw_model == "Mac16,11") return "Apple M4 Pro";   // Mac mini (2024)
    
    // === MacBook Air ===
    if (hw_model == "MacBookAir10,1") return "Apple M1";
    if (hw_model == "Mac14,2") return "Apple M2";
    if (hw_model == "Mac14,15") return "Apple M2";       // 15-inch
    if (hw_model == "Mac15,12") return "Apple M3";
    if (hw_model == "Mac15,13") return "Apple M3";       // 15-inch
    
    // === MacBook Pro ===
    if (hw_model == "MacBookPro17,1") return "Apple M1";
    if (hw_model == "MacBookPro18,1") return "Apple M1 Pro";
    if (hw_model == "MacBookPro18,2") return "Apple M1 Max";
    if (hw_model == "MacBookPro18,3") return "Apple M1 Pro";
    if (hw_model == "MacBookPro18,4") return "Apple M1 Max";
    if (hw_model == "Mac14,5") return "Apple M2 Pro";
    if (hw_model == "Mac14,6") return "Apple M2 Max";
    if (hw_model == "Mac14,9") return "Apple M2 Pro";
    if (hw_model == "Mac14,10") return "Apple M2 Max";
    if (hw_model == "Mac15,3") return "Apple M3 Pro";
    if (hw_model == "Mac15,6") return "Apple M3 Max";
    if (hw_model == "Mac15,7") return "Apple M3 Max";
    if (hw_model == "Mac15,8") return "Apple M3 Pro";
    if (hw_model == "Mac15,9") return "Apple M3 Max";
    if (hw_model == "Mac15,10") return "Apple M3 Pro";
    if (hw_model == "Mac15,11") return "Apple M3 Max";
    // M4 MacBook Pro (14" and 16")
    if (hw_model == "Mac16,1") return "Apple M4 Pro";    // MacBook Pro 14" M4 Pro
    if (hw_model == "Mac16,2") return "Apple M4 Max";    // MacBook Pro 14" M4 Max
    if (hw_model == "Mac16,3") return "Apple M4 Max";    // MacBook Pro 16" M4 Max
    if (hw_model == "Mac16,5") return "Apple M4";        // MacBook Pro 14" M4
    if (hw_model == "Mac16,6") return "Apple M4 Pro";    // MacBook Pro 16" M4 Pro
    if (hw_model == "Mac16,7") return "Apple M4 Max";    // MacBook Pro 14" M4 Max
    if (hw_model == "Mac16,8") return "Apple M4 Max";    // MacBook Pro 16" M4 Max
    
    // === iMac ===
    if (hw_model == "iMac21,1") return "Apple M1";
    if (hw_model == "iMac21,2") return "Apple M1";
    if (hw_model == "Mac15,4") return "Apple M3";
    if (hw_model == "Mac15,5") return "Apple M3";
    
    // === Mac Studio ===
    if (hw_model == "Mac13,1") return "Apple M1 Max";
    if (hw_model == "Mac13,2") return "Apple M1 Ultra";
    if (hw_model == "Mac14,13") return "Apple M2 Max";
    if (hw_model == "Mac14,14") return "Apple M2 Ultra";
    
    // === Mac Pro ===
    if (hw_model == "Mac14,8") return "Apple M2 Ultra";
    
    // If not found in mapping, return with raw model identifier
    // This is more honest than guessing wrong
    if (hw_model.find("Mac16") == 0) return "Apple M4 Series (" + hw_model + ")";
    if (hw_model.find("Mac15") == 0) return "Apple M3 Series (" + hw_model + ")";
    if (hw_model.find("Mac14") == 0) return "Apple M2 Series (" + hw_model + ")";
    if (hw_model.find("Mac13") == 0) return "Apple M1 Series (" + hw_model + ")";
    
    // Return the raw model identifier if no mapping found
    return "Apple Silicon (" + hw_model + ")";
}

static std::string get_cpu_model_macos() {
#if defined(__aarch64__) || defined(_M_ARM64)
    // Apple Silicon detection strategy:
    // 1. Try machdep.cpu.brand_string first (most reliable if available)
    // 2. Fall back to hw.model + mapping table
    
    // Try brand_string first - this gives direct chip name on some systems
    char brand_string[256] = {0};
    size_t len = sizeof(brand_string);
    if (sysctlbyname("machdep.cpu.brand_string", brand_string, &len, nullptr, 0) == 0) {
        std::string brand(brand_string);
        // Check if it's a valid Apple chip name (not empty, contains "Apple")
        if (!brand.empty() && brand.find("Apple") != std::string::npos) {
            return brand;
        }
    }
    
    // Fall back to hw.model and map to human-readable name
    char hw_model[256] = {0};
    len = sizeof(hw_model);
    if (sysctlbyname("hw.model", hw_model, &len, nullptr, 0) == 0) {
        return map_apple_silicon_model(std::string(hw_model));
    }
    
    // Fallback: return arch with core count 
    return get_fallback_cpu_name(get_arch_string(), 
                                  get_physical_cores_macos(), 
                                  get_logical_core_count());
#else
    // Intel Mac: use machdep.cpu.brand_string
    char brand[256] = {0};
    size_t len = sizeof(brand);
    
    if (sysctlbyname("machdep.cpu.brand_string", brand, &len, nullptr, 0) == 0) {
        // Trim leading spaces
        std::string result(brand);
        size_t start = result.find_first_not_of(' ');
        if (start != std::string::npos) {
            return result.substr(start);
        }
        return result;
    }
    
    // Fallback: return arch with core count
    return get_fallback_cpu_name(get_arch_string(), 
                                  get_physical_cores_macos(), 
                                  get_logical_core_count());
#endif
}

static std::string get_cpu_vendor_macos() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "Apple";
#else
    // Intel Mac: get vendor from sysctl
    char vendor[256] = {0};
    size_t len = sizeof(vendor);
    
    if (sysctlbyname("machdep.cpu.vendor", vendor, &len, nullptr, 0) == 0) {
        return std::string(vendor);
    }
    
    return "Intel";
#endif
}

static unsigned get_physical_cores_macos() {
    int physical_cores = 0;
    size_t len = sizeof(physical_cores);
    
    if (sysctlbyname("hw.physicalcpu", &physical_cores, &len, nullptr, 0) == 0) {
        return static_cast<unsigned>(physical_cores);
    }
    
    // Fallback to logical cores
    return get_logical_core_count();
}

CpuInfo get_cpu_info() {
    CpuInfo info;
    
    info.arch = get_arch_string();
    info.logical_cores = get_logical_core_count();
    info.physical_cores = get_physical_cores_macos();
    info.vendor = get_cpu_vendor_macos();
    info.model = get_cpu_model_macos();
    info.cache = get_cache_info();
    
    // Multi-socket detection (macOS typically single socket)
    info.socket_count = get_socket_count();
    
    // Single socket system (typical for macOS)
    SocketInfo socket;
    socket.socket_id = 0;
    socket.model = info.model;
    socket.logical_cores = info.logical_cores;
    socket.physical_cores = info.physical_cores;
    for (unsigned i = 0; i < info.logical_cores; ++i) {
        socket.core_ids.push_back(i);
    }
    info.sockets.push_back(socket);
    
    return info;
}

#else
// Fallback implementation for other platforms

CpuInfo get_cpu_info() {
    CpuInfo info;
    
    info.arch = get_arch_string();
    info.logical_cores = get_logical_core_count();
    info.physical_cores = get_logical_core_count(); // Assume no hyperthreading
    info.vendor = "Unknown";
    // Fallback: return arch with core count 
    info.model = get_fallback_cpu_name(info.arch, info.physical_cores, info.logical_cores);
    info.cache = get_cache_info();
    info.socket_count = 1;
    
    // Single socket
    SocketInfo socket;
    socket.socket_id = 0;
    socket.model = info.model;
    socket.logical_cores = info.logical_cores;
    socket.physical_cores = info.physical_cores;
    for (unsigned i = 0; i < info.logical_cores; ++i) {
        socket.core_ids.push_back(i);
    }
    info.sockets.push_back(socket);
    
    return info;
}

#endif

// ============================================================================
// RAM Detection Implementation 
// ============================================================================

// Get total system RAM in bytes 
size_t get_total_ram() {
#ifdef _WIN32
    // Windows: use GlobalMemoryStatusEx
    MEMORYSTATUSEX mem_info;
    mem_info.dwLength = sizeof(MEMORYSTATUSEX);
    
    if (GlobalMemoryStatusEx(&mem_info)) {
        return static_cast<size_t>(mem_info.ullTotalPhys);
    }
    return 0;
    
#elif defined(__linux__)
    // Linux: read from /proc/meminfo
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") == 0) {
            // Parse "MemTotal:       16384000 kB"
            std::istringstream iss(line);
            std::string label;
            size_t value;
            std::string unit;
            
            iss >> label >> value >> unit;
            
            // Convert to bytes (value is in kB)
            return value * 1024;
        }
    }
    return 0;
    
#elif defined(__APPLE__)
    // macOS: use sysctl hw.memsize
    int64_t mem_size = 0;
    size_t len = sizeof(mem_size);
    
    if (sysctlbyname("hw.memsize", &mem_size, &len, nullptr, 0) == 0) {
        return static_cast<size_t>(mem_size);
    }
    return 0;
    
#else
    // Unknown platform
    return 0;
#endif
}

// Get available system RAM in bytes 
size_t get_available_ram() {
#ifdef _WIN32
    // Windows: use GlobalMemoryStatusEx
    MEMORYSTATUSEX mem_info;
    mem_info.dwLength = sizeof(MEMORYSTATUSEX);
    
    if (GlobalMemoryStatusEx(&mem_info)) {
        return static_cast<size_t>(mem_info.ullAvailPhys);
    }
    return 0;
    
#elif defined(__linux__)
    // Linux: read from /proc/meminfo
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    
    while (std::getline(meminfo, line)) {
        if (line.find("MemAvailable:") == 0) {
            // Parse "MemAvailable:   12345678 kB"
            std::istringstream iss(line);
            std::string label;
            size_t value;
            std::string unit;
            
            iss >> label >> value >> unit;
            
            // Convert to bytes (value is in kB)
            return value * 1024;
        }
    }
    
    // Fallback: try MemFree if MemAvailable is not present (older kernels)
    meminfo.clear();
    meminfo.seekg(0);
    
    while (std::getline(meminfo, line)) {
        if (line.find("MemFree:") == 0) {
            std::istringstream iss(line);
            std::string label;
            size_t value;
            std::string unit;
            
            iss >> label >> value >> unit;
            return value * 1024;
        }
    }
    return 0;
    
#elif defined(__APPLE__)
    // macOS: use vm_statistics64 for available memory
    // Note: macOS doesn't have a simple "available" memory concept
    // We'll use free + inactive pages as an approximation
    
    // First get page size
    int64_t page_size = 0;
    size_t len = sizeof(page_size);
    if (sysctlbyname("hw.pagesize", &page_size, &len, nullptr, 0) != 0) {
        page_size = 4096; // Default page size
    }
    
    // Get VM statistics
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm_stat;
    
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, 
                          reinterpret_cast<host_info64_t>(&vm_stat), &count) == KERN_SUCCESS) {
        // Available = free + inactive (pages that can be reclaimed)
        size_t available = (vm_stat.free_count + vm_stat.inactive_count) * 
                           static_cast<size_t>(page_size);
        return available;
    }
    
    // Fallback: return 0 if we can't get the info
    return 0;
    
#else
    // Unknown platform
    return 0;
#endif
}
