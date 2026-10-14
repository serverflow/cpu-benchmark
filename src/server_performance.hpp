#pragma once

#include <memory>
#include <string>
#include <sstream>

// Active power-plan policy, not proof of actual boost or of BIOS/thermal limits.
// -1 means the setting could not be read. Diagnostics never change the plan.
struct WindowsPowerDiagnostics {
    bool checked = false;
    std::string power_source = "unknown";
    int boost_mode = -1;
    int max_processor_state_percent = -1;
    int frequency_limit_mhz = -1;
};

inline std::string windows_power_warning(const WindowsPowerDiagnostics& power) {
    std::string warning;
    auto append = [&](const std::string& message) {
        if (!warning.empty()) warning += "; ";
        warning += message;
    };
    if (power.boost_mode == 0) append("CPU boost disabled in active Windows power plan");
    if (power.max_processor_state_percent >= 0 && power.max_processor_state_percent < 100)
        append("Maximum processor state below 100% in active Windows power plan");
    if (power.frequency_limit_mhz > 0)
        append("Windows power plan frequency cap: " + std::to_string(power.frequency_limit_mhz) + " MHz");
    return warning;
}

inline std::string windows_power_json(const WindowsPowerDiagnostics& power) {
    std::ostringstream out;
    out << "{\"checked\":" << (power.checked ? "true" : "false")
        << ",\"power_source\":\"" << power.power_source << "\",\"boost_mode\":" << power.boost_mode
        << ",\"max_processor_state_percent\":" << power.max_processor_state_percent
        << ",\"frequency_limit_mhz\":" << power.frequency_limit_mhz << "}";
    return out.str();
}

// Status of the temporary Windows Server performance policy used by compute tests.
// The tuning guard remains inactive on workstations; policy diagnostics still run.
struct ServerPerformanceStatus {
    bool server_os = false;
    bool requested = false;
    bool active = false;
    bool high_performance_scheme = false;
    bool high_qos = false;
    bool high_priority = false;
    std::string warning;
    WindowsPowerDiagnostics power;
};

// Returns true for Windows Server editions (including Server Core and domain controllers).
bool is_windows_server_os();

// Applies HighQoS and high thread priority to the calling benchmark worker.
// The setting only lives for the lifetime of that worker thread.
bool configure_current_thread_for_performance();

// Temporarily optimizes Windows Server for a compute benchmark and restores the previous
// process/thread settings and power scheme when it leaves scope.
class ScopedServerPerformance {
public:
    explicit ScopedServerPerformance(bool enable);
    ~ScopedServerPerformance();

    ScopedServerPerformance(const ScopedServerPerformance&) = delete;
    ScopedServerPerformance& operator=(const ScopedServerPerformance&) = delete;
    ScopedServerPerformance(ScopedServerPerformance&&) = delete;
    ScopedServerPerformance& operator=(ScopedServerPerformance&&) = delete;

    const ServerPerformanceStatus& status() const { return status_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    ServerPerformanceStatus status_;
};
