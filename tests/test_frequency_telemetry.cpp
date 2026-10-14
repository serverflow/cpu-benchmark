#include "platform.hpp"
#include "server_performance.hpp"
#include "result_submission.hpp"
#include "compute_benchmark.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

static void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        check(std::abs(estimate_effective_frequency_mhz(4101, 123) - 5044.23) < 0.001,
              "Turbo percentage must not be capped at 100");
        check(estimate_effective_frequency_mhz(4101, 100) == 4101, "Nominal clock");
        check(estimate_effective_frequency_mhz(4101, 50) == 2050.5, "Below-nominal clock");
        for (double invalid : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity()}) {
            check(estimate_effective_frequency_mhz(invalid, 123) == 0, "Invalid nominal clock");
            check(estimate_effective_frequency_mhz(4101, invalid) == 0, "Invalid performance ratio");
        }
        check(estimate_effective_frequency_mhz(4101, 100000) == 0, "Reject implausible counter value");

        FrequencyStats st;
        st.available = true;
        st.estimated = true;
        st.source = "windows_pdh_processor_performance";
        st.logical_cpu = 0;
        st.min_mhz = 4900; st.max_mhz = 5100; st.avg_mhz = 5000; st.sample_count = 2;
        FrequencyStats mt = st;
        mt.logical_cpu = -1;
        mt.min_mhz = 4500; mt.max_mhz = 4700; mt.avg_mhz = 4600; mt.sample_count = 4;
        FrequencyStats combined = merge_frequency_stats(st, mt);
        check(combined.available && combined.estimated && combined.sample_count == 6, "Merge availability/count");
        check(combined.min_mhz == 4500 && combined.max_mhz == 5100, "Merge extrema");
        check(std::abs(combined.avg_mhz - 4733.3333333) < 0.001, "Sample-weighted average");
        check(!merge_frequency_stats(FrequencyStats{}, FrequencyStats{}).available, "Unavailable stays unavailable");
        check(merge_frequency_stats(FrequencyStats{}, st).avg_mhz == st.avg_mhz, "One valid phase");

        WindowsPowerDiagnostics power;
        check(windows_power_warning(power).empty(), "Unknown policy must not be reported as disabled boost");
        power.boost_mode = 2; power.max_processor_state_percent = 100; power.frequency_limit_mhz = 0;
        check(windows_power_warning(power).empty(), "Unlimited plan must not warn");
        power.boost_mode = 0;
        check(windows_power_warning(power).find("boost disabled") != std::string::npos, "Disabled boost warning");
        power.boost_mode = 2; power.max_processor_state_percent = 99;
        check(windows_power_warning(power).find("below 100%") != std::string::npos, "CPU state warning");
        power.max_processor_state_percent = 100; power.frequency_limit_mhz = 4100;
        check(windows_power_warning(power).find("4100 MHz") != std::string::npos, "Frequency cap warning");

#ifdef _WIN32
        check(sample_current_frequency(1000000) == 0, "Missing counters must not fall back to static MHz");
        FrequencySampler unavailable(1000000);
        unavailable.sample();
        check(!unavailable.get_stats().available, "Missing counters must produce unavailable stats");
        for (unsigned i = 0; i < 100; ++i) {
            FrequencySampler stoppable(1000000);
            stoppable.start_background(1000);
            const auto stop_start = std::chrono::steady_clock::now();
            stoppable.stop_background();
            check(std::chrono::duration<double>(std::chrono::steady_clock::now() - stop_start).count() < 0.5,
                  "Stopping sampler must not wait for the next polling interval");
        }
        ScopedServerPerformance diagnostic_only(false);
        check(!diagnostic_only.status().active && diagnostic_only.status().power.checked,
              "Diagnostics must work without server tuning");
#endif

        ComputeSubmissionData compute;
        compute.frequency = combined; compute.st_frequency = st; compute.mt_frequency = mt;
        compute.windows_power = power;
        Config config;
        CpuInfo cpu;
        cpu.model = "test CPU"; cpu.arch = "x86_64"; cpu.physical_cores = 8; cpu.logical_cores = 16;
        const std::string payload = result_submission::build_compute_payload(compute, config, cpu, "local test");
        check(payload.find("\"st_frequency\":" + frequency_stats_json(st)) != std::string::npos, "ST telemetry submission");
        check(payload.find("\"mt_frequency\":" + frequency_stats_json(mt)) != std::string::npos, "MT telemetry submission");
        check(payload.find("\"frequency\":" + frequency_stats_json(combined)) != std::string::npos, "Compatibility summary");
        check(payload.find("\"windows_power\":" + windows_power_json(power)) != std::string::npos, "Power diagnostics submission");
        FullBenchmarkSubmissionData full;
        full.compute = compute;
        const std::string full_payload = result_submission::build_full_benchmark_payload(full, config, cpu, "local test");
        check(full_payload.find("\"st_frequency\":" + frequency_stats_json(st)) != std::string::npos, "Full suite ST telemetry");
        check(full_payload.find("\"mt_frequency\":" + frequency_stats_json(mt)) != std::string::npos, "Full suite MT telemetry");
        ComputeBenchmarkResults results{};
        results.frequency = combined; results.st_frequency = st; results.mt_frequency = mt;
        results.windows_power = power;
        const std::string local_json = ComputeBenchmark::format_json(results);
        check(local_json.find("\"st_frequency\": " + frequency_stats_json(st)) != std::string::npos, "Local JSON ST frequency");
        check(local_json.find("\"mt_frequency\": " + frequency_stats_json(mt)) != std::string::npos, "Local JSON MT frequency");
        const std::string text = ComputeBenchmark::format_text(results);
        check(text.find("Single Freq (estimated):") != std::string::npos, "Text must label estimated ST clock");
        check(text.find("Multi Freq (estimated):") != std::string::npos, "Text must label estimated MT clock");
        // Print JSON for an additional strict-parser check; never send any results.
        std::cout << payload << '\n' << full_payload << '\n' << local_json;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
