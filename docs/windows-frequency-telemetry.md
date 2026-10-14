# Windows frequency telemetry fix (local beta 0.48 candidate)

## What was wrong

Windows `Processor Frequency`, `CallNtPowerInformation.CurrentMhz` and registry
`~MHz` can stay at nominal frequency on CPUs with autonomous frequency control.
Recording those values as observed clock speeds produces misleading constant
4101 MHz / 4300 MHz results. This does not demonstrate that turbo is disabled.

## Measurement

- Estimate effective active-clock MHz as nominal registry `~MHz` multiplied by
  the matching logical CPU's `% Processor Performance` divided by 100.
- Use `PDH_FMT_NOCAP100`: performance percentages above 100 are normal with boost.
- Single samples the actual pinned ST logical processor. When ST affinity is
  disabled or unsuccessful, it uses the aggregate estimate instead.
- Multi averages per-logical-CPU estimates weighted by `% Processor Time`.
  This avoids weighting idle cores equally and preserves per-CPU nominal clocks
  on hybrid CPUs. It is a system-wide estimate, not exclusively this process's
  clock: other active processes can contribute.
- Sample Windows rate counters at one-second intervals. Keep Single and Multi
  separate; exclude warmup and informational SIMD phases.
- Keep the existing safety restriction on multi-socket/multi-group Windows.
  Missing/invalid counters produce unavailable data, never a static-MHz fallback.
- Mark the estimate with `estimated=true` and
  `source=windows_pdh_processor_performance`. It is not a hardware clock read;
  averaging, nominal-clock calibration and counters can cause small deviations
  from advertised boost MHz. Maximum advertised boost is not guaranteed under
  every workload or on every core.

Linux/macOS frequency providers, compute kernels and score formula are unchanged.
The compute phase sampling windows are now separated on all platforms.

## Power policy

Read the active AC/DC policy on all Windows editions, including when server tuning
is disabled. Report boost mode, maximum processor state and frequency limit.
Unknown settings are `-1`. Warn about disabled boost, a state below 100%, or a
nonzero frequency cap. These are policy checks, not a complete diagnosis of BIOS,
thermal/current limits, battery behavior or Windows power-mode overlays.

Workstation Windows plans are not changed. The existing temporary Server tuning
and restoration behavior is preserved.

## Compatibility and testing

`frequency.min_mhz/max_mhz/avg_mhz/available` remains compatible with the existing
submission schema. Compute JSON and compute/full-suite submission payloads also
include `st_frequency`, `mt_frequency`, sample counts, source and estimate flags,
plus `windows_power` diagnostics. An older server may ignore additional fields;
the frontend/backend must consume them to show phase-specific clocks. This patch
does not edit or deploy the website. Historical submitted frequencies cannot be
reconstructed and are not modified.

Standalone tests do not depend on the absent legacy RapidCheck test suite:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PORTABLE=ON -DBUILD_TESTS=OFF -DBUILD_TELEMETRY_TESTS=ON -DCMAKE_POLICY_DEFAULT_CMP0091=NEW
cmake --build build
ctest --test-dir build --output-on-failure
```

Tests cover above/below nominal clocks, invalid data, weighted summary merging,
unknown/limited power policies, unavailable Windows counters, prompt sampler
shutdown, text labels and local/compute/full-suite JSON telemetry. They do not
submit results or change power settings.

`CMP0091=NEW` ensures the project's static-runtime setting takes effect with
MSVC despite the existing CMake 3.14 minimum. The delivered portable executable
imports only Windows system DLLs, not MSVCP/VCRUNTIME DLLs.

Manual validation: Windows 10 Pro 22H2 (19045), Ryzen 7 8700F, 8C/16T. A full
compute run including SIMD produced:

| Phase | Score | Estimated MHz range | Estimated average MHz |
| --- | ---: | ---: | ---: |
| Single | 1905 | 5017–5043 | 5037 |
| Multi | 15279 | 4797–4818 | 4801 |

Active power policy: AC, boost mode 2, maximum state 100%, frequency cap 0
(uncapped). The active plan was unchanged before/after testing. Separate phase
JSON and compute/full-suite submission JSON passed strict parser checks.
The missing-counter and power-warning cases were tested without disabling boost
on the user's machine. Windows 11 and Server were not hardware-tested in this run.

Reference: [Microsoft PDH formatting flags](https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcountervalue)
and [rate-counter collection](https://learn.microsoft.com/en-us/windows/win32/perfctrs/collecting-performance-data).
