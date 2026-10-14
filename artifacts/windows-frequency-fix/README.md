# Windows frequency-fix candidate

`SFBench-beta-0.48-windows-x64-frequency-fix.exe` is beta 0.48 plus the
frequency/power-diagnostics patch. It is not the previously published beta 0.48
release. Source and this candidate binary are published on `main`; existing
release artifacts and tags are preserved. No new GitHub Release is created.

Portable x64 Release, MSVC 19.50, static C++ runtime, runtime SIMD dispatch,
OpenMP OFF. Only Windows system DLLs are imported; no Visual C++ runtime
redistributable is required.

SHA-256: `97F718A081B624A26C6F6ABB9B628C7A41F74FBA65EA47C13B5F8523F7559C87`

Final binary validation on Ryzen 7 8700F / Windows 10 Pro 22H2:

| Phase | Score | Estimated MHz range | Average estimated MHz |
| --- | ---: | ---: | ---: |
| Single | 1869 | 4993.69–5033.14 | 5013 |
| Multi | 15152 | 4796.98–4797.57 | 4797.06 |

Boost is active. The previous constant 4101 MHz was misleading telemetry, not
evidence of a disabled turbo. Frequency values remain OS-counter estimates.
The active power plan and scoring formula are unchanged. Single-run scores can
vary with background activity and temperature; these runs are functional checks,
not a statistical before/after performance comparison.

See [verification.json](verification.json) for the final binary's test record and
[implementation notes](../../docs/windows-frequency-telemetry.md) for measurement
details, limitations, schema compatibility and test/build instructions.

```powershell
.\SFBench-beta-0.48-windows-x64-frequency-fix.exe
```

Results now contain estimated Single/Multi clocks and Windows power-policy checks.
New results retain the old combined frequency fields for server compatibility.
Phase-specific display on the website requires server/frontend support; this
patch changes the client only and does not repair historical submissions.
